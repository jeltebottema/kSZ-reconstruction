"""
Patchy kinetic Sunyaev-Zel'dovich (kSZ) signal from 21cmFAST output.

Two routes:

1. :func:`ksz_from_coeval` -- simple 2D kSZ proxy from a single coeval
   cube (integrate along the box z-axis).
2. :func:`ksz_from_lightcone` -- full kSZ calculation on a 21cmFAST
   lightcone. :func:`ksz_squared_from_lightcone` exposes the squared
   variant.

The lightcone kSZ implementation is self-contained: the custom
``run_kSZ`` / ``run_kSZ_sq`` / ``_Proj_array`` / ``_KszConstants`` /
``KSZOutput`` have been inlined here from the modified
``py21cmfast.wrapper``. The only piece still imported from stock
py21cmfast is :func:`py21cmfast.wrapper.compute_tau`, which calls the
underlying C extension.

No project-level config dependency; pass parameters explicitly from
the notebook.
"""

from __future__ import annotations

import logging
import random
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
from astropy import constants
from astropy.cosmology import Planck18

logger = logging.getLogger(__name__)


# =====================================================================
# Coeval-cube kSZ output
# =====================================================================
@dataclass
class CoevalKSZMap:
    """2D kSZ map from a single coeval cube."""
    redshift: float
    mean_xHI: float
    dT_map: np.ndarray          # 2D kSZ temperature map (K)
    tau_map: np.ndarray         # 2D optical depth map
    BOX_LEN: float              # cMpc


# =====================================================================
# 1) Simple coeval-cube kSZ
# =====================================================================
def ksz_from_coeval(
    coeval,
    box_len: float,
    hlittle: float = 0.6766,
    OMb: float = 0.04897,
    Y_He: float = 0.245,
) -> CoevalKSZMap:
    """Compute a simple 2D patchy-kSZ map from one coeval cube.

    LOS is the z-axis of the cube; we integrate

        dT/T = -(sigma_T * N_b0 * dR) sum_z (1+delta) x_e (1+z)^2 v_z/c

    and subtract the mean to keep only the patchy fluctuation.

    Parameters
    ----------
    coeval : py21cmfast Coeval
        Must expose ``density``, ``velocity`` (LOS peculiar velocity),
        ``xH_box``, ``redshift``.
    box_len : float
        Comoving box side (cMpc).
    hlittle, OMb, Y_He : cosmology.

    Returns
    -------
    CoevalKSZMap
    """
    delta = np.asarray(coeval.density)
    v_los = np.asarray(coeval.velocity)
    xHI = np.asarray(coeval.xH_box)
    z = float(coeval.redshift)

    N = delta.shape[0]
    dR_cMpc = box_len / N
    cm_per_Mpc = constants.kpc.cgs.value * 1e3
    dR_cm = dR_cMpc * cm_per_Mpc

    rho_b0_cgs = (
        3.0 * (hlittle * 3.2407e-18) ** 2
        / (8.0 * np.pi * constants.G.cgs.value)
        * OMb
        / constants.m_p.cgs.value
    )
    N_b0 = rho_b0_cgs * (1.0 - 0.75 * Y_He) + rho_b0_cgs * Y_He / 4.0

    A = N_b0 * constants.sigma_T.cgs.value * dR_cm
    x_e = (1.0 - xHI) + Y_He / 4.0

    dtau_3d = A * (1.0 + delta) * x_e * (1.0 + z) ** 2  # per cell

    # Robust v/c conversion: 21cmFAST can emit velocity in Mpc/s, m/s
    # or cm/s depending on build.
    c_cgs = constants.c.cgs.value
    scale = np.max(np.abs(v_los)) if v_los.size else 0.0
    if scale > 1e6:            # cm/s
        v_over_c = v_los / c_cgs
    elif scale > 1e2:          # m/s
        v_over_c = v_los * 1e2 / c_cgs
    elif scale > 0:            # Mpc/s or similar
        v_over_c = v_los * cm_per_Mpc / c_cgs
    else:
        v_over_c = np.zeros_like(v_los)

    dT_over_T = -dtau_3d * v_over_c
    dT_map_over_T = np.sum(dT_over_T, axis=2)
    tau_map = np.sum(dtau_3d, axis=2)

    T_cmb = Planck18.Tcmb0.value
    dT_map = dT_map_over_T * T_cmb
    dT_map -= np.mean(dT_map)

    return CoevalKSZMap(
        redshift=z,
        mean_xHI=float(np.mean(xHI)),
        dT_map=dT_map,
        tau_map=tau_map,
        BOX_LEN=box_len,
    )


# =====================================================================
# 2) Lightcone kSZ -- self-contained (inlined from custom wrapper)
# =====================================================================
class _KszConstants:
    """Constants used for kSZ calculation."""

    def __init__(
        self, HII_DIM, BOX_LEN, hlittle, OMb, red_dist, redshift_start, DA_zstart, Y_He
    ):
        RHOb_cgs = (
            3.0
            * (hlittle * 3.2407e-18) ** 2
            / (8.0 * np.pi * constants.G.cgs.value)
            * OMb
            / constants.m_p.cgs.value
        )  # pcm^-3 at z=0
        self.He_No = RHOb_cgs * Y_He / 4.0
        self.N_0 = RHOb_cgs * (1 - 0.75 * Y_He)
        self.N_b0 = self.He_No + self.N_0
        self.dR = BOX_LEN / HII_DIM
        self.CMperMPC = constants.kpc.cgs.value * 1e3
        self.A = self.N_b0 * constants.sigma_T.cgs.value * self.dR * self.CMperMPC
        self.HII_DIM = HII_DIM
        self.BOX_LEN = BOX_LEN
        self.red_dist = red_dist
        self.redshift_start = redshift_start
        self.DA_zstart = DA_zstart
        self.Y_He = Y_He


class KSZOutput:
    """Output class for kSZ effect."""

    def __init__(
        self, kSZ_box, taue, l_s=None, kSZ_power=None, cosmo_params=None, err=None
    ):
        self.kSZ_box = kSZ_box
        self.taue = taue
        self.l_s = l_s
        self.kSZ_power = kSZ_power
        self.cosmo_params = cosmo_params
        self.err = err


def _Proj_array(
    redshifts,
    density,
    velocity,
    xH,
    kSZ_consts,
    PARALLEL_APPROX=False,
    rotation=False,
):
    """Project the lightcone along the LOS into a 2D kSZ map.

    Two modes:

    - Fully analytic (``not PARALLEL_APPROX and not rotation``):
      cumulative tau, exp(-tau) weighting in one shot.
    - Slab-by-slab (either flag set): loops along the lightcone, with
      optional ray-tracing shift (``not PARALLEL_APPROX``) and random
      box rotation every ``HII_DIM`` slabs (``rotation``).
    """
    dtau_3d = (
        kSZ_consts.A * (1.0 + density) * (1.0 + kSZ_consts.Y_He / 4 - xH)
    )  # tau_e contribution (no (1+z)^2 yet)
    if not (PARALLEL_APPROX or rotation):
        # Cumulative tau along LOS (axis=2). `redshifts` is 1D, so
        # broadcasting against the (N, N, red_dist) cube is along the
        # last axis.
        taue_arry = (
            np.cumsum(dtau_3d * (1 + redshifts) ** 2, axis=2)
            + kSZ_consts.mean_taue_curr_z
        )
        # Per-cell kSZ contribution, exp(-tau)-weighted
        Tcmb_3d = dtau_3d * velocity * (1 + redshifts) * np.exp(-taue_arry)
        # Sum along LOS to get a 2D kSZ map, matching the slab branches.
        Tcmb = np.sum(Tcmb_3d, axis=2)
        # Keep only the final cumulative tau map (2D). The original
        # `taue_arry[-1]` indexed the wrong axis; use axis=2.
        taue_arry = taue_arry[..., -1]
    else:
        inc = 1
        inc_displacement = kSZ_consts.dR / kSZ_consts.DA_zstart
        Tcmb_3d = (
            kSZ_consts.A * velocity * (1.08 - xH) * (1.0 + density)
        )  # tcmb contribution (no (1+z) yet)
        taue_arry = np.full(
            (kSZ_consts.HII_DIM, kSZ_consts.HII_DIM), kSZ_consts.mean_taue_curr_z
        )
        Tcmb = np.zeros((kSZ_consts.HII_DIM, kSZ_consts.HII_DIM))
        for k in range(kSZ_consts.red_dist):
            dtau_new = dtau_3d[:, :, k] * (1 + redshifts[k]) ** 2
            Tcmb_new = Tcmb_3d[:, :, k] * (1 + redshifts[k])
            if not PARALLEL_APPROX:
                a = np.round(
                    np.arange(-kSZ_consts.HII_DIM / 2, kSZ_consts.HII_DIM / 2) * inc
                    + kSZ_consts.HII_DIM * 3 / 2
                ).astype(int)
                inc += inc_displacement.value  # increment for ray tracing
                dtau_new = np.take(dtau_new, a, axis=0, mode="wrap")
                dtau_new = np.take(dtau_new, a, axis=1, mode="wrap")
                Tcmb_new = np.take(Tcmb_new, a, axis=0, mode="wrap")
                Tcmb_new = np.take(Tcmb_new, a, axis=1, mode="wrap")
            if rotation:
                if k % kSZ_consts.HII_DIM == 0:
                    tx = int(kSZ_consts.HII_DIM * random.random())
                    ty = int(kSZ_consts.HII_DIM * random.random())
                dtau_new = np.roll(dtau_new, -tx, 0)
                dtau_new = np.roll(dtau_new, -ty, 1)
                Tcmb_new = np.roll(Tcmb_new, -tx, 0)
                Tcmb_new = np.roll(Tcmb_new, -ty, 1)
            taue_arry += dtau_new
            Tcmb += Tcmb_new * np.exp(-taue_arry)
    mean_taue_fin = np.mean(taue_arry)
    Tcmb = Tcmb - np.mean(Tcmb)
    return Tcmb, mean_taue_fin


def _run_kSZ_impl(
    lc,
    z_start: float,
    PARALLEL_APPROX: bool,
    rotation: bool,
    random_seed: int,
    squared: bool,
):
    """Shared implementation for run_kSZ and run_kSZ_sq.

    Inlined from ``py21cmfast.wrapper.run_kSZ`` / ``run_kSZ_sq`` so no
    custom wrapper install is required. ``compute_tau`` is still taken
    from the stock py21cmfast build because it calls the C extension.
    """
    # Stock py21cmfast dependency
    from py21cmfast.wrapper import compute_tau
    # powerbox is installed via requirements.txt
    from powerbox import get_power

    if lc is None:
        raise ValueError("ksz_from_lightcone requires a lightcone object.")

    user_params = lc.user_params
    cosmo_params = lc.cosmo_params

    random.seed(random_seed)

    kSZ_consts = _KszConstants(
        user_params.HII_DIM,
        user_params.BOX_LEN,
        cosmo_params.hlittle,
        cosmo_params.OMb,
        len(lc.lightcone_redshifts),
        z_start,
        lc.lightcone_distances[0],
        0.245,  # Helium fraction
    )

    kSZ_consts.mean_taue_curr_z = compute_tau(
        redshifts=[z_start],
        global_xHI=[1],
        user_params=user_params,
        cosmo_params=cosmo_params,
    )

    Tcmb, mean_taue_fin = _Proj_array(
        lc.lightcone_redshifts,
        lc.density,
        lc.velocity,
        lc.xH_box,
        kSZ_consts,
        PARALLEL_APPROX=PARALLEL_APPROX,
        rotation=rotation,
    )

    CM = kSZ_consts.CMperMPC
    c_cgs = constants.c.cgs.value
    Tcmb0 = Planck18.Tcmb0.value

    if not squared:
        # Linear kSZ map -> map in K, power spectrum in uK^2 (l^2 C_l / 2pi)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            P_k, l_s, err = get_power(
                Tcmb * CM / c_cgs * Tcmb0 / np.sqrt(2 * np.pi),
                user_params.BOX_LEN,
                bins=30,
                log_bins=True,
                get_variance=True,
            )
        l_s *= lc.lightcone_distances[0].value
        P_k = P_k * l_s ** 2
        err = np.sqrt(err) * l_s ** 2

        kSZ_box = Tcmb * CM / c_cgs * Tcmb0
    else:
        # Squared kSZ
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            P_k, l_s, err = get_power(
                (Tcmb * CM / c_cgs) ** 2,
                user_params.BOX_LEN,
                bins=30,
                log_bins=True,
                get_variance=True,
            )
        l_s *= lc.lightcone_distances[0].value
        P_k = P_k / (2 * np.pi) * l_s ** 2 * Tcmb0 ** 4
        err = np.sqrt(err) * l_s ** 2

        kSZ_box = (Tcmb * CM / c_cgs * Tcmb0) ** 2

    mask = np.logical_not(np.isnan(l_s))
    return KSZOutput(
        kSZ_box,
        mean_taue_fin,
        l_s=l_s[mask],
        kSZ_power=P_k[mask],
        err=err[mask],
    )


def ksz_from_lightcone(
    lc,
    rotation: bool = True,
    z_start: float = 5.5,
    parallel_approx: bool = False,
    random_seed: int = 1,
):
    """Patchy kSZ from a 21cmFAST lightcone.

    Inlined from the custom ``py21cmfast.wrapper.run_kSZ`` so the
    package works against a stock ``py21cmfast`` install.

    Parameters
    ----------
    lc : py21cmfast LightCone
        Must expose ``density``, ``velocity`` (LOS), ``xH_box``,
        ``lightcone_redshifts``, ``lightcone_distances``,
        ``user_params``, ``cosmo_params``.
    rotation : bool
        If True, shift each box by a random (tx, ty) every HII_DIM
        slabs to suppress replication artifacts.
    z_start : float
        Starting redshift of the LOS integration.
    parallel_approx : bool
        If True, skip the ray-tracing shift (cheaper, less accurate).
    random_seed : int
        Seed for the rotation RNG.

    Returns
    -------
    KSZOutput with fields ``kSZ_box``, ``taue``, ``l_s``,
    ``kSZ_power``, ``err``.
    """
    logger.info(
        "ksz_from_lightcone: z_start=%.2f, rotation=%s, PARALLEL_APPROX=%s, seed=%d",
        z_start, rotation, parallel_approx, random_seed,
    )
    return _run_kSZ_impl(
        lc=lc,
        z_start=z_start,
        PARALLEL_APPROX=parallel_approx,
        rotation=rotation,
        random_seed=random_seed,
        squared=False,
    )


def ksz_squared_from_lightcone(
    lc,
    rotation: bool = True,
    z_start: float = 5.5,
    parallel_approx: bool = False,
    random_seed: int = 1,
):
    """Squared-kSZ variant, inlined from
    ``py21cmfast.wrapper.run_kSZ_sq``.
    """
    logger.info(
        "ksz_squared_from_lightcone: z_start=%.2f, rotation=%s, PARALLEL_APPROX=%s, seed=%d",
        z_start, rotation, parallel_approx, random_seed,
    )
    return _run_kSZ_impl(
        lc=lc,
        z_start=z_start,
        PARALLEL_APPROX=parallel_approx,
        rotation=rotation,
        random_seed=random_seed,
        squared=True,
    )
