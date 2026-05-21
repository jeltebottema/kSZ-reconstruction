"""
Cosmology utilities.

Default `Constants()` is Planck 2018; `Constants.paper_fiducial()` returns the
GRIZZLY simulation values used by the published kSZ × 21cm paper, per
Shaw et al. 2025 §1 (arXiv:2409.03255) "adapted from Hinshaw et al. 2013".
Always pass an explicit `Constants` instance when the choice matters — do not
rely on the default in physics-critical code paths.

    from ksz_core.cosmology import Constants, H, f_growth, comoving_distance

    c = Constants()                       # Planck 2018
    H(z=6, c=c)                           # km/s/Mpc

    paper = Constants.paper_fiducial()    # H0=70, Om0=0.27, Ob0=0.044
    H(z=6, c=paper)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad


C_LIGHT_KMS = 299792.458  # km/s


@dataclass(frozen=True)
class Constants:
    """Cosmological parameters. Default = Planck 2018."""
    H0: float = 67.4         # km/s/Mpc
    Om0: float = 0.315       # matter density today
    Ob0: float = 0.0493      # baryon density today
    h: float = 0.674         # H0 / 100

    @property
    def OL0(self) -> float:
        """Cosmological constant density today (flat ΛCDM: 1 - Om0)."""
        return 1.0 - self.Om0

    @classmethod
    def paper_fiducial(cls) -> "Constants":
        """
        GRIZZLY simulation cosmology used by the published kSZ × 21cm paper.

        Values per Shaw et al. 2025 §1 (arXiv:2409.03255), stated as
        "adapted from Hinshaw et al. (2013)": H0=70, Om0=0.27, Ob0=0.044, h=0.7.
        The N-body chain is PRACE4LOFAR via cubep3m (Harnois-Déraps et al. 2013).
        """
        return cls(H0=70.0, Om0=0.27, Ob0=0.044, h=0.7)


def H(z: float | np.ndarray, c: Constants = Constants()) -> float | np.ndarray:
    """Hubble parameter at redshift z, km/s/Mpc. Flat ΛCDM."""
    return c.H0 * np.sqrt(c.Om0 * (1.0 + z) ** 3 + c.OL0)


def f_growth(z: float | np.ndarray, c: Constants = Constants()) -> float | np.ndarray:
    """Linear growth rate f(z) ≈ Ω_m(z)^0.55."""
    a = 1.0 / (1.0 + z)
    Om_z = (c.Om0 / a ** 3) / (c.Om0 / a ** 3 + c.OL0)
    return Om_z ** 0.55


def comoving_distance(z: float, c: Constants = Constants()) -> float:
    """Comoving distance from z=0 to z, in Mpc. Flat ΛCDM."""
    if z == 0.0:
        return 0.0
    chi, _ = quad(lambda zp: C_LIGHT_KMS / H(zp, c), 0.0, z)
    return chi


def k_to_ell(k: float | np.ndarray, z: float,
             c: Constants = Constants()) -> float | np.ndarray:
    """Convert wavenumber k [h/Mpc] at redshift z to multipole ℓ.

    ℓ = k · χ(z), with k in h/Mpc and χ in Mpc/h. Internally `comoving_distance`
    returns χ in Mpc, so we multiply by h to get Mpc/h before forming the product.
    """
    chi_mpc = comoving_distance(z, c)
    chi_mpch = chi_mpc * c.h
    return k * chi_mpch


# ─── Optical depth ─────────────────────────────────────────────────────────
#
# Thomson optical depth from line-of-sight integral
#     τ = ∫ n_e σ_T dl
# with proper-length element dl = c/[H(z)(1+z)] dz, so
#     dτ/dz = c σ_T n_e(z) / [H(z) (1+z)].
# Mean electron density today (full H+He ionisation):
#     n_e,0 = Ω_b ρ_crit,0 / m_p · (1 − Y_He/2)
# scaled to z as n_e(z) = n_e,0 (1+z)^3 · n_e_factor.

# CGS constants — kept local to this module for unit transparency.
_SIGMA_T_CM2 = 6.6524587158e-25
_M_PROTON_G = 1.6726219e-24
_MPC_TO_CM = 3.085677581e24
_G_CGS = 6.67430e-8
_C_CMS = 2.99792458e10


def compute_dtau_dz(z: float | np.ndarray, n_e_factor: float = 1.0,
                    c: Constants = Constants(),
                    Y_He: float = 0.24) -> float | np.ndarray:
    """Differential Thomson optical depth dτ/dz at redshift z (dimensionless).

    `n_e_factor` multiplies the mean electron density — pass the ionised
    fraction here when computing patchy-reionisation contributions.
    """
    H0_cgs = c.H0 * 1e5 / _MPC_TO_CM  # s^-1
    rho_crit_0 = 3.0 * H0_cgs**2 / (8.0 * np.pi * _G_CGS)  # g/cm^3
    rho_b_0 = c.Ob0 * rho_crit_0
    n_e_0 = rho_b_0 / _M_PROTON_G * (1.0 - Y_He / 2.0)  # cm^-3
    n_e_z = n_e_0 * (1.0 + z) ** 3 * n_e_factor
    E_z = np.sqrt(c.Om0 * (1.0 + z) ** 3 + c.OL0)
    H_z_cgs = H0_cgs * E_z
    return _C_CMS * _SIGMA_T_CM2 * n_e_z / H_z_cgs / (1.0 + z)


def compute_tau_0_to_z(z: float, c: Constants = Constants(),
                       Y_He: float = 0.24) -> float:
    """Thomson τ from z=0 to z, assuming fully ionised IGM."""
    if z == 0.0:
        return 0.0
    tau, _ = quad(lambda zp: compute_dtau_dz(zp, 1.0, c, Y_He), 0.0, z)
    return tau


def compute_tau_6_to_z(z: float, c: Constants = Constants(),
                       Y_He: float = 0.24) -> float:
    """Thomson τ from z=6 to z (=0 below z=6), assuming fully ionised IGM."""
    if z <= 6.0:
        return 0.0
    tau, _ = quad(lambda zp: compute_dtau_dz(zp, 1.0, c, Y_He), 6.0, z)
    return tau
