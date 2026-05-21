"""
Linear continuity-equation velocity reconstruction.

Implements the first-order relation between matter density and peculiar velocity:

    v_i(k) = i * a * H(z) * f(z) * k_i / k² * δ(k)

where δ is the density contrast (mean-subtracted ρ/ρ̄).

Unifies two prior implementations:
- `ksz_reconstruction/functions/quadratic_estimator.py:1202-1252` (rfft, all 3 components)
- `noisy_reconstruction/code/ml/linear_rec.py:34-63` (fftn, vz-only)

Uses the rfft strategy (half the work for real inputs) with the parameter-driven
cosmology from `ksz_core.cosmology`.

The output velocity is in km/s when `delta` is dimensionless and `box_len_mpc`
is in Mpc.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
from scipy import fft

from ksz_core.cosmology import Constants, H, f_growth


def reconstruct_velocity(
    delta: np.ndarray,
    box_len_mpc: float,
    z: float,
    c: Constants = Constants(),
    components: Literal["vz", "vxyz"] = "vz",
    subtract_mean: bool = True,
) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reconstruct the peculiar velocity field from a 3D density-tracer cube.

    Parameters
    ----------
    delta
        Cubic 3D array. Treated as a density-contrast tracer; if
        `subtract_mean=True` (default), its mean is removed first so the
        function accepts either pre-contrasted δ or a raw tracer (1+δ, Tb, etc.).
    box_len_mpc
        Box side length in Mpc.
    z
        Redshift.
    c
        Cosmology. Defaults to Planck 2018 — pass `Constants.paper_fiducial()`
        to match the published kSZ × 21cm paper (Shaw 2025 values).
    components
        ``"vz"`` returns only the line-of-sight component (3D array).
        ``"vxyz"`` returns a tuple ``(vx, vy, vz)``.
    subtract_mean
        Subtract the spatial mean from `delta` before transforming. Default True.

    Returns
    -------
    Velocity in km/s.
    """
    delta = np.asarray(delta)
    if delta.ndim != 3 or delta.shape[0] != delta.shape[1] or delta.shape[1] != delta.shape[2]:
        raise ValueError(f"expected cubic 3D array, got shape {delta.shape}")
    n = delta.shape[0]

    field = delta - delta.mean() if subtract_mean else delta
    delta_k = fft.rfftn(field, workers=-1).astype(np.complex128)

    dk = 2.0 * np.pi / box_len_mpc
    kx = dk * np.fft.fftfreq(n, d=1.0 / n)
    ky = dk * np.fft.fftfreq(n, d=1.0 / n)
    kz = dk * np.fft.rfftfreq(n, d=1.0 / n)

    k2 = (kx[:, None, None] ** 2
          + ky[None, :, None] ** 2
          + kz[None, None, :] ** 2)
    k2[0, 0, 0] = 1.0  # guard against div-by-zero; DC mode zeroed below

    a = 1.0 / (1.0 + z)
    prefactor = 1j * a * H(z, c) * f_growth(z, c)  # km/s/Mpc

    if components == "vz":
        vz_k = prefactor * kz[None, None, :] / k2 * delta_k
        vz_k[0, 0, 0] = 0.0
        return fft.irfftn(vz_k, s=(n, n, n), workers=-1).real.astype(np.float64)

    if components == "vxyz":
        vx_k = prefactor * kx[:, None, None] / k2 * delta_k
        vy_k = prefactor * ky[None, :, None] / k2 * delta_k
        vz_k = prefactor * kz[None, None, :] / k2 * delta_k
        vx_k[0, 0, 0] = 0.0
        vy_k[0, 0, 0] = 0.0
        vz_k[0, 0, 0] = 0.0
        vx = fft.irfftn(vx_k, s=(n, n, n), workers=-1).real.astype(np.float64)
        vy = fft.irfftn(vy_k, s=(n, n, n), workers=-1).real.astype(np.float64)
        vz = fft.irfftn(vz_k, s=(n, n, n), workers=-1).real.astype(np.float64)
        return vx, vy, vz

    raise ValueError(f"components must be 'vz' or 'vxyz', got {components!r}")
