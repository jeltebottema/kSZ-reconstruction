"""
linear_rec.py
=============
Linear (continuity-equation) v_z reconstruction from a 21cm Tb cube.

    v_z(k) = i * a * H(z) * f(z) * (k_z / k^2) * delta(k)

applied to the mean-subtracted tracer field (Tb - <Tb>). The output
morphology approximates the true v_z; its amplitude is arbitrary
(depends on the unknown tracer bias of delta_Tb vs delta). A scalar
amplitude alpha can be fit on the training set via OLS so that
alpha * vz_linrec is the best scalar match to vz_true.
"""

from __future__ import annotations

import numpy as np

# Planck 2018-ish defaults; override per-call if you use a different cosmology.
H0_DEFAULT = 67.4       # km/s/Mpc
OMEGA_M_DEFAULT = 0.315


def _H_kms_mpc(z: float, H0: float, omega_m: float) -> float:
    return H0 * np.sqrt(omega_m * (1.0 + z) ** 3 + (1.0 - omega_m))


def _f_growth(z: float, omega_m: float) -> float:
    a = 1.0 / (1.0 + z)
    om_a = (omega_m / a ** 3) / (omega_m / a ** 3 + (1.0 - omega_m))
    return om_a ** 0.55


def reconstruct_vz_from_tb(
    tb: np.ndarray,
    z: float,
    box_len_mpc: float,
    H0: float = H0_DEFAULT,
    omega_m: float = OMEGA_M_DEFAULT,
) -> np.ndarray:
    """Linear v_z reconstruction (km/s, up to an unknown amplitude)."""
    tb = np.asarray(tb, dtype=np.float64)
    assert tb.ndim == 3 and tb.shape[0] == tb.shape[1] == tb.shape[2], \
        f"expected cubic box, got {tb.shape}"
    n = tb.shape[0]

    delta_tb = tb - tb.mean()

    k = np.fft.fftfreq(n, d=box_len_mpc / n) * 2.0 * np.pi   # 1/Mpc
    kz = k[None, None, :]
    k2 = (k[:, None, None] ** 2 + k[None, :, None] ** 2 + k[None, None, :] ** 2)
    k2[0, 0, 0] = 1.0   # guard; DC mode zeroed after

    a = 1.0 / (1.0 + z)
    H = _H_kms_mpc(z, H0, omega_m)
    f = _f_growth(z, omega_m)
    prefactor = 1j * a * H * f                              # km/s/Mpc

    delta_k = np.fft.fftn(delta_tb)
    vz_k = prefactor * (kz / k2) * delta_k
    vz_k[0, 0, 0] = 0.0
    vz = np.fft.ifftn(vz_k).real
    return vz.astype(np.float32)


def fit_alpha(vz_true: np.ndarray, vz_lin: np.ndarray) -> float:
    """Least-squares scalar amplitude:  alpha = <y, x> / <x, x>."""
    x = vz_lin.ravel().astype(np.float64)
    y = vz_true.ravel().astype(np.float64)
    denom = float((x * x).sum())
    if denom < 1e-30:
        return 0.0
    return float((x * y).sum() / denom)
