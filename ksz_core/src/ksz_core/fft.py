"""FFT-grid helpers.

`kspace_rfft` builds (kx, ky, kz_half) for use with `numpy.fft.rfftn` on a cube
of side n cells and spacing rc per cell. DC components are guarded against
zero so callers can divide by k² without raising.

`kspace_grid` is the full-grid sibling (`numpy.fft.fftn` shape, no half-axis
truncation) — used where the full conjugate-symmetric grid is needed.
"""
from __future__ import annotations

import numpy as np


def kspace_rfft(n: int, rc: float, dtype=np.float32
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """k-space axes for a cube of side n cells, cell spacing rc.

    Returns (kx, ky, kz_half). Last axis uses `rfftfreq` so its length is
    n//2 + 1. DC components are floored to `np.finfo(dtype).tiny`.
    """
    kx = (2.0 * np.pi * np.fft.fftfreq(n, d=rc)).astype(dtype)
    ky = (2.0 * np.pi * np.fft.fftfreq(n, d=rc)).astype(dtype)
    kz = (2.0 * np.pi * np.fft.rfftfreq(n, d=rc)).astype(dtype)
    tiny = np.finfo(dtype).tiny
    if kx.size:
        kx[0] = max(kx[0], tiny)
    if ky.size:
        ky[0] = max(ky[0], tiny)
    if kz.size:
        kz[0] = max(kz[0], tiny)
    return kx, ky, kz


def kspace_grid(n: int, rc: float, dtype=np.float32
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full-grid k-space axes for a cube of side n cells.

    Same as `kspace_rfft` but the last axis is `fftfreq` (full length n).
    """
    kx = (2.0 * np.pi * np.fft.fftfreq(n, d=rc)).astype(dtype)
    ky = (2.0 * np.pi * np.fft.fftfreq(n, d=rc)).astype(dtype)
    kz = (2.0 * np.pi * np.fft.fftfreq(n, d=rc)).astype(dtype)
    tiny = np.finfo(dtype).tiny
    if kx.size:
        kx[0] = max(kx[0], tiny)
    if ky.size:
        ky[0] = max(ky[0], tiny)
    if kz.size:
        kz[0] = max(kz[0], tiny)
    return kx, ky, kz
