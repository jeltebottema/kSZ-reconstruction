"""Tests for FFT-grid helpers."""
from __future__ import annotations

import numpy as np
import pytest

from ksz_core.fft import kspace_grid, kspace_rfft


def test_kspace_rfft_shape():
    n = 16
    kx, ky, kz = kspace_rfft(n, rc=1.0)
    assert kx.shape == (n,)
    assert ky.shape == (n,)
    assert kz.shape == (n // 2 + 1,)


def test_kspace_rfft_dc_floored_not_zero():
    """DC components are floored to dtype.tiny so callers can divide by k^2."""
    kx, ky, kz = kspace_rfft(8, rc=1.0)
    assert kx[0] > 0
    assert ky[0] > 0
    assert kz[0] > 0


def test_kspace_rfft_matches_numpy_rfftfreq():
    """kz axis (away from DC) should match 2π · rfftfreq(n, d=rc)."""
    n = 32
    rc = 0.5
    _, _, kz = kspace_rfft(n, rc, dtype=np.float64)
    expected = 2.0 * np.pi * np.fft.rfftfreq(n, d=rc)
    # Skip index 0 since it's floored, not 0.
    assert np.allclose(kz[1:], expected[1:], rtol=1e-12)


def test_kspace_rfft_dtype_respected():
    kx, _, _ = kspace_rfft(8, rc=1.0, dtype=np.float64)
    assert kx.dtype == np.float64


def test_kspace_grid_shape():
    n = 16
    kx, ky, kz = kspace_grid(n, rc=1.0)
    assert kx.shape == (n,)
    assert ky.shape == (n,)
    assert kz.shape == (n,)


def test_kspace_grid_matches_numpy_fftfreq():
    n = 16
    rc = 1.0
    _, _, kz = kspace_grid(n, rc, dtype=np.float64)
    expected = 2.0 * np.pi * np.fft.fftfreq(n, d=rc)
    assert np.allclose(kz[1:], expected[1:], rtol=1e-12)


def test_kspace_rfft_half_axis_is_half_of_full():
    """rfft axis has length n//2 + 1 vs full's n."""
    n = 16
    _, _, kz_half = kspace_rfft(n, rc=1.0)
    _, _, kz_full = kspace_grid(n, rc=1.0)
    assert kz_half.size == n // 2 + 1
    assert kz_full.size == n
