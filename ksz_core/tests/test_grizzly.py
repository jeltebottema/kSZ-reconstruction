"""Smoke tests for the GRIZZLY binary readers.

Uses a tiny synthetic binary that matches the file format
(12-byte header + N³ float32 array in Fortran order).
"""
from __future__ import annotations

import numpy as np
import pytest

from ksz_core.loaders.grizzly import brightness_temp, read_den, read_xhi


def _write_den_file(path, arr):
    """Write a synthetic GRIZZLY-style density file: 12-byte header + F-order f4."""
    with open(path, "wb") as f:
        f.write(b"\x00" * 12)
        arr.astype(np.float32).ravel(order="F").tofile(f)


def _write_xhi_file(path, arr):
    """xHI files have no header — just F-order f4."""
    arr.astype(np.float32).ravel(order="F").tofile(path)


def test_read_den_roundtrip(tmp_path):
    n = 8
    rng = np.random.default_rng(42)
    expected = rng.random((n, n, n)).astype(np.float32)

    path = tmp_path / "test_den.bin"
    _write_den_file(path, expected)

    actual = read_den(str(path), nx=n, ny=n, nz=n)
    assert actual.shape == (n, n, n)
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual, expected)


def test_read_xhi_roundtrip(tmp_path):
    n = 8
    rng = np.random.default_rng(7)
    expected = rng.random((n, n, n)).astype(np.float32)

    path = tmp_path / "test_xhi.bin"
    _write_xhi_file(path, expected)

    actual = read_xhi(str(path), nx=n, ny=n, nz=n)
    assert actual.shape == (n, n, n)
    np.testing.assert_array_equal(actual, expected)


def test_read_den_fortran_order_preserved(tmp_path):
    """Element at (i, j, k) should land at the same position after round-trip."""
    n = 4
    expected = np.arange(n ** 3, dtype=np.float32).reshape((n, n, n), order="F")

    path = tmp_path / "test.bin"
    _write_den_file(path, expected)

    actual = read_den(str(path), nx=n, ny=n, nz=n)
    np.testing.assert_array_equal(actual, expected)
    # spot-check: element at (1, 2, 3) should round-trip
    assert actual[1, 2, 3] == expected[1, 2, 3]


def test_brightness_temp_returns_expected_shape_and_sign():
    """Saturation limit: positive, dimensionally sensible, right shape."""
    n = 16
    density = np.ones((n, n, n), dtype=np.float32) + 0.1 * np.random.default_rng(1).standard_normal((n, n, n)).astype(np.float32)
    xHI = np.full((n, n, n), 0.5, dtype=np.float32)
    tb = brightness_temp(density, xHI, z=8.0)
    assert tb.shape == (n, n, n)
    assert tb.dtype == np.float32
    # T_b should be a few mK at z=8 with xHI=0.5; sanity bounds:
    assert 1.0 < tb.mean() < 50.0
    assert np.all(np.isfinite(tb))


def test_brightness_temp_zero_when_fully_ionized():
    n = 8
    density = np.ones((n, n, n), dtype=np.float32)
    xHI = np.zeros((n, n, n), dtype=np.float32)
    tb = brightness_temp(density, xHI, z=8.0)
    np.testing.assert_array_equal(tb, np.zeros_like(tb))
