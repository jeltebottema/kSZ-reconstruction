"""
Tests for `ksz_core.reconstruction.reconstruct_velocity`.

The headline check is the plane-wave amplitude: for δ(z) = A·cos(2π m z / L),
the linear continuity equation gives vz(z) = -(aHf/k_z) · A · sin(2π m z / L).
Both the amplitude relation and the sin / -sin phase are verified.
"""
from __future__ import annotations

import numpy as np
import pytest

from ksz_core.cosmology import Constants, H, f_growth
from ksz_core.reconstruction import reconstruct_velocity


# ── trivial sanity ─────────────────────────────────────────────────────────
def test_zero_field_returns_zero_velocity():
    delta = np.zeros((16, 16, 16))
    vz = reconstruct_velocity(delta, box_len_mpc=100.0, z=6.0)
    np.testing.assert_array_equal(vz, np.zeros_like(vz))


def test_constant_field_returns_zero_velocity_after_mean_subtraction():
    delta = np.full((16, 16, 16), 3.7)
    vz = reconstruct_velocity(delta, box_len_mpc=100.0, z=6.0)
    np.testing.assert_allclose(vz, 0.0, atol=1e-12)


def test_invalid_shape_raises():
    with pytest.raises(ValueError, match="cubic"):
        reconstruct_velocity(np.zeros((8, 8, 4)), box_len_mpc=100.0, z=6.0)


def test_invalid_components_raises():
    with pytest.raises(ValueError, match="components"):
        reconstruct_velocity(np.zeros((8, 8, 8)), box_len_mpc=100.0, z=6.0,
                             components="vxyzt")  # type: ignore[arg-type]


# ── linearity ──────────────────────────────────────────────────────────────
def test_linearity_in_delta():
    rng = np.random.default_rng(0)
    delta = rng.standard_normal((16, 16, 16))
    vz_1 = reconstruct_velocity(delta, box_len_mpc=100.0, z=6.0)
    vz_2 = reconstruct_velocity(2.0 * delta, box_len_mpc=100.0, z=6.0)
    np.testing.assert_allclose(vz_2, 2.0 * vz_1, rtol=1e-10, atol=1e-12)


# ── physics: plane-wave amplitude and phase ────────────────────────────────
def test_plane_wave_amplitude_and_phase_planck():
    """For δ(z)=A cos(2π m z / L), expect vz(z) = -(aHf/k_z)·A·sin(2π m z / L)."""
    n = 32
    L = 100.0   # Mpc
    m = 4       # mode index along z
    A = 0.1
    z_val = 6.0
    c = Constants()  # Planck 2018

    iz = np.arange(n)
    delta_1d = A * np.cos(2 * np.pi * m * iz / n)
    delta = np.broadcast_to(delta_1d[None, None, :], (n, n, n)).copy()

    vz = reconstruct_velocity(delta, box_len_mpc=L, z=z_val, c=c)

    a = 1.0 / (1.0 + z_val)
    k_z = 2.0 * np.pi * m / L
    expected_amp = a * H(z_val, c) * f_growth(z_val, c) * A / k_z
    expected_1d = -expected_amp * np.sin(2 * np.pi * m * iz / n)
    expected_3d = np.broadcast_to(expected_1d[None, None, :], (n, n, n))

    np.testing.assert_allclose(vz, expected_3d, rtol=1e-10, atol=1e-12)


def test_plane_wave_amplitude_paper_fiducial():
    """Same physics check, using the paper's fiducial cosmology (Shaw 2025)."""
    n = 32
    L = 100.0
    m = 4
    A = 0.05
    z_val = 8.0
    c = Constants.paper_fiducial()

    iz = np.arange(n)
    delta_1d = A * np.cos(2 * np.pi * m * iz / n)
    delta = np.broadcast_to(delta_1d[None, None, :], (n, n, n)).copy()

    vz = reconstruct_velocity(delta, box_len_mpc=L, z=z_val, c=c)

    a = 1.0 / (1.0 + z_val)
    k_z = 2.0 * np.pi * m / L
    expected_peak = a * H(z_val, c) * f_growth(z_val, c) * A / k_z
    assert np.abs(vz).max() == pytest.approx(expected_peak, rel=1e-6)


# ── components: vxyz output ────────────────────────────────────────────────
def test_vxyz_returns_three_arrays():
    rng = np.random.default_rng(1)
    delta = rng.standard_normal((16, 16, 16))
    out = reconstruct_velocity(delta, box_len_mpc=100.0, z=6.0, components="vxyz")
    assert isinstance(out, tuple) and len(out) == 3
    vx, vy, vz = out
    assert vx.shape == vy.shape == vz.shape == (16, 16, 16)


def test_vxyz_axis_isolation_for_z_only_wave():
    """For δ varying only in z, vx and vy should be zero (k_x=k_y=0 for that mode)."""
    n = 16
    L = 100.0
    m = 3
    iz = np.arange(n)
    delta_1d = 0.1 * np.cos(2 * np.pi * m * iz / n)
    delta = np.broadcast_to(delta_1d[None, None, :], (n, n, n)).copy()

    vx, vy, vz = reconstruct_velocity(delta, box_len_mpc=L, z=6.0, components="vxyz")
    np.testing.assert_allclose(vx, 0.0, atol=1e-12)
    np.testing.assert_allclose(vy, 0.0, atol=1e-12)
    assert np.abs(vz).max() > 0.0


def test_vxyz_vz_matches_components_vz():
    """Asking for 'vz' must produce the same vz as the third element of 'vxyz'."""
    rng = np.random.default_rng(2)
    delta = rng.standard_normal((16, 16, 16))
    vz_only = reconstruct_velocity(delta, box_len_mpc=100.0, z=6.0, components="vz")
    _, _, vz_xyz = reconstruct_velocity(delta, box_len_mpc=100.0, z=6.0,
                                        components="vxyz")
    np.testing.assert_allclose(vz_only, vz_xyz, rtol=1e-14, atol=1e-14)


# ── subtract_mean flag ─────────────────────────────────────────────────────
def test_subtract_mean_false_passes_field_unchanged():
    """With subtract_mean=False, a pre-contrasted δ gives the same answer as
    the same δ shifted by a constant when subtract_mean=True."""
    rng = np.random.default_rng(3)
    delta = rng.standard_normal((16, 16, 16))
    delta -= delta.mean()  # pre-contrast

    vz_a = reconstruct_velocity(delta, box_len_mpc=100.0, z=6.0,
                                subtract_mean=False)
    vz_b = reconstruct_velocity(delta + 5.0, box_len_mpc=100.0, z=6.0,
                                subtract_mean=True)
    np.testing.assert_allclose(vz_a, vz_b, rtol=1e-12, atol=1e-12)
