"""Tests pin cosmological defaults so future refactors don't silently drift."""
from __future__ import annotations

import math

import numpy as np
import pytest

from ksz_core.cosmology import (
    Constants,
    H,
    comoving_distance,
    compute_dtau_dz,
    compute_tau_0_to_z,
    compute_tau_6_to_z,
    f_growth,
    k_to_ell,
)


# ── Constants ──────────────────────────────────────────────────────────────
def test_planck18_defaults():
    c = Constants()
    assert c.H0 == 67.4
    assert c.Om0 == 0.315
    assert c.Ob0 == 0.0493
    assert c.h == 0.674
    assert c.OL0 == pytest.approx(1.0 - 0.315)


def test_paper_fiducial_defaults():
    c = Constants.paper_fiducial()
    assert c.H0 == 70.0
    assert c.Om0 == 0.27
    assert c.Ob0 == 0.044
    assert c.h == 0.7
    assert c.OL0 == pytest.approx(0.73)


# ── H(z) ───────────────────────────────────────────────────────────────────
def test_H_at_z0_returns_H0_exactly():
    assert H(0.0, Constants()) == pytest.approx(67.4, rel=1e-12)
    assert H(0.0, Constants.paper_fiducial()) == pytest.approx(70.0, rel=1e-12)


def test_H_increases_with_z():
    c = Constants()
    zs = np.array([0.0, 1.0, 6.0, 10.0])
    Hs = H(zs, c)
    assert np.all(np.diff(Hs) > 0)


def test_H_matches_analytic_at_z6_planck():
    c = Constants()
    expected = c.H0 * math.sqrt(c.Om0 * 7.0 ** 3 + c.OL0)
    assert H(6.0, c) == pytest.approx(expected, rel=1e-12)


# ── f_growth(z) ────────────────────────────────────────────────────────────
def test_f_growth_at_z0_equals_Om0_to_055():
    c = Constants()
    assert f_growth(0.0, c) == pytest.approx(c.Om0 ** 0.55, rel=1e-12)


def test_f_growth_approaches_unity_at_high_z():
    """In matter-dominated era, Ω_m(z) → 1 so f → 1."""
    c = Constants()
    assert f_growth(50.0, c) == pytest.approx(1.0, rel=1e-3)


def test_f_growth_array_input():
    c = Constants()
    zs = np.array([0.0, 1.0, 6.0])
    fs = f_growth(zs, c)
    assert fs.shape == zs.shape
    assert np.all(np.diff(fs) > 0)  # increases monotonically in matter era


# ── comoving_distance(z) ───────────────────────────────────────────────────
def test_comoving_distance_zero_at_origin():
    assert comoving_distance(0.0, Constants()) == 0.0


def test_comoving_distance_increases_with_z():
    c = Constants()
    assert comoving_distance(1.0, c) < comoving_distance(2.0, c) < comoving_distance(6.0, c)


def test_comoving_distance_planck_at_z2_matches_astropy():
    """Cross-check against astropy's FlatLambdaCDM with matching parameters."""
    pytest.importorskip("astropy")
    from astropy.cosmology import FlatLambdaCDM

    c = Constants()
    cosmo = FlatLambdaCDM(H0=c.H0, Om0=c.Om0)
    expected = cosmo.comoving_distance(2.0).to("Mpc").value
    assert comoving_distance(2.0, c) == pytest.approx(expected, rel=1e-4)


def test_comoving_distance_paper_fiducial_at_z6_matches_astropy():
    pytest.importorskip("astropy")
    from astropy.cosmology import FlatLambdaCDM

    c = Constants.paper_fiducial()
    cosmo = FlatLambdaCDM(H0=c.H0, Om0=c.Om0)
    expected = cosmo.comoving_distance(6.0).to("Mpc").value
    assert comoving_distance(6.0, c) == pytest.approx(expected, rel=1e-4)


# ── k_to_ell ───────────────────────────────────────────────────────────────
def test_k_to_ell_planck_matches_chi_times_k():
    c = Constants()
    k = 0.1  # h/Mpc
    chi_mpch = comoving_distance(2.0, c) * c.h
    assert k_to_ell(k, 2.0, c) == pytest.approx(k * chi_mpch, rel=1e-12)


def test_k_to_ell_paper_differs_from_planck():
    """Different cosmology -> different chi -> different ell at same k, z."""
    k = 0.1
    z = 2.0
    ell_planck = k_to_ell(k, z, Constants())
    ell_paper = k_to_ell(k, z, Constants.paper_fiducial())
    assert ell_planck != pytest.approx(ell_paper, rel=1e-2)


def test_k_to_ell_array_input():
    ks = np.array([0.01, 0.1, 1.0])
    ells = k_to_ell(ks, 2.0, Constants())
    assert ells.shape == ks.shape
    assert np.all(np.diff(ells) > 0)


# ── Optical depth ──────────────────────────────────────────────────────────
def test_compute_dtau_dz_paper_at_z6_positive():
    val = compute_dtau_dz(6.0, c=Constants.paper_fiducial())
    assert val > 0
    assert np.isfinite(val)


def test_compute_dtau_dz_array_input():
    zs = np.array([0.0, 1.0, 6.0])
    vals = compute_dtau_dz(zs, c=Constants.paper_fiducial())
    assert vals.shape == zs.shape
    assert np.all(vals > 0)


def test_compute_tau_0_to_z_zero_at_origin():
    assert compute_tau_0_to_z(0.0, Constants.paper_fiducial()) == 0.0


def test_compute_tau_0_to_z_paper_in_physical_range():
    """τ(0→6) with WMAP3 baryons + He-doubly-ionised throughout.

    Planck 2018 measures total τ_reion ≈ 0.054 ± 0.007 to z_reion ~ 7-8.
    With paper_fiducial (Ω_b=0.044, lower than Planck's 0.049), τ(0→6) is
    smaller in roughly proportional measure. Verifying the result is finite,
    positive, and in the right neighbourhood — NOT against an external
    hardcoded constant whose Ω_b provenance is unclear.
    """
    c = Constants.paper_fiducial()
    tau_0_6 = compute_tau_0_to_z(6.0, c)
    assert np.isfinite(tau_0_6)
    assert 0.03 < tau_0_6 < 0.07


def test_compute_tau_0_to_z_planck_higher_than_paper():
    """Planck Ω_b > paper Ω_b -> Planck τ > paper τ at same z."""
    z = 6.0
    tau_planck = compute_tau_0_to_z(z, Constants())
    tau_paper = compute_tau_0_to_z(z, Constants.paper_fiducial())
    assert tau_planck > tau_paper


def test_compute_tau_6_to_z_below_z6_is_zero():
    c = Constants.paper_fiducial()
    assert compute_tau_6_to_z(5.0, c) == 0.0
    assert compute_tau_6_to_z(6.0, c) == 0.0


def test_compute_tau_6_to_z_monotone():
    c = Constants.paper_fiducial()
    taus = [compute_tau_6_to_z(z, c) for z in (7.0, 8.0, 10.0, 12.0)]
    assert all(t > 0 for t in taus)
    assert taus[0] < taus[1] < taus[2] < taus[3]
