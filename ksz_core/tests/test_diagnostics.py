"""Tests for diagnostics — pearson_r and jackknife."""
from __future__ import annotations

import numpy as np
import pytest

from ksz_core.diagnostics import (
    block_sums,
    jackknife_pearson_r,
    pearson_from_sums,
    pearson_r,
)


# ── pearson_r ──────────────────────────────────────────────────────────────
def test_pearson_r_self_is_one():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(1000)
    assert pearson_r(x, x) == pytest.approx(1.0, rel=1e-12)


def test_pearson_r_negated_is_minus_one():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(1000)
    assert pearson_r(x, -x) == pytest.approx(-1.0, rel=1e-12)


def test_pearson_r_constant_is_nan():
    x = np.array([1.0, 2.0, 3.0])
    c = np.ones(3)
    assert np.isnan(pearson_r(x, c))


def test_pearson_r_matches_numpy_corrcoef():
    rng = np.random.default_rng(7)
    x = rng.standard_normal(500)
    y = 0.6 * x + 0.4 * rng.standard_normal(500)
    expected = np.corrcoef(x, y)[0, 1]
    assert pearson_r(x, y) == pytest.approx(expected, rel=1e-12)


def test_pearson_r_handles_nan_entries():
    x = np.array([1.0, 2.0, np.nan, 4.0])
    y = np.array([2.0, 4.0, 99.0, 8.0])
    # Linear pair after masking NaN -> r = 1
    assert pearson_r(x, y) == pytest.approx(1.0, rel=1e-12)


def test_pearson_r_all_nan_returns_nan():
    x = np.array([np.nan, np.nan])
    y = np.array([np.nan, np.nan])
    assert np.isnan(pearson_r(x, y))


# ── jackknife ──────────────────────────────────────────────────────────────
def test_block_sums_partition_3d():
    """Sum of all blocks must equal global sum."""
    rng = np.random.default_rng(1)
    a = rng.standard_normal((8, 8, 8))
    b = rng.standard_normal((8, 8, 8))
    blocks = block_sums(a, b, n_per_side=2)
    # Columns: (n, sum_a, sum_b, sum_a2, sum_b2, sum_ab)
    assert blocks.shape == (8, 6)
    assert blocks[:, 0].sum() == a.size
    assert blocks[:, 1].sum() == pytest.approx(a.sum())
    assert blocks[:, 2].sum() == pytest.approx(b.sum())
    assert blocks[:, 3].sum() == pytest.approx((a * a).sum())
    assert blocks[:, 4].sum() == pytest.approx((b * b).sum())
    assert blocks[:, 5].sum() == pytest.approx((a * b).sum())


def test_block_sums_partition_2d():
    rng = np.random.default_rng(2)
    a = rng.standard_normal((8, 8))
    b = rng.standard_normal((8, 8))
    blocks = block_sums(a, b, n_per_side=2)
    assert blocks.shape == (4, 6)
    assert blocks[:, 0].sum() == a.size


def test_pearson_from_sums_matches_pearson_r():
    """Closed-form r from sums should match direct pearson_r."""
    rng = np.random.default_rng(3)
    a = rng.standard_normal(1000)
    b = 0.3 * a + 0.7 * rng.standard_normal(1000)
    n = float(a.size)
    r_sums = pearson_from_sums(
        n, a.sum(), b.sum(),
        (a * a).sum(), (b * b).sum(), (a * b).sum(),
    )
    assert r_sums == pytest.approx(pearson_r(a, b), rel=1e-12)


def test_pearson_from_sums_zero_variance_is_nan():
    n = 4.0
    # b is constant -> sum_b² = n * sum_b² / n => db = 0
    sa, sb = 10.0, 4.0
    saa, sbb = 30.0, 4.0  # 4 ones squared and summed
    sab = 10.0
    assert np.isnan(pearson_from_sums(n, sa, sb, saa, sbb, sab))


def test_jackknife_pearson_r_perfect_correlation():
    """r=1 -> all leave-one-out r values are ~1 -> sigma ~0."""
    rng = np.random.default_rng(4)
    a = rng.standard_normal((8, 8, 8))
    b = a.copy()
    mean, sigma, loo = jackknife_pearson_r(a, b, n_per_side=2)
    assert mean == pytest.approx(1.0, rel=1e-10)
    assert sigma == pytest.approx(0.0, abs=1e-12)
    assert loo.shape == (8,)
    assert np.allclose(loo, 1.0)


def test_jackknife_pearson_r_independent_fields():
    """Independent random fields -> jackknife r near 0 with nonzero sigma."""
    rng = np.random.default_rng(5)
    a = rng.standard_normal((16, 16, 16))
    b = rng.standard_normal((16, 16, 16))
    mean, sigma, loo = jackknife_pearson_r(a, b, n_per_side=2)
    assert abs(mean) < 0.1
    assert sigma > 0
    assert loo.shape == (8,)


def test_jackknife_pearson_r_2d():
    rng = np.random.default_rng(6)
    a = rng.standard_normal((16, 16))
    b = a + 0.1 * rng.standard_normal((16, 16))
    mean, sigma, loo = jackknife_pearson_r(a, b, n_per_side=2)
    assert mean > 0.9
    assert sigma >= 0
    assert loo.shape == (4,)
