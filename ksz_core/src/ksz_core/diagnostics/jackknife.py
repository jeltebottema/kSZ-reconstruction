"""Spatial block jackknife for Pearson r.

Partition the volume into N = n_per_side^D sub-volumes, compute leave-one-out
estimates by summing N-1 blocks, and report

    σ_jk² = (N-1)/N · Σ_i (x_i − x̄)²

with x̄ the mean across the N leave-one-out estimates.

For Pearson r we use the additive-sum trick: per-block sums of A, B, A², B², AB
let the leave-one-out r be reconstructed in O(N) without re-flattening big
arrays.

Scope: only the analyzer-free pieces (`block_sums`, `pearson_from_sums`,
`jackknife_pearson_r`) live here. The r(k) variants in the paper repo depend on
the not-yet-extracted Fourier-correlation machinery and stay there for now;
they migrate when `ksz_core.diagnostics.power` lands.
"""
from __future__ import annotations

import numpy as np


def block_sums(field_a, field_b, n_per_side: int) -> np.ndarray:
    """Per-block sums of A, B, A², B², AB and cell count.

    Returns array of shape (N, 6) where N = n_per_side**D and columns are
    (n_cells, sum(a), sum(b), sum(a*a), sum(b*b), sum(a*b)).

    Works for 2D or 3D fields.
    """
    shape = field_a.shape
    splits = [np.array_split(np.arange(s), n_per_side) for s in shape]
    blocks = []
    if len(shape) == 3:
        for ix in splits[0]:
            for iy in splits[1]:
                for iz in splits[2]:
                    a = field_a[ix[0]:ix[-1] + 1, iy[0]:iy[-1] + 1,
                                iz[0]:iz[-1] + 1].astype(np.float64, copy=False)
                    b = field_b[ix[0]:ix[-1] + 1, iy[0]:iy[-1] + 1,
                                iz[0]:iz[-1] + 1].astype(np.float64, copy=False)
                    blocks.append((a.size, a.sum(), b.sum(),
                                   (a * a).sum(), (b * b).sum(), (a * b).sum()))
    elif len(shape) == 2:
        for ix in splits[0]:
            for iy in splits[1]:
                a = field_a[ix[0]:ix[-1] + 1, iy[0]:iy[-1] + 1].astype(
                    np.float64, copy=False)
                b = field_b[ix[0]:ix[-1] + 1, iy[0]:iy[-1] + 1].astype(
                    np.float64, copy=False)
                blocks.append((a.size, a.sum(), b.sum(),
                               (a * a).sum(), (b * b).sum(), (a * b).sum()))
    else:
        raise ValueError(f"Only 2D/3D fields supported, got shape {shape}")
    return np.array(blocks, dtype=np.float64)


def pearson_from_sums(n: float, sa: float, sb: float,
                      saa: float, sbb: float, sab: float) -> float:
    """Pearson r from pre-aggregated sums. Returns NaN on zero variance."""
    num = n * sab - sa * sb
    da = n * saa - sa * sa
    db = n * sbb - sb * sb
    if da <= 0 or db <= 0:
        return float("nan")
    return float(num / np.sqrt(da * db))


def jackknife_pearson_r(field_a, field_b, n_per_side: int = 2
                        ) -> tuple[float, float, np.ndarray]:
    """Leave-one-out jackknife on Pearson r between two equal-shape fields.

    Works for 2D or 3D; uses n_per_side**D blocks. Returns
    (mean of leave-one-out estimates, jackknife σ, leave-one-out values).
    """
    blocks = block_sums(field_a, field_b, n_per_side)
    total = blocks.sum(axis=0)
    N = blocks.shape[0]
    loo = np.empty(N)
    for i in range(N):
        s = total - blocks[i]
        loo[i] = pearson_from_sums(*s)
    mean = float(np.nanmean(loo))
    sigma = float(np.sqrt((N - 1) / N * np.nansum((loo - mean) ** 2)))
    return mean, sigma, loo
