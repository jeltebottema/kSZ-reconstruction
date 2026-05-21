"""Correlation diagnostics."""
from __future__ import annotations

import numpy as np


def pearson_r(a, b) -> float:
    """Voxel-wise Pearson correlation between two equal-shape arrays.

    Inputs are flattened. NaN/inf entries are ignored pairwise. Returns NaN
    if no finite pairs remain or either side has zero variance.
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    m = np.isfinite(a) & np.isfinite(b)
    if not np.any(m):
        return float("nan")
    a = a[m] - a[m].mean()
    b = b[m] - b[m].mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else float("nan")
