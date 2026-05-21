"""Diagnostics — correlation and resampling statistics."""

from .correlation import pearson_r
from .jackknife import block_sums, jackknife_pearson_r, pearson_from_sums

__all__ = ["pearson_r", "block_sums", "pearson_from_sums", "jackknife_pearson_r"]
