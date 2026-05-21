"""ksz_core — shared utilities for the kSZ × 21cm research stack."""

from .cosmology import (
    Constants,
    H,
    comoving_distance,
    compute_dtau_dz,
    compute_tau_0_to_z,
    compute_tau_6_to_z,
    f_growth,
    k_to_ell,
)
from .diagnostics import (
    block_sums,
    jackknife_pearson_r,
    pearson_from_sums,
    pearson_r,
)
from .fft import kspace_grid, kspace_rfft

__version__ = "0.2.0"

__all__ = [
    "Constants",
    "H",
    "f_growth",
    "comoving_distance",
    "k_to_ell",
    "compute_dtau_dz",
    "compute_tau_0_to_z",
    "compute_tau_6_to_z",
    "pearson_r",
    "block_sums",
    "pearson_from_sums",
    "jackknife_pearson_r",
    "kspace_rfft",
    "kspace_grid",
]
