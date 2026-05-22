"""
noise.py
========
HERA observational noise for 21cm coeval cubes, via

    tuesday.core.observe_coeval

This follows the exact pattern used in
`noise_analysis/notebooks/04_grizzly_uv_sweep.ipynb`. `observe_coeval`
handles uv-sampling + thermal noise, returning a noisy observed Tb cube.

The HERA `Observation` object is built via `ksz_core.noise.hera.
build_hera_observation` — single source of truth shared with the
Wiener-filter / paper pipeline.
"""

from __future__ import annotations

import numpy as np
from astropy import units as un

from ksz_core.noise.hera import build_hera_observation

from .config import DemoConfig


def build_observation(cfg: DemoConfig):
    """Return (observatory, observation, antpos) for cfg.survey via ksz_core."""
    if cfg.survey != "HERA":
        raise NotImplementedError(
            f"survey={cfg.survey!r} not wired up yet; use 'HERA' for the demo."
        )

    return build_hera_observation(
        hex_num=cfg.hera_hex_num,
        split_core=cfg.hera_split_core,
        outriggers=cfg.hera_outriggers,
        dish_size_m=cfg.hera_dish_size_m,
        latitude_deg=cfg.hera_latitude_deg,
        track_hours=cfg.track_hours,
        time_per_day_hours=cfg.time_per_day_hours,
        n_days=cfg.n_days,
    )


# -----------------------------------------------------------------------------
# Core: observe a coeval Tb cube through HERA
# -----------------------------------------------------------------------------
def observe_Tb(Tb_cube: np.ndarray, z: float, cfg: DemoConfig,
               seed: int | None = None) -> np.ndarray:
    """Pass a clean Tb cube through HERA uv-sampling + thermal noise.

    Returns the noisy observed cube, same shape as `Tb_cube`, in mK,
    with any NaN/inf (from modes outside the uv coverage) zeroed.
    """
    from tuesday.core import observe_coeval

    _, obs, _ = build_observation(cfg)
    seed = int(seed if seed is not None else 0)

    observed = observe_coeval(
        box=Tb_cube.astype(np.float32) * un.mK,
        box_length=cfg.box_len * un.Mpc,
        observation=obs,
        redshift=z,
        nrealizations=1,
        remove_wedge=False,
        seed=seed,
    )
    # observe_coeval returns a realisations axis; take the first.
    if isinstance(observed, (list, tuple)):
        arr = observed[0]
    else:
        arr = observed[0] if getattr(observed, "ndim", 0) == 4 else observed

    bt_noisy = arr.to(un.mK).value if hasattr(arr, "to") else np.asarray(arr)
    bt_noisy = np.asarray(bt_noisy, dtype=np.float32)
    bt_noisy = np.nan_to_num(bt_noisy, nan=0.0, posinf=0.0, neginf=0.0)
    return bt_noisy


# -----------------------------------------------------------------------------
# Public API (kept stable so dataset.py / train.py don't need changes)
# -----------------------------------------------------------------------------
def add_observational_noise(Tb_cube: np.ndarray, z: float, cfg: DemoConfig,
                            seed: int | None = None) -> np.ndarray:
    """Return the HERA-observed version of `Tb_cube`.

    Despite the legacy name, this does **not** do `clean + noise`: it
    hands the whole signal box to `tuesday.core.observe_coeval`, which
    applies uv-sampling and adds thermal noise in one step (matching
    `noise_analysis/notebooks/04`). The returned cube is the quantity
    the U-Net should be trained on.
    """
    return observe_Tb(Tb_cube, z, cfg, seed=seed)


def noise_realisation(shape, z: float, cfg: DemoConfig, seed: int | None = None) -> np.ndarray:
    """Pure-noise cube, for debugging / Wiener-filter calibration.

    Mirrors `empirical_noise_2d` usage in notebook 02: feed a zero box
    and read off the noise realisation.
    """
    zeros = np.zeros(shape, dtype=np.float32)
    return observe_Tb(zeros, z, cfg, seed=seed)
