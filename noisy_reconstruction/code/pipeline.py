"""
pipeline.py
===========
Thin end-to-end driver for the noisy kSZ reconstruction pipeline.

Responsibilities:
    - Load a 21cm cube from one of two backends (Grizzly / 21cmFAST).
    - Apply 21cm noise via noise_analysis.noise_filters.
    - Run velocity reconstruction using either:
        * the quadratic estimator from ksz_reconstruction.functions.quadratic_estimator, or
        * the ML model from code/ml_model.py (to be written).
    - Build the patchy kSZ map and add CMB noise from noise_analysis.cmb_noise.
    - Cross-correlate with the full integrated kSZ signal and return r(l).

This is a stub — interfaces to fill in.
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np


class DataBackend(Enum):
    GRIZZLY = "grizzly"
    FAST = "21cmfast"


class Reconstructor(Enum):
    QUADRATIC = "quadratic"
    ML = "ml"


@dataclass
class PipelineConfig:
    backend: DataBackend
    sim_id: str                  # Grizzly sim id or 21cmFAST run name
    redshift: float
    reconstructor: Reconstructor
    noise_21cm: Optional[str] = "SKA1-Low"   # survey key
    noise_cmb:  Optional[str] = "CMB_S4_wide"
    seed: int = 0


def run(cfg: PipelineConfig) -> dict:
    """Run one end-to-end realisation and return diagnostics."""
    # 1. load cube
    # cube, meta = load_cube(cfg.backend, cfg.sim_id, cfg.redshift)

    # 2. add 21cm noise
    # noisy = add_21cm_noise(cube, meta, survey=cfg.noise_21cm, rng=np.random.default_rng(cfg.seed))

    # 3. reconstruct v_z
    # if cfg.reconstructor is Reconstructor.QUADRATIC:
    #     v_rec = quadratic_reconstruct(noisy, meta)
    # else:
    #     v_rec = ml_reconstruct(noisy, meta)

    # 4. build patchy kSZ map, add CMB noise
    # ksz_map = build_ksz_map(noisy, v_rec, meta)
    # ksz_map = add_cmb_noise(ksz_map, meta, survey=cfg.noise_cmb, rng=...)

    # 5. cross-correlate with full integrated kSZ signal
    # r_ell, ell = cross_correlate(ksz_map, full_integrated_ksz(meta))

    raise NotImplementedError("Stub — wire up once noise_analysis and loaders exist.")
