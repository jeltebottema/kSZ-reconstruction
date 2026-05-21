"""
config.py
=========
Single source of truth for the ML demo defaults.

HERA parameters match noise_analysis/src/noise_filters.build_hera_observation
so a single Observation object is valid for both the Wiener-filter work
and the ML pipeline.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DemoConfig:
    # --- simulation ---------------------------------------------------------
    hii_dim: int = 128          # cells per side
    box_len: float = 256.0      # Mpc per side -> dx = 2 Mpc
    target_xHI: float = 0.5     # target neutral fraction for the demo
    z_grid: tuple = (6.0, 7.0, 7.5, 8.0, 8.5, 9.0, 10.0)
    n_train: int = 16
    n_val: int = 4
    seed0: int = 1000           # seeds = seed0 .. seed0 + n_train + n_val - 1

    # --- noise (HERA, via py21cmsense + tuesday.core) -----------------------
    survey: str = "HERA"
    hera_hex_num: int = 11
    hera_split_core: bool = True
    hera_outriggers: int = 2
    hera_dish_size_m: float = 14.0
    hera_latitude_deg: float = -30.0
    track_hours: float = 6.0
    time_per_day_hours: float = 6.0
    n_days: int = 180

    # --- training -----------------------------------------------------------
    batch_size: int = 1         # 128^3 is heavy; use batch=1
    epochs: int = 40
    lr: float = 2e-4
    base_channels: int = 16
    device: str = "cuda"        # falls back to cpu automatically

    # --- paths --------------------------------------------------------------
    data_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2] / "data" / "cubes")
    ckpt_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2] / "data" / "checkpoints")

    @property
    def n_total(self) -> int:
        return self.n_train + self.n_val
