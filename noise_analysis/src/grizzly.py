"""
GRIZZLY data loaders — thin shim re-exporting from ksz_core.

This module historically held the canonical GRIZZLY loaders. As of 2026-05-22
those live in `ksz_core.loaders.grizzly` (the monorepo's shared package).
This file re-exports the public surface so existing consumers don't break.

Prefer `from ksz_core.loaders.grizzly import ...` in new code.
"""
from __future__ import annotations

from ksz_core.loaders.grizzly import (  # noqa: F401
    # Constants
    N_CELL,
    BOX_MPC_H,
    H_LITTLE,
    BOX_LEN_MPC,
    OMEGA_M,
    OMEGA_B,
    NC_PM,
    CROP,
    N_CROP,
    BOX_CROP_MPC,
    XHI_PATTERN,
    # Readers
    read_den,
    read_xhi,
    read_vel,
    # Snapshot index
    available_redshifts,
    mean_xhi_vs_z,
    # Load + brightness temperature
    load_snapshot,
    brightness_temp,
    pick_snapshot_near_xhi,
)
