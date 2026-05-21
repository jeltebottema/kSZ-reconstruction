"""
GRIZZLY (1D radiative transfer) data loading helpers.

GRIZZLY simulations (Ghara et al.) store density, xHI, and velocity for
a sequence of coeval boxes on a 600³ grid in a 500 Mpc/h box.

We crop the central 540³ cells to avoid boundary artefacts.
"""
from __future__ import annotations

import glob
import re
from pathlib import Path

import numpy as np


# ── GRIZZLY simulation constants ──
N_CELL = 600
BOX_MPC_H = 500.0
H_LITTLE = 0.7
BOX_LEN_MPC = BOX_MPC_H / H_LITTLE            # ~714.3 Mpc
OMEGA_M = 0.27
OMEGA_B = 0.045
NC_PM = 13824                                   # PM grid for velocity unit

CROP = slice(30, 570)                            # central 540³
N_CROP = 540
BOX_CROP_MPC = BOX_LEN_MPC * N_CROP / N_CELL    # ~643 Mpc

XHI_PATTERN = ("zeta0.389fesc0.389_Mmin0.120E+10_MminX0.120E+10"
               "_fx0.100E+03_sed3_al1.200xhi.bin")


# ---------------------------------------------------------------------------
# Binary readers (Fortran order, big-box GRIZZLY format)
# ---------------------------------------------------------------------------
def read_den(filename, nx=N_CELL, ny=N_CELL, nz=N_CELL, endian="<"):
    dt = np.dtype(endian + "f4")
    with open(filename, "rb") as f:
        f.seek(12)
        data = np.fromfile(f, dtype=dt, count=nx * ny * nz)
    return data.reshape((nx, ny, nz), order="F")


def read_xhi(filename, nx=N_CELL, ny=N_CELL, nz=N_CELL):
    with open(filename, "rb") as f:
        data = np.fromfile(f, dtype=np.float32, count=nx * ny * nz)
    return data.reshape((nx, ny, nz), order="F")


def read_vel(z, den, filename, n_cell=N_CELL, box=BOX_MPC_H, nc=NC_PM,
             hlittle=H_LITTLE, endian="<"):
    """Return v_z in cm/s (z-component), dividing by density to undo
    GRIZZLY's momentum-like storage."""
    Megaparsec = 3.08568025e24
    Ho = hlittle * 3.2407e-18
    dt = np.dtype(endian + "f4")
    with open(filename, "rb") as f:
        f.seek(12)
        arrv3 = np.fromfile(f, dtype=dt, count=3 * n_cell ** 3)
    arrv3 = arrv3.reshape((3, n_cell, n_cell, n_cell), order="F").astype(np.float32, copy=False)

    len_unit = box * Megaparsec / hlittle / (1.0 + z) / float(nc)
    tau_t = 2.0 / 3.0 / np.sqrt(OMEGA_M * Ho * Ho) / (1.0 + z) ** 2
    vel_unit = len_unit / tau_t

    arrv3 *= np.float32(vel_unit * 8.0)
    eps = np.float32(1e-12)
    den_safe = np.where(den > eps, den, eps)
    vz = (arrv3[2] / den_safe).astype(np.float32, copy=False)
    del arrv3
    return vz


# ---------------------------------------------------------------------------
# Snapshot index
# ---------------------------------------------------------------------------
def available_redshifts(data_dir):
    """Return sorted array of redshifts present in the directory."""
    files = sorted(glob.glob(str(Path(data_dir) / "*n_all.dat")))
    zs = [float(re.search(r"([\d.]+)n_all", f).group(1)) for f in files]
    return np.array(sorted(zs))


def mean_xhi_vs_z(data_dir, redshifts, xhi_pattern=XHI_PATTERN):
    """Mean xHI for each snapshot (uses cropped region)."""
    means = []
    for z in redshifts:
        xhi = read_xhi(str(Path(data_dir) / f"{z:.3f}{xhi_pattern}"))[CROP, CROP, CROP]
        means.append(xhi.mean())
    return np.array(means)


# ---------------------------------------------------------------------------
# Load one snapshot
# ---------------------------------------------------------------------------
def load_snapshot(data_dir, z, xhi_pattern=XHI_PATTERN, crop=True):
    """
    Load (density, xHI, vz_cm_s) for a single GRIZZLY redshift snapshot.

    `density` is ρ/ρ̄ (mean ~ 1 — NOT overdensity δ).
    `xHI` is neutral fraction (0–1).
    `vz_cm_s` is the z-component of peculiar velocity in cm/s.
    """
    data_dir = Path(data_dir)
    den_full = read_den(str(data_dir / f"{z:.3f}n_all.dat"))
    xhi = read_xhi(str(data_dir / f"{z:.3f}{xhi_pattern}"))
    vz = read_vel(z, den_full, str(data_dir / f"{z:.3f}v_all.dat"))

    if crop:
        return den_full[CROP, CROP, CROP], xhi[CROP, CROP, CROP], vz[CROP, CROP, CROP]
    return den_full, xhi, vz


# ---------------------------------------------------------------------------
# 21cm brightness temperature
# ---------------------------------------------------------------------------
def brightness_temp(density, xHI, z, omega_m=OMEGA_M, omega_b=OMEGA_B,
                     h=H_LITTLE):
    """
    High-spin limit 21cm brightness temperature [mK], ignoring velocity gradient:

        T_b ≈ 27 x_HI (1+δ) sqrt((1+z)/10 × 0.15/(Ωm h²)) × (Ωb h²/0.023) [mK]

    Good enough for P(k) shape and noise-filter studies; not absolute-calibrated
    for line-of-sight RSD.
    """
    delta = density / density.mean() - 1.0
    prefactor = 27.0 * (omega_b * h ** 2 / 0.023) \
                * np.sqrt(0.15 / (omega_m * h ** 2) * (1.0 + z) / 10.0)
    return (xHI * (1.0 + delta) * prefactor).astype(np.float32)


def pick_snapshot_near_xhi(data_dir, target_xhi=0.5, xhi_pattern=XHI_PATTERN):
    """Return (z, mean_xhi) for the snapshot whose mean xHI is closest to target."""
    zs = available_redshifts(data_dir)
    xhis = mean_xhi_vs_z(data_dir, zs, xhi_pattern)
    idx = int(np.argmin(np.abs(xhis - target_xhi)))
    return float(zs[idx]), float(xhis[idx])
