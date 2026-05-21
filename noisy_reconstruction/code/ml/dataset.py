"""
dataset.py
==========
PyTorch Dataset over the (noisy Tb, v_z) pairs saved by simulate.py.

    x = noisy Tb cube  (1, N, N, N)
    y = v_z cube       (1, N, N, N)

Both are z-score standardised using statistics computed on the training
set, so the targets the U-Net sees are O(1). The stats are stashed
alongside the dataset so predictions can be un-scaled for evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import DemoConfig
from .noise import observe_Tb   # HERA uv-sampling + thermal noise (tuesday.core)
from .linear_rec import reconstruct_vz_from_tb, fit_alpha


@dataclass
class NormStats:
    x_mean: float
    x_std:  float
    y_mean: float
    y_std:  float

    def to_dict(self) -> dict:
        return dict(x_mean=self.x_mean, x_std=self.x_std,
                    y_mean=self.y_mean, y_std=self.y_std)

    @classmethod
    def from_dict(cls, d) -> "NormStats":
        return cls(**{k: float(d[k]) for k in ("x_mean", "x_std", "y_mean", "y_std")})


def compute_norm_stats(paths: Sequence[Path], cfg: DemoConfig) -> NormStats:
    """Running mean / std over the training set (sim cubes + their noise)."""
    xs, ys = [], []
    for p in paths:
        d = np.load(p)
        Tb_noisy = observe_Tb(d["Tb"], float(d["z"]), cfg, seed=int(d["seed"]) + 10_000)
        xs.append(Tb_noisy.mean()), xs.append(Tb_noisy.std())
        ys.append(d["vz"].mean()),  ys.append(d["vz"].std())
    # crude but fine for a demo: average means/stds across sims
    x_mean = float(np.mean(xs[0::2]))
    x_std  = float(np.mean(xs[1::2]))
    y_mean = float(np.mean(ys[0::2]))
    y_std  = float(np.mean(ys[1::2]))
    return NormStats(x_mean=x_mean, x_std=max(x_std, 1e-6),
                     y_mean=y_mean, y_std=max(y_std, 1e-6))


class TbToVzDataset(Dataset):
    """Box-to-box noisy-Tb -> v_z dataset.

    Each __getitem__ re-draws a fresh noise realisation (seeded on the
    sim seed + epoch counter) so noise is *not* baked into the cubes on
    disk. This gives free data augmentation.
    """

    def __init__(self, paths: Sequence[Path], cfg: DemoConfig,
                 stats: NormStats, epoch: int = 0):
        self.paths = [Path(p) for p in paths]
        self.cfg = cfg
        self.stats = stats
        self.epoch = epoch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        d = np.load(self.paths[idx])
        seed = int(d["seed"]) * 1000 + self.epoch
        Tb_noisy = observe_Tb(d["Tb"], float(d["z"]), self.cfg, seed=seed)
        vz = np.asarray(d["vz"], dtype=np.float32)

        x = (Tb_noisy - self.stats.x_mean) / self.stats.x_std
        y = (vz       - self.stats.y_mean) / self.stats.y_std

        x = torch.from_numpy(x).unsqueeze(0).contiguous()   # (1, N, N, N)
        y = torch.from_numpy(y).unsqueeze(0).contiguous()
        return x, y


# -----------------------------------------------------------------------------
# Residual variant: U-Net predicts (v_z_true - alpha * v_z_linrec)
# -----------------------------------------------------------------------------
def fit_residual_alpha(paths: Sequence[Path], cfg: DemoConfig,
                       n_noise_seeds: int = 1) -> float:
    """Fit a single scalar alpha on the training set.

    Runs the linear reconstruction on *noisy* Tb (matching training conditions),
    averaged over `n_noise_seeds` noise draws per cube, then least-squares
    against true v_z pooled across all cubes.
    """
    xs, ys = [], []
    for p in paths:
        d = np.load(p)
        z = float(d["z"]); bl = float(d["box_len"])
        for s in range(n_noise_seeds):
            Tb_noisy = observe_Tb(d["Tb"], z, cfg, seed=int(d["seed"]) + 10_000 + s)
            xs.append(reconstruct_vz_from_tb(Tb_noisy, z, bl).ravel())
            ys.append(np.asarray(d["vz"], dtype=np.float32).ravel())
    x = np.concatenate(xs).astype(np.float64)
    y = np.concatenate(ys).astype(np.float64)
    return fit_alpha(y, x)


def compute_residual_norm_stats(paths: Sequence[Path], cfg: DemoConfig,
                                alpha: float) -> NormStats:
    """Norm stats with residual target: y_target = vz - alpha * vz_linrec(Tb_noisy)."""
    xs, ys = [], []
    for p in paths:
        d = np.load(p)
        z = float(d["z"]); bl = float(d["box_len"])
        Tb_noisy = observe_Tb(d["Tb"], z, cfg, seed=int(d["seed"]) + 10_000)
        vz_lin = reconstruct_vz_from_tb(Tb_noisy, z, bl)
        vz_true = np.asarray(d["vz"], dtype=np.float32)
        resid   = vz_true - alpha * vz_lin
        xs += [Tb_noisy.mean(), Tb_noisy.std()]
        ys += [resid.mean(),    resid.std()]
    return NormStats(
        x_mean=float(np.mean(xs[0::2])),
        x_std =max(float(np.mean(xs[1::2])), 1e-6),
        y_mean=float(np.mean(ys[0::2])),
        y_std =max(float(np.mean(ys[1::2])), 1e-6),
    )


class ResidualTbToVzDataset(Dataset):
    """Noisy Tb -> (v_z - alpha * linear-recon(Tb_noisy)) residual.

    At eval, final prediction is:  alpha * linrec(Tb_noisy) + model(Tb_noisy) * y_std + y_mean
    """

    def __init__(self, paths: Sequence[Path], cfg: DemoConfig,
                 stats: NormStats, alpha: float, epoch: int = 0):
        self.paths = [Path(p) for p in paths]
        self.cfg = cfg
        self.stats = stats
        self.alpha = float(alpha)
        self.epoch = epoch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        d = np.load(self.paths[idx])
        z = float(d["z"]); bl = float(d["box_len"])
        seed = int(d["seed"]) * 1000 + self.epoch

        Tb_noisy = observe_Tb(d["Tb"], z, self.cfg, seed=seed)
        vz_lin   = reconstruct_vz_from_tb(Tb_noisy, z, bl)
        vz_true  = np.asarray(d["vz"], dtype=np.float32)
        resid    = vz_true - self.alpha * vz_lin

        x = ((Tb_noisy - self.stats.x_mean) / self.stats.x_std).astype(np.float32)
        y = ((resid    - self.stats.y_mean) / self.stats.y_std).astype(np.float32)
        x = torch.from_numpy(x).unsqueeze(0).contiguous()
        y = torch.from_numpy(y).unsqueeze(0).contiguous()
        return x, y
