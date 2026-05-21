"""
train.py
========
Training loop for the box-to-box U-Net.

Loss    : MSE in standardised space.
Metric  : Pearson r between predicted and true v_z (un-standardised),
          averaged over the validation set per epoch.
Optim   : Adam, cosine LR schedule.
Outputs : best.pt + last.pt checkpoints in <data_dir>/../checkpoints/.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import DemoConfig
from .dataset import (NormStats, TbToVzDataset, compute_norm_stats,
                      ResidualTbToVzDataset, compute_residual_norm_stats,
                      fit_residual_alpha)
from .unet3d import UNet3D, count_parameters


def _pearsonr_flat(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().float()
    b = b.flatten().float()
    am, bm = a - a.mean(), b - b.mean()
    denom = (am.norm() * bm.norm()).clamp_min(1e-12)
    return float((am * bm).sum() / denom)


def pick_device(preferred: str) -> torch.device:
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preferred == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train(cfg: DemoConfig, train_paths: Sequence[Path], val_paths: Sequence[Path]) -> dict:
    cfg.ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"Computing normalisation stats on {len(train_paths)} train cubes ...")
    stats = compute_norm_stats(train_paths, cfg)
    print(f"  x: mean={stats.x_mean:.3f} std={stats.x_std:.3f}")
    print(f"  y: mean={stats.y_mean:.3f} std={stats.y_std:.3f}")

    train_ds = TbToVzDataset(train_paths, cfg, stats)
    val_ds   = TbToVzDataset(val_paths,   cfg, stats)
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    device = pick_device(cfg.device)
    print(f"Using device: {device}")

    model = UNet3D(in_channels=1, out_channels=1, base=cfg.base_channels).to(device)
    print(f"U-Net parameters: {count_parameters(model):,}")

    opt   = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    mse   = torch.nn.MSELoss()

    history = {"train_loss": [], "val_loss": [], "val_r": [], "lr": []}
    best_r  = -np.inf

    for epoch in range(cfg.epochs):
        t0 = time.time()
        train_ds.set_epoch(epoch)

        # ---- train ----
        model.train()
        running = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(x)
            loss = mse(pred, y)
            loss.backward()
            opt.step()
            running += float(loss) * x.size(0)
        tr_loss = running / len(train_ds)

        # ---- val ----
        model.eval()
        running = 0.0
        rs = []
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = mse(pred, y)
                running += float(loss) * x.size(0)
                # un-standardise for Pearson r in physical units
                pred_phys = pred * stats.y_std + stats.y_mean
                y_phys    = y    * stats.y_std + stats.y_mean
                rs.append(_pearsonr_flat(pred_phys, y_phys))
        val_loss = running / len(val_ds)
        val_r = float(np.mean(rs))

        sched.step()
        lr_now = opt.param_groups[0]["lr"]
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["val_r"].append(val_r)
        history["lr"].append(lr_now)

        print(f"epoch {epoch:3d}  train {tr_loss:.4f}  "
              f"val {val_loss:.4f}  r={val_r:+.3f}  "
              f"lr={lr_now:.2e}  ({time.time() - t0:.1f}s)")

        # checkpoint
        ckpt = {
            "model": model.state_dict(),
            "stats": stats.to_dict(),
            "cfg":   cfg.__dict__ | {"data_dir": str(cfg.data_dir), "ckpt_dir": str(cfg.ckpt_dir)},
            "epoch": epoch,
        }
        torch.save(ckpt, cfg.ckpt_dir / "last.pt")
        if val_r > best_r:
            best_r = val_r
            torch.save(ckpt, cfg.ckpt_dir / "best.pt")

    with open(cfg.ckpt_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"done.  best val r = {best_r:+.3f}")
    return history


def load_best(cfg: DemoConfig, device: torch.device | None = None):
    device = device or pick_device(cfg.device)
    ckpt = torch.load(cfg.ckpt_dir / "best.pt", map_location=device)
    model = UNet3D(in_channels=1, out_channels=1, base=cfg.base_channels).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    stats = NormStats.from_dict(ckpt["stats"])
    return model, stats, device


# -----------------------------------------------------------------------------
# Residual training: target = v_z_true - alpha * linrec(Tb_noisy)
# -----------------------------------------------------------------------------
def train_residual(cfg: DemoConfig,
                   train_paths: Sequence[Path],
                   val_paths: Sequence[Path]) -> dict:
    """Same loop as `train`, but the target is the residual after the linear
    (continuity-equation) velocity reconstruction. Checkpoints land in
    cfg.ckpt_dir / 'linrec'. `alpha` is stashed alongside the norm stats.
    """
    ckpt_dir = cfg.ckpt_dir / "linrec"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fitting linear-reconstruction alpha on {len(train_paths)} train cubes ...")
    alpha = fit_residual_alpha(train_paths, cfg)
    print(f"  alpha = {alpha:+.3e}")

    print("Computing residual-target normalisation stats ...")
    stats = compute_residual_norm_stats(train_paths, cfg, alpha=alpha)
    print(f"  x: mean={stats.x_mean:.3f} std={stats.x_std:.3f}")
    print(f"  y: mean={stats.y_mean:.3f} std={stats.y_std:.3f}  (residual target)")

    train_ds = ResidualTbToVzDataset(train_paths, cfg, stats, alpha=alpha)
    val_ds   = ResidualTbToVzDataset(val_paths,   cfg, stats, alpha=alpha)
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    device = pick_device(cfg.device)
    print(f"Using device: {device}")

    model = UNet3D(in_channels=1, out_channels=1, base=cfg.base_channels).to(device)
    print(f"U-Net parameters: {count_parameters(model):,}")

    opt   = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    mse   = torch.nn.MSELoss()

    history = {"train_loss": [], "val_loss": [], "val_r_resid": [], "lr": []}
    best_r = -np.inf

    for epoch in range(cfg.epochs):
        t0 = time.time()
        train_ds.set_epoch(epoch)
        val_ds.set_epoch(epoch)   # val noise also varies epoch-to-epoch

        model.train()
        running = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(x)
            loss = mse(pred, y)
            loss.backward(); opt.step()
            running += float(loss) * x.size(0)
        tr_loss = running / len(train_ds)

        model.eval()
        running = 0.0; rs = []
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                running += float(mse(pred, y)) * x.size(0)
                # Pearson r on the residual target (pred vs y, in std-space is fine)
                rs.append(_pearsonr_flat(pred, y))
        val_loss = running / len(val_ds)
        val_r = float(np.mean(rs))

        sched.step()
        lr_now = opt.param_groups[0]["lr"]
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["val_r_resid"].append(val_r)
        history["lr"].append(lr_now)

        print(f"epoch {epoch:3d}  train {tr_loss:.4f}  val {val_loss:.4f}  "
              f"r_resid={val_r:+.3f}  lr={lr_now:.2e}  ({time.time()-t0:.1f}s)")

        ckpt = {
            "model": model.state_dict(),
            "stats": stats.to_dict(),
            "alpha": alpha,
            "cfg":   cfg.__dict__ | {"data_dir": str(cfg.data_dir), "ckpt_dir": str(cfg.ckpt_dir)},
            "epoch": epoch,
        }
        torch.save(ckpt, ckpt_dir / "last.pt")
        if val_r > best_r:
            best_r = val_r
            torch.save(ckpt, ckpt_dir / "best.pt")

    with open(ckpt_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"done.  best residual r = {best_r:+.3f}")
    return history


def load_best_residual(cfg: DemoConfig, device: torch.device | None = None):
    """Returns (model, stats, alpha, device) for the residual checkpoint."""
    device = device or pick_device(cfg.device)
    ckpt = torch.load(cfg.ckpt_dir / "linrec" / "best.pt", map_location=device)
    model = UNet3D(in_channels=1, out_channels=1, base=cfg.base_channels).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    stats = NormStats.from_dict(ckpt["stats"])
    alpha = float(ckpt["alpha"])
    return model, stats, alpha, device
