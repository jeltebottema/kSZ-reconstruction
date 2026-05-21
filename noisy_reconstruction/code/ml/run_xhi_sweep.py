"""
run_xhi_sweep.py
================
Supercomputer runner for the noisy-21cm -> v_z U-Net at three EoR epochs:

    <x_HI> ~ 0.50   (mid-reionisation)
    <x_HI> ~ 0.82   (near the Tb-delta zero cross-correlation)
    <x_HI> ~ 0.98   (very beginning of EoR)

Stages:
    sim    : run 21cmFAST coeval cubes for each target xHI
    train  : train one U-Net per target xHI
    eval   : self + CROSS-epoch evaluation matrix (model_A on data_B)
    all    : sim -> train -> eval

The 21cmFAST auto-cache is redirected to local scratch (default
$SLURM_TMPDIR / $TMPDIR) so $HOME does not fill.  Each IC file is ~100 MB;
with 20 seeds x 3 xHI that would be ~6 GB in $HOME otherwise.

Usage (on the cluster):
    python code/ml/run_xhi_sweep.py \
        --stage all \
        --work-dir $SCRATCH/xhi_sweep \
        --cache-dir $SLURM_TMPDIR/21cmfast-cache
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# Make the project importable regardless of where we are launched from.
THIS = Path(__file__).resolve()
PROJECT   = THIS.parent.parent.parent                # .../noisy_reconstruction
PHD_ROOT  = PROJECT.parent                            # .../PhD
sys.path.insert(0, str(PHD_ROOT))

from noisy_reconstruction.code.ml.config   import DemoConfig
from noisy_reconstruction.code.ml.simulate import generate_dataset
from noisy_reconstruction.code.ml.noise    import observe_Tb
from noisy_reconstruction.code.ml.train    import train, load_best


TARGETS = (0.50, 0.82, 0.98)
# Wide enough to bracket xHI=0.98 (z~12-14) and xHI=0.50 (z~7-8) with default
# 21cmFAST params.
Z_GRID  = (7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0)


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

def tag_for(x: float) -> str:
    return f"xhi{int(round(x * 100)):02d}"          # 0.82 -> 'xhi82'


def make_cfg(x: float, work_dir: Path, **overrides) -> DemoConfig:
    kw = dict(
        target_xHI = x,
        z_grid     = Z_GRID,
        data_dir   = work_dir / "cubes"       / tag_for(x),
        ckpt_dir   = work_dir / "checkpoints" / tag_for(x),
    )
    kw.update(overrides)
    return DemoConfig(**kw)


def configure_21cmfast_cache(scratch: Path):
    """Redirect 21cmFAST's auto-HDF5 cache to local scratch (auto-purged)."""
    import py21cmfast as p21c
    scratch.mkdir(parents=True, exist_ok=True)
    try:
        p21c.config["direc"] = scratch
    except Exception:
        p21c.config.direc = scratch  # older attribute-style fallback
    print(f"[cache] 21cmFAST auto-cache -> {scratch}", flush=True)


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def stage_sim(cfgs: dict[float, DemoConfig]) -> dict[float, dict]:
    paths = {}
    for x, cfg in cfgs.items():
        print(f"\n======== simulate xHI={x} ========", flush=True)
        t0 = time.time()
        paths[x] = generate_dataset(cfg)
        manifest = {
            "z":     float(paths[x]["z"]),
            "train": [str(p) for p in paths[x]["train"]],
            "val":   [str(p) for p in paths[x]["val"]],
        }
        (cfg.data_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"[sim xHI={x}] done in {time.time() - t0:.0f}s", flush=True)
    return paths


def load_paths(cfgs: dict[float, DemoConfig]) -> dict[float, dict]:
    """Reload manifests written by stage_sim (so `train`/`eval` can run alone)."""
    out = {}
    for x, cfg in cfgs.items():
        m = json.loads((cfg.data_dir / "manifest.json").read_text())
        out[x] = dict(
            z     = m["z"],
            train = [Path(p) for p in m["train"]],
            val   = [Path(p) for p in m["val"]],
        )
    return out


def stage_train(cfgs: dict[float, DemoConfig], paths: dict[float, dict]):
    for x, cfg in cfgs.items():
        print(f"\n======== train xHI={x} ========", flush=True)
        t0 = time.time()
        history = train(cfg, paths[x]["train"], paths[x]["val"])
        (cfg.ckpt_dir / "history.json").write_text(json.dumps(history, indent=2))
        print(f"[train xHI={x}] done in {time.time() - t0:.0f}s", flush=True)


def _predict_vz(model, stats, Tb_obs: np.ndarray, device) -> np.ndarray:
    import torch
    xn = (Tb_obs - stats.x_mean) / stats.x_std
    xn = torch.from_numpy(xn).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(xn).squeeze().cpu().numpy()
    return pred * stats.y_std + stats.y_mean


def stage_eval(cfgs: dict[float, DemoConfig],
               paths: dict[float, dict],
               out_file: Path) -> dict:
    """
    Cross-epoch evaluation.

    For each (x_train, x_eval) pair:
        - load the model trained at x_train (its own normalisation stats)
        - run it on every val cube generated at x_eval
        - record per-cube Pearson r between predicted v_z and truth
    """
    xs = sorted(cfgs)                                   # [0.5, 0.82, 0.98]
    matrix_r = np.full((len(xs), len(xs)), np.nan, dtype=np.float32)

    models = {}
    for x in xs:
        m, s, d = load_best(cfgs[x])
        models[x] = (m, s, d)
        print(f"[eval] loaded model xHI={x}  from {cfgs[x].ckpt_dir}", flush=True)

    per_cube = []
    for i, x_train in enumerate(xs):
        model, stats, device = models[x_train]
        for j, x_eval in enumerate(xs):
            rs = []
            for vp in paths[x_eval]["val"]:
                val = np.load(vp)
                Tb_obs  = observe_Tb(val["Tb"], float(val["z"]),
                                     cfgs[x_eval], seed=9999)
                vz_pred = _predict_vz(model, stats, Tb_obs, device)
                vz_true = val["vz"]
                r = float(np.corrcoef(vz_true.ravel(), vz_pred.ravel())[0, 1])
                rs.append(r)
                per_cube.append(dict(
                    train_xhi = x_train, eval_xhi = x_eval,
                    val_file  = str(vp),  r = r,
                ))
            matrix_r[i, j] = float(np.mean(rs))
            print(f"[eval] train xHI={x_train} -> eval xHI={x_eval}  "
                  f"<r>={matrix_r[i, j]:+.3f}  (N={len(rs)})", flush=True)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps({
        "xhi_targets": xs,
        "matrix_r":    matrix_r.tolist(),
        "per_cube":    per_cube,
    }, indent=2))
    np.save(out_file.with_suffix(".npy"), matrix_r)

    print("\n=== cross-eval matrix <Pearson r> ===")
    print("          " + "  ".join(f"eval={x:4.2f}" for x in xs))
    for i, xt in enumerate(xs):
        row = "  ".join(f"{matrix_r[i, j]:+6.3f} " for j in range(len(xs)))
        print(f"train={xt:4.2f}   {row}")
    return {"xhi_targets": xs, "matrix_r": matrix_r}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _power_1d(cube: np.ndarray, box_len: float):
    n = cube.shape[0]
    k1 = np.fft.fftfreq(n, d=box_len / n) * 2 * np.pi
    kx, ky, kz = np.meshgrid(k1, k1, k1, indexing="ij")
    kmag = np.sqrt(kx**2 + ky**2 + kz**2)
    fft  = np.fft.fftn(cube - cube.mean())
    pk3  = (np.abs(fft)**2) * (box_len / n)**3 / (n**3)
    bins = np.linspace(kmag[kmag > 0].min(), kmag.max(), 25)
    k_c  = 0.5 * (bins[1:] + bins[:-1])
    idx  = np.digitize(kmag.ravel(), bins)
    pk1  = np.array([pk3.ravel()[idx == i].mean() if np.any(idx == i) else np.nan
                     for i in range(1, len(bins))])
    return k_c, pk1


def stage_plot(cfgs: dict[float, DemoConfig],
               paths: dict[float, dict],
               work_dir: Path):
    """Produce PNGs under {work_dir}/plots/.  Headless-safe (Agg backend)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = work_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    xs = sorted(cfgs)

    # ---- 1. training curves ------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
    for x in xs:
        hp = cfgs[x].ckpt_dir / "history.json"
        if not hp.exists():
            print(f"[plot] missing {hp}; skip training curve for xHI={x}", flush=True)
            continue
        h = json.loads(hp.read_text())
        axes[0].plot(h["val_loss"],   label=f"xHI={x} val")
        axes[0].plot(h["train_loss"], ls=":", alpha=0.6, label=f"xHI={x} train")
        axes[1].plot(h["val_r"],      label=f"xHI={x}")
    axes[0].set(xlabel="epoch", ylabel="MSE (standardised)", yscale="log")
    axes[0].legend(fontsize=8)
    axes[1].set(xlabel="epoch", ylabel="val Pearson r")
    axes[1].axhline(0, color="k", lw=0.5)
    axes[1].legend()
    fig.savefig(plots / "training_curves.png", dpi=140)
    plt.close(fig)

    # ---- 2. cross-eval heatmap --------------------------------------------
    ce_file = work_dir / "cross_eval.json"
    if ce_file.exists():
        ce = json.loads(ce_file.read_text())
        M  = np.asarray(ce["matrix_r"], dtype=float)
        xs_m = ce["xhi_targets"]
        fig, ax = plt.subplots(figsize=(4.6, 4.0), constrained_layout=True)
        vlim = float(max(0.05, np.nanmax(np.abs(M))))
        im = ax.imshow(M, cmap="RdBu_r", vmin=-vlim, vmax=vlim)
        ax.set_xticks(range(len(xs_m)), [f"{x:.2f}" for x in xs_m])
        ax.set_yticks(range(len(xs_m)), [f"{x:.2f}" for x in xs_m])
        ax.set_xlabel("eval  <x_HI>")
        ax.set_ylabel("train <x_HI>")
        ax.set_title("Cross-epoch Pearson r  (v_z_true, v_z_pred)")
        for i in range(len(xs_m)):
            for j in range(len(xs_m)):
                ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center",
                        color="k", fontsize=10)
        plt.colorbar(im, ax=ax, shrink=0.85)
        fig.savefig(plots / "cross_eval_heatmap.png", dpi=140)
        plt.close(fig)
    else:
        print(f"[plot] no {ce_file}; run --stage eval first to get the heatmap.")

    # ---- 3. slice matrix: rows = train xHI, cols = eval xHI ---------------
    # Extra top row shows the truth for each eval cube (same val-cube seed for all).
    import torch
    models = {x: load_best(cfgs[x]) for x in xs}

    truths = {}
    preds  = {}    # preds[(x_train, x_eval)] = vz_pred
    for x_eval in xs:
        val = np.load(paths[x_eval]["val"][0])
        Tb_obs = observe_Tb(val["Tb"], float(val["z"]),
                            cfgs[x_eval], seed=9999)
        truths[x_eval] = val["vz"]
        for x_train in xs:
            m, s, dev = models[x_train]
            preds[(x_train, x_eval)] = _predict_vz(m, s, Tb_obs, dev)

    nrow, ncol = 1 + len(xs), len(xs)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4*ncol, 3.2*nrow),
                             constrained_layout=True, squeeze=False)
    for j, x_eval in enumerate(xs):
        mid  = cfgs[x_eval].hii_dim // 2
        vmax = float(np.percentile(np.abs(truths[x_eval]), 99))
        im = axes[0, j].imshow(truths[x_eval][:, :, mid].T, origin="lower",
                               cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axes[0, j].set_title(f"truth  eval xHI={x_eval}")
        axes[0, j].set_xticks([]); axes[0, j].set_yticks([])
        plt.colorbar(im, ax=axes[0, j], shrink=0.85)

        for i, x_train in enumerate(xs):
            ax = axes[1 + i, j]
            p  = preds[(x_train, x_eval)]
            r  = float(np.corrcoef(truths[x_eval].ravel(), p.ravel())[0, 1])
            im = ax.imshow(p[:, :, mid].T, origin="lower",
                           cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax.set_title(f"train {x_train} -> eval {x_eval}\nr={r:+.3f}",
                         fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            plt.colorbar(im, ax=ax, shrink=0.85)
    fig.savefig(plots / "vz_slices_matrix.png", dpi=140)
    plt.close(fig)

    # ---- 4. 1D P(k) per epoch, diagonal (train=eval) ----------------------
    fig, axes = plt.subplots(1, len(xs), figsize=(4.8*len(xs), 3.6),
                             constrained_layout=True, squeeze=False)
    for ax, x in zip(axes[0], xs):
        vz_true = truths[x]
        vz_pred = preds[(x, x)]
        k, pk_t = _power_1d(vz_true, cfgs[x].box_len)
        _, pk_p = _power_1d(vz_pred, cfgs[x].box_len)
        ax.loglog(k, pk_t, label="true")
        ax.loglog(k, pk_p, ls="--", label="pred (train=eval)")
        ax.set(xlabel=r"$k\ [\mathrm{Mpc}^{-1}]$",
               ylabel=r"$P_{v_z}(k)\ [(\mathrm{km/s})^2\,\mathrm{Mpc}^3]$",
               title=f"xHI={x}")
        ax.legend(fontsize=8)
    fig.savefig(plots / "power_spectra.png", dpi=140)
    plt.close(fig)

    print(f"[plot] wrote figures to {plots}", flush=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["sim", "train", "eval", "plot", "all"], default="all")
    ap.add_argument("--work-dir", type=Path, default=Path("./xhi_sweep_work"),
                    help="Where cubes + checkpoints + results live (use $SCRATCH).")
    ap.add_argument("--cache-dir", type=Path,
                    default=Path(os.environ.get("SLURM_TMPDIR",
                                  os.environ.get("TMPDIR", "/tmp"))) / "21cmfast-cache",
                    help="Where 21cmFAST dumps its auto HDF5 cache.")
    ap.add_argument("--targets", type=float, nargs="+", default=list(TARGETS))
    args = ap.parse_args()

    args.work_dir.mkdir(parents=True, exist_ok=True)
    configure_21cmfast_cache(args.cache_dir)

    cfgs = {x: make_cfg(x, args.work_dir) for x in args.targets}
    for x, c in cfgs.items():
        print(f"[cfg xHI={x}] data={c.data_dir}  ckpt={c.ckpt_dir}", flush=True)

    if args.stage in ("sim", "all"):
        paths = stage_sim(cfgs)
    else:
        paths = load_paths(cfgs)

    if args.stage in ("train", "all"):
        stage_train(cfgs, paths)

    if args.stage in ("eval", "all"):
        stage_eval(cfgs, paths, args.work_dir / "cross_eval.json")

    if args.stage in ("plot", "all"):
        stage_plot(cfgs, paths, args.work_dir)


if __name__ == "__main__":
    main()
