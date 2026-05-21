# ml/ — box-to-box U-Net velocity reconstruction

Pipeline: **clean 21cmFAST Tb cube → HERA-observed Tb cube via `tuesday.core.observe_coeval`
→ 3D U-Net → line-of-sight velocity v_z cube.**

No patching, no lightcone stitching — every sample is a full `(N, N, N)`
coeval cube at one redshift picked so that `<x_HI> ≈ 0.5` (mid-EoR).

## Components (live under `noisy_reconstruction/code/ml/`)

- `config.py`    — `DemoConfig` defaults (128³ / 256 Mpc, HERA, 16+4 sims).
- `simulate.py`  — runs 21cmFAST coeval cubes and saves `(Tb, xHI, vz)` `.npz`.
- `noise.py`    — HERA observation via `tuesday.core.observe_coeval`, using
                   `build_hera_observation` imported from
                   `noise_analysis/src/noise_filters.py`. Matches the pattern
                   in `noise_analysis/notebooks/04_grizzly_uv_sweep.ipynb`.
- `dataset.py`   — `TbToVzDataset` with per-epoch fresh noise draws + z-scoring.
- `unet3d.py`    — small 3D U-Net (~6 M params at base=16, depth=4).
- `train.py`     — Adam + cosine LR + MSE loss, Pearson-r val metric, checkpoints.

## Notebooks

- `ml_pipeline_demo.ipynb` — end-to-end demo:
  generate sims → observe through HERA → train U-Net → evaluate.
- `05_unet_21cm_to_vz.ipynb` — earlier starter (kept for reference).

## How to run

### 1. Install the environment

The `noise_analysis/` project already has an env with `py21cmfast`,
`21cmsense`, and `tuesday-eor` (see `noise_analysis/pyproject.toml`).
We reuse it and just add PyTorch on top.

```bash
cd ~/Documents/PhD/noise_analysis
uv sync                                    # install existing deps
uv add py21cmfast torch                    # py21cmfast for sims, torch for U-Net
```

If `py21cmfast` needs the C deps (`gsl`, `fftw3`), install those with
your OS package manager (`brew install gsl fftw` on macOS,
`apt install libgsl-dev libfftw3-dev` on Debian/Ubuntu).

### 2. Generate the dataset

From the ML folder, using the same env:

```bash
cd ~/Documents/PhD/noisy_reconstruction
uv run --project ../noise_analysis python -c "
from code.ml.config   import DemoConfig
from code.ml.simulate import generate_dataset
paths = generate_dataset(DemoConfig())
print('done:', paths['z'])
"
```

That scans z on seed 1000, picks the redshift where `<x_HI> ≈ 0.5`, and
runs 20 coeval cubes (16 train + 4 val). Cubes land in
`noisy_reconstruction/data/cubes/sim_seed<N>.npz`. Re-running skips
existing files.

### 3. Train the U-Net

```bash
uv run --project ../noise_analysis python -c "
from code.ml.config import DemoConfig
from code.ml.simulate import generate_dataset
from code.ml.train   import train
cfg = DemoConfig()
paths = generate_dataset(cfg)             # skipped if already done
train(cfg, paths['train'], paths['val'])
"
```

Or, interactively:

```bash
uv run --project ../noise_analysis jupyter lab
# -> open noisy_reconstruction/notebooks/ml/ml_pipeline_demo.ipynb
# -> Run All
```

Checkpoints land in `noisy_reconstruction/data/checkpoints/{best,last}.pt`
and training curves in `history.json`.

### 4. Evaluate

The last two cells of `ml_pipeline_demo.ipynb` load `best.pt`, run on a
held-out validation cube, and show slice comparisons + 1D power spectra.
Or programmatically:

```python
from code.ml.train import load_best
model, stats, device = load_best(DemoConfig())
```

## Demo defaults (tunable in `DemoConfig`)

| knob                                | value                         |
| ----------------------------------- | ----------------------------- |
| `hii_dim`                           | 128                           |
| `box_len` [Mpc]                     | 256                           |
| `target_xHI`                        | 0.5                           |
| train / val seeds                   | 16 / 4                        |
| survey                              | HERA (hex_num=11, split_core, outriggers=2) |
| dish, latitude                      | 14 m, -30°                    |
| track / time_per_day / n_days       | 6 h / 6 h / 180 days          |
| U-Net base channels                 | 16                            |
| depth                               | 4                             |
| batch size / epochs                 | 1 / 40                        |
| optimizer                           | Adam(2e-4)                    |

## Data on disk

```
noisy_reconstruction/data/
├── cubes/                              # sim_seed<N>.npz (Tb, xHI, vz, z, seed, ...)
└── checkpoints/                        # best.pt, last.pt, history.json
```

## Known simplifications (deliberate, for the demo)

- Single-redshift coeval, not lightcone — no evolution across the box.
- `remove_wedge=False` — thermal noise + uv-sampling only; wedge removal
  can be toggled on in `noise.py::observe_Tb` once the baseline works.
- Batch size 1 — fine for 128³; bump up once input is downsampled or the
  base channels are trimmed.
- Velocity is the full 21cmFAST `velocity_z`, not the Ma+/quadratic
  reconstruction — the network is asked to infer v_z from the observed
  Tb directly.
