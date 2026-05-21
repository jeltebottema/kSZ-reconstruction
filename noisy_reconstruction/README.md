# noisy_reconstruction

The `ksz_reconstruction/` pipeline, but with realistic noise included on
both the 21cm input and the kSZ side. Two reconstruction approaches live
side by side here so we can compare them on the same noisy inputs:

1. **Quadratic estimator** — direct port of the Ma+/Smith-style estimator
   used in the noise-free paper (`../ksz_reconstruction`), now run on
   wedge-filtered + thermal-noise 21cm cubes.
2. **Machine-learning reconstruction** — neural-net mapping
   (21cm cube + noise) → reconstructed v_z, intended to either replace
   or complement the quadratic estimator. The U-Net starter from
   `../noise_analysis/notebooks/05_unet_21cm_to_vz.ipynb` lives in
   `notebooks/ml/`.

## Pipeline

```
21cm signal  +  21cm noise (uv, wedge)
        │
        ▼
   reconstruction  ──►  v_z reconstructed
   (quadratic OR ML)
        │
        ▼
   patchy kSZ map  +  CMB noise (ILC residual etc.)
        │
        ▼
   cross-correlation with full integrated kSZ signal
```

Noise models for both 21cm and CMB come from `../noise_analysis/`
through a shared `NoiseModel` interface.

## Data paths (dual)

- **Grizzly** lightcones — `data/grizzly` symlink → `../ksz_reconstruction/data_raghu`.
- **21cmFAST** — drop new sim outputs into `data/` (lightcone or coeval).

The pipeline code in `code/` should accept either backend through the
same loader interface.

## Layout

```
noisy_reconstruction/
├── code/                     # importable pipeline modules
├── scripts/                  # CLI entry points / batch runs
├── notebooks/
│   ├── quadratic/            # noisy quadratic-estimator pipeline
│   └── ml/                   # ML reconstruction (U-Net, …)
├── data/
│   └── grizzly -> ../ksz_reconstruction/data_raghu
├── plots/
├── notes/
└── README.md
```

## Status (2026-04-19)

- Just scaffolded.
- Starter notebook for the noisy quadratic pipeline:
  `notebooks/quadratic/01_noisy_pipeline_smoketest.ipynb`.
- ML reconstruction starter: `notebooks/ml/05_unet_21cm_to_vz.ipynb`
  (ported in from `noise_analysis/`).

## Next steps

1. Wire in `noise_analysis.NoiseModel` for both 21cm and CMB sides.
2. Re-run the four paper figures from `ksz_reconstruction/plots_paper/`
   with noise turned on, so we can quote a noise-degradation factor.
3. Train the U-Net on the same train/test split used for the quadratic
   estimator to make the comparison fair.
