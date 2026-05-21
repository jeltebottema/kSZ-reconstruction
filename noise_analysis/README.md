# noise_analysis

Realistic observational noise modelling for 21cm interferometry and CMB experiments,
for use as an input to downstream reconstruction pipelines (e.g.
`noisy_reconstruction/`) and forecast work.

## Science goal

Produce physically motivated, survey-specific noise realisations and power spectra
for

- **21cm** (HERA, SKA1-Low, …): system temperature model, uv-coverage from antenna
  layout, thermal + sample-variance noise, foreground-wedge geometry.
- **CMB** (SO, S4, Planck, …): beam-deconvolved white + 1/f noise, ILC residual
  after component separation, lensing reconstruction noise.

Outputs are consistent noise power spectra P_N(k) / C_ell^N and map-level
realisations that can be added to signal cubes / maps downstream.

## Data paths

This folder is designed to work with two simulation sources interchangeably:

- **Grizzly** lightcones — via the `data_grizzly` symlink to
  `../ksz_reconstruction/data_raghu` (Raghunath's runs).
- **21cmFAST** — `data_nikos/` holds a starter lightcone; new runs can be
  dropped into `data/` (21cmFAST coeval/lightcone outputs).

## Layout

```
noise_analysis/
├── src/                      # installable modules (grizzly.py, noise_filters.py)
├── scripts/                  # runnable pipelines (appendix_b_model_dependence.py, …)
├── notebooks/                # exploratory / figure notebooks
│   ├── 01_noise_in_21cmfast.ipynb
│   ├── 02_noise_filter.ipynb
│   ├── 03_uv_sweep.ipynb
│   └── 04_grizzly_uv_sweep.ipynb
├── plots/                    # generated figures
├── data/                     # new 21cm sim outputs (empty on first use)
├── data_nikos/               # Nikos' 21cmFAST starter lightcone
├── data_grizzly -> ../ksz_reconstruction/data_raghu
├── notes/                    # THEORY_NOTES.md, MEETING_NOTES.md, theory_notes.tex
├── main.py
├── pyproject.toml / uv.lock  # uv-managed env
└── README.md
```

## Status (2026-04-19)

- 21cm noise side: working. `src/noise_filters.py` + `src/grizzly.py`
  provide uv-based thermal noise, wedge masks, and Wiener filters. Covered
  by notebooks 01–04 and the `appendix_b_model_dependence.py` script.
- CMB noise side: **to be built from scratch.** Targets: SO / CMB-S4 survey
  specs, ILC residuals, compatibility with `orphics`-style C_ell conventions.

## Next steps

1. Add `src/cmb_noise.py` — survey-spec white+1/f noise, beam, ILC residual.
2. Match both 21cm and CMB outputs to a common `NoiseModel` interface so
   `noisy_reconstruction/` can consume them without knowing the backend.
3. Port `05_unet_21cm_to_vz.ipynb` → `noisy_reconstruction/notebooks/ml/`
   (reconstruction, not noise modelling).
