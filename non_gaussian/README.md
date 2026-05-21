# non_gaussian

Non-Gaussianity of the patchy kSZ signal through the Epoch of Reionisation.
Joint project with Will.

> **Scope:** placeholder. The precise physical questions and target
> observables will be pinned down with Will. For now, the folder holds
> the existing starter material (inherited from the former `code will/`)
> and is structured so work can continue in place.

## Starting material (inherited from `code will/`)

- `21cmfast_ksz_eor/` — self-contained mini-pipeline:
  21cmFAST coeval cubes + lightcones → patchy kSZ maps →
  non-Gaussian statistics (PDF, variance, skewness, kurtosis, 1D power
  spectrum) as a function of mean x_HI across the EoR.
  See `21cmfast_ksz_eor/README.md` for details.
- `notebooks/will.ipynb`                  — first pass with Will.
- `notebooks/04_ksz_non_gaussianity.ipynb` — non-Gaussianity analysis.
- `notebooks/coeval_ksz_nongauss.ipynb`   — coeval-cube based variant.
- `notebooks/grizzly_ksz.ipynb`           — Grizzly-input version.
- `notebooks/open_nikos.ipynb`            — lightcone loading / sanity checks.

## Data paths (dual)

Designed to work with both

- **Grizzly** lightcones — `data/grizzly` symlink → `../ksz_reconstruction/data_raghu`
- **21cmFAST** — coeval/lightcone outputs from the embedded
  `21cmfast_ksz_eor/run_pipeline.ipynb` or new runs dropped into `data/`.

## Layout

```
non_gaussian/
├── 21cmfast_ksz_eor/         # self-contained 21cmFAST + kSZ + NG mini-pipeline
├── code/                     # shared non-Gaussianity utilities (empty for now)
├── scripts/                  # CLI entry points
├── notebooks/                # analysis notebooks
├── data/
│   └── grizzly -> ../ksz_reconstruction/data_raghu
├── plots/
├── notes/                    # meeting notes with Will go here
└── README.md
```

## Open scope questions (for the Will kickoff)

- Which non-Gaussian statistic is the primary target? (PDF moments /
  bispectrum / Minkowski functionals / scattering transform / …)
- What does "detection" mean — forecast against CMB-S4 ILC residuals,
  or against 21cm-deprojected maps from `noisy_reconstruction/`?
- Do we compare `Grizzly` and `21cmFAST` head-to-head to quantify
  model dependence? (There's already scaffolding for both here.)

Once decided, write the scope up in `notes/SCOPE.md` and replace this
section.
