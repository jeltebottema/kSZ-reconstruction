# kSZ-reconstruction monorepo

The research codebase for the kSZ × 21cm cross-correlation work — five subfolders managed as a single uv workspace.

## Layout

| Subfolder | Purpose |
|---|---|
| `ksz_core/` | Shared utilities (cosmology, GRIZZLY loaders, linear continuity reconstruction, HERA noise, FFT helpers, diagnostics, plotting). v0.2.0. |
| `ksz_reconstruction/` | Paper-side analysis code that produced the MNRAS draft. |
| `noise_analysis/` | HERA / SKA-low noise via `py21cmsense`, plus the canonical GRIZZLY loader. |
| `noisy_reconstruction/` | Dual QE + ML reconstruction pipeline under realistic noise. |
| `non_gaussian/` | Exploratory kSZ non-Gaussianity (Will collaboration, scope being defined). |

## Setup

```bash
# 1. (one-time, macOS only) make sure brew has libomp + gsl + fftw
brew install libomp gsl fftw

# 2. point the compiler at libomp's keg-only headers when syncing
export CPPFLAGS="-I$(brew --prefix libomp)/include -I/opt/homebrew/include"
export LDFLAGS="-L$(brew --prefix libomp)/lib -L/opt/homebrew/lib"

# 3. install everything into one .venv at the monorepo root
uv sync --all-packages

# 4. verify
.venv/bin/python -c "import ksz_core; from ksz_core.cosmology import Constants; print(Constants.paper_fiducial())"
```

After step 3 there's one `.venv/` at the root with all five workspace members installed editable. Sister-subfolder imports work without `sys.path` gymnastics.

`py21cmfast` (a transitive dep via `tuesday-eor`) is what needs `libomp` + `gsl`; without the env-var dance in step 2, its build fails. Once it's built and cached, subsequent `uv sync` runs are fast and don't need the env vars.

## Cosmology

Fiducial values per **Shaw et al. 2025 §1** (arXiv:2409.03255), "adapted from Hinshaw et al. 2013" (WMAP9):

```python
H_0 = 70 km/s/Mpc        Ω_m = 0.27
Ω_Λ = 0.73               Ω_b = 0.044
h   = 0.7                Y_He = 0.24
```

In `ksz_core`:

```python
from ksz_core.cosmology import Constants
c = Constants.paper_fiducial()    # the values above
```

N-body chain: PRACE4LOFAR / `cubep3m` (Harnois-Déraps et al. 2013). 500 h⁻¹Mpc box, 600³ voxels, 63 coeval cubes z=6.1–15.6.

## Running tests

```bash
uv run --with pytest python -m pytest ksz_core/tests/ -q
# 60 passed
```

## Project tracking

Strategic context, daily logs, and per-project tracking live in `~/Documents/claude-workspace/`. The `context/projects/ksz-21cm-crosscorr.md` file in that workspace points back here for code.
