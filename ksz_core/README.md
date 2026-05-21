# ksz_core

Shared utilities for the kSZ × 21cm research stack.

## What lives here

### `ksz_core.cosmology`

`Constants` dataclass (Planck 2018 default, `Constants.paper_fiducial()`
override for the published kSZ paper's WMAP3-ish values), plus:

- `H(z, c)` — Hubble parameter, km/s/Mpc
- `f_growth(z, c)` — linear growth rate ≈ Ω_m(z)^0.55
- `comoving_distance(z, c)` — χ(z), Mpc
- `k_to_ell(k, z, c)` — k [h/Mpc] → multipole ℓ via χ(z)
- `compute_dtau_dz(z, n_e_factor, c, Y_He)` — Thomson dτ/dz
- `compute_tau_0_to_z(z, c, Y_He)` — integrated τ from origin
- `compute_tau_6_to_z(z, c, Y_He)` — integrated τ from z=6

### `ksz_core.loaders.grizzly`

Binary readers for the Ghara et al. GRIZZLY simulations: `read_den`,
`read_xhi`, `read_vel`, `load_snapshot`, `brightness_temp`.

### `ksz_core.reconstruction`

Linear continuity velocity reconstruction (rfft strategy, configurable
cosmology). Returns `vz` or `(vx, vy, vz)` via the `components` switch.

### `ksz_core.noise.hera`

`build_hera_observation` — HERA-like array via py21cmsense. Optional
extra `[hera]`.

### `ksz_core.diagnostics`

- `pearson_r(a, b)` — voxel-wise Pearson correlation, NaN-safe
- `block_sums(field_a, field_b, n_per_side)` — per-block aggregate sums
- `pearson_from_sums(n, sa, sb, saa, sbb, sab)` — r from pre-aggregated sums
- `jackknife_pearson_r(field_a, field_b, n_per_side)` — leave-one-out jackknife
  with mean + sigma + per-block values

### `ksz_core.fft`

- `kspace_rfft(n, rc, dtype)` — rfft-side k-axes with DC floored
- `kspace_grid(n, rc, dtype)` — full-grid k-axes

### `ksz_core.plotting`

- `set_default_style()` — shared matplotlib rcParams. Optional extra `[plotting]`.

## Consumers

- `ksz_reconstruction/` — paper-producing repo
- `noise_analysis/` — noise infrastructure
- `noisy_reconstruction/` — dual quadratic-estimator + ML pipeline
- `non_gaussian/` — exploratory non-Gaussianity work

## Install

Standalone (during the Chunk-A/A.5 review window):

```bash
cd ~/Documents/PhD/ksz_core
uv sync                       # base deps only
uv sync --extra hera          # adds 21cmsense for HERA bits
uv sync --extra plotting      # adds matplotlib for set_default_style
uv sync --extra dev           # adds pytest + matplotlib
uv run pytest                 # 60 tests
```

From a consumer repo (after monorepo Chunk B):

```toml
# in the consumer's pyproject.toml
dependencies = ["ksz-core"]

[tool.uv.sources]
ksz-core = { workspace = true }    # once inside the monorepo workspace
```

## Cosmology defaults

Default `Constants()` matches **Planck 2018**: H0=67.4, Ω_m=0.315,
Ω_b=0.0493, h=0.674.

For the published kSZ × 21cm paper (GRIZZLY simulation cosmology, per
Shaw et al. 2025 §1 / Hinshaw et al. 2013):

```python
from ksz_core.cosmology import Constants, H
c = Constants.paper_fiducial()    # H0=70, Om0=0.27, Ob0=0.044, h=0.7
H(z=6, c=c)
```

Do not silently switch the paper repo to Planck 2018 — pass
`Constants.paper_fiducial()` explicitly.
