# Noise Analysis Project — Meeting Notes

## Goal
Build correct noise behaviour in 21cmFAST simulations to study how instrumental noise degrades the kSZ reconstruction cross-correlation. Extends the existing kSZ reconstruction work to include realistic HERA noise.

## Setup

- **Packages**: `tuesday` (coeval-obs branch) + `21cmSense` (fast-hist branch) + `py21cmfast` v4.1.1
- **Simulation**: BOX_LEN=600 Mpc, DIM=1024, HII_DIM=512, z=8.0, seed=42
- **Instrument**: HERA-350 (hex_num=11, split core, 2 outriggers), 14m dishes, 6h tracks, 180 days
- **Noise model**: `tuesday.core.observe_coeval()` adds thermal noise consistent with UV coverage + optional foreground wedge removal

## What works

### 1. Simulation + noise pipeline
- 21cmFAST coeval box generates density, velocity, neutral fraction, brightness temperature
- `observe_coeval` takes the brightness temp box, adds realistic thermal noise in UV space, optionally removes the foreground wedge
- Multiple noise realizations can be generated with different seeds
- Wedge removal uses a hard mask in delay space: modes where |tau| < wedge_slope * k_perp / f0 are zeroed

### 2. kSZ reconstruction pipeline
- Velocity reconstruction from 21cm tracer field via continuity equation in Fourier space: v_z(k) = a * H(z) * f(z) * i*k_z * T(k) / k^2
- Tracer field: T = (1+delta) * x_HI (21cm brightness temperature proxy)
- kSZ projection: sum of x_e * (1+delta) * v_z along LOS, with physical prefactor (-T_CMB * sigma_T * n_e * dl / c)
- Cross-correlation: r(k) = P_cross / sqrt(P_11 * P_22) in Fourier space + real-space Pearson r
- Pipeline runs on both clean and noisy boxes to measure degradation

### 3. Instrument visualisation
- Antenna layout, baseline histogram, instantaneous UV coverage
- Earth-rotation synthesis UV coverage + density map
- Primary beam profile (Gaussian, FWHM from dish size and frequency)

### 4. 2D noise filter f(k_perp, k_parallel)
Three methods compared for computing the Wiener noise filter f = P_signal / (P_signal + P_noise):

| Method | How P_noise is obtained | Result |
|--------|------------------------|--------|
| **Analytical** | `compute_thermal_rms_uvgrid` from tuesday, squared. Flat in k_parallel. | Smooth, matches paper approach |
| **Averaged** (10 realizations) | Pass zero box through observe_coeval 10x, average P_noise | Converges to analytical |
| **Single realization** | One noise draw | Noisy, dark patches at high k_parallel |

## Key findings / things that now make sense

### UV coverage → noise shape
- HERA's compact hexagonal core gives massive redundancy at short baselines (~14m multiples)
- This means LOW noise at intermediate k_perp (many baselines) and HIGH noise at large k_perp (sparse long baselines)
- The noise is set by: P_noise(k_perp) ~ 1 / N_baselines(k_perp)
- Noise is flat in k_parallel because thermal noise is white in frequency

### Why single-realization filter looks wrong
- The variance of a power spectrum estimate in a bin scales as ~1/N_modes in that bin
- At **high k_perp**: many modes per bin (many integer combinations of nx, ny give similar k_perp) → single realization averages down fine, matches analytical
- At **low k_perp**: few modes per bin (small rings in kx-ky plane) → large scatter per bin
- At **high k_parallel**: fewer kz cells near the Nyquist edge → fewer modes per bin
- Where N_modes is small, random upward fluctuations in P_noise push f → 0 → dark patches
- The paper uses the analytical (ensemble-averaged) P_noise, which is perfectly smooth
- Averaging over many realizations converges to the analytical result, but this is just approximating what the analytical method gives directly

### Foreground wedge
- Modes where k_parallel < wedge_slope * k_perp are contaminated by foregrounds
- tuesday implements this as a hard binary mask (zero all modes inside wedge)
- wedge_slope=1.0 corresponds to the horizon limit (moderate choice)
- Combined filter: f_total = f_noise * f_fore shows which modes survive BOTH noise and foregrounds

## Differences from the paper (kSZ2-21cm2 cross-correlations)

1. **Coeval vs lightcone**: We use a single-redshift coeval box. The paper uses a lightcone with redshift evolution and a redshift window function. This affects which k_parallel modes are available.

2. **Direct field vs squared field**: The paper computes kSZ^2 x 21cm^2 cross-correlations (squared fields). We compute direct kSZ cross-correlation (unsquared). The squaring step changes which modes contribute — the paper explicitly notes that projection must happen AFTER squaring to preserve radial modes.

3. **Filter definition**: The paper defines f_noise as a Wiener filter from the analytical noise PS. Our analytical method reproduces this. The single-realization method does not.

4. **Resolution**: Paper likely uses finer grids. Our BOX_LEN=600, HII_DIM=512 gives:
   - dk = 2*pi/600 * h = 0.007 h/Mpc (k-space resolution)
   - k_par_max = pi * 512/600 * h = 1.8 h/Mpc (matches paper range)
   - Some white stripes remain at low k_perp due to discrete k-grid + log-spaced bins (box length issue, not dimension issue)

5. **Units**: Paper uses h/Mpc, we now match this (multiply k by h=0.6736).

## Technical details

### 21cmFAST velocity units
- `coeval.velocity_z` is in comoving Mpc/s
- Convert to physical km/s: v_phys = a * v_raw * 3.0857e19 (Mpc_to_km)

### Accessing fields in py21cmfast v4.1.1
- `coeval.brightness_temp` — works via __getattr__ proxy
- `coeval.density` — overdensity delta
- `coeval.velocity_z` — LOS velocity
- `coeval.ionized_box.get("neutral_fraction")` — need .get() because it's an Array wrapper, not raw numpy
- `coeval.ionized_box.global_xH` — mean neutral fraction as float

### powerbox get_power (dev branch)
- Returns tuple of 4+: (P, k, var, sumweights), NOT just (P, k)
- Use `result[0], result[1]` to unpack
- `dimensionless=False` needed for brightness temperature (can have zero/negative mean)

### White stripes in 2D plots
- Empty bins where no Fourier grid points land
- k_perp = sqrt(kx^2 + ky^2) has irregular spacing at low values
- Log-spaced bins at low k_perp can be narrower than the grid spacing dk
- Fix: larger BOX_LEN (finer dk), NOT larger N (which only extends k_max)

## Questions for supervisor

1. Should we implement the full squared-field pipeline (kSZ^2 x 21cm^2) from the paper, or is the direct cross-correlation sufficient for our purposes?
2. The paper uses a lightcone with redshift windows — how important is this vs our single-z coeval approach?
3. For the Wiener filter: should we use the analytical P_noise (smooth, like the paper) or is there value in using the actual noisy observation (single realization) since that's what you'd have in practice?
4. The wedge_slope=1.0 and wedge_buffer=0 are "moderate" choices — what range should we explore?
5. Should we also look at SKA-Low for comparison (the paper shows both HERA and SKA)?

## Files
- Notebook: `notebooks/01_noise_in_21cmfast.ipynb`
- Based on: Alex's pipeline in `~/Documents/PhD/Bachelor student/work alex/` (compute_ksz.py, ksz_pipeline.py)
- Reference: `~/Documents/PhD/ksz_reconstruction/` for simulation patterns
</content>
</invoke>