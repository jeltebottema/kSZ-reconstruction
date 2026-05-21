# Theory: Thermal Noise in 21cm Interferometry

Step-by-step derivation of the noise filter f(k_perp, k_parallel), building from first principles to the final 2D plot.

---

## 1. What does an interferometer measure?

Each pair of antennas (a "baseline") measures one visibility:

```
V(u,v,ν) = ∫ I(l,m,ν) · B(l,m,ν) · exp(-2πi(ul + vm)) dl dm
```

- **I(l,m,ν)**: sky brightness at angular position (l,m) and frequency ν
- **B(l,m,ν)**: primary beam (antenna response pattern)
- **(u,v)**: baseline vector in units of wavelengths = d/λ
- The visibility is one Fourier coefficient of the sky

A baseline of length d probes angular scale θ ~ λ/d. Longer baselines → smaller angular scales → higher k_perp.

## 2. From baseline to wavenumber

The observed frequency maps to redshift: ν = 1420 MHz / (1+z)

Two independent Fourier axes:

**Transverse (angular → k_perp):**
```
k_perp = 2π u / D_M(z)
```
where D_M(z) is the comoving transverse distance. The paper calls this X = D_M(z).

**Radial (frequency → k_parallel):**
```
k_parallel = 2π η · H(z) ν₂₁ / (c (1+z)²)
```
where η is the Fourier dual of frequency (delay, in seconds). The paper calls the conversion factor Y = c(1+z)² / (H(z) ν₂₁) [Mpc/MHz].

Key point: k_perp is set by which baseline you use, k_parallel is set by which frequency channel you look at.

## 3. Thermal noise per visibility

Each visibility measurement has thermal noise with RMS:

```
σ_vis = T_sys / √(2 · Δν · t_int)
```

- **T_sys**: system temperature [K] — dominated by galactic synchrotron at EoR frequencies
- **Δν**: channel bandwidth [Hz]
- **t_int**: integration time per visibility [s]
- Factor 2: two polarisations

At z=8: T_sys ≈ 237 + 1.6 × (158/300)^(-5.23) ≈ 700 K (Eq. 19 of the paper)

This noise is **independent** between different baselines and different frequency channels. This is crucial — it means the noise is white in (u,v,ν) space.

## 4. From visibility noise to power spectrum noise

The noise power spectrum P_N measures how much noise power there is per Fourier mode.

For a single baseline measuring mode (k_perp, k_parallel):

```
P_N = (T_sys)² × (beam factor) × X² × Y / (t_int × N_pol × N_bl(u))
```

This is **Eq. 14** of the paper (Parsons+ 2014 formalism for HERA):

```
P_N(k_perp) = T_sys² × (Ω_p²/Ω_pp) × X² × Y / (t_int × N_pol × N_bl(u))
```

Breaking this down:

| Factor | What it does | Value at z=8 |
|--------|-------------|--------------|
| T_sys² | Noise amplitude squared | ~(700 K)² |
| Ω_p²/Ω_pp | Beam correction (≈ 2 Ω_p for Gaussian) | ~0.037 sr |
| X² | Converts angles → comoving Mpc² | ~(6500 Mpc)² |
| Y | Converts freq → comoving Mpc/MHz | ~12 Mpc/MHz |
| t_int | More time → less noise | 200 hr = 720000 s |
| N_pol | Two polarisations | 2 |
| N_bl(u) | More baselines at this u → less noise | depends on k_perp |

### Why P_N depends on k_perp but NOT on k_parallel:

- k_perp is set by which baseline you use → N_bl(u) varies with baseline length
- k_parallel is set by frequency → thermal noise is white in frequency → flat after FFT
- **Result: P_N = P_N(k_perp) only** — constant along k_parallel

This is why the 2D filter has vertical band structure.

## 5. The baseline density N_bl(u)

N_bl(u) = number of baseline pairs sampling a given (u,v) cell. This depends entirely on the antenna layout:

**HERA** (compact hexagonal, 350 dishes, 14m spacing):
- Many short baselines (14m, 28m, 42m...) → dense coverage at small u (low k_perp)
- Few long baselines → sparse coverage at large u (high k_perp)
- N_bl peaks at u ~ 50 wavelengths (k_perp ~ 0.05 h/Mpc)

**SKA-Low** (extended, 512 stations, up to 65 km):
- Few short baselines → sparse at small u
- Many intermediate baselines → peak at u ~ 300
- Long baselines out to u ~ 10000

This is why HERA's noise suppresses **large k_perp** (sparse long baselines) while SKA suppresses **small k_perp** (sparse short baselines).

## 6. The Wiener noise filter

The noise filter tells you what fraction of the signal survives:

```
f_noise(k_perp) = P_signal(k) / (P_signal(k) + P_noise(k_perp))
```

This is **Eq. 13** — a Wiener filter (optimal linear filter in the MSE sense).

- Where P_signal >> P_noise: f → 1 (signal dominates, mode survives)
- Where P_signal << P_noise: f → 0 (noise dominates, mode lost)
- Where P_signal = P_noise: f = 0.5 (crossover)

Since P_noise depends only on k_perp, the filter has the same value for all k_parallel at a given k_perp → **vertical bands** in the 2D plot.

## 7. The foreground wedge

Bright foregrounds (galactic synchrotron, point sources) are spectrally smooth — they have power only at low k_parallel. But the interferometer's chromatic response (beam width changes with frequency) leaks foreground power from low k_parallel into a wedge-shaped region:

```
k_parallel < m(z) × k_perp
```

where the wedge slope m(z) is set by geometry (**Eq. 12**):

```
m(z) = H(z) D_M(z) / (c (1+z))
```

At z=8: m ≈ 0.32 (exact value depends on cosmology).

Modes inside the wedge are contaminated and must be discarded:

```
f_fore = 1  if |k_parallel| > m × k_perp  (EoR window — safe)
f_fore = 0  if |k_parallel| < m × k_perp  (inside wedge — lost)
```

The wedge slope m(z) ~ 0.32 from cosmology. But in practice a larger slope (m=1.0, the "horizon limit") is often used as a conservative choice. tuesday uses m=1.0 by default.

## 8. The combined filter

```
f_21cm(k_perp, k_parallel) = f_fore(k_perp, k_parallel) × f_noise(k_perp)
```

This is **Eq. 18** — the total filter applied to the 21cm field.

Reading the 2D plot:
- **Bottom-left triangle** (low k_par, low k_perp): killed by wedge (f_fore = 0)
- **Left column** (low k_perp, outside wedge): killed by noise for HERA (few short baselines) or surviving for SKA
- **Right column** (high k_perp, outside wedge): killed by noise for SKA (few long baselines) or surviving for HERA's core
- **Middle band** (intermediate k_perp, outside wedge): **EoR window** — where signal survives

## 9. Computing P_noise: three approaches

### A. Paper-analytical (Eq. 14)
Use the formula directly with an analytic approximation for N_bl(u). The paper uses a log-normal fit tuned to match HERA's layout. This gives the smoothest result.

Pros: no simulation needed, fast, smooth, reproducible.
Cons: N_bl(u) is approximate (not the real antenna positions).

### B. Tuesday-analytical
Use `compute_thermal_rms_uvgrid()` which computes σ_noise per UV cell from the actual HERA antenna positions and observation parameters. Square to get P_noise.

Pros: uses real antenna layout, accounts for rotation synthesis.
Cons: only non-zero where baselines exist (sparse grid), needs binning.

### C. Noise realizations
Pass a zero-signal box through `observe_coeval` → pure noise output. Compute its power spectrum.

Pros: includes all effects (beam, gridding, interpolation).
Cons: single realization is noisy (variance ~ 1/N_modes per bin); need many realizations to converge.

The variance of P in a bin with N_modes is: σ(P) ~ P / √N_modes.
At high k_perp: many modes (many integer combinations give similar k_perp) → single realization is fine.
At low k_perp: few modes → large scatter → dark patches in the filter.

## 10. Order of operations matters

The paper (Section 3, procedure steps 1-6) emphasises:

1. Filter the 3D 21cm field in Fourier space (apply f_21cm)
2. Inverse FFT back to real space
3. **Square** the field in real space
4. **Then** project to 2D by integrating along the LOS

If you project first and then square, you lose the radial modes that survived the filter. The squaring converts k-space filtering into configuration-space weighting, which preserves more information about the 3D mode structure.

This is specifically for the kSZ²×21cm² cross-correlation. For our direct cross-correlation (no squaring), this ordering issue doesn't apply.

## Summary of key numbers at z = 8

| Quantity | Value |
|----------|-------|
| Observed frequency | 158 MHz |
| Wavelength | 1.9 m |
| T_sys | ~700 K |
| D_M (comoving distance) | ~6500 Mpc |
| X = D_M | 6500 Mpc |
| Y = c(1+z)²/(H ν₂₁) | ~12 Mpc/MHz |
| Wedge slope m(z) | ~0.32 (cosmological), 1.0 (horizon limit) |
| HERA dishes | 350, diameter 14m |
| Integration time | 200h (paper) or 180 days × 6h (our setup) |
</content>