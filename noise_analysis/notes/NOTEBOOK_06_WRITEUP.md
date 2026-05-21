# Notebook 06 — EoR evolution of kSZ reconstruction with noise

## Goal

Quantify how well the reionization kSZ signal can be reconstructed from a
**noisy 21cm observation** of GRIZZLY snapshots, as a function of the mean
neutral fraction $\langle x_{HI}\rangle$.

Four scenarios per snapshot: **Clean / Noisy / Wiener / Wiener+wedge**.

---

## Pipeline (math only)

### 1. Velocity from a tracer (continuity equation)

Given any tracer field $T(\mathbf{x})$ (clean or filtered $T_b$), reconstruct
the line-of-sight velocity under linear continuity:

$$
\tilde v_z(\mathbf{k}) \;=\; a\,H(z)\,f(z)\;\frac{i\,k_z}{k^2}\,\tilde T(\mathbf{k}),
\qquad
f(z) \approx \Omega_m(z)^{0.55}.
$$

### 2. kSZ projection

$$
\Delta T_{kSZ}(\hat n) \;=\; -\,T_{CMB}\,\sigma_T\,\bar n_{e,0}(1+z)^3
\int \frac{dl}{c}\,x_e(\mathbf{x})\,\big[1+\delta(\mathbf{x})\big]\,v_z(\mathbf{x}),
$$
with $x_e = 1 - x_{HI}$. Implemented as a sum over the LOS axis.

### 3. Observed 21cm cube

$$
T_b^{\text{obs}}(\mathbf{x}) \;=\; T_b(\mathbf{x}) \;+\; n(\mathbf{x}) \;+\; F(\mathbf{x}),
$$
- $n$: HERA thermal noise from `tuesday.observe_coeval` (uv-sampled, redshift-dep.).
- $F$: toy foreground, Gaussian with $k_\perp^{-1}$ colouring, confined to
  the **wedge** $|k_\|| \le \mu_w k_\perp$ (slope $\mu_w=1$), scaled so
  $\sigma_F = 1000\,\sigma_{T_b}$.

---

## 4. Wiener filter (the core of denoising)

The Wiener filter is the **minimum-MSE** estimator of the signal given a
noisy linear measurement $d = s + n$ with independent $s$ and $n$:

$$
\boxed{\;W(\mathbf{k}) \;=\; \frac{P_s(\mathbf{k})}{P_s(\mathbf{k}) + P_n(\mathbf{k})}\;}
\qquad\Longrightarrow\qquad
\hat s(\mathbf{k}) = W(\mathbf{k})\,d(\mathbf{k}).
$$

Properties: $W\to 1$ where $P_s \gg P_n$ (keep), $W\to 0$ where $P_n \gg P_s$
(kill), smoothly down-weighted in between. It is *diagonal in Fourier space*
— i.e. a multiplicative mask per mode.

**Signal power** (per voxel, from the clean $T_b$ cube of that snapshot):
$$
P_s(\mathbf{k}) \;=\; \frac{\big|\widetilde{T_b}(\mathbf{k})\big|^2}{V},
\qquad V = L^3.
$$

**Noise power** (HERA thermal RMS on the uv grid):
$$
P_n(\mathbf{k}) \;=\; \sigma_{uv}^2(k_\perp),
$$
assumed flat in $k_\|$ (white along LOS, rms from
`compute_thermal_rms_uvgrid`). In practice $\sigma_{uv}^2$ is binned in
$k_\perp$ and interpolated onto the full 3D rfft grid. Unsampled baselines
$\Rightarrow P_n = 10^{30}$ so $W\approx 0$ there.

**Application:**
$$
\hat T_b^{\text{Wiener}}(\mathbf{x}) = \mathcal{F}^{-1}\!\big[W(\mathbf{k})\,\tilde d(\mathbf{k})\big].
$$

**Wiener + wedge** multiplies by the wedge-exclusion mask
$\mathbb{1}[\,|k_\|| > \mu_w k_\perp\,]$ before the inverse FFT:
$$
\hat T_b^{\text{W+wedge}}(\mathbf{x}) = \mathcal{F}^{-1}\!\big[W(\mathbf{k})\,\mathbb{1}_{\text{wedge}}(\mathbf{k})\,\tilde d(\mathbf{k})\big].
$$

---

## 5. Correlation diagnostics

Both are applied to the 2D kSZ maps $\kappa_\star = \Delta T_{kSZ}^\star$.

**Real-space Pearson:**
$$
r(x) \;=\; \frac{\langle\,\kappa_{\text{true}}\,\kappa_{\text{rec}}\,\rangle_{\mathbf x}}
{\sigma_{\kappa_{\text{true}}}\,\sigma_{\kappa_{\text{rec}}}}.
$$

**Scale-dependent (Fourier) correlation, binned in $|k|$:**
$$
r(k) \;=\; \frac{\mathrm{Re}\!\sum_{|\mathbf k'|\in k}\!\tilde\kappa_{\text{true}}\tilde\kappa_{\text{rec}}^\ast}
{\sqrt{\sum|\tilde\kappa_{\text{true}}|^2\,\sum|\tilde\kappa_{\text{rec}}|^2}},
\qquad
\ell = k\,\chi(z).
$$

Reported scalar: $r(\ell=3000)$ via linear interpolation of $r(k)$ to
$k_\star = 3000/\chi(z)$.

Comoving distance
$\chi(z) = \dfrac{c}{H_0}\int_0^z\dfrac{dz'}{\sqrt{\Omega_m(1+z')^3+\Omega_\Lambda}}$.

---

## 6. Output

Two evolution curves vs $\langle x_{HI}\rangle$ (one marker per GRIZZLY
snapshot, four scenarios):

1. $r(x)$ — real-space Pearson on the kSZ map.
2. $r(\ell=3000)$ — CMB-relevant scale.

Plus one diagnostic $r(\ell)$ curve at the mid-EoR snapshot
($\langle x_{HI}\rangle\!\approx\!0.5$).

**Expected ordering:** Clean ≳ Wiener+wedge > Wiener > Noisy, with the gap
widening toward late reionization where $P_s$ drops and $W$ kills more modes.
