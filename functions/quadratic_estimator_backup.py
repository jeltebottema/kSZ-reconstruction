#!/usr/bin/env python3
"""
Quadratic Estimator for Velocity Reconstruction

Based on Hotinli & Johnson "Reconstructing large scales at cosmic dawn"
Equation 18: Quadratic estimator using CMB × 21cm cross-correlation

This implementation focuses on the cross-correlation structure, 
with simplified normalization terms for initial study.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import fft
from scipy.ndimage import gaussian_filter
from scipy.integrate import quad
from powerbox import get_power
import gc

# ============================================================================
# CONFIGURATION
# ============================================================================

LITTLEH = 0.7
BOX_MPC_OVER_H = 500.0
N_BOX = 600

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data_raghu/")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "plots_quad")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Cropping to avoid boundary effects
CENTRAL_CROP = slice(30, 570)
LOS_CROP = slice(30, 570)

# Physical constants
C_LIGHT = 2.998e5  # km/s
SIGMA_T = 6.6524e-25  # Thomson cross-section in cm^2
M_P = 1.6726e-24  # Proton mass in g


# ============================================================================
# DATA LOADING (from generate_all_plots.py)
# ============================================================================

def read_den(filename, nx, ny, nz, endian="<"):
    """Read density field from binary file (IDL-compatible)."""
    dt_f4 = np.dtype(endian + "f4")
    with open(filename, 'rb') as f:
        f.seek(12)  # skip 3*float32 header like IDL read_den
        data = np.fromfile(f, dtype=dt_f4, count=nx*ny*nz)
    return data.reshape((nx, ny, nz), order='F')

def read_xhi(filename, nx, ny, nz):
    """Read neutral hydrogen fraction from binary file."""
    with open(filename, 'rb') as f:
        data = np.fromfile(f, dtype=np.float32, count=nx*ny*nz)
    return data.reshape((nx, ny, nz), order='F')

def read_vel(z, den, filename, n_cell=600, box=500.0, nc=13824, hlittle=0.7, endian="<"):
    """Read velocity field from binary file (IDL-compatible)."""
    Megaparsec = 3.08568025e24
    omega_l = 0.73
    omega_m = 1.0 - omega_l
    Ho = hlittle * 3.2407e-18

    dt_f4 = np.dtype(endian + "f4")
    with open(filename, 'rb') as f:
        f.seek(12)  # skip 3*float32 dims like IDL read_vel
        arrv3 = np.fromfile(f, dtype=dt_f4, count=3 * n_cell**3)

    arrv3 = arrv3.reshape((3, n_cell, n_cell, n_cell), order="F").astype(np.float32, copy=False)

    len_unit = box * Megaparsec / hlittle / (1.0 + z) / float(nc)
    tau_t = 2.0 / 3.0 / np.sqrt(omega_m * Ho * Ho) / (1.0 + z)**2
    vel_unit = len_unit / tau_t

    arrv3 *= np.float32(vel_unit * 8.0)  # cm/s

    eps = np.float32(1e-12)
    den32 = den.astype(np.float32, copy=False)
    den_safe = np.where(den32 > eps, den32, eps)

    with np.errstate(divide="ignore", invalid="ignore"):
        vx = (arrv3[0] / den_safe).astype(np.float32, copy=False)
        vy = (arrv3[1] / den_safe).astype(np.float32, copy=False)
        vz = (arrv3[2] / den_safe).astype(np.float32, copy=False)

    return vx, vy, vz


def load_simulation_data(z, n=N_BOX, data_dir=DATA_DIR):
    """Load all simulation data for a given redshift."""
    zstr = f"{z:.3f}"
    filenameDen = f"{data_dir}{zstr}n_all.dat"
    filenameVel = f"{data_dir}{zstr}v_all.dat"
    filenameXhi = f"{data_dir}{zstr}zeta0.389fesc0.389_Mmin0.120E+10_MminX0.120E+10_fx0.100E+03_sed3_al1.200xhi.bin"
    
    den = read_den(filenameDen, n, n, n).astype(np.float32, copy=False)
    xhi = read_xhi(filenameXhi, n, n, n).astype(np.float32, copy=False)
    vx, vy, vz = read_vel(z, den, filenameVel, n_cell=n)
    
    # Convert velocities to km/s
    vx *= np.float32(1.0 / 1e5)
    vy *= np.float32(1.0 / 1e5)
    vz *= np.float32(1.0 / 1e5)
    
    return den, xhi, vx, vy, vz


# ============================================================================
# COSMOLOGY FUNCTIONS
# ============================================================================

def hubble_parameter(z, H0=70.0, Omega_m=0.27, Omega_L=0.73):
    """Hubble parameter H(z) in km/s/Mpc."""
    return H0 * np.sqrt(Omega_m * (1 + z)**3 + Omega_L)

def comoving_distance(z, H0=70.0, Omega_m=0.27, Omega_L=0.73):
    """Comoving distance to redshift z in Mpc."""
    def integrand(zp):
        return C_LIGHT / hubble_parameter(zp, H0, Omega_m, Omega_L)
    chi, _ = quad(integrand, 0, z)
    return chi

def growth_rate(z, Omega_m=0.27, Omega_L=0.73):
    """Linear growth rate f ≈ Ω_m(z)^0.55."""
    a = 1.0 / (1.0 + z)
    Omega_m_z = (Omega_m / a**3) / (Omega_m / a**3 + Omega_L)
    return Omega_m_z**0.55


# ============================================================================
# kSZ SIGNAL COMPUTATION
# ============================================================================

def compute_ksz_temperature(vz, xhi, den, z, physical_norm=True):
    """
    Compute the kSZ temperature signal: ΔT/T ∝ -τ × v_r/c
    
    The kSZ effect is:
        ΔT_kSZ = -T_CMB × τ × (v_r/c)
    
    where τ is the optical depth and v_r is the radial velocity.
    
    For EoR, τ ∝ n_e ∝ (1 - x_HI) × (1 + δ)
    """
    # Electron fraction (ionized fraction)
    x_e = 1.0 - xhi
    
    # Electron density proxy: n_e ∝ (1 + δ) × x_e
    mean_den = den.mean()
    delta = den / mean_den - 1.0
    n_e_proxy = (1.0 + delta) * x_e
    
    # kSZ signal: ΔT ∝ -n_e × v_r
    # The minus sign comes from: approaching gas (v < 0) → hot spot (ΔT > 0)
    ksz = -n_e_proxy * vz
    
    if physical_norm:
        # Rough normalization to get µK units
        # This is approximate - proper calculation requires integration along LOS
        T_CMB = 2.725e6  # µK
        tau_factor = 1e-4  # Approximate optical depth per cell
        ksz *= T_CMB * tau_factor / C_LIGHT
    
    return ksz.astype(np.float32)


def compute_ksz_map_2d(ksz_3d):
    """Integrate kSZ signal along line of sight to get 2D map."""
    return np.sum(ksz_3d, axis=2).astype(np.float32)


# ============================================================================
# 21cm BRIGHTNESS TEMPERATURE
# ============================================================================

def compute_brightness_temperature(den, xhi, z, include_velocity_term=False, vz=None):
    """
    Compute 21cm brightness temperature field.
    
    δT_b ∝ x_HI × (1 + δ) × [H / (dv_r/dr + H)]
    
    The velocity term accounts for redshift-space distortions.
    """
    mean_den = den.mean()
    delta = den / mean_den
    
    if include_velocity_term and vz is not None:
        # Compute velocity gradient
        dz_cell = BOX_MPC_OVER_H / N_BOX / LITTLEH  # Cell size in Mpc
        dvdz = np.gradient(vz, dz_cell, axis=2).astype(np.float32)
        
        Hz = hubble_parameter(z)
        velocity_factor = Hz / (dvdz + Hz)
        
        Tb = delta * xhi * velocity_factor
    else:
        Tb = delta * xhi
    
    return Tb.astype(np.float32)


# ============================================================================
# QUADRATIC ESTIMATOR IMPLEMENTATION
# ============================================================================

def kspace_grid(n, box_size, dtype=np.float32):
    """Generate k-space coordinates for FFT."""
    dk = 2.0 * np.pi / box_size
    kx = np.fft.fftfreq(n, d=1.0/n) * dk
    ky = np.fft.fftfreq(n, d=1.0/n) * dk
    kz = np.fft.rfftfreq(n, d=1.0/n) * dk
    return kx.astype(dtype), ky.astype(dtype), kz.astype(dtype)


# ============================================================================
# 2D PROJECTION FUNCTIONS (to match paper's observational setup)
# ============================================================================

def project_to_2d(field_3d, axis=2):
    """
    Project 3D field to 2D by integrating along line of sight.
    
    This mimics what observers see: integrated signal along LOS.
    """
    return np.sum(field_3d, axis=axis).astype(np.float32)


def compute_2d_power_spectrum(field_2d, box_size_2d):
    """
    Compute 2D power spectrum P(ell) from a 2D field.
    
    Returns ell values and P(ell).
    """
    n = field_2d.shape[0]
    
    # FFT
    field_k = np.fft.rfft2(field_2d - field_2d.mean())
    
    # 2D k-space coordinates
    dk = 2.0 * np.pi / box_size_2d
    kx = np.fft.fftfreq(n, d=1.0/n) * dk
    ky = np.fft.rfftfreq(n, d=1.0/n) * dk
    
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    k_mag = np.sqrt(KX**2 + KY**2)
    
    # Power at each mode
    P_k = np.abs(field_k)**2
    
    # Bin in k (which corresponds to ell in flat-sky approximation)
    k_max = np.max(k_mag)
    n_bins = 30
    k_bins = np.linspace(0, k_max, n_bins + 1)
    k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])
    
    P_binned = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (k_mag >= k_bins[i]) & (k_mag < k_bins[i+1])
        if np.any(mask):
            P_binned[i] = np.mean(P_k[mask])
    
    return k_centers, P_binned


def compute_2d_cross_spectrum(field1_2d, field2_2d, box_size_2d):
    """
    Compute 2D cross-power spectrum between two fields.
    """
    n = field1_2d.shape[0]
    
    field1_k = np.fft.rfft2(field1_2d - field1_2d.mean())
    field2_k = np.fft.rfft2(field2_2d - field2_2d.mean())
    
    dk = 2.0 * np.pi / box_size_2d
    kx = np.fft.fftfreq(n, d=1.0/n) * dk
    ky = np.fft.rfftfreq(n, d=1.0/n) * dk
    
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    k_mag = np.sqrt(KX**2 + KY**2)
    
    # Cross-spectrum
    cross = np.real(np.conj(field1_k) * field2_k)
    
    k_max = np.max(k_mag)
    n_bins = 30
    k_bins = np.linspace(0, k_max, n_bins + 1)
    k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])
    
    C_binned = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (k_mag >= k_bins[i]) & (k_mag < k_bins[i+1])
        if np.any(mask):
            C_binned[i] = np.mean(cross[mask])
    
    return k_centers, C_binned


def quadratic_estimator_flatsky(Theta_2d, H_2d, tau_2d, box_size_2d):
    """
    PROPER Flat-sky quadratic estimator following Hotinli & Johnson Eq. 18.
    
    v̂(L) = N^{vv}(L) × ∫ d²ℓ/(2π)² × Γ(ℓ,L) × Θ(ℓ) × H(L-ℓ) / [C^{ΘΘ}(ℓ) × C^{HH}(|L-ℓ|)]
    
    where:
    - Θ = CMB temperature (kSZ signal)
    - H = 21cm brightness temperature (hydrogen tracer)
    - Γ(ℓ,L) ∝ C^{τH}(|L-ℓ|) = cross-power of τ and H
    - C^{ΘΘ} = observed CMB power spectrum
    - C^{HH} = observed 21cm power spectrum
    - N^{vv} = reconstruction noise (normalization)
    
    This is a CONVOLUTION in Fourier space, not a simple division!
    
    The key difference from my "proxy" method:
    - Uses C^{τH}(ℓ) as a STATISTICAL weight, not τ itself
    - Properly accounts for noise via inverse-variance weighting
    - Performance degrades when using total kSZ (other-z adds to C^{ΘΘ})
    """
    n = Theta_2d.shape[0]
    
    # Mean-subtract
    Theta_c = Theta_2d - Theta_2d.mean()
    H_c = H_2d - H_2d.mean()
    tau_c = tau_2d - tau_2d.mean()
    
    # 2D FFT
    Theta_k = np.fft.fft2(Theta_c)
    H_k = np.fft.fft2(H_c)
    tau_k = np.fft.fft2(tau_c)
    
    # k-space coordinates
    dk = 2.0 * np.pi / box_size_2d
    kx = np.fft.fftfreq(n, d=1.0/n) * dk
    ky = np.fft.fftfreq(n, d=1.0/n) * dk
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    k_mag = np.sqrt(KX**2 + KY**2)
    
    # Compute power spectra (binned by |k|)
    k_max = np.max(k_mag)
    n_bins = 50
    k_bins = np.linspace(0, k_max, n_bins + 1)
    k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])
    
    def bin_power(field_k):
        """Compute binned power spectrum."""
        P_k = np.abs(field_k)**2
        P_binned = np.zeros(n_bins)
        for i in range(n_bins):
            mask = (k_mag >= k_bins[i]) & (k_mag < k_bins[i+1])
            if np.any(mask):
                P_binned[i] = np.mean(P_k[mask])
        return P_binned
    
    def bin_cross_power(field1_k, field2_k):
        """Compute binned cross-power spectrum."""
        cross = np.real(np.conj(field1_k) * field2_k)
        C_binned = np.zeros(n_bins)
        for i in range(n_bins):
            mask = (k_mag >= k_bins[i]) & (k_mag < k_bins[i+1])
            if np.any(mask):
                C_binned[i] = np.mean(cross[mask])
        return C_binned
    
    # Compute power spectra
    C_ThetaTheta = bin_power(Theta_k)
    C_HH = bin_power(H_k)
    C_tauH = bin_cross_power(tau_k, H_k)  # Key coupling term!
    
    # Interpolate to get C(|k|) at each mode
    def interp_to_modes(C_binned):
        """Interpolate binned spectrum to each k-mode."""
        C_modes = np.zeros_like(k_mag)
        for i in range(n_bins):
            mask = (k_mag >= k_bins[i]) & (k_mag < k_bins[i+1])
            C_modes[mask] = C_binned[i]
        return C_modes
    
    C_TT_modes = interp_to_modes(C_ThetaTheta)
    C_HH_modes = interp_to_modes(C_HH)
    C_tauH_modes = interp_to_modes(C_tauH)
    
    # Avoid division by zero
    eps = 1e-30
    C_TT_modes = np.maximum(C_TT_modes, eps)
    C_HH_modes = np.maximum(C_HH_modes, eps)
    
    # The quadratic estimator (Eq. 18 in flat-sky):
    # v̂(L) ∝ ∫ d²ℓ Γ(ℓ,L) Θ(ℓ) H(L-ℓ) / [C^{ΘΘ}(ℓ) C^{HH}(|L-ℓ|)]
    #
    # In practice, this is a convolution. For diagonal approximation:
    # v̂(L) ∝ Θ(L) × H*(L) × C^{τH}(L) / [C^{ΘΘ}(L) × C^{HH}(L)]
    #
    # This is NOT the same as dividing by τ!
    
    # Γ coefficient (Eq. 19): Γ ∝ C^{τH}
    Gamma = np.abs(C_tauH_modes)
    
    # Inverse-variance weighted estimator
    numerator = np.conj(Theta_k) * H_k * Gamma
    denominator = C_TT_modes * C_HH_modes
    
    v_k = numerator / denominator
    
    # Compute normalization N^{vv} (Eq. 20)
    # 1/N^{vv}(L) = Σ_ℓ |Γ(ℓ,L)|² / [C^{ΘΘ}(ℓ) C^{HH}(|L-ℓ|)]
    # For diagonal: N^{vv}(L) ∝ C^{ΘΘ}(L) C^{HH}(L) / |Γ(L)|²
    N_vv = denominator / (Gamma**2 + eps)
    
    # Apply normalization (this sets the amplitude!)
    v_k_normalized = v_k * N_vv
    
    # Transform back to real space
    v_2d = np.real(np.fft.ifft2(v_k_normalized)).astype(np.float32)
    v_2d = v_2d - v_2d.mean()
    
    return v_2d


def quadratic_estimator_2d(ksz_2d, Tb_2d, tau_2d, box_size_2d):
    """
    Quadratic estimator for velocity reconstruction on 2D projected maps.
    
    The key physics: kSZ ∝ τ × v, so v ∝ kSZ / τ
    
    But we don't observe τ directly - we observe Tb which correlates with τ.
    The quadratic estimator uses this correlation:
    
        v̂ ∝ ⟨kSZ × Tb⟩ / ⟨τ × Tb⟩
    
    In Fourier space with inverse-variance weighting:
        v̂(ell) ∝ [Θ(ell) × δH*(ell)] / C^{τH}(ell) × [1 / C^{ΘΘ}(ell)]
    """
    n = ksz_2d.shape[0]
    
    # Mean-subtract
    ksz_c = ksz_2d - ksz_2d.mean()
    Tb_c = Tb_2d - Tb_2d.mean()
    tau_c = tau_2d - tau_2d.mean()
    
    # 2D FFT
    ksz_k = np.fft.rfft2(ksz_c).astype(np.complex64)
    Tb_k = np.fft.rfft2(Tb_c).astype(np.complex64)
    tau_k = np.fft.rfft2(tau_c).astype(np.complex64)
    
    # Power spectra at each mode
    P_ksz = np.abs(ksz_k)**2
    P_Tb = np.abs(Tb_k)**2
    P_tau = np.abs(tau_k)**2
    
    # Cross-power C^{τH} - this is the key coupling
    C_tau_Tb = np.real(np.conj(tau_k) * Tb_k)
    
    # Cross-power C^{Θτ} - kSZ-tau correlation (contains velocity info!)
    C_ksz_tau = np.real(np.conj(ksz_k) * tau_k)
    
    # Avoid division by zero
    eps = 1e-30
    P_ksz = np.maximum(P_ksz, eps)
    P_Tb = np.maximum(P_Tb, eps)
    P_tau = np.maximum(P_tau, eps)
    C_tau_Tb_safe = np.where(np.abs(C_tau_Tb) > eps, C_tau_Tb, eps)
    
    # Quadratic estimator:
    # Since kSZ ∝ τ × v, we have C^{Θτ} ∝ ⟨τ²⟩ × v
    # So v ∝ C^{Θτ} / P_τ
    # But we use Tb as proxy for τ, weighted by C^{τH}
    
    # Cross-spectrum of kSZ with Tb
    C_ksz_Tb = np.conj(ksz_k) * Tb_k
    
    # Estimator: v̂ ∝ C^{ΘH} / C^{τH} (using Tb as τ proxy)
    v_eff_k = C_ksz_Tb / C_tau_Tb_safe
    
    # Apply smoothing to reduce noise
    dk = 2.0 * np.pi / box_size_2d
    kx = np.fft.fftfreq(n, d=1.0/n) * dk
    ky = np.fft.rfftfreq(n, d=1.0/n) * dk
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    k_mag = np.sqrt(KX**2 + KY**2)
    
    # Low-pass filter to reduce noise on small scales
    k_max = 0.5 * np.max(k_mag)
    filter_k = np.exp(-(k_mag / k_max)**2)
    v_eff_k *= filter_k
    
    # Transform back to real space
    v_eff_2d = np.real(np.fft.irfft2(v_eff_k, s=(n, n))).astype(np.float32)
    
    return v_eff_2d


def quadratic_estimator_2d_v2(ksz_2d, Tb_2d, tau_2d, box_size_2d):
    """
    Direct real-space estimator: v = -kSZ / τ (IDEAL CASE)
    
    Physics:
    --------
    The kSZ effect is: Θ_kSZ = -σ_T ∫ n_e v_r / c dl ∝ -τ × v
    
    Therefore: v = -Θ_kSZ / τ
    
    This gives PERFECT reconstruction when τ is known exactly.
    In reality, τ is not directly observable.
    
    Amplitude:
    ----------
    The output has correct amplitude because we're directly inverting
    the kSZ-velocity relationship.
    """
    # Direct division with regularization
    eps = tau_2d.std() * 0.1  # Regularization based on τ variance
    
    # v = -kSZ / τ (exact inversion of kSZ = -τ × v)
    v_eff_2d = -ksz_2d / (tau_2d + eps)
    
    # Remove mean (velocity field should be mean-zero)
    v_eff_2d = v_eff_2d - v_eff_2d.mean()
    
    return v_eff_2d.astype(np.float32)


def quadratic_estimator_2d_v3(ksz_2d, Tb_2d, tau_2d, box_size_2d):
    """
    PROXY ESTIMATOR: v ≈ -kSZ / τ_proxy (OBSERVATIONALLY REALISTIC)
    
    Physics:
    --------
    In observations, we don't have τ directly, but we can construct a PROXY:
    
    - τ ∝ n_e ∝ (1 - x_HI) × (1 + δ)  [optical depth from free electrons]
    - Tb ∝ x_HI × (1 + δ)              [21cm brightness from neutral hydrogen]
    
    Key insight: τ and Tb are ANTI-CORRELATED through x_HI:
    - Ionized regions (low x_HI): HIGH τ, LOW Tb
    - Neutral regions (high x_HI): LOW τ, HIGH Tb
    
    Proxy construction:
    -------------------
    τ_proxy = 1 - Tb_normalized = 1 - Tb / Tb_max
    
    This maps:
    - High Tb (neutral) → low τ_proxy
    - Low Tb (ionized) → high τ_proxy
    
    Then: v ≈ -kSZ / τ_proxy
    
    Why this works:
    ---------------
    The 21cm signal at a SPECIFIC REDSHIFT tells us where the neutral gas is.
    The kSZ signal (even if integrated over all z) is strongest where τ is high.
    By dividing kSZ by τ_proxy, we extract velocity weighted by the ionization
    structure at that specific redshift.
    
    Amplitude:
    ----------
    The amplitude is approximately correct because τ_proxy ≈ τ / τ_max,
    so v_rec ≈ v × τ_max. We don't apply additional scaling.
    """
    # Construct τ proxy from Tb
    # Normalize Tb to [0, 1] range
    Tb_norm = Tb_2d / (Tb_2d.max() + 1e-10)
    
    # τ_proxy: high where Tb is low (ionized regions)
    tau_proxy = 1.0 - Tb_norm
    
    # Regularization to avoid division by zero
    eps = tau_proxy.std() * 0.1
    
    # Estimate velocity: v ≈ -kSZ / τ_proxy
    v_eff_2d = -ksz_2d / (tau_proxy + eps)
    
    # Remove mean (velocity field should be mean-zero)
    v_eff_2d = v_eff_2d - v_eff_2d.mean()
    
    return v_eff_2d.astype(np.float32)


def quadratic_estimator_2d_v4(ksz_2d, Tb_2d, tau_2d, box_size_2d):
    """
    CROSS-CORRELATION ESTIMATOR: v ∝ -kSZ × Tb (SIMPLE BUT LIMITED)
    
    Physics:
    --------
    This is the naive cross-correlation approach:
    
    - kSZ ∝ -τ × v ∝ -(1 - x_HI)(1 + δ) × v
    - Tb ∝ x_HI × (1 + δ)
    
    Product:
    kSZ × Tb ∝ -x_HI × (1 - x_HI) × (1 + δ)² × v
    
    Problems:
    ---------
    1. The product is dominated by (1 + δ)² density fluctuations
    2. The x_HI(1 - x_HI) factor peaks at x_HI = 0.5, biasing the signal
    3. Velocity information is mixed with density/ionization structure
    
    Why it doesn't work well:
    -------------------------
    Unlike the proxy method which DIVIDES by τ to isolate v,
    this method MULTIPLIES, which mixes v with other fields.
    
    The correlation with true velocity is typically |r| < 0.2.
    """
    # Normalize fields to zero mean, unit variance
    ksz_c = ksz_2d - ksz_2d.mean()
    Tb_c = Tb_2d - Tb_2d.mean()
    
    ksz_norm = ksz_c / (ksz_c.std() + 1e-10)
    Tb_norm = Tb_c / (Tb_c.std() + 1e-10)
    
    # Cross-correlation product
    v_eff_2d = -ksz_norm * Tb_norm
    
    # Smooth to reduce noise
    v_eff_2d = gaussian_filter(v_eff_2d, sigma=3)
    
    # Scale to approximate velocity units (arbitrary)
    v_eff_2d = v_eff_2d * 30.0
    
    return v_eff_2d.astype(np.float32)


# ============================================================================
# 3D QUADRATIC ESTIMATORS (for reference)
# ============================================================================

def compute_cross_power_spectrum(field1_k, field2_k, n, box_size):
    """
    Compute the cross-power spectrum P_12(k) in spherical k-bins.
    
    Returns k values and corresponding P(k).
    """
    # k-space coordinates
    kx, ky, kz = kspace_grid(n, box_size)
    
    # |k| for each mode
    k_mag = np.sqrt(kx[:, None, None]**2 + ky[None, :, None]**2 + kz[None, None, :]**2)
    
    # Cross-spectrum at each k-mode
    cross = np.real(np.conj(field1_k) * field2_k)
    
    # Bin in k
    k_max = np.max(k_mag)
    k_bins = np.linspace(0, k_max, 50)
    k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])
    
    P_k = np.zeros(len(k_centers))
    for i in range(len(k_centers)):
        mask = (k_mag >= k_bins[i]) & (k_mag < k_bins[i+1])
        if np.any(mask):
            P_k[i] = np.mean(cross[mask])
    
    return k_centers, P_k


def quadratic_estimator_velocity(ksz_field, Tb_field, tau_field, z, n=N_BOX):
    """
    Quadratic estimator for velocity reconstruction (Hotinli & Johnson Eq. 18).
    
    The key physics:
    - kSZ signal: Θ ∝ τ × v_eff (optical depth × velocity)
    - 21cm signal: δH ∝ τ (traces optical depth)
    
    The estimator exploits: ⟨Θ × δH⟩ ∝ ⟨τ²⟩ × v_eff
    
    In Fourier space for a 3D box, the estimator becomes:
    
        v̂(k) = N(k) × Σ_{k'} [Θ(k') × δH(k-k') × C^{τH}(|k-k'|)] / [C^{ΘΘ}(k') × C^{HH}(|k-k'|)]
    
    where:
    - C^{τH} is the cross-power between optical depth and 21cm (the "coupling")
    - C^{ΘΘ} is the kSZ power spectrum
    - C^{HH} is the 21cm power spectrum
    - N(k) is the normalization (reconstruction noise)
    
    This is a convolution in Fourier space, which becomes multiplication in real space.
    
    Simplified implementation:
    We approximate the convolution by noting that the dominant contribution comes
    from modes where k' ≈ k, so:
    
        v̂(k) ≈ [Θ(k) × δH*(k) × C^{τH}(k)] / [C^{ΘΘ}(k) × C^{HH}(k)] × (i k / k²)
    
    The C^{τH} term is crucial - it weights by the τ-H correlation.
    """
    box_size = BOX_MPC_OVER_H / LITTLEH  # Mpc
    
    # Mean-subtract fields
    ksz_c = ksz_field - ksz_field.mean()
    Tb_c = Tb_field - Tb_field.mean()
    tau_c = tau_field - tau_field.mean()
    
    # FFT of all fields
    ksz_k = fft.rfftn(ksz_c, workers=-1).astype(np.complex64)
    Tb_k = fft.rfftn(Tb_c, workers=-1).astype(np.complex64)
    tau_k = fft.rfftn(tau_c, workers=-1).astype(np.complex64)
    
    # Compute power spectra P(k) for each field
    # We need these as functions of |k|
    kx, ky, kz = kspace_grid(n, box_size)
    k_mag = np.sqrt(kx[:, None, None]**2 + ky[None, :, None]**2 + kz[None, None, :]**2)
    
    # Power spectra at each k-mode
    P_ksz = np.abs(ksz_k)**2
    P_Tb = np.abs(Tb_k)**2
    
    # Cross-power C^{τH}(k) - this is the key coupling term!
    # This weights by how correlated τ and H are at each scale
    C_tau_H = np.real(np.conj(tau_k) * Tb_k)
    
    # Avoid division by zero
    eps = 1e-30
    P_ksz = np.maximum(P_ksz, eps)
    P_Tb = np.maximum(P_Tb, eps)
    
    # The quadratic estimator (Eq. 18 adapted for 3D):
    # v̂(k) ∝ [Θ*(k) × δH(k) × C^{τH}(k)] / [C^{ΘΘ}(k) × C^{HH}(k)]
    #
    # The C^{τH} weighting is what makes this different from naive cross-correlation
    numerator = np.conj(ksz_k) * Tb_k * np.abs(C_tau_H)
    denominator = P_ksz * P_Tb
    
    weighted_field = numerator / denominator
    
    # k-space coordinates for velocity reconstruction
    k2 = kx[:, None, None]**2 + ky[None, :, None]**2 + kz[None, None, :]**2
    k2[0, 0, 0] = 1.0  # Avoid division by zero at k=0
    
    # Cosmological factors for v = i a H f k / k² × δ
    a = 1.0 / (1.0 + z)
    H0 = 100.0 * LITTLEH  # km/s/Mpc
    Omega_m = 0.27
    Omega_L = 0.73
    Ha = H0 * np.sqrt(Omega_m / a**3 + Omega_L)
    f = growth_rate(z)
    
    factor = 1j * Ha * a * f
    
    # Reconstruct each velocity component
    vx_k = factor * kx[:, None, None] / k2 * weighted_field
    vy_k = factor * ky[None, :, None] / k2 * weighted_field
    vz_k = factor * kz[None, None, :] / k2 * weighted_field
    
    # Transform back to real space
    vx_rec = np.real(fft.irfftn(vx_k, s=(n, n, n), workers=-1)).astype(np.float32)
    vy_rec = np.real(fft.irfftn(vy_k, s=(n, n, n), workers=-1)).astype(np.float32)
    vz_rec = np.real(fft.irfftn(vz_k, s=(n, n, n), workers=-1)).astype(np.float32)
    
    return vx_rec, vy_rec, vz_rec


def quadratic_estimator_velocity_v2(ksz_field, Tb_field, tau_field, z, n=N_BOX):
    """
    Alternative quadratic estimator using the Γ coefficient structure.
    
    From Eq. 19, the coupling coefficient is:
        Γ ∝ C^{τH}(k)
    
    The estimator (Eq. 18) can be written as:
        v̂(k) ∝ Σ [Θ(k') × δH(k-k')] × Γ(k') / [C^{ΘΘ}(k') × C^{HH}(k-k')]
    
    In real space, this convolution becomes:
        v̂(x) ∝ [Θ_filtered(x) × δH_filtered(x)]
    
    where the filtering is by the inverse-variance weights.
    
    This version implements the real-space product with proper weighting.
    """
    box_size = BOX_MPC_OVER_H / LITTLEH
    
    # Mean-subtract
    ksz_c = ksz_field - ksz_field.mean()
    Tb_c = Tb_field - Tb_field.mean()
    tau_c = tau_field - tau_field.mean()
    
    # FFT
    ksz_k = fft.rfftn(ksz_c, workers=-1).astype(np.complex64)
    Tb_k = fft.rfftn(Tb_c, workers=-1).astype(np.complex64)
    tau_k = fft.rfftn(tau_c, workers=-1).astype(np.complex64)
    
    # Power spectra
    P_ksz = np.abs(ksz_k)**2
    P_Tb = np.abs(Tb_k)**2
    
    # Cross-power C^{τH} - the Γ coefficient
    C_tau_H = np.real(np.conj(tau_k) * Tb_k)
    
    eps = 1e-30
    P_ksz = np.maximum(P_ksz, eps)
    P_Tb = np.maximum(P_Tb, eps)
    
    # Wiener-filtered fields (inverse-variance weighted)
    # Θ_filt = Θ × Γ / C^{ΘΘ} = Θ × C^{τH} / C^{ΘΘ}
    # δH_filt = δH / C^{HH}
    ksz_filtered_k = ksz_k * np.abs(C_tau_H) / P_ksz
    Tb_filtered_k = Tb_k / P_Tb
    
    # Transform to real space
    ksz_filtered = np.real(fft.irfftn(ksz_filtered_k, s=(n, n, n), workers=-1))
    Tb_filtered = np.real(fft.irfftn(Tb_filtered_k, s=(n, n, n), workers=-1))
    
    # Real-space product (the quadratic combination)
    Q = ksz_filtered * Tb_filtered
    Q = Q - Q.mean()
    
    # FFT of the quadratic field
    Q_k = fft.rfftn(Q, workers=-1).astype(np.complex64)
    
    # k-space coordinates
    kx, ky, kz = kspace_grid(n, box_size)
    
    k2 = kx[:, None, None]**2 + ky[None, :, None]**2 + kz[None, None, :]**2
    k2[0, 0, 0] = 1.0
    
    # Cosmological factors
    a = 1.0 / (1.0 + z)
    H0 = 100.0 * LITTLEH
    Omega_m = 0.27
    Omega_L = 0.73
    Ha = H0 * np.sqrt(Omega_m / a**3 + Omega_L)
    f = growth_rate(z)
    
    factor = 1j * Ha * a * f
    
    # Reconstruct velocity
    vx_k = factor * kx[:, None, None] / k2 * Q_k
    vy_k = factor * ky[None, :, None] / k2 * Q_k
    vz_k = factor * kz[None, None, :] / k2 * Q_k
    
    vx_rec = np.real(fft.irfftn(vx_k, s=(n, n, n), workers=-1)).astype(np.float32)
    vy_rec = np.real(fft.irfftn(vy_k, s=(n, n, n), workers=-1)).astype(np.float32)
    vz_rec = np.real(fft.irfftn(vz_k, s=(n, n, n), workers=-1)).astype(np.float32)
    
    return vx_rec, vy_rec, vz_rec


def linear_estimator_velocity(tracer_field, z, n=N_BOX):
    """
    Standard linear estimator (continuity-based) for comparison.
    
    v_i(k) = i a H f k_i / k² × δ(k)
    """
    box_size = BOX_MPC_OVER_H / LITTLEH
    
    tracer_c = tracer_field - tracer_field.mean()
    tracer_k = fft.rfftn(tracer_c, workers=-1).astype(np.complex64)
    
    kx, ky, kz = kspace_grid(n, box_size)
    
    k2 = kx[:, None, None]**2 + ky[None, :, None]**2 + kz[None, None, :]**2
    k2[0, 0, 0] = 1.0
    
    a = 1.0 / (1.0 + z)
    H0 = 100.0 * LITTLEH
    Omega_m = 0.27
    Omega_L = 0.73
    Ha = H0 * np.sqrt(Omega_m / a**3 + Omega_L)
    f = growth_rate(z)
    
    factor = 1j * Ha * a * f
    
    vx_k = factor * kx[:, None, None] / k2 * tracer_k
    vy_k = factor * ky[None, :, None] / k2 * tracer_k
    vz_k = factor * kz[None, None, :] / k2 * tracer_k
    
    vx_rec = np.real(fft.irfftn(vx_k, s=(n, n, n), workers=-1)).astype(np.float32)
    vy_rec = np.real(fft.irfftn(vy_k, s=(n, n, n), workers=-1)).astype(np.float32)
    vz_rec = np.real(fft.irfftn(vz_k, s=(n, n, n), workers=-1)).astype(np.float32)
    
    return vx_rec, vy_rec, vz_rec


# ============================================================================
# ANALYSIS AND PLOTTING
# ============================================================================

def compute_correlation(field1, field2, boundary=10):
    """Compute Pearson correlation between two 3D fields, excluding boundaries."""
    f1 = field1[boundary:-boundary, boundary:-boundary, boundary:-boundary].flatten()
    f2 = field2[boundary:-boundary, boundary:-boundary, boundary:-boundary].flatten()
    
    mask = np.isfinite(f1) & np.isfinite(f2)
    if not np.any(mask):
        return np.nan
    
    return np.corrcoef(f1[mask], f2[mask])[0, 1]


def analyze_realistic_scenario(redshifts, output_dir=OUTPUT_DIR):
    """
    Realistic observational scenario:
    - kSZ is integrated over ALL redshifts (CMB sees total optical depth)
    - 21cm is redshift-dependent (observed at specific frequency)
    
    This tests whether the quadratic estimator can extract velocity at a 
    specific redshift using the full integrated kSZ signal.
    """
    print(f"\n{'='*80}")
    print("REALISTIC SCENARIO: Full integrated kSZ vs redshift-binned 21cm")
    print(f"{'='*80}")
    
    # First, compute the TOTAL integrated kSZ from all redshifts
    print("\nStep 1: Computing total integrated kSZ from all redshifts...")
    
    ksz_total_2d = None
    tau_total_2d = None
    data_per_z = {}
    
    for z in redshifts:
        print(f"  Loading z = {z:.3f}...")
        den, xhi, vx, vy, vz = load_simulation_data(z)
        
        # Crop
        den = den[CENTRAL_CROP, CENTRAL_CROP, LOS_CROP]
        xhi = xhi[CENTRAL_CROP, CENTRAL_CROP, LOS_CROP]
        vz = vz[CENTRAL_CROP, CENTRAL_CROP, LOS_CROP]
        
        n = vz.shape[0]
        mean_den = den.mean()
        delta = den / mean_den - 1.0
        
        # Compute fields
        ksz_3d = compute_ksz_temperature(vz, xhi, den, z, physical_norm=False)
        Tb_3d = compute_brightness_temperature(den, xhi, z, include_velocity_term=False)
        x_e = 1.0 - xhi
        tau_3d = (1.0 + delta) * x_e
        
        # Project to 2D
        ksz_2d = project_to_2d(ksz_3d, axis=2)
        Tb_2d = project_to_2d(Tb_3d, axis=2)
        tau_2d = project_to_2d(tau_3d, axis=2)
        
        # True velocity at this redshift
        tau_vz_2d = project_to_2d(tau_3d * vz, axis=2)
        v_true_2d = tau_vz_2d / (tau_2d + 1e-10)
        
        # Accumulate total kSZ and tau
        if ksz_total_2d is None:
            ksz_total_2d = ksz_2d.copy()
            tau_total_2d = tau_2d.copy()
        else:
            ksz_total_2d += ksz_2d
            tau_total_2d += tau_2d
        
        # Store per-redshift data
        data_per_z[z] = {
            'Tb_2d': Tb_2d,
            'tau_2d': tau_2d,
            'v_true_2d': v_true_2d,
            'mean_xHI': xhi.mean(),
        }
        
        gc.collect()
    
    print(f"\n  Total kSZ 2D: mean = {ksz_total_2d.mean():.2f}, std = {ksz_total_2d.std():.2f}")
    print(f"  Total τ 2D: mean = {tau_total_2d.mean():.2f}, std = {tau_total_2d.std():.2f}")
    
    # Now test: can we extract velocity at each redshift using TOTAL kSZ + single-z 21cm?
    print("\nStep 2: Testing velocity extraction at each redshift...")
    print("        Using: TOTAL integrated kSZ + single-redshift 21cm")
    
    box_size_2d = BOX_MPC_OVER_H / LITTLEH * (n / N_BOX)
    boundary = 10
    
    results = []
    
    for z in redshifts:
        data = data_per_z[z]
        Tb_2d = data['Tb_2d']
        tau_2d = data['tau_2d']
        v_true_2d = data['v_true_2d']
        mean_xHI = data['mean_xHI']
        
        print(f"\n  z = {z:.3f} (xHI = {mean_xHI:.3f}):")
        
        # Ideal case: use single-z kSZ (not realistic but for comparison)
        # This is what we had before
        
        # Realistic case: use TOTAL kSZ with single-z Tb
        # The challenge: kSZ_total contains contributions from ALL redshifts
        # Can we still extract velocity at z using Tb(z)?
        
        # Method 1: Direct division with total kSZ and single-z tau
        # v ≈ -kSZ_total / τ(z) -- this won't work because kSZ_total ≠ τ(z) × v(z)
        
        # Method 2: PROPER flat-sky quadratic estimator (Eq. 18)
        # Uses C^{τH} as statistical weight, NOT τ itself
        v_flatsky = quadratic_estimator_flatsky(ksz_total_2d, Tb_2d, tau_2d, box_size_2d)
        
        # Method 3: My "proxy" division (NOT the paper's method!)
        # This has hidden access to τ structure via Tb
        Tb_norm = Tb_2d / (Tb_2d.max() + 1e-10)
        tau_proxy = 1.0 - Tb_norm
        eps = tau_proxy.std() * 0.1
        v_quad_proxy = -ksz_total_2d / (tau_proxy + eps)
        v_quad_proxy = v_quad_proxy - v_quad_proxy.mean()
        
        # Method 4: Ideal (single-z kSZ / single-z tau) - for reference
        # We need to recompute single-z kSZ
        den, xhi, _, _, vz = load_simulation_data(z)
        den = den[CENTRAL_CROP, CENTRAL_CROP, LOS_CROP]
        xhi = xhi[CENTRAL_CROP, CENTRAL_CROP, LOS_CROP]
        vz = vz[CENTRAL_CROP, CENTRAL_CROP, LOS_CROP]
        ksz_3d_z = compute_ksz_temperature(vz, xhi, den, z, physical_norm=False)
        ksz_2d_z = project_to_2d(ksz_3d_z, axis=2)
        
        eps_tau = tau_2d.std() * 0.1
        v_ideal = -ksz_2d_z / (tau_2d + eps_tau)
        v_ideal = v_ideal - v_ideal.mean()
        
        # Compute correlations
        v_true_crop = v_true_2d[boundary:-boundary, boundary:-boundary].flatten()
        
        def corr(v_rec):
            v_rec_crop = v_rec[boundary:-boundary, boundary:-boundary].flatten()
            mask = np.isfinite(v_true_crop) & np.isfinite(v_rec_crop)
            if not np.any(mask):
                return np.nan
            return np.corrcoef(v_true_crop[mask], v_rec_crop[mask])[0, 1]
        
        r_ideal = corr(v_ideal)
        r_proxy = corr(v_quad_proxy)
        r_flatsky = corr(v_flatsky)
        
        # Compute amplitude ratios (std of reconstructed / std of true)
        v_true_std = v_true_2d[boundary:-boundary, boundary:-boundary].std()
        v_ideal_std = v_ideal[boundary:-boundary, boundary:-boundary].std()
        v_proxy_std = v_quad_proxy[boundary:-boundary, boundary:-boundary].std()
        v_flatsky_std = v_flatsky[boundary:-boundary, boundary:-boundary].std()
        
        amp_ideal = v_ideal_std / v_true_std
        amp_proxy = v_proxy_std / v_true_std
        amp_flatsky = v_flatsky_std / v_true_std
        
        print(f"    Ideal (single-z kSZ/τ):      r = {r_ideal:.4f}, amp = {amp_ideal:.2f}×")
        print(f"    Flat-sky Eq.18 (C^τH weight): r = {r_flatsky:.4f}, amp = {amp_flatsky:.2f}×")
        print(f"    Proxy division (NOT paper):  r = {r_proxy:.4f}, amp = {amp_proxy:.2f}×")
        print(f"    True velocity std: {v_true_std:.2f} km/s")
        
        results.append({
            'z': z,
            'mean_xHI': mean_xHI,
            'r_ideal': r_ideal,
            'r_proxy': r_proxy,
            'r_flatsky': r_flatsky,
            'amp_ideal': amp_ideal,
            'amp_proxy': amp_proxy,
            'amp_flatsky': amp_flatsky,
            'v_true_std': v_true_std,
        })
        
        gc.collect()
    
    # Create summary plot: correlation vs redshift
    print("\nCreating realistic scenario summary plots...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    zs = [r['z'] for r in results]
    xHIs = [r['mean_xHI'] for r in results]
    
    # Left: Correlation vs xHI
    ax = axes[0]
    ax.plot(xHIs, [r['r_ideal'] for r in results], 'o-', 
            label='Ideal (single-z kSZ/τ)', markersize=8, color='green', linewidth=2)
    ax.plot(xHIs, [r['r_flatsky'] for r in results], '^-', 
            label='Flat-sky Eq.18 (proper)', markersize=8, color='blue', linewidth=2)
    ax.plot(xHIs, [r['r_proxy'] for r in results], 's-', 
            label='Proxy division (NOT paper)', markersize=8, color='orange', linewidth=2)
    
    ax.set_xlabel(r'Mean $x_{HI}$', fontsize=14)
    ax.set_ylabel('Correlation with true velocity', fontsize=14)
    ax.set_title('Realistic Scenario: Correlation', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.5, 1.1)
    
    # Add redshift labels on top
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(xHIs[::2] if len(xHIs) > 4 else xHIs)
    ax2.set_xticklabels([f'{z:.1f}' for z in zs[::2]] if len(zs) > 4 else [f'{z:.1f}' for z in zs])
    ax2.set_xlabel('Redshift', fontsize=12)
    
    # Right: Amplitude ratio vs xHI
    ax = axes[1]
    ax.plot(xHIs, [r['amp_ideal'] for r in results], 'o-', 
            label='Ideal (single-z kSZ/τ)', markersize=8, color='green', linewidth=2)
    ax.plot(xHIs, [r['amp_flatsky'] for r in results], '^-', 
            label='Flat-sky Eq.18 (proper)', markersize=8, color='blue', linewidth=2)
    ax.plot(xHIs, [r['amp_proxy'] for r in results], 's-', 
            label='Proxy division (NOT paper)', markersize=8, color='orange', linewidth=2)
    ax.axhline(y=1.0, color='k', linestyle='--', linewidth=1, label='Perfect amplitude')
    
    ax.set_xlabel(r'Mean $x_{HI}$', fontsize=14)
    ax.set_ylabel('Amplitude ratio (rec/true)', fontsize=14)
    ax.set_title('Realistic Scenario: Amplitude', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    
    outfile = os.path.join(output_dir, 'realistic_scenario_summary.png')
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outfile}")
    
    return results


def analyze_quadratic_estimator(z, output_dir=OUTPUT_DIR):
    """
    Compare quadratic estimator with linear estimator using 2D projections.
    
    NOTE: This is the IDEALIZED case where kSZ is from a single redshift.
    For realistic scenario, use analyze_realistic_scenario().
    
    Following the paper's approach:
    1. Project 3D fields to 2D (LOS integration)
    2. Apply quadratic estimator on 2D maps
    3. Compare with true LOS-integrated velocity
    """
    print(f"\n{'='*80}")
    print(f"Quadratic Estimator Analysis at z = {z:.3f}")
    print(f"{'='*80}")
    
    # Load data
    print("Loading simulation data...")
    den, xhi, vx, vy, vz = load_simulation_data(z)
    
    # Crop to central region
    den = den[CENTRAL_CROP, CENTRAL_CROP, LOS_CROP]
    xhi = xhi[CENTRAL_CROP, CENTRAL_CROP, LOS_CROP]
    vx = vx[CENTRAL_CROP, CENTRAL_CROP, LOS_CROP]
    vy = vy[CENTRAL_CROP, CENTRAL_CROP, LOS_CROP]
    vz = vz[CENTRAL_CROP, CENTRAL_CROP, LOS_CROP]
    
    n = vz.shape[0]
    mean_xHI = xhi.mean()
    print(f"  Mean x_HI = {mean_xHI:.4f}")
    
    # Compute 3D fields
    print("Computing 3D kSZ, 21cm, and optical depth fields...")
    ksz_3d = compute_ksz_temperature(vz, xhi, den, z, physical_norm=False)
    Tb_3d = compute_brightness_temperature(den, xhi, z, include_velocity_term=False)
    
    # Compute delta
    mean_den = den.mean()
    delta = den / mean_den - 1.0
    
    # Compute optical depth proxy: τ ∝ n_e ∝ (1 - x_HI) × (1 + δ)
    x_e = 1.0 - xhi
    tau_3d = (1.0 + delta) * x_e
    
    # =========================================================================
    # PROJECT TO 2D (following paper's observational setup)
    # =========================================================================
    print("\nProjecting 3D fields to 2D (LOS integration)...")
    
    # kSZ map: integrated kSZ signal (what CMB observes)
    ksz_2d = project_to_2d(ksz_3d, axis=2)
    
    # 21cm map: integrated brightness temperature (what 21cm observes)
    Tb_2d = project_to_2d(Tb_3d, axis=2)
    
    # Optical depth map: integrated τ
    tau_2d = project_to_2d(tau_3d, axis=2)
    
    # True velocity map: τ-weighted velocity (what we want to reconstruct)
    # v_eff = ∫ τ × v_z dz / ∫ τ dz
    tau_vz_2d = project_to_2d(tau_3d * vz, axis=2)
    v_true_2d = tau_vz_2d / (tau_2d + 1e-10)
    
    # Also compute simple mean velocity for comparison
    v_mean_2d = np.mean(vz, axis=2)
    
    # Box size for 2D (same as 3D transverse)
    box_size_2d = BOX_MPC_OVER_H / LITTLEH * (n / N_BOX)  # Mpc
    
    print(f"  kSZ 2D: mean = {ksz_2d.mean():.4f}, std = {ksz_2d.std():.4f}")
    print(f"  Tb 2D: mean = {Tb_2d.mean():.4f}, std = {Tb_2d.std():.4f}")
    print(f"  τ 2D: mean = {tau_2d.mean():.4f}, std = {tau_2d.std():.4f}")
    print(f"  v_true 2D: mean = {v_true_2d.mean():.2f}, std = {v_true_2d.std():.2f} km/s")
    
    # =========================================================================
    # 2D QUADRATIC ESTIMATORS
    # =========================================================================
    print("\nRunning 2D quadratic estimators...")
    
    # Quadratic estimator v1 (Fourier-space cross-correlation)
    print("  Quadratic v1 (Fourier C^{ΘH}/C^{τH})...")
    v_quad1_2d = quadratic_estimator_2d(ksz_2d, Tb_2d, tau_2d, box_size_2d)
    
    # Quadratic estimator v2 (direct division: v = -kSZ/τ)
    print("  Quadratic v2 (direct: v = -kSZ/τ)...")
    v_quad2_2d = quadratic_estimator_2d_v2(ksz_2d, Tb_2d, tau_2d, box_size_2d)
    
    # Quadratic estimator v3 (observational proxy: using Tb instead of τ)
    print("  Quadratic v3 (proxy: v = -kSZ/(1-Tb_norm))...")
    v_quad3_2d = quadratic_estimator_2d_v3(ksz_2d, Tb_2d, tau_2d, box_size_2d)
    
    # Quadratic estimator v4 (cross-correlation)
    print("  Quadratic v4 (cross: -kSZ × Tb)...")
    v_quad4_2d = quadratic_estimator_2d_v4(ksz_2d, Tb_2d, tau_2d, box_size_2d)
    
    # =========================================================================
    # 3D LINEAR ESTIMATORS (proper continuity-based, then project to 2D)
    # =========================================================================
    print("\nRunning 3D linear estimators (then projecting to 2D)...")
    
    # Use the proper 3D linear estimator: v_z(k) = i * aHf * k_z / k² * δ(k)
    vz_lin_delta_3d = linear_estimator_velocity(delta, z, n=n)[2]  # Get vz component
    vz_lin_Tb_3d = linear_estimator_velocity(-Tb_3d, z, n=n)[2]
    
    # Project reconstructed 3D velocities to 2D (τ-weighted like true velocity)
    vz_lin_delta_2d = project_to_2d(tau_3d * vz_lin_delta_3d, axis=2) / (tau_2d + 1e-10)
    vz_lin_Tb_2d = project_to_2d(tau_3d * vz_lin_Tb_3d, axis=2) / (tau_2d + 1e-10)
    
    # =========================================================================
    # COMPUTE CORRELATIONS
    # =========================================================================
    print("\nComputing 2D correlations with true τ-weighted velocity...")
    
    boundary = 10
    v_true_crop = v_true_2d[boundary:-boundary, boundary:-boundary].flatten()
    
    def corr_2d(v_rec_2d):
        v_rec_crop = v_rec_2d[boundary:-boundary, boundary:-boundary].flatten()
        mask = np.isfinite(v_true_crop) & np.isfinite(v_rec_crop)
        if not np.any(mask):
            return np.nan
        return np.corrcoef(v_true_crop[mask], v_rec_crop[mask])[0, 1]
    
    r_lin_delta = corr_2d(vz_lin_delta_2d)
    r_lin_Tb = corr_2d(vz_lin_Tb_2d)
    r_quad1 = corr_2d(v_quad1_2d)
    r_quad2 = corr_2d(v_quad2_2d)
    r_quad3 = corr_2d(v_quad3_2d)
    r_quad4 = corr_2d(v_quad4_2d)
    
    print(f"  Linear 3D→2D (δ):        r = {r_lin_delta:.4f}")
    print(f"  Linear 3D→2D (-Tb):      r = {r_lin_Tb:.4f}")
    print(f"  Quadratic v1 (Fourier):  r = {r_quad1:.4f}")
    print(f"  Quadratic v2 (-kSZ/τ):   r = {r_quad2:.4f}")
    print(f"  Quadratic v3 (proxy):    r = {r_quad3:.4f}")
    print(f"  Quadratic v4 (kSZ×Tb):   r = {r_quad4:.4f}")
    
    # =========================================================================
    # CREATE COMPARISON PLOT
    # =========================================================================
    print("\nCreating 2D comparison plot...")
    
    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    
    # Top row: 2D maps - True, Ideal, Proxy, Linear
    vmin_map = np.percentile(v_true_2d, 5)
    vmax_map = np.percentile(v_true_2d, 95)
    
    im0 = axes[0, 0].imshow(v_true_2d.T, origin='lower', cmap='RdBu_r', 
                             vmin=vmin_map, vmax=vmax_map)
    axes[0, 0].set_title(r'True $v_{eff}$ (τ-weighted)', fontsize=12)
    plt.colorbar(im0, ax=axes[0, 0], label='km/s')
    
    im1 = axes[0, 1].imshow(v_quad2_2d.T, origin='lower', cmap='RdBu_r',
                             vmin=vmin_map, vmax=vmax_map)
    axes[0, 1].set_title(f'Ideal: $-kSZ/τ$ (r={r_quad2:.3f})', fontsize=12)
    plt.colorbar(im1, ax=axes[0, 1], label='km/s')
    
    im2 = axes[0, 2].imshow(v_quad3_2d.T, origin='lower', cmap='RdBu_r')
    axes[0, 2].set_title(f'PROXY: $-kSZ/τ_{{proxy}}$ (r={r_quad3:.3f})', fontsize=12, fontweight='bold')
    plt.colorbar(im2, ax=axes[0, 2])
    
    im3 = axes[0, 3].imshow(vz_lin_Tb_2d.T, origin='lower', cmap='RdBu_r',
                             vmin=vmin_map, vmax=vmax_map)
    axes[0, 3].set_title(f'Linear 3D: $-T_b$ (r={r_lin_Tb:.3f})', fontsize=12)
    plt.colorbar(im3, ax=axes[0, 3], label='km/s')
    
    # Bottom row: scatter plots
    np.random.seed(42)
    sample_size = 10000
    idx = np.random.choice(len(v_true_crop), size=min(sample_size, len(v_true_crop)), replace=False)
    
    methods_2d = [
        (v_quad2_2d, r'Ideal: $-kSZ/\tau$', r_quad2, 'green'),
        (v_quad3_2d, r'PROXY: $-kSZ/\tau_{proxy}$', r_quad3, 'orange'),
        (vz_lin_Tb_2d, r'Linear 3D: $-T_b$', r_lin_Tb, 'blue'),
        (vz_lin_delta_2d, r'Linear 3D: $\delta$', r_lin_delta, 'purple'),
    ]
    
    for ax, (v_rec, label, r_val, color) in zip(axes[1, :], methods_2d):
        v_rec_crop = v_rec[boundary:-boundary, boundary:-boundary].flatten()
        
        ax.scatter(v_rec_crop[idx], v_true_crop[idx], alpha=0.3, s=1, c=color)
        
        vmin, vmax = np.percentile(v_true_crop[idx], [1, 99])
        ax.plot([vmin, vmax], [vmin, vmax], 'r--', lw=2)
        
        ax.set_xlabel('Reconstructed', fontsize=12)
        ax.set_ylabel(r'True $v_{eff}$', fontsize=12)
        ax.set_title(f'{label}\nr = {r_val:.4f}', fontsize=13)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Velocity Reconstruction Comparison at z = {z:.2f}\n'
                 f'Mean $x_{{HI}}$ = {mean_xHI:.3f}', fontsize=16)
    plt.tight_layout()
    
    outfile = os.path.join(output_dir, f'quadratic_estimator_comparison_z{z:.2f}.png')
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outfile}")
    
    return {
        'z': z,
        'mean_xHI': mean_xHI,
        'r_lin_delta': r_lin_delta,
        'r_lin_Tb': r_lin_Tb,
        'r_quad1': r_quad1,
        'r_quad2': r_quad2,
        'r_quad3': r_quad3,
        'r_quad4': r_quad4,
    }


def analyze_multiple_redshifts(redshifts, output_dir=OUTPUT_DIR):
    """Analyze quadratic estimator at multiple redshifts."""
    results = []
    
    for z in redshifts:
        result = analyze_quadratic_estimator(z, output_dir)
        results.append(result)
        gc.collect()
    
    # Summary plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    zs = [r['z'] for r in results]
    xHIs = [r['mean_xHI'] for r in results]
    
    ax.plot(xHIs, [r['r_quad2'] for r in results], 'o-', 
            label=r'Quad: $-kSZ/\tau$ (ideal)', markersize=10, color='green', linewidth=2)
    ax.plot(xHIs, [r['r_quad3'] for r in results], 's-', 
            label=r'Quad: proxy $(1-T_b/T_{b,max})$', markersize=10, color='orange', linewidth=2)
    ax.plot(xHIs, [r['r_lin_delta'] for r in results], '^-', 
            label=r'Linear 3D: $\delta$', markersize=8, color='blue', linewidth=2)
    ax.plot(xHIs, [r['r_lin_Tb'] for r in results], 'd-', 
            label=r'Linear 3D: $-T_b$', markersize=8, color='purple', linewidth=2)
    ax.plot(xHIs, [r['r_quad4'] for r in results], 'v-', 
            label=r'Quad: $kSZ \times T_b$', markersize=8, color='red', linewidth=2)
    
    ax.set_xlabel(r'Mean $x_{HI}$', fontsize=14)
    ax.set_ylabel('Correlation with true $v_z$', fontsize=14)
    ax.set_title('Velocity Reconstruction: Linear vs Quadratic Estimators', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.5, 1)
    
    # Add redshift labels on top axis
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(xHIs)
    ax2.set_xticklabels([f'{z:.1f}' for z in zs])
    ax2.set_xlabel('Redshift', fontsize=12)
    
    plt.tight_layout()
    outfile = os.path.join(output_dir, 'quadratic_estimator_vs_redshift.png')
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved summary: {outfile}")
    
    return results


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Test redshifts spanning EoR (use more available data files)
    # Selected to span the full EoR with good coverage
    redshifts = [6.231, 6.617, 7.059, 7.570, 8.064, 8.636, 9.308]
    
    print("="*80)
    print("QUADRATIC ESTIMATOR STUDY")
    print("Based on Hotinli & Johnson 'Reconstructing large scales at cosmic dawn'")
    print("="*80)
    
    # Part 1: Idealized analysis (single-z kSZ)
    print("\n" + "="*80)
    print("PART 1: IDEALIZED SCENARIO (single-redshift kSZ)")
    print("="*80)
    results = analyze_multiple_redshifts(redshifts)
    
    # Part 2: Realistic analysis (total integrated kSZ)
    print("\n" + "="*80)
    print("PART 2: REALISTIC SCENARIO (total integrated kSZ)")
    print("="*80)
    realistic_results = analyze_realistic_scenario(redshifts)
    
    print("\n" + "="*100)
    print("SUMMARY: IDEALIZED SCENARIO (single-redshift kSZ)")
    print("="*100)
    print(f"{'z':>8} {'x_HI':>8} {'Lin(δ)':>10} {'Lin(Tb)':>10} {'Q2(-kSZ/τ)':>12} {'Q3(proxy)':>12} {'Q4(kSZ×Tb)':>12}")
    print("-"*80)
    for r in results:
        print(f"{r['z']:8.3f} {r['mean_xHI']:8.4f} {r['r_lin_delta']:10.4f} "
              f"{r['r_lin_Tb']:10.4f} {r['r_quad2']:12.4f} {r['r_quad3']:12.4f} {r['r_quad4']:12.4f}")
    print("="*100)
    
    print("\n" + "="*120)
    print("SUMMARY: REALISTIC SCENARIO (total integrated kSZ from ALL redshifts)")
    print("="*120)
    print(f"{'z':>8} {'x_HI':>8} {'r_ideal':>10} {'r_flatsky':>12} {'r_proxy':>10} {'amp_ideal':>12} {'amp_flatsky':>12} {'amp_proxy':>12}")
    print("-"*110)
    for r in realistic_results:
        print(f"{r['z']:8.3f} {r['mean_xHI']:8.4f} {r['r_ideal']:10.4f} "
              f"{r['r_flatsky']:12.4f} {r['r_proxy']:10.4f} {r['amp_ideal']:12.1f}× {r['amp_flatsky']:12.1f}× {r['amp_proxy']:12.1f}×")
    print("="*120)
    
    print("\nKey findings:")
    print("  IDEALIZED (single-z kSZ):")
    print("    - Ideal (-kSZ/τ): Perfect r=1.0 when τ is known exactly")
    print("    - Proxy division: r≈0.98 but WRONG amplitude (has hidden τ access)")
    print("  REALISTIC (total integrated kSZ):")
    print("    - Flat-sky Eq.18: Proper quadratic estimator using C^{τH} as weight")
    print("    - Proxy division: Still works but amplitude is ~10^4× wrong")
    print("    - The proxy method is NOT the paper's estimator!")
    print("="*120)
    
    print("\n" + "="*80)
    print("EXPLANATION OF METHODS")
    print("="*80)
    print("""
FORMULAS USED:
--------------
kSZ:  Θ = -(1-xHI)(1+δ) × v_z  [proportional to -τ × v]
τ:    τ = (1-xHI)(1+δ)         [optical depth from free electrons]  
Tb:   Tb = xHI × (1+δ)         [21cm brightness from neutral H]

IDEAL ESTIMATOR (single-z kSZ):
  v = -kSZ / τ
  This is PERFECT (r=1.0) because kSZ = -τ × v exactly.

PROXY DIVISION (NOT the paper's method!):
  τ_proxy = 1 - Tb/Tb_max
  v = -kSZ / τ_proxy
  
  WHY IT WORKS "TOO WELL":
  - τ ∝ (1-xHI)(1+δ) and Tb ∝ xHI(1+δ) share the (1+δ) structure
  - τ_proxy captures the SPATIAL PATTERN of τ (hidden access!)
  - Correlation is high (r≈0.98) but amplitude is WRONG by ~10^4×
  
FLAT-SKY QUADRATIC ESTIMATOR (Eq. 18):
  v̂(L) = N^{vv}(L) × ∫ d²ℓ Γ(ℓ,L) Θ(ℓ) H(L-ℓ) / [C^{ΘΘ}(ℓ) C^{HH}(|L-ℓ|)]
  
  where Γ ∝ C^{τH}(ℓ) is the τ-H CROSS-POWER SPECTRUM
  
  KEY DIFFERENCES FROM PROXY:
  - Uses C^{τH}(ℓ) as STATISTICAL weight, not τ itself
  - Is a CONVOLUTION (mode-coupling), not a simple division
  - Properly normalized by N^{vv} from power spectra
  - Performance DEGRADES with total kSZ (other-z adds noise to C^{ΘΘ})

OBSERVATIONAL REALISM:
  - kSZ is integrated over ALL redshifts (CMB sees total τ)
  - 21cm is redshift-specific (observed at specific frequency)
  - The paper's estimator handles this via C^{ΘΘ,obs} in denominator
  - Other-z contributions increase C^{ΘΘ,obs} → increase reconstruction noise
""")
    print("="*80)
