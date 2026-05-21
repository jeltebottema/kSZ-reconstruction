#!/usr/bin/env python3
"""
================================================================================
QUADRATIC ESTIMATOR FOR VELOCITY RECONSTRUCTION
================================================================================

Based on Hotinli & Johnson "Reconstructing large scales at cosmic dawn"
Equations (18)-(20): Quadratic estimator using CMB × 21cm cross-correlation

OBSERVATIONALLY-VIABLE comparison between:
  (A) Linear continuity-based velocity reconstruction from 3D density
  (B) Paper-style quadratic estimator for the remote dipole field

================================================================================
MATHEMATICAL BACKGROUND
================================================================================

THE QUADRATIC ESTIMATOR (Flat-sky version of Eq. 18):
------------------------------------------------------
    v̂^α(L) = b_v^α × N_α(L) × ∫ d²ℓ/(2π)² × Γ^α(ℓ,L) × Θ_total(ℓ) × H_α(L-ℓ)
                                            / [C_ΘΘ^obs(ℓ) × C_HH,α^obs(|L-ℓ|)]

where:
  - Θ_total = TOTAL kSZ temperature (integrated over ALL redshifts)
  - H_α = 21cm brightness in redshift bin α
  - Γ^α ∝ C_τH,α = cross-power of optical depth and 21cm in bin α
  - N_α(L) = normalization from Eq. 20

NORMALIZATION (Eq. 20):
    1/N_α(L) = ∫ d²ℓ/(2π)² × |Γ^α|² / [C_ΘΘ^obs(ℓ) × C_HH,α^obs(|L-ℓ|)]

TRUE TARGET FIELD:
    v_eff^α(n̂) = [∫_α dχ τ(χ,n̂) v_r(χ,n̂)] / [∫_α dχ τ(χ,n̂)]

WHY CROSS-CORRELATION, NOT DIVISION:
1. Division v = -Θ/τ is NOT observable (τ not measured directly)
2. Θ_total includes ALL redshifts; division cannot separate by bin α
3. The estimator uses off-diagonal covariance ⟨Θ(ℓ₁) H_α(ℓ₂)⟩
4. Other redshifts enter as "noise" via C_ΘΘ^obs

================================================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from scipy import fft
from scipy.ndimage import gaussian_filter
from scipy.integrate import quad
from scipy.interpolate import interp1d
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


def safe_real(arr):
    """Safely extract real part."""
    return np.real(arr).astype(np.float32, copy=False)


def kspace_rfft_3d(n, box_size, dtype=np.float32):
    """Generate k-space coordinates for 3D rfft."""
    rc = box_size / float(n)
    kx = 2.0 * np.pi * np.fft.fftfreq(n, d=rc).astype(dtype)
    ky = 2.0 * np.pi * np.fft.fftfreq(n, d=rc).astype(dtype)
    kz = 2.0 * np.pi * np.fft.rfftfreq(n, d=rc).astype(dtype)
    # Protect against divide by zero
    tiny = np.finfo(dtype).tiny
    if kz.size: kz[0] = max(kz[0], tiny)
    if kx.size: kx[0] = max(kx[0], tiny)
    if ky.size: ky[0] = max(ky[0], tiny)
    return kx, ky, kz


# ============================================================================
# MAP BUILDING FUNCTIONS (per-redshift)
# ============================================================================

def build_maps_for_z(z, crop=True):
    """
    Build all 2D maps for a single redshift bin.
    
    Returns:
    --------
    dict with keys:
        'tau_2d': optical depth map (for truth only, NOT used in estimators)
        'Tb_2d': 21cm brightness temperature map (H_α)
        'ksz_2d': kSZ temperature map (contribution from this z)
        'v_true_2d': true τ-weighted velocity
        'delta_2d': density contrast map
        'mean_xHI': mean neutral fraction
        'z': redshift
    """
    den, xhi, vx, vy, vz = load_simulation_data(z)
    
    if crop:
        den = den[CENTRAL_CROP, CENTRAL_CROP, LOS_CROP]
        xhi = xhi[CENTRAL_CROP, CENTRAL_CROP, LOS_CROP]
        vz = vz[CENTRAL_CROP, CENTRAL_CROP, LOS_CROP]
    
    n = den.shape[0]
    mean_den = den.mean()
    delta = den / mean_den - 1.0
    mean_xHI = xhi.mean()
    
    # Optical depth: τ ∝ n_e ∝ (1 - x_HI)(1 + δ)
    x_e = 1.0 - xhi
    tau_3d = (1.0 + delta) * x_e
    
    # 21cm brightness: Tb ∝ x_HI(1 + δ)
    Tb_3d = (1.0 + delta) * xhi
    
    # kSZ: Θ ∝ -τ × v_r
    ksz_3d = -tau_3d * vz
    
    # Project to 2D (LOS integration along z-axis)
    tau_2d = np.sum(tau_3d, axis=2).astype(np.float32)
    Tb_2d = np.sum(Tb_3d, axis=2).astype(np.float32)
    ksz_2d = np.sum(ksz_3d, axis=2).astype(np.float32)
    delta_2d = np.sum(delta, axis=2).astype(np.float32)
    
    # True τ-weighted velocity: v_eff = ∫τv dχ / ∫τ dχ
    tau_v_2d = np.sum(tau_3d * vz, axis=2).astype(np.float32)
    v_true_2d = tau_v_2d / (tau_2d + 1e-10)
    
    return {
        'tau_2d': tau_2d,
        'Tb_2d': Tb_2d,
        'ksz_2d': ksz_2d,
        'v_true_2d': v_true_2d,
        'delta_2d': delta_2d,
        'delta_3d': delta,
        'tau_3d': tau_3d,
        'vz_3d': vz,
        'mean_xHI': mean_xHI,
        'z': z,
        'n': n,
    }


def build_ksz_total(redshifts):
    """
    Build the TOTAL integrated kSZ map from all redshifts.
    
    This is what CMB observes: Θ_total = Σ_α Θ_α
    
    Returns:
    --------
    Theta_total_2d: 2D kSZ map integrated over all redshifts
    data_per_z: dict of per-redshift data for later use
    """
    print(f"\nBuilding total kSZ from {len(redshifts)} redshifts...")
    
    Theta_total_2d = None
    tau_total_2d = None
    data_per_z = {}
    
    for z in redshifts:
        print(f"  z = {z:.3f}...", end=" ")
        data = build_maps_for_z(z)
        data_per_z[z] = data
        
        if Theta_total_2d is None:
            Theta_total_2d = data['ksz_2d'].copy()
            tau_total_2d = data['tau_2d'].copy()
        else:
            Theta_total_2d += data['ksz_2d']
            tau_total_2d += data['tau_2d']
        
        print(f"xHI = {data['mean_xHI']:.3f}")
        gc.collect()
    
    print(f"  Total Θ: mean = {Theta_total_2d.mean():.2f}, std = {Theta_total_2d.std():.2f}")
    
    return Theta_total_2d, tau_total_2d, data_per_z


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


def quadratic_estimator_flatsky(Theta_total_2d, H_alpha_2d, C_tauH_func, 
                                  C_TT_obs_func, C_HH_obs_func, box_size_2d,
                                  b_v=1.0, N_TT=0.0, N_HH=0.0):
    """
    Flat-sky quadratic estimator following Hotinli & Johnson Eq. 18-20.
    
    v̂^α(L) = N_α(L) × ∫ d²ℓ/(2π)² × Γ^α(ℓ,L) × Θ(ℓ) × H(L-ℓ) 
                                   / [C_ΘΘ^obs(ℓ) × C_HH^obs(|L-ℓ|)]
    
    This implementation uses the convolution theorem:
    - The integral is a convolution of filtered Θ and filtered H
    - Convolution in Fourier space = product in real space
    
    Parameters:
    -----------
    Theta_total_2d : TOTAL kSZ map (integrated over ALL redshifts)
    H_alpha_2d : 21cm brightness map for bin α
    C_tauH_func, C_TT_obs_func, C_HH_obs_func : Power spectrum functions
    box_size_2d : Box size in Mpc
    
    Returns:
    --------
    v_hat_2d, N_vv_2d : Reconstructed velocity and noise maps
    """
    n = Theta_total_2d.shape[0]
    
    # Mean-subtract
    Theta_c = Theta_total_2d - Theta_total_2d.mean()
    H_c = H_alpha_2d - H_alpha_2d.mean()
    
    # 2D FFT
    Theta_k = np.fft.fft2(Theta_c)
    H_k = np.fft.fft2(H_c)
    
    # k-space coordinates
    dk = 2.0 * np.pi / box_size_2d
    kx = np.fft.fftfreq(n, d=1.0/n) * dk
    ky = np.fft.fftfreq(n, d=1.0/n) * dk
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    ell = np.sqrt(KX**2 + KY**2)
    
    # Get power spectra at each mode
    eps = 1e-30
    C_TT = np.maximum(C_TT_obs_func(ell), eps) + N_TT
    C_HH = np.maximum(C_HH_obs_func(ell), eps) + N_HH
    C_tauH = C_tauH_func(ell)
    
    # Γ coefficient: C_τH (negative because τ and H are anti-correlated)
    Gamma = C_tauH
    
    # =========================================================================
    # CONVOLUTION-BASED ESTIMATOR
    # =========================================================================
    # v̂(L) = N(L) × ∫ d²ℓ [Θ(ℓ)/C_TT(ℓ)] × [H(L-ℓ) × Γ(L-ℓ)/C_HH(L-ℓ)]
    #
    # This is [Θ_filt] ⊛ [H × Γ_filt] where ⊛ is convolution
    # Convolution theorem: FFT(A ⊛ B) = FFT(A) × FFT(B) for cyclic convolution
    # But we want: [A ⊛ B](L) = ∫ A(ℓ) B(L-ℓ) dℓ
    # In real space: A(x) × B(x) then FFT gives the convolution
    # =========================================================================
    
    # Filter 1: Θ(ℓ) / C_TT(ℓ)
    Theta_filt_k = Theta_k / C_TT
    
    # Filter 2: H(ℓ) × Γ(ℓ) / C_HH(ℓ)
    H_Gamma_filt_k = H_k * Gamma / C_HH
    
    # To real space
    Theta_filt_x = np.fft.ifft2(Theta_filt_k)
    H_Gamma_filt_x = np.fft.ifft2(H_Gamma_filt_k)
    
    # Product in real space = convolution in Fourier space
    Q_x = Theta_filt_x * H_Gamma_filt_x
    Q_k = np.fft.fft2(Q_x)
    
    # =========================================================================
    # NORMALIZATION
    # =========================================================================
    # 1/N(L) = ∫ d²ℓ |Γ(L-ℓ)|² / [C_TT(ℓ) × C_HH(L-ℓ)]
    # This is also a convolution: [1/C_TT] ⊛ [Γ²/C_HH]
    # =========================================================================
    
    term1_k = 1.0 / C_TT
    term2_k = Gamma**2 / C_HH
    
    term1_x = np.fft.ifft2(term1_k)
    term2_x = np.fft.ifft2(term2_k)
    inv_N_x = term1_x * term2_x
    inv_N_k = np.fft.fft2(inv_N_x)
    
    # N(L) with regularization
    N_vv_k = 1.0 / (np.abs(inv_N_k) + eps)
    
    # Apply normalization
    v_hat_k = b_v * N_vv_k * Q_k
    
    # Back to real space
    v_hat_2d = np.real(np.fft.ifft2(v_hat_k)).astype(np.float32)
    v_hat_2d = v_hat_2d - v_hat_2d.mean()
    
    N_vv_2d = np.real(np.fft.ifft2(N_vv_k)).astype(np.float32)
    
    return v_hat_2d, N_vv_2d


def proxy_estimator(Theta_2d, H_2d):
    """
    Proxy-based velocity estimator: v̂ ∝ -Θ / H
    
    This uses H (21cm brightness) as a proxy for τ (optical depth).
    
    Physics:
    - kSZ: Θ = -τ × v where τ ∝ (1-xHI)(1+δ)
    - 21cm: H ∝ xHI(1+δ)
    - Proxy: -Θ/H ∝ [(1-xHI)(1+δ) × v] / [xHI(1+δ)] = (1-xHI)/xHI × v
    
    This works well (r≈0.93) because H captures the (1+δ) structure.
    
    WARNING: This is NOT the paper's quadratic estimator! It implicitly
    uses the spatial structure of τ through H, which is not observable
    in the same way. The amplitude will be wrong.
    
    Parameters:
    -----------
    Theta_2d : kSZ temperature map (can be total or single-z)
    H_2d : 21cm brightness map for the target redshift bin
    
    Returns:
    --------
    v_hat_2d : Reconstructed velocity (shape only, amplitude is wrong)
    """
    # Use percentile-based threshold to avoid extreme values
    H_threshold = np.percentile(np.abs(H_2d), 5)  # 5th percentile
    H_threshold = max(H_threshold, 1.0)  # At least 1.0
    
    # Clip H to avoid division by very small values
    H_clipped = np.clip(H_2d, H_threshold, None)
    
    v_hat_2d = -Theta_2d / H_clipped
    v_hat_2d = v_hat_2d - v_hat_2d.mean()
    
    # Clip extreme outliers
    v_clip = np.percentile(np.abs(v_hat_2d), 99.5)
    v_hat_2d = np.clip(v_hat_2d, -v_clip, v_clip)
    
    return v_hat_2d.astype(np.float32)


def quadratic_estimator_diagonal(Theta_total_2d, H_alpha_2d, tau_alpha_2d, box_size_2d):
    """
    Simplified diagonal quadratic estimator for comparison.
    
    This uses a diagonal approximation where L ≈ ℓ:
    v̂(L) ∝ Θ(L) × H*(L) × Γ(L) / [C_TT(L) × C_HH(L)]
    
    This is NOT the full paper estimator but provides a useful comparison.
    It essentially does inverse-variance weighted cross-correlation.
    """
    n = Theta_total_2d.shape[0]
    
    # Mean-subtract
    Theta_c = Theta_total_2d - Theta_total_2d.mean()
    H_c = H_alpha_2d - H_alpha_2d.mean()
    tau_c = tau_alpha_2d - tau_alpha_2d.mean()
    
    # FFT
    Theta_k = np.fft.fft2(Theta_c)
    H_k = np.fft.fft2(H_c)
    tau_k = np.fft.fft2(tau_c)
    
    # Power spectra at each mode
    eps = 1e-30
    C_TT = np.abs(Theta_k)**2
    C_HH = np.abs(H_k)**2
    C_tauH = np.real(np.conj(tau_k) * H_k)
    
    # Smooth power spectra (azimuthal average approximation)
    dk = 2.0 * np.pi / box_size_2d
    kx = np.fft.fftfreq(n, d=1.0/n) * dk
    ky = np.fft.fftfreq(n, d=1.0/n) * dk
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    ell = np.sqrt(KX**2 + KY**2)
    
    # Bin and smooth
    n_bins = 30
    ell_max = np.max(ell)
    ell_bins = np.linspace(0, ell_max, n_bins + 1)
    
    C_TT_smooth = np.zeros_like(ell)
    C_HH_smooth = np.zeros_like(ell)
    C_tauH_smooth = np.zeros_like(ell)
    
    for i in range(n_bins):
        mask = (ell >= ell_bins[i]) & (ell < ell_bins[i+1])
        if np.any(mask):
            C_TT_smooth[mask] = np.mean(C_TT[mask])
            C_HH_smooth[mask] = np.mean(C_HH[mask])
            C_tauH_smooth[mask] = np.mean(C_tauH[mask])
    
    C_TT_smooth = np.maximum(C_TT_smooth, eps)
    C_HH_smooth = np.maximum(C_HH_smooth, eps)
    
    # Diagonal estimator: v̂(L) ∝ Θ(L) × H*(L) × |Γ(L)| / [C_TT(L) × C_HH(L)]
    Gamma = np.abs(C_tauH_smooth)
    
    v_hat_k = Theta_k * np.conj(H_k) * Gamma / (C_TT_smooth * C_HH_smooth)
    
    # Normalization: N(L) = C_TT(L) × C_HH(L) / Γ(L)²
    N_vv = C_TT_smooth * C_HH_smooth / (Gamma**2 + eps)
    v_hat_k = v_hat_k * N_vv
    
    v_hat_2d = np.real(np.fft.ifft2(v_hat_k)).astype(np.float32)
    v_hat_2d = v_hat_2d - v_hat_2d.mean()
    
    N_vv_2d = np.real(np.fft.ifft2(N_vv)).astype(np.float32)
    
    return v_hat_2d, N_vv_2d


def compute_power_spectrum_1d(field_2d, box_size_2d, n_bins=30):
    """
    Compute 1D power spectrum P(ell) from a 2D field.
    
    Returns ell_centers and P(ell) as arrays.
    """
    n = field_2d.shape[0]
    field_k = np.fft.fft2(field_2d - field_2d.mean())
    
    dk = 2.0 * np.pi / box_size_2d
    kx = np.fft.fftfreq(n, d=1.0/n) * dk
    ky = np.fft.fftfreq(n, d=1.0/n) * dk
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    ell = np.sqrt(KX**2 + KY**2)
    
    P_k = np.abs(field_k)**2
    
    ell_max = np.max(ell)
    ell_bins = np.linspace(0, ell_max, n_bins + 1)
    ell_centers = 0.5 * (ell_bins[:-1] + ell_bins[1:])
    
    P_binned = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (ell >= ell_bins[i]) & (ell < ell_bins[i+1])
        if np.any(mask):
            P_binned[i] = np.mean(P_k[mask])
    
    return ell_centers, P_binned


def compute_cross_spectrum_1d(field1_2d, field2_2d, box_size_2d, n_bins=30):
    """Compute 1D cross-power spectrum between two 2D fields."""
    n = field1_2d.shape[0]
    field1_k = np.fft.fft2(field1_2d - field1_2d.mean())
    field2_k = np.fft.fft2(field2_2d - field2_2d.mean())
    
    dk = 2.0 * np.pi / box_size_2d
    kx = np.fft.fftfreq(n, d=1.0/n) * dk
    ky = np.fft.fftfreq(n, d=1.0/n) * dk
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    ell = np.sqrt(KX**2 + KY**2)
    
    cross = np.real(np.conj(field1_k) * field2_k)
    
    ell_max = np.max(ell)
    ell_bins = np.linspace(0, ell_max, n_bins + 1)
    ell_centers = 0.5 * (ell_bins[:-1] + ell_bins[1:])
    
    C_binned = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (ell >= ell_bins[i]) & (ell < ell_bins[i+1])
        if np.any(mask):
            C_binned[i] = np.mean(cross[mask])
    
    return ell_centers, C_binned


def make_power_spectrum_interpolator(ell_centers, P_ell):
    """Create an interpolator function for P(ell)."""
    # Extend to ell=0 and high ell
    ell_ext = np.concatenate([[0], ell_centers, [2*ell_centers[-1]]])
    P_ext = np.concatenate([[P_ell[0]], P_ell, [P_ell[-1]]])
    
    return interp1d(ell_ext, P_ext, kind='linear', 
                    bounds_error=False, fill_value=(P_ell[0], P_ell[-1]))


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


def continuity_reconstruct_v_from_delta_3d(delta_3d, z, n=None):
    """
    Continuity-based velocity reconstruction from 3D density field.
    
    v_i(k) = i × a × H × f × k_i / k² × δ(k)
    
    This is the LINEAR estimator from the continuity equation.
    
    Parameters:
    -----------
    delta_3d : 3D array
        Density contrast field δ = ρ/ρ̄ - 1
    z : float
        Redshift
    
    Returns:
    --------
    vx_rec, vy_rec, vz_rec : 3D arrays
        Reconstructed velocity components in km/s
    """
    if n is None:
        n = delta_3d.shape[0]
    box_size = BOX_MPC_OVER_H / LITTLEH  # Mpc
    
    delta_c = delta_3d - delta_3d.mean()
    delta_k = fft.rfftn(delta_c, workers=-1).astype(np.complex64)
    
    kx, ky, kz = kspace_rfft_3d(n, box_size)
    
    k2 = kx[:, None, None]**2 + ky[None, :, None]**2 + kz[None, None, :]**2
    k2[0, 0, 0] = 1.0  # Avoid division by zero
    
    # Cosmological factors
    a = 1.0 / (1.0 + z)
    H0 = 100.0 * LITTLEH  # km/s/Mpc
    Omega_m = 0.27
    Omega_L = 0.73
    Ha = H0 * np.sqrt(Omega_m / a**3 + Omega_L)
    f = growth_rate(z)
    
    factor = 1j * Ha * a * f
    
    vx_k = factor * kx[:, None, None] / k2 * delta_k
    vy_k = factor * ky[None, :, None] / k2 * delta_k
    vz_k = factor * kz[None, None, :] / k2 * delta_k
    
    vx_rec = safe_real(fft.irfftn(vx_k, s=(n, n, n), workers=-1))
    vy_rec = safe_real(fft.irfftn(vy_k, s=(n, n, n), workers=-1))
    vz_rec = safe_real(fft.irfftn(vz_k, s=(n, n, n), workers=-1))
    
    return vx_rec, vy_rec, vz_rec


def project_and_tau_weight(v_3d, tau_3d):
    """
    Project 3D velocity to 2D with τ-weighting.
    
    v_eff(n̂) = [∫ τ(χ,n̂) v(χ,n̂) dχ] / [∫ τ(χ,n̂) dχ]
    
    This is the TRUE target field that both linear and quadratic 
    estimators should reconstruct.
    
    Parameters:
    -----------
    v_3d : 3D array
        Velocity field (e.g., vz in km/s)
    tau_3d : 3D array
        Optical depth field τ ∝ (1-xHI)(1+δ)
    
    Returns:
    --------
    v_eff_2d : 2D array
        τ-weighted velocity projected along LOS
    """
    tau_v_2d = np.sum(tau_3d * v_3d, axis=2).astype(np.float32)
    tau_2d = np.sum(tau_3d, axis=2).astype(np.float32)
    
    v_eff_2d = tau_v_2d / (tau_2d + 1e-10)
    
    return v_eff_2d


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_map_robust(ax, data, title, cmap='RdBu_r', percentile_clip=1.0, 
                    use_symlog=False, linthresh=1.0):
    """
    Plot a 2D map with robust visualization.
    
    Parameters:
    -----------
    ax : matplotlib axis
    data : 2D array
    title : str
    cmap : str
    percentile_clip : float
        Clip at this percentile (e.g., 1.0 means 1st and 99th percentiles)
    use_symlog : bool
        If True, use symmetric log normalization
    linthresh : float
        Linear threshold for symlog normalization
    """
    if use_symlog:
        vmax = np.percentile(np.abs(data), 100 - percentile_clip)
        norm = SymLogNorm(linthresh=linthresh, vmin=-vmax, vmax=vmax)
        im = ax.imshow(data.T, origin='lower', cmap=cmap, norm=norm)
    else:
        vmin = np.percentile(data, percentile_clip)
        vmax = np.percentile(data, 100 - percentile_clip)
        im = ax.imshow(data.T, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
    
    ax.set_title(title, fontsize=11)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)
    
    return im


def compute_correlation_2d(field1, field2, boundary=10):
    """Compute Pearson correlation between two 2D fields, excluding boundaries."""
    if boundary > 0:
        f1 = field1[boundary:-boundary, boundary:-boundary].flatten()
        f2 = field2[boundary:-boundary, boundary:-boundary].flatten()
    else:
        f1 = field1.flatten()
        f2 = field2.flatten()
    
    mask = np.isfinite(f1) & np.isfinite(f2)
    if not np.any(mask):
        return np.nan
    
    return np.corrcoef(f1[mask], f2[mask])[0, 1]


def compute_scale_dependent_correlation(field1_2d, field2_2d, box_size_2d, n_bins=15):
    """
    Compute scale-dependent correlation r(ell) between two 2D fields.
    
    r(ell) = P_{12}(ell) / sqrt(P_{11}(ell) × P_{22}(ell))
    """
    n = field1_2d.shape[0]
    
    f1_k = np.fft.fft2(field1_2d - field1_2d.mean())
    f2_k = np.fft.fft2(field2_2d - field2_2d.mean())
    
    dk = 2.0 * np.pi / box_size_2d
    kx = np.fft.fftfreq(n, d=1.0/n) * dk
    ky = np.fft.fftfreq(n, d=1.0/n) * dk
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    ell = np.sqrt(KX**2 + KY**2)
    
    P11 = np.abs(f1_k)**2
    P22 = np.abs(f2_k)**2
    P12 = np.real(np.conj(f1_k) * f2_k)
    
    ell_max = np.max(ell)
    ell_bins = np.linspace(0, ell_max, n_bins + 1)
    ell_centers = 0.5 * (ell_bins[:-1] + ell_bins[1:])
    
    r_ell = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (ell >= ell_bins[i]) & (ell < ell_bins[i+1])
        if np.any(mask):
            P11_bin = np.mean(P11[mask])
            P22_bin = np.mean(P22[mask])
            P12_bin = np.mean(P12[mask])
            
            denom = np.sqrt(P11_bin * P22_bin)
            if denom > 0:
                r_ell[i] = P12_bin / denom
    
    return ell_centers, r_ell


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
# MAIN PAPER-CONSISTENT ANALYSIS
# ============================================================================

def run_paper_consistent_analysis(redshifts, output_dir=OUTPUT_DIR):
    """
    Run the full paper-consistent analysis comparing:
    (A) Linear continuity-based velocity reconstruction
    (B) Quadratic estimator (Eq. 18-20)
    
    Both methods are compared on the SAME target field:
    v_eff^α(n̂) = [∫_α dχ τ(χ,n̂) v_r(χ,n̂)] / [∫_α dχ τ(χ,n̂)]
    """
    print("="*80)
    print("PAPER-CONSISTENT QUADRATIC ESTIMATOR ANALYSIS")
    print("Hotinli & Johnson: Reconstructing large scales at cosmic dawn")
    print("="*80)
    
    # Box size for 2D maps (after cropping)
    n_crop = 540  # 570 - 30
    box_size_2d = BOX_MPC_OVER_H / LITTLEH * (n_crop / N_BOX)  # Mpc
    
    # =========================================================================
    # STEP 1: Build total kSZ from ALL redshifts
    # =========================================================================
    Theta_total_2d, tau_total_2d, data_per_z = build_ksz_total(redshifts)
    
    print(f"\nIncluded redshifts: {redshifts}")
    print(f"Box size (2D): {box_size_2d:.1f} Mpc")
    
    # =========================================================================
    # STEP 2: Compute power spectra from the data
    # =========================================================================
    print("\nComputing power spectra...")
    
    # Total kSZ power spectrum (C_ΘΘ^obs)
    ell_TT, C_TT = compute_power_spectrum_1d(Theta_total_2d, box_size_2d)
    C_TT_func = make_power_spectrum_interpolator(ell_TT, C_TT)
    
    results = []
    
    # =========================================================================
    # STEP 3: For each redshift bin α, compare methods
    # =========================================================================
    for z in redshifts:
        print(f"\n{'='*60}")
        print(f"Analyzing redshift bin α: z = {z:.3f}")
        print(f"{'='*60}")
        
        data = data_per_z[z]
        H_alpha_2d = data['Tb_2d']  # 21cm brightness (H_α)
        tau_alpha_2d = data['tau_2d']  # τ for this bin (for truth only!)
        v_true_2d = data['v_true_2d']  # True τ-weighted velocity
        delta_3d = data['delta_3d']
        tau_3d = data['tau_3d']
        vz_3d = data['vz_3d']
        mean_xHI = data['mean_xHI']
        n = data['n']
        
        print(f"  Mean x_HI = {mean_xHI:.3f}")
        print(f"  True v_eff: mean = {v_true_2d.mean():.2f}, std = {v_true_2d.std():.2f} km/s")
        
        # Compute per-bin power spectra
        ell_HH, C_HH = compute_power_spectrum_1d(H_alpha_2d, box_size_2d)
        C_HH_func = make_power_spectrum_interpolator(ell_HH, C_HH)
        
        ell_tauH, C_tauH = compute_cross_spectrum_1d(tau_alpha_2d, H_alpha_2d, box_size_2d)
        C_tauH_func = make_power_spectrum_interpolator(ell_tauH, C_tauH)
        
        # ---------------------------------------------------------------------
        # METHOD A: Linear continuity reconstruction
        # ---------------------------------------------------------------------
        print("  Running linear continuity reconstruction...")
        vx_lin, vy_lin, vz_lin = continuity_reconstruct_v_from_delta_3d(delta_3d, z, n)
        
        # Project with τ-weighting (same as truth)
        v_lin_2d = project_and_tau_weight(vz_lin, tau_3d)
        
        r_linear = compute_correlation_2d(v_true_2d, v_lin_2d)
        amp_linear = v_lin_2d.std() / v_true_2d.std()
        print(f"    Linear: r = {r_linear:.4f}, amplitude ratio = {amp_linear:.2f}")
        
        # ---------------------------------------------------------------------
        # METHOD B: Quadratic estimator (Eq. 18) - full convolution
        # ---------------------------------------------------------------------
        print("  Running quadratic estimator (Eq. 18)...")
        v_quad_2d, N_vv_2d = quadratic_estimator_flatsky(
            Theta_total_2d, H_alpha_2d, C_tauH_func,
            C_TT_func, C_HH_func, box_size_2d
        )
        
        r_quad = compute_correlation_2d(v_true_2d, v_quad_2d)
        amp_quad = v_quad_2d.std() / v_true_2d.std()
        print(f"    Quadratic (conv): r = {r_quad:.4f}, amplitude ratio = {amp_quad:.2f}")
        
        # ---------------------------------------------------------------------
        # METHOD C: Proxy estimator (-Θ/H)
        # ---------------------------------------------------------------------
        print("  Running proxy estimator (-Θ/H)...")
        v_proxy_2d = proxy_estimator(Theta_total_2d, H_alpha_2d)
        
        r_proxy = compute_correlation_2d(v_true_2d, v_proxy_2d)
        amp_proxy = v_proxy_2d.std() / v_true_2d.std()
        print(f"    Proxy (-Θ/H): r = {r_proxy:.4f}, amplitude ratio = {amp_proxy:.2f}")
        
        # ---------------------------------------------------------------------
        # Scale-dependent correlation
        # ---------------------------------------------------------------------
        ell_r, r_ell_lin = compute_scale_dependent_correlation(v_true_2d, v_lin_2d, box_size_2d)
        _, r_ell_quad = compute_scale_dependent_correlation(v_true_2d, v_quad_2d, box_size_2d)
        _, r_ell_proxy = compute_scale_dependent_correlation(v_true_2d, v_proxy_2d, box_size_2d)
        
        # ---------------------------------------------------------------------
        # Create diagnostic plots
        # ---------------------------------------------------------------------
        print("  Creating diagnostic plots...")
        
        fig, axes = plt.subplots(2, 4, figsize=(18, 10))
        
        # Top row: Maps
        plot_map_robust(axes[0, 0], v_true_2d, f'True $v_{{eff}}$ (z={z:.2f})', percentile_clip=1)
        plot_map_robust(axes[0, 1], v_lin_2d, f'Linear (r={r_linear:.3f})', percentile_clip=1)
        plot_map_robust(axes[0, 2], v_quad_2d, f'Quad conv (r={r_quad:.3f})', percentile_clip=1)
        plot_map_robust(axes[0, 3], v_proxy_2d, f'Proxy (r={r_proxy:.3f})', percentile_clip=1)
        
        # Bottom row: Scatter plots and diagnostics
        boundary = 10
        v_true_crop = v_true_2d[boundary:-boundary, boundary:-boundary].flatten()
        v_lin_crop = v_lin_2d[boundary:-boundary, boundary:-boundary].flatten()
        v_quad_crop = v_quad_2d[boundary:-boundary, boundary:-boundary].flatten()
        
        np.random.seed(42)
        idx = np.random.choice(len(v_true_crop), size=min(5000, len(v_true_crop)), replace=False)
        
        # Linear scatter
        axes[1, 0].scatter(v_lin_crop[idx], v_true_crop[idx], alpha=0.3, s=1, c='blue')
        vmin, vmax = np.percentile(v_true_crop, [1, 99])
        axes[1, 0].plot([vmin, vmax], [vmin, vmax], 'r--', lw=2)
        axes[1, 0].set_xlabel('Linear reconstruction')
        axes[1, 0].set_ylabel('True $v_{eff}$')
        axes[1, 0].set_title(f'Linear: r = {r_linear:.4f}')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Quadratic scatter
        axes[1, 1].scatter(v_quad_crop[idx], v_true_crop[idx], alpha=0.3, s=1, c='orange')
        axes[1, 1].plot([vmin, vmax], [vmin, vmax], 'r--', lw=2)
        axes[1, 1].set_xlabel('Quadratic reconstruction')
        axes[1, 1].set_ylabel('True $v_{eff}$')
        axes[1, 1].set_title(f'Quadratic: r = {r_quad:.4f}')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Scale-dependent correlation
        axes[1, 2].plot(ell_r, r_ell_lin, 'b-o', label='Linear', markersize=4)
        axes[1, 2].plot(ell_r, r_ell_quad, 'orange', marker='s', label='Quad conv', markersize=4)
        axes[1, 2].plot(ell_r, r_ell_proxy, 'g-^', label='Proxy', markersize=4)
        axes[1, 2].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        axes[1, 2].set_xlabel(r'$\ell$ (multipole)')
        axes[1, 2].set_ylabel(r'$r(\ell)$')
        axes[1, 2].set_title('Scale-dependent correlation')
        axes[1, 2].legend(fontsize=8)
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].set_ylim(-0.5, 1.1)
        
        # Power spectra
        ell_v, P_v_true = compute_power_spectrum_1d(v_true_2d, box_size_2d)
        _, P_v_lin = compute_power_spectrum_1d(v_lin_2d, box_size_2d)
        _, P_v_quad = compute_power_spectrum_1d(v_quad_2d, box_size_2d)
        
        axes[1, 3].loglog(ell_v, P_v_true, 'k-', label='True', linewidth=2)
        axes[1, 3].loglog(ell_v, P_v_lin, 'b--', label='Linear', linewidth=2)
        axes[1, 3].loglog(ell_v, P_v_quad, 'orange', linestyle='--', label='Quadratic', linewidth=2)
        axes[1, 3].set_xlabel(r'$\ell$')
        axes[1, 3].set_ylabel(r'$C_\ell^{vv}$')
        axes[1, 3].set_title('Power spectra')
        axes[1, 3].legend()
        axes[1, 3].grid(True, alpha=0.3)
        
        plt.suptitle(f'Velocity Reconstruction: z = {z:.2f}, $x_{{HI}}$ = {mean_xHI:.3f}\n'
                    f'Using TOTAL kSZ (integrated over {len(redshifts)} redshifts)', fontsize=14)
        plt.tight_layout()
        
        outfile = os.path.join(output_dir, f'paper_comparison_z{z:.2f}.png')
        plt.savefig(outfile, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Saved: {outfile}")
        
        results.append({
            'z': z,
            'mean_xHI': mean_xHI,
            'r_linear': r_linear,
            'r_quad': r_quad,
            'r_proxy': r_proxy,
            'amp_linear': amp_linear,
            'amp_quad': amp_quad,
            'amp_proxy': amp_proxy,
            'v_true_std': v_true_2d.std(),
        })
        
        gc.collect()
    
    # =========================================================================
    # STEP 4: Summary plot
    # =========================================================================
    print("\nCreating summary plot...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    zs = [r['z'] for r in results]
    xHIs = [r['mean_xHI'] for r in results]
    
    # Correlation
    ax = axes[0]
    ax.plot(xHIs, [r['r_linear'] for r in results], 'b-o', 
            label='Linear (continuity)', markersize=10, linewidth=2)
    ax.plot(xHIs, [r['r_quad'] for r in results], 'orange', marker='s',
            label='Quad (conv)', markersize=10, linewidth=2)
    ax.plot(xHIs, [r['r_proxy'] for r in results], 'g-^',
            label='Proxy (-Θ/H)', markersize=10, linewidth=2)
    ax.set_xlabel(r'Mean $x_{HI}$', fontsize=14)
    ax.set_ylabel('Correlation with true $v_{eff}$', fontsize=14)
    ax.set_title('Pixel Correlation', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.2, 1.1)
    
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(xHIs[::2] if len(xHIs) > 4 else xHIs)
    ax2.set_xticklabels([f'{z:.1f}' for z in zs[::2]] if len(zs) > 4 else [f'{z:.1f}' for z in zs])
    ax2.set_xlabel('Redshift', fontsize=12)
    
    # Amplitude
    ax = axes[1]
    ax.plot(xHIs, [r['amp_linear'] for r in results], 'b-o', 
            label='Linear', markersize=10, linewidth=2)
    ax.plot(xHIs, [r['amp_quad'] for r in results], 'orange', marker='s',
            label='Quad (conv)', markersize=10, linewidth=2)
    ax.plot(xHIs, [r['amp_proxy'] for r in results], 'g-^',
            label='Proxy', markersize=10, linewidth=2)
    ax.axhline(y=1.0, color='k', linestyle='--', linewidth=1, label='Perfect')
    ax.set_xlabel(r'Mean $x_{HI}$', fontsize=14)
    ax.set_ylabel('Amplitude ratio (rec/true)', fontsize=14)
    ax.set_title('Amplitude Recovery', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.suptitle(f'Linear vs Quadratic Estimator Comparison\n'
                f'Using TOTAL kSZ integrated over {len(redshifts)} redshifts', fontsize=14)
    plt.tight_layout()
    
    outfile = os.path.join(output_dir, 'paper_comparison_summary.png')
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {outfile}")
    
    # =========================================================================
    # STEP 5: Print summary table
    # =========================================================================
    print("\n" + "="*110)
    print("SUMMARY: Linear vs Quadratic Estimator (using TOTAL integrated kSZ)")
    print("="*110)
    print(f"{'z':>8} {'x_HI':>8} {'r_linear':>10} {'r_quad':>10} {'r_proxy':>10} {'amp_lin':>10} {'amp_quad':>10} {'amp_proxy':>10}")
    print("-"*90)
    for r in results:
        print(f"{r['z']:8.3f} {r['mean_xHI']:8.3f} {r['r_linear']:10.4f} {r['r_quad']:10.4f} "
              f"{r['r_proxy']:10.4f} {r['amp_linear']:10.2f} {r['amp_quad']:10.2f} {r['amp_proxy']:10.2f}")
    print("="*110)
    
    return results


if __name__ == "__main__":
    # Redshifts spanning EoR
    redshifts = [6.231, 6.617, 7.059, 7.570, 8.064, 8.636, 9.308]
    
    # Run the paper-consistent analysis
    results = run_paper_consistent_analysis(redshifts)
