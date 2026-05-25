#!/usr/bin/env python3
"""
Comprehensive kSZ Reconstruction Analysis Pipeline
Processes multiple redshifts and generates all analysis plots
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import fft
from scipy.ndimage import gaussian_filter
from scipy.integrate import quad
from scipy.stats import pearsonr
from powerbox import get_power
import gc
from matplotlib.lines import Line2D

# ksz_core is the single source of truth for cosmology (Limber unit convention).
# Paper-code wrappers below preserve the old signatures but delegate to ksz_core.
from ksz_core.cosmology import (
    Constants as _KSZ_Constants,
    chi_mpch as _ksz_chi_mpch,
    k_to_ell as _ksz_k_to_ell,
)

_PAPER_COSMO = _KSZ_Constants.paper_fiducial()  # H0=70, Om0=0.27, h=0.7
 
LITTLEH = 0.7
BOX_MPC_OVER_H = 500.0
K_MIN = 0.012  # Minimum k for Fourier analysis [h/Mpc]
K_MAX_PLOT = 1.0  # Maximum k for plotting [h/Mpc]
FOURIER_BINS = 50  # Number of bins for Fourier correlation
SMOOTH_SIGMA = 5.0  # Smoothing scale in Mpc/h
PHYSICAL_NORM = True # Set to True for physical units (µK), False for arbitrary units
CENTRAL_CROP = slice(30, 570)  # Use central 500x500 region to avoid boundary effects
LOS_CROP = slice(30, 570)  # Exclude boundary slices along line of sight

# ============================================================================
# 21cm BRIGHTNESS TEMPERATURE NORMALIZATION STUDY
# ============================================================================
# Three normalization methods for the linear continuity reconstruction:
#
# (A) RAW δTb [mK]:
#     The raw 21cm brightness temperature field.
#     Units: mK (or arbitrary simulation units)
#     This is the baseline - what we currently use.
#
# (B) MEAN-NORMALIZED (dimensionless):
#     δTb_dim = (δTb - ⟨δTb⟩) / ⟨δTb⟩
#     This is dimensionally clean and removes the mean.
#     CAVEAT: ⟨δTb⟩ is NOT directly observable - it requires knowing the
#     global mean brightness temperature, which depends on cosmology and
#     reionization history. This is a SIMULATION-ONLY test.
#
# (C) FIXED-SCALE NORMALIZED (observable-motivated):
#     δTb_mK = δTb / T_ref
#     where T_ref is a fixed reference temperature (default: 10 mK).
#     This is dimensionless and uses only observable quantities.
#     CAVEAT: The amplitude is arbitrary (depends on T_ref choice).
#     The CORRELATION COEFFICIENT is the robust metric here.
#
# KEY INSIGHT: The linear continuity equation v(k) = -i aHf k/k² × δ(k)
# gives v ∝ input_field. Different normalizations change the AMPLITUDE
# of the reconstructed velocity but NOT its MORPHOLOGY (correlation).
# Therefore, Pearson r is the primary robust metric for comparison.
# ============================================================================

T_REF_MK = 10.0  # Reference temperature for fixed-scale normalization [mK]

# Get the absolute path to the project root directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # Go up one level from functions/
DATA_DIR = os.path.join(PROJECT_ROOT, "data_raghu/")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "plots_paper_new")

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def safe_real(arr):
    """Safely extract real part and handle precision."""
    return np.real(arr).astype(np.float32, copy=False)

def kspace_rfft(n, rc, dtype=np.float32):
    """Generate k-space coordinates for rfft (with zero-protection)."""
    kx = 2.0 * np.pi * np.fft.fftfreq(n, d=rc).astype(dtype)
    ky = 2.0 * np.pi * np.fft.fftfreq(n, d=rc).astype(dtype)
    kz = 2.0 * np.pi * np.fft.rfftfreq(n, d=rc).astype(dtype)
    # Protect against divide by zero in k-space
    tiny = np.finfo(dtype).tiny
    if kz.size: kz[0] = max(kz[0], tiny)
    if kx.size: kx[0] = max(kx[0], tiny)
    if ky.size: ky[0] = max(ky[0], tiny)
    return kx, ky, kz

def pearson_r(field1, field2):
    """Compute Pearson correlation coefficient between two fields."""
    f1_flat = field1.ravel()
    f2_flat = field2.ravel()
    mask = np.isfinite(f1_flat) & np.isfinite(f2_flat)
    if not np.any(mask):
        return np.nan
    return np.corrcoef(f1_flat[mask], f2_flat[mask])[0, 1]

def k_to_ell(k, z, littleh=0.7):
    """Convert k [h/Mpc] to multipole ℓ via the Limber identity ℓ = k · χ.

    Delegates to ksz_core.cosmology.k_to_ell, which enforces consistent units
    (k in h/Mpc, χ in Mpc/h). The old local implementation here used χ in Mpc
    and was missing a factor of h — see outputs/2026-05-22 daily log.
    """
    if littleh == 0.7:
        c = _PAPER_COSMO
    else:
        c = _KSZ_Constants(H0=100.0 * littleh, Om0=0.27, Ob0=0.044, h=littleh)
    return _ksz_k_to_ell(k, z, c)


# ============================================================================
# OPTICAL DEPTH FUNCTIONS
# ============================================================================


def compute_dtau_dz(z, n_e_factor=1.0, H0=70.0, Omega_b=0.044, Omega_m=0.27, Y_He=0.24):
    """
    Compute the differential Thomson optical depth dτ/dz at redshift z.
    
    The optical depth is defined as:
        τ = ∫ n_e * σ_T * dl
    
    where dl is proper distance. Using dl/dz = c / H(z) / (1+z):
        dτ/dz = c * σ_T * n_e / H(z) / (1+z)
    
    Parameters:
    -----------
    z : float
        Redshift
    n_e_factor : float
        Factor to multiply mean electron density (e.g., ionization fraction)
    H0 : float
        Hubble constant in km/s/Mpc
    Omega_b : float
        Baryon density parameter
    Omega_m : float
        Matter density parameter
    Y_He : float
        Helium mass fraction
    
    Returns:
    --------
    dtau_dz : float
        Differential optical depth per unit redshift (dimensionless)
    """
    # Physical constants (CGS for clarity)
    c_cgs = 2.99792458e10           # Speed of light [cm/s]
    sigma_T = 6.6524587158e-25      # Thomson cross-section [cm^2]
    m_p = 1.6726219e-24             # Proton mass [g]
    Mpc_to_cm = 3.085677581e24      # Mpc to cm
    
    Omega_L = 1.0 - Omega_m
    
    # Critical density today [g/cm^3]
    # ρ_crit,0 = 3H0²/(8πG)
    H0_cgs = H0 * 1e5 / Mpc_to_cm   # H0 in s^-1
    G_cgs = 6.67430e-8              # cm^3 g^-1 s^-2
    rho_crit_0 = 3 * H0_cgs**2 / (8 * np.pi * G_cgs)  # g/cm^3
    
    # Mean baryon density today [g/cm^3]
    rho_b_0 = Omega_b * rho_crit_0
    
    # Mean electron density today (assuming fully ionized H + He)
    # n_e = n_H + 2*n_He = ρ_b/m_p * (X_H + Y_He/2) = ρ_b/m_p * (1 - Y_He/2)
    n_e_0 = rho_b_0 / m_p * (1.0 - Y_He / 2.0)  # cm^-3
    
    # Mean electron density at redshift z (scales as (1+z)^3)
    n_e_z = n_e_0 * (1.0 + z)**3 * n_e_factor  # cm^-3
    
    # Hubble parameter at redshift z [s^-1]
    E_z = np.sqrt(Omega_m * (1 + z)**3 + Omega_L)
    H_z_cgs = H0_cgs * E_z  # s^-1
    
    # dτ/dz = c * σ_T * n_e / H(z) / (1+z)
    # Units: [cm/s] * [cm^2] * [cm^-3] / [s^-1] = dimensionless ✓
    dtau_dz = c_cgs * sigma_T * n_e_z / H_z_cgs / (1.0 + z)
    
    return dtau_dz


def compute_tau_0_to_z(z, H0=70.0, Omega_b=0.044, Omega_m=0.27, Y_He=0.24):
    """
    Compute the Thomson optical depth from z=0 to redshift z,
    assuming the universe is fully ionized.
    
    Parameters:
    -----------
    z : float
        Upper redshift limit
    
    Returns:
    --------
    tau : float
        Thomson optical depth from 0 to z
    """
    result, _ = quad(lambda zp: compute_dtau_dz(zp, n_e_factor=1.0, H0=H0, 
                                                  Omega_b=Omega_b, Omega_m=Omega_m, 
                                                  Y_He=Y_He), 0, z)
    return result


# Mean Thomson optical depth from z=0 to z=6 (assuming fully ionized)
# This is τ̄₀₆ from the paper, approximately 0.0517
TAU_0_6 = 0.0517


def compute_tau_6_to_z(z, H0=70.0, Omega_b=0.044, Omega_m=0.27, Y_He=0.24):
    """
    Compute the Thomson optical depth from z=6 to redshift z,
    assuming the universe is fully ionized.
    
    This is τ₆z from the paper.
    
    Parameters:
    -----------
    z : float
        Upper redshift limit (must be >= 6)
    
    Returns:
    --------
    tau : float
        Thomson optical depth from 6 to z
    """
    if z <= 6:
        return 0.0
    result, _ = quad(lambda zp: compute_dtau_dz(zp, n_e_factor=1.0, H0=H0,
                                                  Omega_b=Omega_b, Omega_m=Omega_m,
                                                  Y_He=Y_He), 6, z)
    return result

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def read_den(filename, nx, ny, nz, endian="<"):
    """Read density field from binary file (IDL-compatible)."""
    dt_f4 = np.dtype(endian + "f4")
    with open(filename, 'rb') as f:
        f.seek(12)  # skip 3*float32 header like IDL read_den
        data = np.fromfile(f, dtype=dt_f4, count=nx*ny*nz)
    return data.reshape((nx, ny, nz), order='F')

def read_xhi(filename, nx, ny, nz):
    """Read neutral fraction field from binary file."""
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

    arrv3 *= np.float32(vel_unit * 8.0)  # cm/s like IDL

    eps = np.float32(1e-12)
    den32 = den.astype(np.float32, copy=False)
    den_safe = np.where(den32 > eps, den32, eps)

    with np.errstate(divide="ignore", invalid="ignore"):
        vx = (arrv3[0] / den_safe).astype(np.float32, copy=False)
        vy = (arrv3[1] / den_safe).astype(np.float32, copy=False)
        vz = (arrv3[2] / den_safe).astype(np.float32, copy=False)

    return vx, vy, vz

def reconstruct_velocities(z, n=600, data_dir=DATA_DIR, include_velocity_term=True,
                           T_ref_mK=T_REF_MK):
    """Load data and reconstruct velocities for a single redshift.
    
    This function computes velocity reconstructions using the linear continuity
    equation with MULTIPLE normalizations of the 21cm brightness temperature:
    
    (A) Raw δTb: The raw brightness temperature field [mK units]
    (B) Mean-normalized: (δTb - ⟨δTb⟩) / ⟨δTb⟩ [dimensionless, simulation-only]
    (C) Fixed-scale: δTb / T_ref [dimensionless, observable-motivated]
    
    Parameters:
    -----------
    z : float
        Redshift
    n : int
        Grid size (default 600)
    data_dir : str
        Data directory path
    include_velocity_term : bool
        If True, include H/(dv/dr + H) velocity gradient term in tracer field
    T_ref_mK : float
        Reference temperature for fixed-scale normalization [mK]
    
    Returns:
    --------
    Tuple of arrays:
        den, xhi, vx, vy, vz : Original simulation fields
        vx_rec, vy_rec, vz_rec : Reconstruction from δ (density)
        vx_recx, vy_recx, vz_recx : Reconstruction from raw δTb
        vx_recx_norm, vy_recx_norm, vz_recx_norm : Reconstruction from δTb/<xHI>
        vz_rec_mean_norm : Reconstruction from mean-normalized δTb (LOS only)
        vz_rec_fixed_mK : Reconstruction from fixed-scale δTb (LOS only)
    """
    zstr = f"{z:.3f}"
    filenameDen = f"{data_dir}{zstr}n_all.dat"
    filenameVel = f"{data_dir}{zstr}v_all.dat"
    filenameXhi = f"{data_dir}{zstr}zeta0.389fesc0.389_Mmin0.120E+10_MminX0.120E+10_fx0.100E+03_sed3_al1.200xhi.bin"
    
    den = read_den(filenameDen, n, n, n).astype(np.float32, copy=False)
    xhi = read_xhi(filenameXhi, n, n, n).astype(np.float32, copy=False)
    vx, vy, vz = read_vel(z, den, filenameVel, n_cell=n)
    
    # Convert to km/s
    vx *= np.float32(1.0 / 1e5)
    vy *= np.float32(1.0 / 1e5)
    vz *= np.float32(1.0 / 1e5)
    
    # Reconstruct velocities
    mean_den = den.mean(dtype=np.float64).astype(np.float32)
    delta = (den / mean_den).astype(np.float32, copy=False)
    
    # Compute velocity gradient term if requested
    # δTb ∝ xHI × (1 + δ) × H / (dv_r/dr + H)
    # where H is the Hubble parameter and dv_r/dr is the LOS velocity gradient
    if include_velocity_term:
        box_mpc_over_h = 500.0  # Box size in Mpc/h
        dz_cell = box_mpc_over_h / n / LITTLEH  # Cell size in Mpc (physical units)
        # vz is in km/s (already converted above)
        dvdz = np.gradient(vz, dz_cell, axis=2).astype(np.float32)  # km/s/Mpc
        
        # Hubble parameter at z: H(z) = H0 * E(z)
        H0 = 100.0 * LITTLEH  # km/s/Mpc
        omega_m0 = 0.27
        omega_l0 = 0.73
        Hz = H0 * np.sqrt(omega_m0 * (1 + z)**3 + omega_l0)  # km/s/Mpc
        
        # Velocity factor: H / (dv_r/dr + H)
        velocity_factor = Hz / (dvdz + Hz)
        tracer_field_raw = delta * xhi * velocity_factor.astype(np.float32)
    else:
        tracer_field_raw = delta * xhi
    
    # =========================================================================
    # NORMALIZATION METHOD A: Raw δTb (current baseline)
    # =========================================================================
    # Mean-subtract to remove DC offset
    tracer_field = tracer_field_raw - tracer_field_raw.mean()
    
    # =========================================================================
    # NORMALIZATION METHOD B: Mean-normalized (dimensionless)
    # δTb_dim = (δTb - ⟨δTb⟩) / ⟨δTb⟩
    # 
    # This is theoretically clean (dimensionless) but NOT observable because
    # ⟨δTb⟩ depends on the global mean brightness temperature which requires
    # knowledge of cosmology and reionization history.
    # =========================================================================
    mean_Tb = tracer_field_raw.mean()
    if np.abs(mean_Tb) > 1e-10:
        tracer_field_mean_norm = (tracer_field_raw - mean_Tb) / mean_Tb
    else:
        # Fallback if mean is near zero (shouldn't happen for 21cm)
        tracer_field_mean_norm = tracer_field_raw - mean_Tb
    
    # =========================================================================
    # NORMALIZATION METHOD C: Fixed-scale normalized (observable-motivated)
    # δTb_mK = δTb / T_ref
    #
    # This is dimensionless and uses only observable quantities (the field itself).
    # The amplitude is arbitrary (depends on T_ref choice), but the MORPHOLOGY
    # (and hence correlation coefficient) is preserved.
    # =========================================================================
    tracer_field_fixed_mK = tracer_field_raw / T_ref_mK
    tracer_field_fixed_mK = tracer_field_fixed_mK - tracer_field_fixed_mK.mean()
    
    # Also keep the old normalization by mean xHI for backward compatibility
    mean_xHI = xhi.mean()
    if include_velocity_term:
        tracer_field_norm = delta * xhi * velocity_factor.astype(np.float32) / mean_xHI
    else:
        tracer_field_norm = delta * xhi / mean_xHI
    tracer_field_norm = tracer_field_norm - tracer_field_norm.mean()
    
    # =========================================================================
    # FFT all tracer fields
    # =========================================================================
    dltk = fft.rfftn(delta - 1.0, workers=-1).astype(np.complex64, copy=False)
    dltXhk = fft.rfftn(tracer_field, workers=-1).astype(np.complex64, copy=False)
    dltXhk_norm = fft.rfftn(tracer_field_norm, workers=-1).astype(np.complex64, copy=False)
    dltXhk_mean_norm = fft.rfftn(tracer_field_mean_norm, workers=-1).astype(np.complex64, copy=False)
    dltXhk_fixed_mK = fft.rfftn(tracer_field_fixed_mK, workers=-1).astype(np.complex64, copy=False)
    
    rc = 500.0 / float(n) / LITTLEH
    kx, ky, kz = kspace_rfft(n, rc, dtype=np.float32)
    
    a = 1.0 / (1.0 + z)
    H0 = 100.0 * LITTLEH
    omega_l0 = 0.73
    omega_m0 = 1.0 - omega_l0
    Ha = np.float32(H0 * np.sqrt(omega_m0 / a**3 + omega_l0))
    Omega_m_a = (omega_m0 / a**3) / (omega_m0 / a**3 + omega_l0)
    f_omega = np.float32(Omega_m_a**0.55)
    factor = np.complex64(Ha * a * f_omega) * 1j
    
    kx2 = (kx * kx).astype(np.float32, copy=False)
    ky2 = (ky * ky).astype(np.float32, copy=False)
    kz2 = (kz * kz).astype(np.float32, copy=False)
    
    def reconstruct_one(axis, dlt_r):
        tmp = dlt_r.astype(np.complex64, copy=True)
        np.multiply(tmp, factor, out=tmp)
        if axis == "x":
            np.multiply(tmp, kx[:, None, None], out=tmp)
        elif axis == "y":
            np.multiply(tmp, ky[None, :, None], out=tmp)
        else:
            np.multiply(tmp, kz[None, None, :], out=tmp)
        
        absk2 = kx2[:, None, None] + ky2[None, :, None] + kz2[None, None, :]
        np.divide(tmp, absk2, out=tmp, where=absk2 != 0)
        rec = safe_real(fft.irfftn(tmp, s=(n, n, n), workers=-1))
        return rec
    
    # Reconstruction from δ (density contrast)
    vx_rec = reconstruct_one("x", dltk)
    vy_rec = reconstruct_one("y", dltk)
    vz_rec = reconstruct_one("z", dltk)
    
    # Reconstruction from raw δTb (Method A)
    vx_recx = reconstruct_one("x", dltXhk)
    vy_recx = reconstruct_one("y", dltXhk)
    vz_recx = reconstruct_one("z", dltXhk)
    
    # Reconstruction from Tb / <xHI> (backward compatibility)
    vx_recx_norm = reconstruct_one("x", dltXhk_norm)
    vy_recx_norm = reconstruct_one("y", dltXhk_norm)
    vz_recx_norm = reconstruct_one("z", dltXhk_norm)
    
    # Reconstruction from mean-normalized δTb (Method B) - LOS only to save memory
    vz_rec_mean_norm = reconstruct_one("z", dltXhk_mean_norm)
    
    # Reconstruction from fixed-scale δTb (Method C) - LOS only to save memory
    vz_rec_fixed_mK = reconstruct_one("z", dltXhk_fixed_mK)
    
    return (den, xhi, vx, vy, vz, 
            vx_rec, vy_rec, vz_rec,           # from δ
            vx_recx, vy_recx, vz_recx,        # from raw δTb (Method A)
            vx_recx_norm, vy_recx_norm, vz_recx_norm,  # from δTb/<xHI>
            vz_rec_mean_norm,                 # from mean-normalized δTb (Method B)
            vz_rec_fixed_mK)                  # from fixed-scale δTb (Method C)

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def compute_ksz_maps(vz, xhi, den, z=None, physical_norm=False, use_optical_depth=False):
    """
    Compute kSZ map from velocity, ionization, and density fields.
    
    Parameters:
    -----------
    vz : ndarray
        Line-of-sight velocity field in km/s
    xhi : ndarray
        Neutral fraction field
    den : ndarray
        Density field
    z : float, optional
        Redshift (required if physical_norm=True or use_optical_depth=True)
    physical_norm : bool, optional
        If True, apply physical normalization to get µK units
        If False (default), return arbitrary units
    use_optical_depth : bool, optional
        If True, weight the kSZ signal by dτ/dz (optical depth formulation)
        This follows Eq. 4 and 11 from Jelić et al.:
            ΔT_kSZ/T_CMB = -∫ (v_r/c) * x_e * (1+δ) * dτ
        where dτ = (dτ/dz) * dz
    
    Returns:
    --------
    ksz_map : ndarray (2D)
        Integrated kSZ map
    """
    # Compute density contrast
    mean_den = den.mean()
    delta = den / mean_den - 1.0
    
    # Ionized fraction weight (dimensionless)
    # x_e = (1 - x_HI) is ionized fraction
    # n_e/n̄_e = x_e * (1 + δ) where n̄_e is mean electron density
    xe_delta = (1.0 - xhi) * (1.0 + delta)
    
    # Physical normalization factor
    pref = 1.0
    if physical_norm or use_optical_depth:
        if z is None:
            raise ValueError("Redshift z must be provided when physical_norm=True or use_optical_depth=True")
        
        # Physical constants
        c_km_s = 2.99792458e5       # Speed of light [km/s]
        T_CMB = 2.725e6             # CMB temperature [µK]
        
        if use_optical_depth:
            # Optical depth formulation (Eq. 4 from paper):
            # ΔT_kSZ/T_CMB = -∫ (v_r/c) * x_e * (1+δ) * dτ
            # where dτ = (dτ/dz) * dz
            # 
            # For a single redshift slice with thickness Δz:
            # ΔT = -T_CMB * (1/c) * (dτ/dz) * Δz * Σ[x_e * (1+δ) * v_r]
            #
            # We need to estimate Δz for the box at this redshift
            # Using dz/dl = H(z)/c * (1+z) where dl is proper distance
            
            # Cosmological parameters
            H0 = 70.0  # km/s/Mpc
            Omega_m = 0.27
            Omega_L = 1.0 - Omega_m
            
            # Hubble parameter at z
            E_z = np.sqrt(Omega_m * (1 + z)**3 + Omega_L)
            H_z = H0 * E_z  # km/s/Mpc
            
            # Box size in proper Mpc
            n_cells = vz.shape[2]
            box_proper_Mpc = BOX_MPC_OVER_H / LITTLEH / (1.0 + z)
            dl_proper = box_proper_Mpc / n_cells  # Mpc per cell
            
            # dz per cell: dz = H(z)/c * (1+z) * dl_proper
            dz_per_cell = H_z / c_km_s * (1 + z) * dl_proper
            
            # dτ/dz at this redshift (for mean ionized universe)
            dtau_dz = compute_dtau_dz(z)
            
            # Prefactor: -T_CMB * (v/c) * dτ
            # where dτ = (dτ/dz) * dz_per_cell for each cell
            # v is already in km/s, c in km/s, so v/c is dimensionless
            pref = -T_CMB * dtau_dz * dz_per_cell / c_km_s  # [µK]
            
        else:
            # Original physical normalization (direct n_e integration)
            sigma_T = 6.6524587158e-25  # Thomson cross-section [cm^2]
            c_cm_s = 2.99792458e10      # Speed of light [cm/s]
            m_p = 1.6726219e-24         # Proton mass [g]
            Mpc_to_cm = 3.085677581e24  # Mpc to cm
            
            H0 = 70.0  # km/s/Mpc
            Omega_b = 0.044  # Baryon density parameter
            Omega_m = 0.27
            Y_He = 0.24  # Helium mass fraction
            
            # Critical density today [g/cm^3]
            H0_cgs = H0 * 1e5 / Mpc_to_cm  # s^-1
            G_cgs = 6.67430e-8  # cm^3 g^-1 s^-2
            rho_crit_0 = 3 * H0_cgs**2 / (8 * np.pi * G_cgs)  # g/cm^3
            rho_b_0 = Omega_b * rho_crit_0  # g/cm^3
            
            # Mean electron density today (assuming fully ionized H + He)
            # n_e = n_H + 2*n_He = ρ_b/m_p * (1 - Y_He/2)
            n_e_0 = rho_b_0 / m_p * (1.0 - Y_He / 2.0)  # cm^-3
            
            # Mean electron density at redshift z
            n_e_mean = n_e_0 * (1.0 + z)**3  # cm^-3
            
            # Proper cell size along LOS [cm]
            n_cells = vz.shape[2]
            box_proper_Mpc = BOX_MPC_OVER_H / LITTLEH / (1.0 + z)
            dl_proper = box_proper_Mpc / n_cells * Mpc_to_cm  # cm
            
            # kSZ: ΔT = -T_CMB * σ_T * n̄_e * (1/c) * Σ[xe_delta * v_r * dl]
            # v_r in km/s -> cm/s: multiply by 1e5
            pref = -T_CMB * sigma_T * n_e_mean * dl_proper / c_cm_s * 1e5  # [µK]
    
    # Compute kSZ map: Σ[x_e * (1+δ) * v_los]
    # For physical_norm=True, result is in µK
    # For physical_norm=False, result is in arbitrary units
    
    return pref * np.sum(xe_delta * vz, axis=2)

def compute_fourier_correlation_coefficient(field1, field2, boxlength, bins=FOURIER_BINS):
    """Compute Fourier-space correlation coefficient r(k) for 2D fields.
    
    Mean-subtracts fields before FFT to avoid DC component issues.
    Excludes k=0 mode and starts binning from k_fundamental.
    """
    # Mean-subtract to remove DC component
    f1 = field1 - np.mean(field1)
    f2 = field2 - np.mean(field2)
    
    fft1 = np.fft.rfft2(f1)
    fft2 = np.fft.rfft2(f2)
    
    ny, nx = field1.shape
    dx, dy = boxlength[1] / nx, boxlength[0] / ny
    
    kx = 2.0 * np.pi * np.fft.fftfreq(ny, d=dx)
    ky = 2.0 * np.pi * np.fft.rfftfreq(nx, d=dy)
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing='ij')
    k_mag = np.sqrt(kx_grid**2 + ky_grid**2)
    
    # Start from k_fundamental (exclude k=0 DC mode)
    k_fundamental = 2.0 * np.pi / max(boxlength)
    k_max = k_mag.max()
    k_bins = np.linspace(k_fundamental, k_max, bins + 1)
    k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])
    
    r_k = np.zeros(bins)
    for i in range(bins):
        mask = (k_mag >= k_bins[i]) & (k_mag < k_bins[i+1])
        if np.any(mask):
            fft1_bin = fft1[mask]
            fft2_bin = fft2[mask]
            cross = np.real(fft1_bin * np.conj(fft2_bin))
            auto1 = np.real(fft1_bin * np.conj(fft1_bin))
            auto2 = np.real(fft2_bin * np.conj(fft2_bin))
            denom = np.sqrt(auto1.sum() * auto2.sum())
            if denom > 0:
                r_k[i] = cross.sum() / denom
    
    return k_centers, r_k


def compute_fourier_correlation_coefficient_3d(field1, field2, boxlength, bins=FOURIER_BINS):
    """Compute Fourier-space correlation coefficient r(k) for 3D fields.
    
    Mean-subtracts fields before FFT to avoid DC component issues.
    Uses float64 to avoid overflow in auto-power products.
    Excludes k=0 mode and starts binning from k_fundamental.
    
    Parameters:
    -----------
    field1, field2 : 3D arrays
        The two fields to correlate
    boxlength : float or list
        Box size in Mpc/h. If float, assumes cubic box.
    bins : int
        Number of k bins
    
    Returns:
    --------
    k_centers : array
        Centers of k bins [h/Mpc]
    r_k : array
        Correlation coefficient in each k bin
    """
    # Mean-subtract to remove DC component and convert to float64 to avoid overflow
    f1 = (field1 - np.mean(field1)).astype(np.float64)
    f2 = (field2 - np.mean(field2)).astype(np.float64)
    
    fft1 = np.fft.rfftn(f1)
    fft2 = np.fft.rfftn(f2)
    
    nx, ny, nz = field1.shape
    if isinstance(boxlength, (int, float)):
        dx = dy = dz = boxlength / nx
        box_size = boxlength
    else:
        dx = boxlength[0] / nx
        dy = boxlength[1] / ny
        dz = boxlength[2] / nz
        box_size = max(boxlength)
    
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)
    kz = 2.0 * np.pi * np.fft.rfftfreq(nz, d=dz)
    kx_grid, ky_grid, kz_grid = np.meshgrid(kx, ky, kz, indexing='ij')
    k_mag = np.sqrt(kx_grid**2 + ky_grid**2 + kz_grid**2)
    
    # Start from k_fundamental (exclude k=0 DC mode)
    k_fundamental = 2.0 * np.pi / box_size
    k_max = min(k_mag.max(), 2.0)  # Cap at k=2 h/Mpc for reasonable binning
    k_bins = np.linspace(k_fundamental, k_max, bins + 1)
    k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])
    
    r_k = np.zeros(bins)
    for i in range(bins):
        mask = (k_mag >= k_bins[i]) & (k_mag < k_bins[i+1])
        if np.any(mask):
            fft1_bin = fft1[mask]
            fft2_bin = fft2[mask]
            # Use float64 sums to avoid overflow
            cross = np.real(fft1_bin * np.conj(fft2_bin)).sum()
            auto1 = np.real(fft1_bin * np.conj(fft1_bin)).sum()
            auto2 = np.real(fft2_bin * np.conj(fft2_bin)).sum()
            denom = np.sqrt(auto1 * auto2)
            if denom > 0:
                r_k[i] = cross / denom
    
    return k_centers, r_k

# ============================================================================
# JACKKNIFE HELPERS
# ============================================================================
# Spatial block jackknife: partition the volume into N cube/tile sub-volumes,
# compute the statistic on each leave-one-out subset (N-1 blocks), and report
#     σ_jk² = (N-1)/N · Σ_i (x_i − x̄)²
# with x̄ the mean across the N leave-one-out estimates.
#
# For Pearson r we use the additive-sum trick: per-block sums of A, B, A², B²,
# AB allow the leave-one-out r to be reconstructed in O(N) without re-flattening
# big arrays. For r(k) we compute the spectrum on each sub-cube separately,
# interpolate onto the full-box k-grid, and average over the (N-1) blocks.
# ============================================================================

def _block_sums(field_a, field_b, n_per_side):
    """Per-block sums of A, B, A², B², AB and cell count for jackknife."""
    shape = field_a.shape
    splits = [np.array_split(np.arange(s), n_per_side) for s in shape]
    blocks = []
    for ix in splits[0]:
        for iy in splits[1]:
            if len(shape) == 3:
                for iz in splits[2]:
                    a = field_a[ix[0]:ix[-1]+1, iy[0]:iy[-1]+1,
                                iz[0]:iz[-1]+1].astype(np.float64, copy=False)
                    b = field_b[ix[0]:ix[-1]+1, iy[0]:iy[-1]+1,
                                iz[0]:iz[-1]+1].astype(np.float64, copy=False)
                    blocks.append((a.size, a.sum(), b.sum(),
                                   (a*a).sum(), (b*b).sum(), (a*b).sum()))
            else:
                a = field_a[ix[0]:ix[-1]+1, iy[0]:iy[-1]+1].astype(np.float64,
                                                                    copy=False)
                b = field_b[ix[0]:ix[-1]+1, iy[0]:iy[-1]+1].astype(np.float64,
                                                                    copy=False)
                blocks.append((a.size, a.sum(), b.sum(),
                               (a*a).sum(), (b*b).sum(), (a*b).sum()))
    return np.array(blocks, dtype=np.float64)


def _pearson_from_sums(n, sa, sb, saa, sbb, sab):
    num = n*sab - sa*sb
    da = n*saa - sa*sa
    db = n*sbb - sb*sb
    if da <= 0 or db <= 0:
        return np.nan
    return float(num / np.sqrt(da * db))


def jackknife_pearson_r(field_a, field_b, n_per_side=2):
    """Leave-one-out jackknife on Pearson r between two equal-shape fields.

    Works for 2D or 3D; uses n_per_side**D blocks. Returns
    (mean of leave-one-out estimates, jackknife σ, leave-one-out values).
    """
    blocks = _block_sums(field_a, field_b, n_per_side)
    total = blocks.sum(axis=0)
    N = blocks.shape[0]
    loo = np.empty(N)
    for i in range(N):
        s = total - blocks[i]
        loo[i] = _pearson_from_sums(*s)
    mean = float(np.nanmean(loo))
    sigma = float(np.sqrt((N - 1) / N * np.nansum((loo - mean)**2)))
    return mean, sigma, loo


def jackknife_r_k_3d(field_a, field_b, boxlength, n_per_side=2,
                     bins=FOURIER_BINS, k_grid=None):
    """Octant-block jackknife on the Fourier correlation r(k) of 3D fields.

    Computes r(k) on each sub-cube, interpolates onto `k_grid` (if given,
    otherwise the first sub-cube's k-centres), then forms leave-one-out
    averages and the jackknife σ band.
    """
    shape = field_a.shape
    splits = [np.array_split(np.arange(s), n_per_side) for s in shape]
    sub_L = boxlength / n_per_side
    per_block = []
    for ix in splits[0]:
        for iy in splits[1]:
            for iz in splits[2]:
                a_sub = field_a[ix[0]:ix[-1]+1, iy[0]:iy[-1]+1, iz[0]:iz[-1]+1]
                b_sub = field_b[ix[0]:ix[-1]+1, iy[0]:iy[-1]+1, iz[0]:iz[-1]+1]
                k_s, r_s = compute_fourier_correlation_coefficient_3d(
                    a_sub, b_sub, boxlength=sub_L, bins=bins
                )
                per_block.append((k_s, r_s))

    if k_grid is None:
        k_grid = per_block[0][0]

    rks = np.array([np.interp(k_grid, k_s, r_s, left=np.nan, right=np.nan)
                    for k_s, r_s in per_block])
    N = rks.shape[0]
    # Leave-one-out average across blocks (per-k)
    with np.errstate(invalid='ignore'):
        total = np.nansum(rks, axis=0)
        count = np.sum(np.isfinite(rks), axis=0)
        loo = np.empty_like(rks)
        for i in range(N):
            num = total - np.where(np.isfinite(rks[i]), rks[i], 0.0)
            den = np.maximum(count - np.isfinite(rks[i]).astype(int), 1)
            loo[i] = num / den
    mean = np.nanmean(loo, axis=0)
    sigma = np.sqrt((N - 1) / N * np.nansum((loo - mean)**2, axis=0))
    return k_grid, mean, sigma


def jackknife_r_k_2d(map_a, map_b, boxlength, n_per_side=2,
                     bins=FOURIER_BINS, k_grid=None):
    """Tile-block jackknife on the Fourier correlation r(k) of 2D maps."""
    shape = map_a.shape
    splits = [np.array_split(np.arange(s), n_per_side) for s in shape]
    sub_L = [boxlength[0] / n_per_side, boxlength[1] / n_per_side]
    per_block = []
    for ix in splits[0]:
        for iy in splits[1]:
            a_sub = map_a[ix[0]:ix[-1]+1, iy[0]:iy[-1]+1]
            b_sub = map_b[ix[0]:ix[-1]+1, iy[0]:iy[-1]+1]
            k_s, r_s = compute_fourier_correlation_coefficient(
                a_sub, b_sub, boxlength=sub_L, bins=bins
            )
            per_block.append((k_s, r_s))

    if k_grid is None:
        k_grid = per_block[0][0]

    rks = np.array([np.interp(k_grid, k_s, r_s, left=np.nan, right=np.nan)
                    for k_s, r_s in per_block])
    N = rks.shape[0]
    with np.errstate(invalid='ignore'):
        total = np.nansum(rks, axis=0)
        count = np.sum(np.isfinite(rks), axis=0)
        loo = np.empty_like(rks)
        for i in range(N):
            num = total - np.where(np.isfinite(rks[i]), rks[i], 0.0)
            den = np.maximum(count - np.isfinite(rks[i]).astype(int), 1)
            loo[i] = num / den
    mean = np.nanmean(loo, axis=0)
    sigma = np.sqrt((N - 1) / N * np.nansum((loo - mean)**2, axis=0))
    return k_grid, mean, sigma


# ============================================================================
# R(k,z) MEASUREMENT — ionisation-to-density ratio
# ============================================================================

def measure_R_kz_single(z, n=600, bins=40, use_crop=True):
    """
    Measure the ionisation-to-density ratio R(k,z) at a single redshift.

    R(k,z) = P_{δx_HI, δ}(k) / [ x̄_HI(z) · P_{δδ}(k) ]

    From the paper:
        Δ₂₁(k,z) ≈ [1 + R(k,z)] δ(k,z)

    Parameters
    ----------
    z : float
        Redshift.
    n : int
        Grid size (default 600).
    bins : int
        Number of spherical k-bins.
    use_crop : bool
        If True, apply CENTRAL_CROP to reduce boundary effects.

    Returns
    -------
    dict with keys: z, mean_xHI, k, R_k, P_cross, P_auto
    """
    zstr = f"{z:.3f}"
    filenameDen = f"{DATA_DIR}{zstr}n_all.dat"
    filenameXhi = f"{DATA_DIR}{zstr}zeta0.389fesc0.389_Mmin0.120E+10_MminX0.120E+10_fx0.100E+03_sed3_al1.200xhi.bin"

    den = read_den(filenameDen, n, n, n).astype(np.float64)
    xhi = read_xhi(filenameXhi, n, n, n).astype(np.float64)

    if use_crop:
        den = den[CENTRAL_CROP, CENTRAL_CROP, LOS_CROP]
        xhi = xhi[CENTRAL_CROP, CENTRAL_CROP, LOS_CROP]

    # Overdensity δ = ρ/ρ̄ − 1
    mean_den = den.mean()
    delta = den / mean_den - 1.0

    # Neutral fraction fluctuation
    xhi_mean = float(xhi.mean())
    delta_xhi = xhi - xhi_mean

    # 3D rFFT
    delta_k = np.fft.rfftn(delta)
    delta_xhi_k = np.fft.rfftn(delta_xhi)

    # k-space grid [h/Mpc]
    nx, ny, nz_r = delta_k.shape
    nz_full = delta.shape[2]
    dx = BOX_MPC_OVER_H / n  # cell size in Mpc/h
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dx)
    kz = 2.0 * np.pi * np.fft.rfftfreq(nz_full, d=dx)
    kx_g, ky_g, kz_g = np.meshgrid(kx, ky, kz, indexing='ij')
    k_mag = np.sqrt(kx_g**2 + ky_g**2 + kz_g**2)

    # Bin edges
    k_fundamental = 2.0 * np.pi / (dx * nx)  # ~ 2π / box_crop
    k_max = min(k_mag.max(), 2.0)  # cap at 2 h/Mpc
    k_edges = np.linspace(k_fundamental, k_max, bins + 1)
    k_centers = 0.5 * (k_edges[:-1] + k_edges[1:])

    # Cross-power and auto-power in shells
    P_cross = np.zeros(bins)
    P_auto = np.zeros(bins)
    R_k = np.full(bins, np.nan)

    for i in range(bins):
        mask = (k_mag >= k_edges[i]) & (k_mag < k_edges[i + 1])
        if not np.any(mask):
            continue
        cross = np.real(delta_xhi_k[mask] * np.conj(delta_k[mask]))
        auto = np.real(delta_k[mask] * np.conj(delta_k[mask]))
        P_cross[i] = cross.mean()
        P_auto[i] = auto.mean()
        if P_auto[i] > 0 and xhi_mean > 1e-6:
            R_k[i] = P_cross[i] / (xhi_mean * P_auto[i])

    print(f"  z={z:7.3f}  x̄_HI={xhi_mean:.4f}  "
          f"R(k→0)≈{np.nanmean(R_k[:3]):.3f}  "
          f"1+R(k→0)≈{1 + np.nanmean(R_k[:3]):.3f}")

    del den, xhi, delta, delta_xhi, delta_k, delta_xhi_k
    gc.collect()

    return {
        'z': z,
        'mean_xHI': xhi_mean,
        'k': k_centers,
        'R_k': R_k,
        'P_cross': P_cross,
        'P_auto': P_auto,
    }


def measure_R_kz_all(redshifts, n=600, bins=40):
    """
    Measure R(k,z) at multiple redshifts.

    Parameters
    ----------
    redshifts : list of float
    n : int
        Grid size.
    bins : int
        Number of k-bins.

    Returns
    -------
    list of dict (one per redshift, same format as measure_R_kz_single).
    """
    print("\n" + "="*80)
    print("MEASURING R(k,z) — ionisation-to-density ratio")
    print("="*80)

    R_results = []
    for z in redshifts:
        res = measure_R_kz_single(z, n=n, bins=bins)
        R_results.append(res)
    return R_results


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_velocity_correlation_vs_neutral_fraction(results, output_dir=OUTPUT_DIR):
    """Plot velocity reconstruction correlation vs neutral fraction with redshift colorbar."""
    print("\n" + "="*80)
    print("Plotting: Velocity Correlation vs Neutral Fraction")
    print("="*80)
    
    neutral_fractions = np.array([r['mean_xHI'] for r in results])
    r_3d_values = np.array([r['r_3d'] for r in results])
    r_2d_values = np.array([r['r_2d'] for r in results])
    redshifts = np.array([r['z'] for r in results])
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    ax.tick_params(axis='both', which='major', labelsize=20, length=8, width=2, pad=10)
    
    # Plot dashed gray lines connecting points
    ax.plot(neutral_fractions, r_3d_values, '--', linewidth=2, color='gray', alpha=0.6, zorder=1)
    ax.plot(neutral_fractions, r_2d_values, '--', linewidth=2, color='gray', alpha=0.6, zorder=1)
    
    # Scatter with redshift color coding - simple markers
    sc1 = ax.scatter(neutral_fractions, r_3d_values, c=redshifts, cmap='viridis', 
                     s=60, zorder=5, marker='o')
    sc2 = ax.scatter(neutral_fractions, r_2d_values, c=redshifts, cmap='viridis', 
                     s=60, zorder=5, marker='s')
    
    # Colorbar for redshift
    cbar = plt.colorbar(sc1, ax=ax, pad=0.02)
    cbar.set_label('Redshift z', fontsize=20)
    cbar.ax.tick_params(labelsize=18)
    
    # Custom legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
               markersize=10, label='3D field'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', 
               markersize=10, label='2D integrated')
    ]
    ax.legend(handles=legend_elements, fontsize=14, loc='best', framealpha=0.9)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax.axhline(y=1, color='gray', linestyle='--', linewidth=1, alpha=0.3)
    ax.set_xlabel('Mean Neutral Fraction $\\langle x_{HI} \\rangle$', fontsize=22)
    ax.set_ylabel('Pearson coefficient r(info)', fontsize=22)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1.0, 1.0)
    ax.set_xlim([min(neutral_fractions) - 0.05, max(neutral_fractions) + 0.05])
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}velocity_correlation_vs_neutral_fraction.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}velocity_correlation_vs_neutral_fraction.png")


def plot_velocity_correlation_combined(results, output_dir=OUTPUT_DIR):
    """Create 2-panel figure: Left = real-space velocity correlation vs xHI, Right = r(k) vs k for selected redshifts."""
    print("\n" + "="*80)
    print("Plotting: Velocity Correlation Combined (Real + Fourier)")
    print("="*80)
    
    neutral_fractions = np.array([r['mean_xHI'] for r in results])
    # Negate to show correlation without the minus sign on reconstruction
    r_3d_values = -np.array([r['r_3d'] for r in results])
    r_2d_values = -np.array([r['r_2d'] for r in results])
    redshifts = np.array([r['z'] for r in results])
    
    # Select 4 redshifts based on target neutral fractions for right panel
    # Exclude z=6.83 (xHI~0.47)
    target_xHI_plot = [0.95, 0.55, 0.25, 0.05]
    selected_results = []
    for target in target_xHI_plot:
        best_idx = min(range(len(results)), 
                       key=lambda i: abs(results[i]['mean_xHI'] - target))
        if results[best_idx] not in selected_results:
            selected_results.append(results[best_idx])
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    # Left: Real-space velocity correlation vs neutral fraction
    ax = axes[0]
    ax.plot(neutral_fractions, r_3d_values, '--', linewidth=2, color='gray', alpha=0.6, zorder=1)
    ax.plot(neutral_fractions, r_2d_values, '--', linewidth=2, color='gray', alpha=0.6, zorder=1)
    sc1 = ax.scatter(neutral_fractions, r_3d_values, c=redshifts, cmap='viridis', 
                     s=60, zorder=5, marker='o')
    sc2 = ax.scatter(neutral_fractions, r_2d_values, c=redshifts, cmap='viridis', 
                     s=60, zorder=5, marker='s')
    cbar = plt.colorbar(sc1, ax=ax, pad=0.02)
    cbar.set_label('Redshift z', fontsize=20)
    cbar.ax.tick_params(labelsize=18)
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
               markersize=10, label='3D field'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', 
               markersize=10, label='2D integrated')
    ]
    ax.legend(handles=legend_elements, fontsize=18, loc='upper left', framealpha=0.9)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax.set_xlabel('Mean Neutral Fraction $\\langle x_{HI} \\rangle$', fontsize=22)
    ax.set_ylabel('$r(x)$', fontsize=22)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-1.0, 1.0])
    ax.set_xlim([0.0, 1.0])
    ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.tick_params(axis='both', which='major', labelsize=20)
    
    # Right: Fourier velocity correlation r(k) vs k for selected redshifts
    # Show both 3D (dashed) and 2D projected (solid) correlations
    # Start from k=0.1 because large scales (k<0.1) are poorly sampled in the box
    # and dominated by sample variance / edge effects
    ax = axes[1]
    norm = plt.Normalize(vmin=redshifts.min(), vmax=redshifts.max())
    cmap = plt.cm.viridis
    
    # Plot each redshift with both 3D (dashed) and 2D (solid)
    # Use k <= 1.0 cutoff only (no lower cutoff, let xlim handle display)
    for result in selected_results:
        color = cmap(norm(result['z']))
        
        # 3D Fourier correlation (dashed line)
        k_mask_3d = result['k_values_vel_3d'] <= 1.0
        ax.plot(result['k_values_vel_3d'][k_mask_3d], -result['r_k_vel_3d'][k_mask_3d],
                color=color, linewidth=2.0, linestyle='--', alpha=0.8)
        
        # 2D Fourier correlation (solid line)
        k_mask_2d = result['k_values_vel_2d'] <= 1.0
        ax.plot(result['k_values_vel_2d'][k_mask_2d], -result['r_k_vel_2d'][k_mask_2d],
                color=color, linewidth=2.0, linestyle='-',
                label=f"z={result['z']:.2f}, $x_{{HI}}$={result['mean_xHI']:.2f}")
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax.set_xlabel('k [h/Mpc]', fontsize=22)
    ax.set_ylabel('$r(k)$', fontsize=22)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-1.01, 1.01])
    ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.set_xlim([0.0, 1.0])
    
    # Create combined legend with 2D/3D info on top, then redshift info below
    # All with same font size and consistent formatting
    style_legend = [
        Line2D([0], [0], color='gray', linewidth=2, linestyle='-', label='2D projected'),
        Line2D([0], [0], color='gray', linewidth=2, linestyle='--', label='3D field')
    ]
    # Get the redshift legend handles from the plot
    handles, labels = ax.get_legend_handles_labels()
    # Combine: style legend on top, then redshift legend
    all_handles = style_legend + handles
    all_labels = ['2D projected', '3D field'] + labels
    ax.legend(all_handles, all_labels, fontsize=18, loc='upper right', framealpha=0.9)
    ax.tick_params(axis='both', which='major', labelsize=20)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/velocity_correlation_combined.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/velocity_correlation_combined.png")


def plot_velocity_correlation_jk(results, output_dir=OUTPUT_DIR):
    """Fig 3 with jackknife error bars.

    Left  panel: real-space r(3D), r(2D) vs ⟨x_HI⟩ with σ_jk error bars.
    Right panel: r(k) vs k for selected redshifts with ±σ_jk shaded bands.
    Saves to <output_dir>/velocity_correlation_jk.{png,pdf}.
    """
    print("\n" + "=" * 80)
    print("Plotting: Velocity Correlation Combined (jackknife)")
    print("=" * 80)

    neutral_fractions = np.array([r['mean_xHI'] for r in results])
    r_3d_values = -np.array([r['r_3d'] for r in results])
    r_2d_values = -np.array([r['r_2d'] for r in results])
    r_3d_sigma = np.array([r.get('r_3d_jk_sigma', np.nan) for r in results])
    r_2d_sigma = np.array([r.get('r_2d_jk_sigma', np.nan) for r in results])
    redshifts = np.array([r['z'] for r in results])

    target_xHI_plot = [0.95, 0.55, 0.25, 0.05]
    selected_results = []
    for target in target_xHI_plot:
        best_idx = min(range(len(results)),
                       key=lambda i: abs(results[i]['mean_xHI'] - target))
        if results[best_idx] not in selected_results:
            selected_results.append(results[best_idx])

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # Left panel: real-space Pearson r with jackknife error bars
    ax = axes[0]
    ax.plot(neutral_fractions, r_3d_values, '--', lw=2, color='gray',
            alpha=0.6, zorder=1)
    ax.plot(neutral_fractions, r_2d_values, '--', lw=2, color='gray',
            alpha=0.6, zorder=1)
    norm = plt.Normalize(vmin=redshifts.min(), vmax=redshifts.max())
    colors_pts = plt.cm.viridis(norm(redshifts))
    ax.errorbar(neutral_fractions, r_3d_values, yerr=r_3d_sigma,
                fmt='none', ecolor='gray', alpha=0.6, zorder=3)
    ax.errorbar(neutral_fractions, r_2d_values, yerr=r_2d_sigma,
                fmt='none', ecolor='gray', alpha=0.6, zorder=3)
    sc1 = ax.scatter(neutral_fractions, r_3d_values, c=redshifts, cmap='viridis',
                     s=60, zorder=5, marker='o')
    ax.scatter(neutral_fractions, r_2d_values, c=redshifts, cmap='viridis',
               s=60, zorder=5, marker='s')
    cbar = plt.colorbar(sc1, ax=ax, pad=0.02)
    cbar.set_label('Redshift z', fontsize=20)
    cbar.ax.tick_params(labelsize=18)
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
               markersize=10, label='3D field'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='gray',
               markersize=10, label='2D integrated'),
    ]
    ax.legend(handles=legend_elements, fontsize=18, loc='upper left',
              framealpha=0.9)
    ax.axhline(y=0, color='black', lw=1, alpha=0.5)
    ax.set_xlabel('Mean Neutral Fraction $\\langle x_{HI} \\rangle$', fontsize=22)
    ax.set_ylabel('$r(x)$', fontsize=22)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-1.05, 1.05])
    ax.set_xlim([0.0, 1.0])
    ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.tick_params(axis='both', which='major', labelsize=20)

    # Right panel: r(k) curves with jackknife error bands
    ax = axes[1]
    cmap = plt.cm.viridis
    for result in selected_results:
        color = cmap(norm(result['z']))
        k3d = result['k_values_vel_3d']
        r3d = -result['r_k_vel_3d']
        s3d = result.get('r_k_vel_3d_jk_sigma', np.zeros_like(r3d))
        m3d = (k3d <= 1.0) & np.isfinite(r3d)
        ax.fill_between(k3d[m3d], (r3d - s3d)[m3d], (r3d + s3d)[m3d],
                        color=color, alpha=0.15, zorder=1)
        ax.plot(k3d[m3d], r3d[m3d], color=color, lw=2.0, linestyle='--',
                alpha=0.85)

        k2d = result['k_values_vel_2d']
        r2d = -result['r_k_vel_2d']
        s2d = result.get('r_k_vel_2d_jk_sigma', np.zeros_like(r2d))
        m2d = (k2d <= 1.0) & np.isfinite(r2d)
        ax.fill_between(k2d[m2d], (r2d - s2d)[m2d], (r2d + s2d)[m2d],
                        color=color, alpha=0.15, zorder=1)
        ax.plot(k2d[m2d], r2d[m2d], color=color, lw=2.0, linestyle='-',
                label=f"z={result['z']:.2f}, $x_{{HI}}$={result['mean_xHI']:.2f}")

    ax.axhline(y=0, color='black', lw=0.8, alpha=0.5)
    ax.set_xlabel('k [h/Mpc]', fontsize=22)
    ax.set_ylabel('$r(k)$', fontsize=22)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-1.05, 1.05])
    ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.set_xlim([0.0, 1.0])
    style_legend = [
        Line2D([0], [0], color='gray', lw=2, linestyle='-', label='2D projected'),
        Line2D([0], [0], color='gray', lw=2, linestyle='--', label='3D field'),
    ]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(style_legend + handles,
              ['2D projected', '3D field'] + labels,
              fontsize=18, loc='upper right', framealpha=0.9)
    ax.tick_params(axis='both', which='major', labelsize=20)

    plt.tight_layout()
    for ext in ('png', 'pdf'):
        out = os.path.join(output_dir, f'velocity_correlation_jk.{ext}')
        plt.savefig(out, dpi=300, bbox_inches='tight')
        print(f"  Saved: {out}")
    plt.close()


def plot_2d_velocity_scatter(z, output_dir=OUTPUT_DIR):
    """Create scatter plot for 2D integrated velocity maps at a specific redshift.
    
    Shows both raw and amplitude-normalized reconstructions to demonstrate that
    the δTb reconstruction has correct morphology (high r) but wrong amplitude,
    which can be fixed by simple rescaling.
    """
    print("\n" + "="*80)
    print(f"Plotting: 2D Integrated Velocity Scatter (z={z:.3f})")
    print("="*80)
    
    # Load data
    (den, xhi, vx, vy, vz, vx_rec, vy_rec, vz_rec, 
     vx_recx, vy_recx, vz_recx, vx_recx_norm, vy_recx_norm, vz_recx_norm,
     _, _) = reconstruct_velocities(z)
    n = den.shape[0]
    
    # Crop to central region
    den = den[CENTRAL_CROP, CENTRAL_CROP, LOS_CROP]
    xhi = xhi[CENTRAL_CROP, CENTRAL_CROP, LOS_CROP]
    vz = vz[CENTRAL_CROP, CENTRAL_CROP, LOS_CROP]
    vz_recx = vz_recx[CENTRAL_CROP, CENTRAL_CROP, LOS_CROP]
    vz_rec = vz_rec[CENTRAL_CROP, CENTRAL_CROP, LOS_CROP]
    
    mean_xHI = xhi.mean()
    
    # 2D integrated velocity maps
    vz_map = np.sum(vz, axis=2)
    vz_rec_map = np.sum(-vz_recx, axis=2)  # Using -vz_recx as in the analysis
    vz_rec_unfiltered_map = np.sum(vz_rec, axis=2)  # Unfiltered reconstruction
    
    # Exclude boundaries (10 pixels on each side) to avoid edge effects from gradient
    boundary = 10
    vz_map = vz_map[boundary:-boundary, boundary:-boundary]
    vz_rec_map = vz_rec_map[boundary:-boundary, boundary:-boundary]
    vz_rec_unfiltered_map = vz_rec_unfiltered_map[boundary:-boundary, boundary:-boundary]
    
    # Flatten for scatter plot
    vz_flat = vz_map.flatten()
    vz_rec_flat = vz_rec_map.flatten()
    vz_rec_unfilt_flat = vz_rec_unfiltered_map.flatten()
    
    # =========================================================================
    # AMPLITUDE NORMALIZATION
    # =========================================================================
    # The δTb reconstruction has correct morphology but wrong amplitude.
    # We can fix this by rescaling to match the true velocity variance.
    # This is equivalent to finding the best-fit slope.
    # =========================================================================
    
    # Compute best-fit slope for amplitude correction
    mask = np.isfinite(vz_flat) & np.isfinite(vz_rec_flat)
    if np.any(mask):
        slope, intercept = np.polyfit(vz_rec_flat[mask], vz_flat[mask], 1)
        vz_rec_normalized = slope * vz_rec_flat + intercept
    else:
        slope, intercept = 1.0, 0.0
        vz_rec_normalized = vz_rec_flat
    
    # Compute metrics
    def compute_metrics(y_true, y_pred):
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        if not np.any(mask):
            return np.nan, np.nan
        y_t, y_p = y_true[mask], y_pred[mask]
        r = np.corrcoef(y_t, y_p)[0, 1] if len(y_t) > 1 else np.nan
        rmse = np.sqrt(np.mean((y_p - y_t)**2))
        return float(r), float(rmse)
    
    r_unfilt, rmse_unfilt = compute_metrics(vz_flat, vz_rec_unfilt_flat)
    r_filt, rmse_filt = compute_metrics(vz_flat, vz_rec_flat)
    r_norm, rmse_norm = compute_metrics(vz_flat, vz_rec_normalized)
    
    print(f"  δ reconstruction: r={r_unfilt:.4f}, RMSE={rmse_unfilt:.2f}")
    print(f"  δTb reconstruction (raw): r={r_filt:.4f}, RMSE={rmse_filt:.2f}")
    print(f"  δTb reconstruction (normalized): r={r_norm:.4f}, RMSE={rmse_norm:.2f}")
    print(f"  Amplitude correction: slope={slope:.4f}, intercept={intercept:.2f}")
    
    # Create scatter plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Sample points for plotting (to avoid overplotting)
    n_points = len(vz_flat)
    sample_size = min(50000, n_points)
    np.random.seed(42)
    idx = np.random.choice(n_points, sample_size, replace=False)
    
    # Plot - show normalized version instead of raw δTb
    ax.scatter(vz_rec_unfilt_flat[idx], vz_flat[idx], alpha=0.3, s=1, c='blue', 
               label=f'from $\\delta$: r={r_unfilt:.3f}')
    ax.scatter(vz_rec_normalized[idx], vz_flat[idx], alpha=0.3, s=1, c='green',
               label=f'from $-\\delta T_b$ (normalized): r={r_norm:.3f}')
    
    # y=x line
    lim = np.nanpercentile(vz_flat, [1, 99])
    ax.plot([lim[0], lim[1]], [lim[0], lim[1]], 'k--', linewidth=2, alpha=0.7, label='y = x')
    
    ax.set_xlabel('Reconstructed $v_z$ (2D integrated) [km/s]', fontsize=20)
    ax.set_ylabel('True $v_z$ (2D integrated) [km/s]', fontsize=20)
    ax.set_title(f'z = {z:.2f}, $\\langle x_{{HI}} \\rangle$ = {mean_xHI:.3f}', fontsize=18)
    ax.legend(fontsize=14, loc='upper left', markerscale=10)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    outfile = os.path.join(output_dir, f'velocity_2d_scatter_z{z:.3f}.png')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outfile}")
    
    # =========================================================================
    # SECOND PLOT: Show raw vs normalized side by side
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Left: Raw (showing amplitude offset)
    ax = axes[0]
    ax.scatter(vz_rec_unfilt_flat[idx], vz_flat[idx], alpha=0.3, s=1, c='blue', 
               label=f'from $\\delta$: r={r_unfilt:.3f}')
    ax.scatter(vz_rec_flat[idx], vz_flat[idx], alpha=0.3, s=1, c='orange',
               label=f'from $-\\delta T_b$ (raw): r={r_filt:.3f}')
    all_vals = np.concatenate([vz_flat, vz_rec_flat, vz_rec_unfilt_flat])
    lim_raw = np.nanpercentile(all_vals, [1, 99])
    ax.plot([lim_raw[0], lim_raw[1]], [lim_raw[0], lim_raw[1]], 'k--', linewidth=2, alpha=0.7, label='y = x')
    ax.set_xlabel('Reconstructed $v_z$ (2D integrated)', fontsize=16)
    ax.set_ylabel('True $v_z$ (2D integrated)', fontsize=16)
    ax.set_title('Raw amplitude (offset visible)', fontsize=16)
    ax.legend(fontsize=12, loc='upper left', markerscale=10)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_xlim(lim_raw)
    ax.set_ylim(lim_raw)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Right: Normalized (amplitude corrected)
    ax = axes[1]
    ax.scatter(vz_rec_unfilt_flat[idx], vz_flat[idx], alpha=0.3, s=1, c='blue', 
               label=f'from $\\delta$: r={r_unfilt:.3f}')
    ax.scatter(vz_rec_normalized[idx], vz_flat[idx], alpha=0.3, s=1, c='green',
               label=f'from $-\\delta T_b$ (normalized): r={r_norm:.3f}')
    ax.plot([lim[0], lim[1]], [lim[0], lim[1]], 'k--', linewidth=2, alpha=0.7, label='y = x')
    ax.set_xlabel('Reconstructed $v_z$ (2D integrated)', fontsize=16)
    ax.set_ylabel('True $v_z$ (2D integrated)', fontsize=16)
    ax.set_title(f'Amplitude normalized (slope={slope:.3f})', fontsize=16)
    ax.legend(fontsize=12, loc='upper left', markerscale=10)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'2D Integrated Velocity: z = {z:.2f}, $\\langle x_{{HI}} \\rangle$ = {mean_xHI:.3f}', fontsize=18)
    plt.tight_layout()
    outfile = os.path.join(output_dir, f'velocity_2d_scatter_raw_vs_norm_z{z:.3f}.png')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outfile}")
    
    # Clean up
    del den, xhi, vz, vz_recx, vz_rec, vz_map, vz_rec_map, vz_rec_unfiltered_map
    gc.collect()
    
    return {'z': z, 'mean_xHI': mean_xHI, 'r_unfilt': r_unfilt, 'r_filt': r_filt,
            'r_norm': r_norm, 'rmse_unfilt': rmse_unfilt, 'rmse_filt': rmse_filt,
            'rmse_norm': rmse_norm, 'slope': slope}


def plot_velocity_and_ksz_comparison(z1=6.2, z2=7.5, output_dir=OUTPUT_DIR):
    """Create 2x2 comparison plot: velocity (top) and kSZ (bottom) at two redshifts.
    
    Each panel shows both δ and -δTb reconstruction overlaid.
    Left column: z1 (e.g., 6.2)
    Right column: z2 (e.g., 7.5)
    Top row: 3D voxel-by-voxel velocity comparison
    Bottom row: kSZ signal (ne × vz) comparison
    """
    print("\n" + "="*80)
    print(f"Plotting: Velocity and kSZ Comparison (z={z1:.3f}, z={z2:.3f})")
    print("="*80)
    
    def compute_metrics(y_true, y_pred):
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        if not np.any(mask):
            return np.nan, np.nan
        y_t, y_p = y_true[mask], y_pred[mask]
        r = np.corrcoef(y_t, y_p)[0, 1] if len(y_t) > 1 else np.nan
        return float(r)
    
    def load_and_process(z):
        """Load data and compute all needed quantities for one redshift."""
        (den, xhi, vx, vy, vz, vx_rec, vy_rec, vz_rec, 
         vx_recx, vy_recx, vz_recx, vx_recx_norm, vy_recx_norm, vz_recx_norm,
         _, _) = reconstruct_velocities(z)
        
        # Crop to central region
        den = den[CENTRAL_CROP, CENTRAL_CROP, LOS_CROP]
        xhi = xhi[CENTRAL_CROP, CENTRAL_CROP, LOS_CROP]
        vz = vz[CENTRAL_CROP, CENTRAL_CROP, LOS_CROP]
        vz_rec = vz_rec[CENTRAL_CROP, CENTRAL_CROP, LOS_CROP]
        vz_recx = vz_recx[CENTRAL_CROP, CENTRAL_CROP, LOS_CROP]
        vz_recx_norm = vz_recx_norm[CENTRAL_CROP, CENTRAL_CROP, LOS_CROP]
        
        mean_xHI = xhi.mean()
        
        # Exclude boundaries
        boundary = 10
        vz_crop = vz[boundary:-boundary, boundary:-boundary, boundary:-boundary]
        vz_rec_crop = vz_rec[boundary:-boundary, boundary:-boundary, boundary:-boundary]
        vz_recx_crop = vz_recx[boundary:-boundary, boundary:-boundary, boundary:-boundary]
        vz_recx_norm_crop = vz_recx_norm[boundary:-boundary, boundary:-boundary, boundary:-boundary]
        den_crop = den[boundary:-boundary, boundary:-boundary, boundary:-boundary]
        xhi_crop = xhi[boundary:-boundary, boundary:-boundary, boundary:-boundary]
        
        # Compute electron density ne = (1 - xHI) × (1 + δ)
        mean_den = den_crop.mean()
        delta = den_crop / mean_den
        ne = (1 - xhi_crop) * delta
        
        # Flatten
        vz_flat = vz_crop.flatten()
        vz_rec_flat = vz_rec_crop.flatten()
        vz_recx_flat = (-vz_recx_crop).flatten()
        vz_recx_norm_flat = (-vz_recx_norm_crop).flatten()
        
        # kSZ signals
        ksz_sim = (ne * vz_crop).flatten()
        ksz_rec_delta = (ne * vz_rec_crop).flatten()
        ksz_rec_tb = (ne * (-vz_recx_crop)).flatten()
        ksz_rec_tb_norm = (ne * (-vz_recx_norm_crop)).flatten()
        
        # Compute correlations
        r_vel_delta = compute_metrics(vz_flat, vz_rec_flat)
        r_vel_tb = compute_metrics(vz_flat, vz_recx_flat)
        r_vel_tb_norm = compute_metrics(vz_flat, vz_recx_norm_flat)
        r_ksz_delta = compute_metrics(ksz_sim, ksz_rec_delta)
        r_ksz_tb = compute_metrics(ksz_sim, ksz_rec_tb)
        r_ksz_tb_norm = compute_metrics(ksz_sim, ksz_rec_tb_norm)
        
        return {
            'z': z, 'mean_xHI': mean_xHI,
            'vz_flat': vz_flat, 'vz_rec_flat': vz_rec_flat, 
            'vz_recx_flat': vz_recx_flat, 'vz_recx_norm_flat': vz_recx_norm_flat,
            'ksz_sim': ksz_sim, 'ksz_rec_delta': ksz_rec_delta, 
            'ksz_rec_tb': ksz_rec_tb, 'ksz_rec_tb_norm': ksz_rec_tb_norm,
            'r_vel_delta': r_vel_delta, 'r_vel_tb': r_vel_tb, 'r_vel_tb_norm': r_vel_tb_norm,
            'r_ksz_delta': r_ksz_delta, 'r_ksz_tb': r_ksz_tb, 'r_ksz_tb_norm': r_ksz_tb_norm
        }
    
    # Load data for both redshifts
    data1 = load_and_process(z1)
    data2 = load_and_process(z2)
    
    # Create 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    sample_size = 30000
    np.random.seed(42)
    
    for col, data in enumerate([data1, data2]):
        z = data['z']
        mean_xHI = data['mean_xHI']
        
        n_pts = len(data['vz_flat'])
        idx = np.random.choice(n_pts, min(sample_size, n_pts), replace=False)
        
        # Top row: 3D velocity comparison (δ, -δTb, and -δTb/<xHI> overlaid)
        ax = axes[0, col]
        ax.scatter(data['vz_rec_flat'][idx], data['vz_flat'][idx], alpha=0.3, s=1, c='blue',
                   label=r'$\delta$: r = ' + f'{data["r_vel_delta"]:.3f}')
        ax.scatter(data['vz_recx_flat'][idx], data['vz_flat'][idx], alpha=0.3, s=1, c='orange',
                   label=r'$-\delta T_b$: r = ' + f'{data["r_vel_tb"]:.3f}')
        ax.scatter(data['vz_recx_norm_flat'][idx], data['vz_flat'][idx], alpha=0.3, s=1, c='green',
                   label=r'$-\delta T_b / \langle x_{HI} \rangle$: r = ' + f'{data["r_vel_tb_norm"]:.3f}')
        
        all_vals = np.concatenate([data['vz_flat'], data['vz_rec_flat'], data['vz_recx_flat'], data['vz_recx_norm_flat']])
        lim = np.nanpercentile(all_vals, [1, 99])
        ax.plot([lim[0], lim[1]], [lim[0], lim[1]], 'k--', linewidth=2, alpha=0.7)
        
        ax.set_xlabel(r'Reconstructed $v_z$ [km/s]', fontsize=16)
        ax.set_ylabel(r'Simulated $v_z$ [km/s]', fontsize=16)
        ax.set_title(f'z = {z:.2f}, ' + r'$\langle x_{HI} \rangle$' + f' = {mean_xHI:.2f}', fontsize=16)
        ax.legend(fontsize=10, loc='upper left', markerscale=5)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Bottom row: kSZ comparison (δ, -δTb, and -δTb/<xHI> overlaid)
        ax = axes[1, col]
        ax.scatter(data['ksz_rec_delta'][idx], data['ksz_sim'][idx], alpha=0.3, s=1, c='blue',
                   label=r'$\delta$: r = ' + f'{data["r_ksz_delta"]:.3f}')
        ax.scatter(data['ksz_rec_tb'][idx], data['ksz_sim'][idx], alpha=0.3, s=1, c='orange',
                   label=r'$-\delta T_b$: r = ' + f'{data["r_ksz_tb"]:.3f}')
        ax.scatter(data['ksz_rec_tb_norm'][idx], data['ksz_sim'][idx], alpha=0.3, s=1, c='green',
                   label=r'$-\delta T_b / \langle x_{HI} \rangle$: r = ' + f'{data["r_ksz_tb_norm"]:.3f}')
        
        all_vals = np.concatenate([data['ksz_sim'], data['ksz_rec_delta'], data['ksz_rec_tb'], data['ksz_rec_tb_norm']])
        lim = np.nanpercentile(all_vals, [1, 99])
        ax.plot([lim[0], lim[1]], [lim[0], lim[1]], 'k--', linewidth=2, alpha=0.7)
        
        ax.set_xlabel(r'Reconstructed kSZ ($n_e \times v_z$)', fontsize=16)
        ax.set_ylabel(r'Simulated kSZ ($n_e \times v_z$)', fontsize=16)
        ax.set_title(f'z = {z:.2f}, ' + r'$\langle x_{HI} \rangle$' + f' = {mean_xHI:.2f}', fontsize=16)
        ax.legend(fontsize=10, loc='upper left', markerscale=5)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    outfile = os.path.join(output_dir, f'velocity_ksz_3methods_z{z1:.2f}_z{z2:.2f}.png')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outfile}")
    
    # Clean up
    del data1, data2
    gc.collect()
    
    return {'z1': z1, 'z2': z2}


def plot_velocity_term_comparison(z, output_dir=OUTPUT_DIR):
    """Compare velocity reconstruction with and without velocity gradient term.
    
    Creates a 2-panel figure showing scatter plots for both cases.
    """
    print("\n" + "="*80)
    print(f"Plotting: Velocity Term Comparison (z={z:.3f})")
    print("="*80)
    
    def compute_metrics(y_true, y_pred):
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        if not np.any(mask):
            return np.nan, np.nan
        y_t, y_p = y_true[mask], y_pred[mask]
        r = np.corrcoef(y_t, y_p)[0, 1] if len(y_t) > 1 else np.nan
        rmse = np.sqrt(np.mean((y_p - y_t)**2))
        return float(r), float(rmse)
    
    # Load data with velocity term (default)
    print("  Loading with velocity term...")
    (den_v, xhi_v, vx_v, vy_v, vz_v, vx_rec_v, vy_rec_v, vz_rec_v, 
     vx_recx_v, vy_recx_v, vz_recx_v, _, _, _, _, _) = reconstruct_velocities(z, include_velocity_term=True)
    
    # Load data without velocity term
    print("  Loading without velocity term...")
    (den_nv, xhi_nv, vx_nv, vy_nv, vz_nv, vx_rec_nv, vy_rec_nv, vz_rec_nv, 
     vx_recx_nv, vy_recx_nv, vz_recx_nv, _, _, _, _, _) = reconstruct_velocities(z, include_velocity_term=False)
    
    # Crop to central region
    xhi = xhi_v[CENTRAL_CROP, CENTRAL_CROP, LOS_CROP]
    vz = vz_v[CENTRAL_CROP, CENTRAL_CROP, LOS_CROP]
    vz_recx_v = vz_recx_v[CENTRAL_CROP, CENTRAL_CROP, LOS_CROP]
    vz_recx_nv = vz_recx_nv[CENTRAL_CROP, CENTRAL_CROP, LOS_CROP]
    
    mean_xHI = xhi.mean()
    
    # Exclude boundaries
    boundary = 10
    vz_crop = vz[:, :, boundary:-boundary]
    vz_recx_v_crop = vz_recx_v[:, :, boundary:-boundary]
    vz_recx_nv_crop = vz_recx_nv[:, :, boundary:-boundary]
    
    # Flatten for scatter
    vz_flat = vz_crop.flatten()
    vz_recx_v_flat = (-vz_recx_v_crop).flatten()
    vz_recx_nv_flat = (-vz_recx_nv_crop).flatten()
    
    # Compute metrics
    r_with_v, rmse_with_v = compute_metrics(vz_flat, vz_recx_v_flat)
    r_no_v, rmse_no_v = compute_metrics(vz_flat, vz_recx_nv_flat)
    
    print(f"  With velocity term: r={r_with_v:.4f}, RMSE={rmse_with_v:.2f}")
    print(f"  Without velocity term: r={r_no_v:.4f}, RMSE={rmse_no_v:.2f}")
    
    # Create 1x2 figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    sample_size = 50000
    np.random.seed(42)
    n_points = len(vz_flat)
    idx = np.random.choice(n_points, min(sample_size, n_points), replace=False)
    
    # Left: Without velocity term
    ax = axes[0]
    ax.scatter(vz_recx_nv_flat[idx], vz_flat[idx], alpha=0.3, s=1, c='blue')
    lim = np.nanpercentile(np.concatenate([vz_flat, vz_recx_nv_flat]), [1, 99])
    ax.plot([lim[0], lim[1]], [lim[0], lim[1]], 'r--', linewidth=2, alpha=0.7, label='y = x')
    ax.set_xlabel('Reconstructed vz', fontsize=18)
    ax.set_ylabel('Original vz', fontsize=18)
    ax.set_title(f'WITHOUT velocity term\nr={r_no_v:.4f}, RMSE={rmse_no_v:.1f}', fontsize=16)
    ax.legend(fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Right: With velocity term
    ax = axes[1]
    ax.scatter(vz_recx_v_flat[idx], vz_flat[idx], alpha=0.3, s=1, c='orange')
    lim = np.nanpercentile(np.concatenate([vz_flat, vz_recx_v_flat]), [1, 99])
    ax.plot([lim[0], lim[1]], [lim[0], lim[1]], 'r--', linewidth=2, alpha=0.7, label='y = x')
    ax.set_xlabel('Reconstructed vz', fontsize=18)
    ax.set_ylabel('Original vz', fontsize=18)
    ax.set_title(f'WITH velocity term H/(dv/dr+H)\nr={r_with_v:.4f}, RMSE={rmse_with_v:.1f}', fontsize=16)
    ax.legend(fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Velocity Reconstruction Comparison (z={z:.3f}, mean xHI={mean_xHI:.3f})', 
                 fontsize=18, fontweight='bold')
    plt.tight_layout()
    outfile = os.path.join(output_dir, f'velocity_term_comparison_z{z:.3f}.png')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outfile}")
    
    # Clean up
    del den_v, xhi_v, vz_v, vz_recx_v, den_nv, xhi_nv, vz_nv, vz_recx_nv
    gc.collect()
    
    return {'z': z, 'mean_xHI': mean_xHI, 
            'r_with_v': r_with_v, 'r_no_v': r_no_v,
            'rmse_with_v': rmse_with_v, 'rmse_no_v': rmse_no_v}


def plot_velocity_scatter_comparison(z1, z2, output_dir=OUTPUT_DIR):
    """Create 6-panel figure (3 rows × 2 redshifts).

    Row 1: 3D velocity scatter (reconstructed vs true v_z, voxel-by-voxel)
    Row 2: 2D projected velocity scatter (LOS-summed v_z)
    Row 3: 2D kSZ scatter (true vs reconstructed kSZ pixel-by-pixel)
           — added 2026-05-25 per Saleem's request

    NORMALIZATION:
    -------------
    Velocity panels (rows 1-2): only shift the mean (intercept) to align the
    data, NOT the slope. Preserves true amplitude. δTb-based reconstruction
    can additionally be pre-normalized (std-matched to true vz) before projection.

    kSZ panel (row 3): kSZ maps already encode the v×x_e×(1+δ) physics, so
    no separate normalization — just raw scatter.
    """
    print("\n" + "="*80)
    print(f"Plotting: Velocity Scatter Comparison (z={z1:.3f}, z={z2:.3f})")
    print("="*80)
    
    def compute_metrics(y_true, y_pred):
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        if not np.any(mask):
            return np.nan, np.nan
        y_t, y_p = y_true[mask], y_pred[mask]
        r = np.corrcoef(y_t, y_p)[0, 1] if len(y_t) > 1 else np.nan
        rmse = np.sqrt(np.mean((y_p - y_t)**2))
        return float(r), float(rmse)
    
    def load_and_process(z):
        (den, xhi, vx, vy, vz, vx_rec, vy_rec, vz_rec,
         vx_recx, vy_recx, vz_recx, _, _, _, _, _) = reconstruct_velocities(z)

        # Crop to central region
        den_c = den[CENTRAL_CROP, CENTRAL_CROP, LOS_CROP]
        xhi = xhi[CENTRAL_CROP, CENTRAL_CROP, LOS_CROP]
        vz = vz[CENTRAL_CROP, CENTRAL_CROP, LOS_CROP]
        vz_recx = vz_recx[CENTRAL_CROP, CENTRAL_CROP, LOS_CROP]
        vz_rec = vz_rec[CENTRAL_CROP, CENTRAL_CROP, LOS_CROP]

        mean_xHI = xhi.mean()

        # ─── kSZ maps: true vs reconstructed (paper-relevant: δTb-based) ──
        # compute_ksz_maps integrates v × x_e × (1+δ) along LOS → 2D map.
        # Paper kSZ convention (see plot_true_vs_reconstructed_ksz_maps L2956
        # and analyze_stitched_full_ksz_vs_individual L2390): use vz_recx
        # directly, no sign flip. The minus sign in rows 1-2's velocity
        # scatter is a separate display convention for the "−δT_b" overlay.
        ksz_true_2d_map = compute_ksz_maps(vz, xhi, den_c, z=z,
                                            physical_norm=PHYSICAL_NORM)
        ksz_rec_2d_map  = compute_ksz_maps(vz_recx, xhi, den_c, z=z,
                                            physical_norm=PHYSICAL_NORM)
        
        # Exclude boundaries for 3D data (10 cells on each side along LOS)
        boundary = 10
        vz_crop = vz[:, :, boundary:-boundary]
        vz_rec_crop = vz_rec[:, :, boundary:-boundary]
        vz_recx_crop = -vz_recx[:, :, boundary:-boundary]  # Apply sign here
        
        # 3D data (flatten)
        vz_3d = vz_crop.flatten()
        vz_rec_3d = vz_rec_crop.flatten()
        vz_recx_3d = vz_recx_crop.flatten()
        
        # =====================================================================
        # METHOD 1: Standard - project raw fields, then shift mean
        # =====================================================================
        vz_2d_map = np.sum(vz, axis=2)
        vz_rec_2d_map = np.sum(vz_rec, axis=2)
        vz_recx_2d_map = np.sum(-vz_recx, axis=2)
        
        # Exclude boundaries from 2D maps
        vz_2d = vz_2d_map[boundary:-boundary, boundary:-boundary].flatten()
        vz_rec_2d = vz_rec_2d_map[boundary:-boundary, boundary:-boundary].flatten()
        vz_recx_2d = vz_recx_2d_map[boundary:-boundary, boundary:-boundary].flatten()
        
        # =====================================================================
        # METHOD 2: Normalize δTb in 3D BEFORE projection
        # Scale 3D δTb field to have same std as true velocity, then project
        # =====================================================================
        # Normalize 3D δTb to match true velocity scale (voxel-by-voxel)
        std_vz = np.std(vz_crop)
        std_recx = np.std(vz_recx_crop)
        if std_recx > 0:
            scale_factor = std_vz / std_recx
            vz_recx_prenorm = vz_recx_crop * scale_factor
        else:
            vz_recx_prenorm = vz_recx_crop
        
        # Now project the pre-normalized field
        vz_recx_prenorm_full = -vz_recx * scale_factor  # Apply to full field
        vz_recx_prenorm_2d_map = np.sum(vz_recx_prenorm_full, axis=2)
        vz_recx_prenorm_2d = vz_recx_prenorm_2d_map[boundary:-boundary, boundary:-boundary].flatten()
        
        # kSZ scatter inputs — exclude boundary cells for consistency
        ksz_true_2d = ksz_true_2d_map[boundary:-boundary, boundary:-boundary].flatten()
        ksz_rec_2d  = ksz_rec_2d_map[boundary:-boundary, boundary:-boundary].flatten()

        # Clean up
        del den, den_c, xhi, vz, vz_recx, vz_rec, ksz_true_2d_map, ksz_rec_2d_map
        gc.collect()

        return {
            'mean_xHI': mean_xHI, 'z': z,
            'vz_3d': vz_3d, 'vz_rec_3d': vz_rec_3d, 'vz_recx_3d': vz_recx_3d,
            'vz_2d': vz_2d, 'vz_rec_2d': vz_rec_2d, 'vz_recx_2d': vz_recx_2d,
            'vz_recx_prenorm_2d': vz_recx_prenorm_2d,
            'scale_factor': scale_factor,
            'ksz_true_2d': ksz_true_2d, 'ksz_rec_2d': ksz_rec_2d,
        }
    
    # Load data for both redshifts
    data1 = load_and_process(z1)
    data2 = load_and_process(z2)
    
    # 3 rows × 2 redshifts: row 0 = 3D vel, row 1 = 2D proj vel, row 2 = kSZ (NEW)
    fig, axes = plt.subplots(3, 2, figsize=(14, 18))
    
    # More points for 3D to show density better
    sample_3d = 150000
    sample_2d = 50000
    np.random.seed(42)
    
    for col, data in enumerate([data1, data2]):
        mean_xHI = data['mean_xHI']
        z = data['z']
        
        # =====================================================================
        # Top row: 3D velocity scatter
        # =====================================================================
        ax = axes[0, col]
        n_3d = len(data['vz_3d'])
        idx_3d = np.random.choice(n_3d, min(sample_3d, n_3d), replace=False)
        
        # Compute metrics
        r_unfilt_3d, _ = compute_metrics(data['vz_3d'], data['vz_rec_3d'])
        r_filt_3d, _ = compute_metrics(data['vz_3d'], data['vz_recx_3d'])
        
        # Shift mean only (intercept), NOT slope
        # This centers both distributions without changing amplitude
        shift_delta_3d = np.mean(data['vz_3d']) - np.mean(data['vz_rec_3d'])
        shift_Tb_3d = np.mean(data['vz_3d']) - np.mean(data['vz_recx_3d'])
        
        vz_rec_3d_shifted = data['vz_rec_3d'] + shift_delta_3d
        vz_recx_3d_shifted = data['vz_recx_3d'] + shift_Tb_3d
        
        # Plot with paper-quality formatting
        # Plot yellow/orange FIRST so blue overlaps on top
        ax.scatter(vz_recx_3d_shifted[idx_3d], data['vz_3d'][idx_3d], 
                   alpha=0.15, s=0.5, c='#FFB000', rasterized=True,
                   label=r'$-\delta T_b$')
        ax.scatter(vz_rec_3d_shifted[idx_3d], data['vz_3d'][idx_3d], 
                   alpha=0.15, s=0.5, c='#0066FF', rasterized=True,
                   label=r'$\delta$')
        
        # Set limits based on true velocity
        lim = np.nanpercentile(data['vz_3d'], [0.5, 99.5])
        ax.plot([lim[0], lim[1]], [lim[0], lim[1]], 'k--', linewidth=2, alpha=0.8)
        
        ax.set_xlabel(r'Reconstructed $v_z$ [km/s]', fontsize=20)
        ax.set_ylabel(r'True $v_z$ [km/s]', fontsize=20)
        ax.set_title(f'3D: z={z:.2f}, $\\langle x_{{HI}} \\rangle$={mean_xHI:.2f}', fontsize=20)
        leg = ax.legend(fontsize=18, loc='upper left', markerscale=20, framealpha=0.95)
        for lh in leg.legend_handles:
            lh.set_alpha(1.0)
        ax.tick_params(axis='both', which='major', labelsize=18)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # =====================================================================
        # Bottom row: 2D projected velocity scatter
        # Compare: standard projection vs pre-normalized projection
        # =====================================================================
        ax = axes[1, col]
        n_2d = len(data['vz_2d'])
        idx_2d = np.random.choice(n_2d, min(sample_2d, n_2d), replace=False)
        
        # Compute metrics for both methods
        r_unfilt_2d, _ = compute_metrics(data['vz_2d'], data['vz_rec_2d'])
        r_filt_2d, _ = compute_metrics(data['vz_2d'], data['vz_recx_2d'])
        r_prenorm_2d, _ = compute_metrics(data['vz_2d'], data['vz_recx_prenorm_2d'])
        
        # Shift mean only (intercept), NOT slope
        shift_delta_2d = np.mean(data['vz_2d']) - np.mean(data['vz_rec_2d'])
        shift_Tb_2d = np.mean(data['vz_2d']) - np.mean(data['vz_recx_2d'])
        shift_prenorm_2d = np.mean(data['vz_2d']) - np.mean(data['vz_recx_prenorm_2d'])
        
        vz_rec_2d_shifted = data['vz_rec_2d'] + shift_delta_2d
        vz_recx_2d_shifted = data['vz_recx_2d'] + shift_Tb_2d
        vz_recx_prenorm_2d_shifted = data['vz_recx_prenorm_2d'] + shift_prenorm_2d
        
        # Plot with paper-quality formatting
        # Plot yellow/orange FIRST so blue overlaps on top
        ax.scatter(vz_recx_prenorm_2d_shifted[idx_2d], data['vz_2d'][idx_2d], 
                   alpha=0.3, s=1, c='#FFB000', rasterized=True,
                   label=r'$-\delta T_b$')
        ax.scatter(vz_rec_2d_shifted[idx_2d], data['vz_2d'][idx_2d], 
                   alpha=0.3, s=1, c='#0066FF', rasterized=True,
                   label=r'$\delta$')
        
        # Set limits based on true velocity
        lim_2d = np.nanpercentile(data['vz_2d'], [0.5, 99.5])
        ax.plot([lim_2d[0], lim_2d[1]], [lim_2d[0], lim_2d[1]], 'k--', linewidth=2, alpha=0.8)
        
        ax.set_xlabel(r'Reconstructed $v_z$ [km/s]', fontsize=20)
        ax.set_ylabel(r'True $v_z$ [km/s]', fontsize=20)
        ax.set_title(f'2D Projected: z={z:.2f}, $\\langle x_{{HI}} \\rangle$={mean_xHI:.2f}', fontsize=20)
        leg = ax.legend(fontsize=18, loc='upper left', markerscale=20, framealpha=0.95)
        for lh in leg.legend_handles:
            lh.set_alpha(1.0)
        ax.tick_params(axis='both', which='major', labelsize=18)
        ax.set_xlim(lim_2d)
        ax.set_ylim(lim_2d)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # =====================================================================
        # Row 3 (NEW, Saleem 2026-05-25): true vs reconstructed kSZ scatter
        # =====================================================================
        ax = axes[2, col]
        ksz_true = data['ksz_true_2d']
        ksz_rec  = data['ksz_rec_2d']
        n_ksz = len(ksz_true)
        idx_ksz = np.random.choice(n_ksz, min(sample_2d, n_ksz), replace=False)

        r_ksz, rmse_ksz = compute_metrics(ksz_true, ksz_rec)

        ax.scatter(ksz_rec[idx_ksz], ksz_true[idx_ksz],
                   alpha=0.3, s=1, c='#1F8A4A', rasterized=True,
                   label=r'kSZ')

        # symmetric limits on true kSZ
        lim_ksz = np.nanpercentile(ksz_true, [0.5, 99.5])
        ax.plot([lim_ksz[0], lim_ksz[1]], [lim_ksz[0], lim_ksz[1]],
                'k--', linewidth=2, alpha=0.8)

        ksz_unit = "[µK]" if PHYSICAL_NORM else "[a.u.]"
        ax.set_xlabel(f'Reconstructed kSZ {ksz_unit}', fontsize=20)
        ax.set_ylabel(f'True kSZ {ksz_unit}', fontsize=20)
        ax.set_title(f'kSZ: z={z:.2f}, $\\langle x_{{HI}} \\rangle$={mean_xHI:.2f}  '
                     f'(r={r_ksz:+.3f})', fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=18)
        ax.set_xlim(lim_ksz)
        ax.set_ylim(lim_ksz)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        # Print metrics for reference
        print(f"  z={z:.2f}: 3D r(δ)={r_unfilt_3d:.3f}, r(δTb)={r_filt_3d:.3f}")
        print(f"          2D r(δ)={r_unfilt_2d:.3f}, r(δTb)={r_filt_2d:.3f}, r(δTb pre-norm)={r_prenorm_2d:.3f}")
        print(f"          kSZ r={r_ksz:+.4f}, RMSE={rmse_ksz:.3e}")
        print(f"          3D scale factor: {data['scale_factor']:.4f}")
    
    plt.tight_layout()
    outfile = os.path.join(output_dir, 'velocity_scatter_3d_vs_2d_comparison.png')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outfile}")
    
    # Clean up
    del data1, data2
    gc.collect()


def plot_fourier_ksz_correlation(all_results, k_min=K_MIN, k_max=K_MAX_PLOT, output_dir=OUTPUT_DIR):
    """Plot Fourier kSZ correlation (unsmoothed only).
    
    Selects 6 redshifts based on target neutral fractions: 0.95, 0.75, 0.55, 0.45, 0.25, 0.05
    """
    print("\n" + "="*80)
    print("Plotting: Fourier kSZ Correlation")
    print("="*80)
    
    # Select 6 redshifts based on target neutral fractions
    target_xHI = [0.95, 0.75, 0.55, 0.45, 0.25, 0.05]
    selected_results = []
    for target in target_xHI:
        # Find result with closest neutral fraction
        best_result = min(all_results, key=lambda r: abs(r['mean_xHI'] - target))
        if best_result not in selected_results:
            selected_results.append(best_result)
    
    print(f"  Selected {len(selected_results)} redshifts based on neutral fraction:")
    for r in selected_results:
        print(f"    z={r['z']:.3f}, <xHI>={r['mean_xHI']:.3f}")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    n_selected = len(selected_results)
    cmap = plt.cm.viridis
    colors = [cmap(i / max(n_selected - 1, 1)) for i in range(n_selected)]
    
    # Plot 1: k range up to k_max
    ax1 = axes[0]
    for idx, result in enumerate(selected_results):
        mask_u = (result['k_values'] >= k_min) & (result['k_values'] <= k_max)
        ax1.plot(result['k_values'][mask_u], result['r_k'][mask_u],
                 color=colors[idx], linestyle='-', linewidth=2.5, alpha=0.9,
                 label=f'z={result["z"]:.2f}, xHI={result["mean_xHI"]:.2f}')
    
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax1.axhline(y=1, color='gray', linestyle='--', linewidth=1, alpha=0.3)
    ax1.set_xlabel('k [h/Mpc]', fontsize=22)
    ax1.set_ylabel('Correlation coefficient r(k)', fontsize=22)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([-1.01, 1.01])
    ax1.set_xlim([k_min, k_max])
    ax1.tick_params(axis='both', which='major', labelsize=20)
    ax1.legend(fontsize=14, loc='lower right', frameon=True)
    
    # Plot 2: ℓ = 2000-4000 range
    ax2 = axes[1]
    for idx, result in enumerate(selected_results):
        mask_u = (result['k_values'] >= k_min) & \
                 (result['ell_values'] >= 2000) & (result['ell_values'] <= 4000)
        if np.any(mask_u):
            ax2.plot(result['ell_values'][mask_u], result['r_k'][mask_u],
                     color=colors[idx], linestyle='-', linewidth=2.5, alpha=0.9,
                     label=f'z={result["z"]:.2f}, xHI={result["mean_xHI"]:.2f}')
    
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax2.axhline(y=1, color='gray', linestyle='--', linewidth=1, alpha=0.3)
    ax2.set_xlabel('Multipole ℓ', fontsize=22)
    ax2.set_ylabel('Correlation coefficient r(ℓ)', fontsize=22)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([-1.01, 1.01])
    ax2.set_xlim([2000, 4000])
    ax2.tick_params(axis='both', which='major', labelsize=20)
    ax2.legend(fontsize=14, loc='lower right', frameon=True)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}fourier_ksz_correlation.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}fourier_ksz_correlation.png")

def plot_ksz_correlation_vs_neutral_fraction(all_results, smooth_sigma=SMOOTH_SIGMA,
                                             output_dir=OUTPUT_DIR):
    """Plot kSZ correlation vs neutral fraction with redshift colorbar."""
    print("\n" + "="*80)
    print("Plotting: kSZ Correlation vs Neutral Fraction")
    print("="*80)
    
    neutral_fractions = np.array([r['mean_xHI'] for r in all_results])
    r_unsmooth = np.array([r['r_ksz_unsmooth'] for r in all_results])
    r_smooth = np.array([r['r_ksz_smooth'] for r in all_results])
    redshifts = np.array([r['z'] for r in all_results])
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    ax.tick_params(axis='both', which='major', labelsize=20, length=8, width=2, pad=10)
    
    # Plot dashed gray line connecting points
    ax.plot(neutral_fractions, r_unsmooth, '--', linewidth=2, color='gray', alpha=0.6, zorder=1)
    
    # Scatter with redshift color coding - simple markers
    sc = ax.scatter(neutral_fractions, r_unsmooth, c=redshifts, cmap='viridis', 
                    s=60, zorder=5)
    
    # Colorbar for redshift
    cbar = plt.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label('Redshift z', fontsize=20)
    cbar.ax.tick_params(labelsize=18)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax.axhline(y=1, color='gray', linestyle='--', linewidth=1, alpha=0.3)
    ax.set_xlabel('Mean Neutral Fraction $\\langle x_{HI} \\rangle$', fontsize=22)
    ax.set_ylabel('Real-Space Correlation r(kSZ, kSZ$_{rec}$)', fontsize=22)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([min(r_unsmooth) - 0.1, max(r_unsmooth) + 0.15])
    ax.set_xlim([min(neutral_fractions) - 0.05, max(neutral_fractions) + 0.05])
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}ksz_correlation_vs_neutral_fraction.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}ksz_correlation_vs_neutral_fraction.png")

def plot_R_kz_vs_k(R_results, output_dir=OUTPUT_DIR):
    """
    Plot R(k,z) and 1+R(k,z) vs k at several redshifts.

    Two-panel figure:
        Left:  R(k,z) vs k
        Right: 1+R(k,z) vs k   (the effective 21cm bias factor)
    """
    print("\n" + "="*80)
    print("Plotting: R(k,z) vs k")
    print("="*80)

    # Sort by redshift (high z first for legend ordering)
    R_sorted = sorted(R_results, key=lambda r: r['z'], reverse=True)

    redshifts = np.array([r['z'] for r in R_sorted])
    cmap = plt.cm.coolwarm_r
    norm = plt.Normalize(vmin=redshifts.min(), vmax=redshifts.max())

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Left panel: R(k,z)
    ax = axes[0]
    for res in R_sorted:
        k, R_k = res['k'], res['R_k']
        mask = np.isfinite(R_k) & (k > 0) & (k <= K_MAX_PLOT)
        color = cmap(norm(res['z']))
        ax.semilogx(k[mask], R_k[mask], '-', color=color, lw=1.5,
                     label=f"z={res['z']:.2f} ($\\bar{{x}}_{{HI}}$={res['mean_xHI']:.2f})")

    ax.axhline(0, color='grey', ls='--', lw=0.8, alpha=0.5)
    ax.axhline(-1, color='grey', ls=':', lw=0.8, alpha=0.5)
    ax.set_xlabel('k [h/Mpc]', fontsize=22)
    ax.set_ylabel('$R(k,z)$', fontsize=22)
    ax.set_title('Ionisation-to-density ratio $R(k,z)$', fontsize=16)
    ax.legend(fontsize=9, ncol=2, loc='lower left')
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.grid(True, alpha=0.3)

    # Right panel: 1 + R(k,z)
    ax2 = axes[1]
    for res in R_sorted:
        k, R_k = res['k'], res['R_k']
        mask = np.isfinite(R_k) & (k > 0) & (k <= K_MAX_PLOT)
        color = cmap(norm(res['z']))
        ax2.semilogx(k[mask], 1.0 + R_k[mask], '-', color=color, lw=1.5,
                      label=f"z={res['z']:.2f}")

    ax2.axhline(1, color='grey', ls='--', lw=0.8, alpha=0.5)
    ax2.axhline(0, color='grey', ls=':', lw=0.8, alpha=0.5)
    ax2.set_xlabel('k [h/Mpc]', fontsize=22)
    ax2.set_ylabel('$1 + R(k,z)$', fontsize=22)
    ax2.set_title('Effective 21cm bias $[1+R(k,z)]$', fontsize=16)
    ax2.legend(fontsize=9, ncol=2, loc='lower left')
    ax2.tick_params(axis='both', which='major', labelsize=20)
    ax2.grid(True, alpha=0.3)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.85, pad=0.02)
    cbar.set_label('Redshift $z$', fontsize=20)
    cbar.ax.tick_params(labelsize=18)

    plt.tight_layout()
    outfile = os.path.join(output_dir, 'R_kz_vs_k.png')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    outfile_pdf = os.path.join(output_dir, 'R_kz_vs_k.pdf')
    plt.savefig(outfile_pdf, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outfile}")
    print(f"  Saved: {outfile_pdf}")


def plot_one_plus_R_vs_neutral_fraction(R_results, output_dir=OUTPUT_DIR):
    """
    Plot 1+R(k,z) vs mean neutral fraction with redshift colorbar.

    Matches the style of plot_velocity_correlation_vs_neutral_fraction.
    Shows three scale bins (large, mid, small) with different markers.
    """
    print("\n" + "="*80)
    print("Plotting: 1+R vs Neutral Fraction")
    print("="*80)

    neutral_fractions = []
    redshifts = []
    one_plus_R_large = []   # k < 0.1 h/Mpc
    one_plus_R_mid = []     # 0.1 < k < 0.3 h/Mpc
    one_plus_R_small = []   # 0.3 < k < 1.0 h/Mpc

    for res in R_results:
        k = res['k']
        R_k = res['R_k']
        neutral_fractions.append(res['mean_xHI'])
        redshifts.append(res['z'])

        mask_large = (k > 0) & (k < 0.1) & np.isfinite(R_k)
        mask_mid   = (k >= 0.1) & (k < 0.3) & np.isfinite(R_k)
        mask_small = (k >= 0.3) & (k < 1.0) & np.isfinite(R_k)

        one_plus_R_large.append(1 + np.nanmean(R_k[mask_large]) if mask_large.any() else np.nan)
        one_plus_R_mid.append(1 + np.nanmean(R_k[mask_mid]) if mask_mid.any() else np.nan)
        one_plus_R_small.append(1 + np.nanmean(R_k[mask_small]) if mask_small.any() else np.nan)

    neutral_fractions = np.array(neutral_fractions)
    redshifts = np.array(redshifts)
    one_plus_R_large = np.array(one_plus_R_large)
    one_plus_R_mid = np.array(one_plus_R_mid)
    one_plus_R_small = np.array(one_plus_R_small)

    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    ax.tick_params(axis='both', which='major', labelsize=20, length=8, width=2, pad=10)

    # Dashed gray lines connecting points
    ax.plot(neutral_fractions, one_plus_R_large, '--', linewidth=2, color='gray', alpha=0.6, zorder=1)
    ax.plot(neutral_fractions, one_plus_R_mid, '--', linewidth=2, color='gray', alpha=0.6, zorder=1)
    ax.plot(neutral_fractions, one_plus_R_small, '--', linewidth=2, color='gray', alpha=0.6, zorder=1)

    # Scatter with redshift color coding
    sc1 = ax.scatter(neutral_fractions, one_plus_R_large, c=redshifts, cmap='viridis',
                     s=100, zorder=5, marker='o', edgecolors='k', linewidths=0.5)
    sc2 = ax.scatter(neutral_fractions, one_plus_R_mid, c=redshifts, cmap='viridis',
                     s=80, zorder=5, marker='s', edgecolors='k', linewidths=0.5)
    sc3 = ax.scatter(neutral_fractions, one_plus_R_small, c=redshifts, cmap='viridis',
                     s=60, zorder=5, marker='^', edgecolors='k', linewidths=0.5)

    # Colorbar for redshift
    cbar = plt.colorbar(sc1, ax=ax, pad=0.02)
    cbar.set_label('Redshift z', fontsize=20)
    cbar.ax.tick_params(labelsize=18)

    # Reference lines
    ax.axhline(1, color='k', ls='--', lw=1, alpha=0.4)
    ax.axhline(0, color='k', ls=':', lw=1.5, alpha=0.6)

    # Legend for scale markers
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
               markersize=10, label='$k < 0.1$ h/Mpc'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='gray',
               markersize=9, label='$0.1 < k < 0.3$ h/Mpc'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='gray',
               markersize=9, label='$0.3 < k < 1.0$ h/Mpc'),
    ]
    ax.legend(handles=legend_elements, fontsize=14, loc='lower right', framealpha=0.9)

    ax.set_xlabel('Mean Neutral Fraction $\\langle x_{HI} \\rangle$', fontsize=22)
    ax.set_ylabel('$1 + R(k,z)$', fontsize=22)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.02, 1.02])

    plt.tight_layout()
    outfile = os.path.join(output_dir, 'one_plus_R_vs_neutral_fraction.png')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    outfile_pdf = os.path.join(output_dir, 'one_plus_R_vs_neutral_fraction.pdf')
    plt.savefig(outfile_pdf, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outfile}")
    print(f"  Saved: {outfile_pdf}")


def analyze_stitched_full_ksz_vs_individual(redshifts_stitch=None, n_stitch=51,
                                             lightcone_slab=False):
    """Stitch real kSZ from multiple coeval boxes and compare to individual
    reconstructions.

    Uses central cropping in (x,y) to reduce boundary effects.

    Parameters
    ----------
    lightcone_slab : bool
        If True, use the thin-slab lightcone approximation.
        If False (default, as of 2026-05-22), each coeval box is treated as a
        full coeval cube — no lightcone slabbing.

        PROBLEM WITH NAIVE STACKING
        ---------------------------
        Naively summing the full kSZ map from each coeval box integrates
        through the entire box depth (L_box) at every redshift.  Stacking
        N_z such maps therefore integrates over N_z × L_box of LOS
        structure.  For 51 snapshots of a 714 Mpc box this is ~36 Gpc,
        whereas the true comoving distance from z=6 to z=12 is only
        ~2.5 Gpc — an over-count by a factor ~15, producing inflated
        amplitudes of ±300 µK.

        THIN-SLAB LIGHTCONE APPROXIMATION
        ----------------------------------
        In a real lightcone, each redshift z_i contributes a slab of
        comoving thickness

            Δχ_i = χ(z_{i+1/2}) − χ(z_{i−1/2})

        where z_{i±1/2} are the midpoints to adjacent snapshots.

        The kSZ contribution from that slab is:

            ΔT_i(x,y) = −(T_CMB σ_T / c) n̄_e(z_i) (Δx_proper)
                         × Σ_{j ∈ slab_i} x_e(x,y,j) [1+δ(x,y,j)] v_r(x,y,j)

        where the sum runs over only N_slab,i cells along axis 2:

            N_slab,i = round(Δχ_i / Δx_comoving)

        To choose WHICH cells, we tile through the periodic box along
        the LOS: the cumulative comoving distance from z_0 determines
        the offset into the box,

            j_start,i = floor[ (χ(z_i) − χ(z_0)) mod L_box / Δx_comoving ]

        This preserves spatial correlation between adjacent redshift
        slices (they sample adjacent cells) while wrapping around the
        periodic boundary when the LOS exceeds one box length — mimicking
        passage through independent large-scale structure.

        NOTE: This applies equally to Grizzly AND 21cmFAST coeval boxes.
        21cmFAST lightcones do not need this because the LOS is already
        constructed correctly.
    """
    # All 51 redshifts with complete data
    all_redshifts = [
        6.056, 6.113, 6.172, 6.231, 6.292, 6.354, 6.418, 6.483, 6.549,
        6.617, 6.686, 6.757, 6.830, 6.905, 6.981, 7.059, 7.139, 7.221,
        7.305, 7.391, 7.480, 7.570, 7.664, 7.760, 7.859, 7.960, 8.064,
        8.172, 8.283, 8.397, 8.515, 8.636, 8.762, 8.892, 9.026, 9.164,
        9.308, 9.457, 9.611, 9.771, 9.938, 10.110, 10.290, 10.478,
        10.673, 10.877, 11.090, 11.313, 11.546, 11.791, 12.048
    ]
    # all_redshifts = [
    #     6.056
    # ]

    
    if redshifts_stitch is None:
        # Select n_stitch redshifts evenly spaced (by index for now, will be ~evenly spaced in xHI)
        step = max(1, len(all_redshifts) // n_stitch)
        redshifts_stitch = all_redshifts[::step][:n_stitch]
    
    print("\n" + "="*80)
    print("INTEGRATED kSZ ANALYSIS: REAL vs RECONSTRUCTION")
    print("="*80)
    print(f"Using {len(redshifts_stitch)} redshifts")
    print(f"Lightcone thin-slab: {'ON' if lightcone_slab else 'OFF (full box per z)'}")
    print("Computing kSZ maps per redshift and summing (memory efficient)...")

    # ── Pre-compute thin-slab parameters ────────────────────────────────
    if lightcone_slab:
        from scipy.integrate import quad as _quad

        def _chi(z_val):
            """Comoving distance in Mpc (not Mpc/h)."""
            c_km_s = 299792.458
            H0_loc = 100.0 * LITTLEH
            Om = 0.27
            integrand = lambda zp: 1.0 / np.sqrt(Om * (1 + zp)**3 + (1 - Om))
            result, _ = _quad(integrand, 0, z_val)
            return c_km_s / H0_loc * result

        cell_comoving = BOX_MPC_OVER_H / LITTLEH / 600  # Mpc per cell (comoving)
        box_comoving = BOX_MPC_OVER_H / LITTLEH          # full box in Mpc

        # Comoving distances for all snapshots
        chi_all = np.array([_chi(z) for z in redshifts_stitch])
        chi_0 = chi_all[0]

        # Comoving slab thickness Δχ_i (Voronoi-style: midpoints to neighbours)
        dchi = np.zeros(len(redshifts_stitch))
        for i in range(len(redshifts_stitch)):
            if i == 0:
                z_lo = redshifts_stitch[0]
            else:
                z_lo = 0.5 * (redshifts_stitch[i - 1] + redshifts_stitch[i])
            if i == len(redshifts_stitch) - 1:
                z_hi = redshifts_stitch[-1]
            else:
                z_hi = 0.5 * (redshifts_stitch[i] + redshifts_stitch[i + 1])
            dchi[i] = _chi(z_hi) - _chi(z_lo)

        total_los = chi_all[-1] - chi_all[0]
        print(f"  Total comoving LOS: {total_los:.0f} Mpc  (box = {box_comoving:.0f} Mpc)")
        print(f"  LOS / box = {total_los / box_comoving:.1f} box lengths")
        print(f"  Mean slab thickness: {dchi.mean():.1f} Mpc  "
              f"({dchi.mean() / cell_comoving:.0f} cells)")

    # ── Main loop ───────────────────────────────────────────────────────
    ksz_map_full_real = None
    ksz_map_full_rec = None
    individual_results = []
    n_full = None

    for idx_z, z in enumerate(redshifts_stitch):
        print(f"\nLoading and processing box at z={z}...")
        (den, xhi, vx, vy, vz, vx_rec, vy_rec, vz_rec,
         vx_recx, vy_recx, vz_recx, _, _, _, _, _) = reconstruct_velocities(z)

        if n_full is None:
            n_full = vz.shape[0]

        # Central crop in (x,y) to avoid boundary effects
        den = den[CENTRAL_CROP, CENTRAL_CROP, :]
        xhi = xhi[CENTRAL_CROP, CENTRAL_CROP, :]
        vz = vz[CENTRAL_CROP, CENTRAL_CROP, :]
        vz_recx = vz_recx[CENTRAL_CROP, CENTRAL_CROP, :]

        mean_xhi = xhi.mean()

        if lightcone_slab:
            # Select thin slab along axis 2
            n_slab = max(1, round(dchi[idx_z] / cell_comoving))

            # Tile through periodic box: offset = cumulative distance mod box
            j_start = int(((chi_all[idx_z] - chi_0) % box_comoving) / cell_comoving)
            j_start = j_start % n_full  # wrap

            # Extract slab with periodic wrapping
            indices = np.arange(j_start, j_start + n_slab) % n_full
            den_slab = den[:, :, indices]
            xhi_slab = xhi[:, :, indices]
            vz_slab = vz[:, :, indices]
            vz_recx_slab = vz_recx[:, :, indices]

            print(f"  Slab: {n_slab} cells (Δχ={dchi[idx_z]:.1f} Mpc), "
                  f"offset j={j_start}, mean xHI: {mean_xhi:.4f}")

            ksz_real_z = compute_ksz_maps(vz_slab, xhi_slab, den_slab,
                                          z=z, physical_norm=PHYSICAL_NORM)
            ksz_rec_z = compute_ksz_maps(vz_recx_slab, xhi_slab, den_slab,
                                         z=z, physical_norm=PHYSICAL_NORM)
        else:
            # Full box integration (original behaviour)
            den = den[:, :, LOS_CROP]
            xhi = xhi[:, :, LOS_CROP]
            vz = vz[:, :, LOS_CROP]
            vz_recx = vz_recx[:, :, LOS_CROP]
            print(f"  Shape: {vz.shape}, mean xHI: {mean_xhi:.4f}")

            ksz_real_z = compute_ksz_maps(vz, xhi, den,
                                          z=z, physical_norm=PHYSICAL_NORM)
            ksz_rec_z = compute_ksz_maps(vz_recx, xhi, den,
                                         z=z, physical_norm=PHYSICAL_NORM)

        # Store per-redshift results (full-box maps for Fourier correlation)
        individual_results.append({
            'z': z,
            'mean_xhi': mean_xhi,
            'ksz_real': ksz_real_z.copy(),
            'ksz_rec': ksz_rec_z.copy(),
        })

        # Accumulate for integrated maps
        if ksz_map_full_real is None:
            ksz_map_full_real = ksz_real_z.copy()
            ksz_map_full_rec = ksz_rec_z.copy()
        else:
            ksz_map_full_real += ksz_real_z
            ksz_map_full_rec += ksz_rec_z

        del vx, vy, vx_rec, vy_rec, vz_rec, vx_recx, vy_recx, vz_recx
        del den, xhi, vz, ksz_real_z, ksz_rec_z
        gc.collect()

    print(f"\nIntegrated kSZ maps computed (summed over {len(redshifts_stitch)} redshifts)")
    print(f"  Range: [{ksz_map_full_real.min():.2f}, {ksz_map_full_real.max():.2f}] µK")
    print(f"  Std:   {ksz_map_full_real.std():.2f} µK")

    # Box size for the cropped region
    n_crop = CENTRAL_CROP.stop - CENTRAL_CROP.start
    dx_full = BOX_MPC_OVER_H / float(n_full)
    dy_full = BOX_MPC_OVER_H / float(n_full)
    Lx_crop = dx_full * n_crop
    Ly_crop = dy_full * n_crop

    # =========================================================================
    # ANALYSIS 1: Per-redshift reconstruction vs integrated real kSZ
    # =========================================================================
    print("\n" + "-"*60)
    print("ANALYSIS 1: Per-redshift reconstruction vs integrated real kSZ")
    print("-"*60)

    cross_corr_results_individual = []

    for result in individual_results:
        z = result['z']
        mean_xhi = result['mean_xhi']
        ksz_rec_z = result['ksz_rec']
        ksz_real_z = result['ksz_real']

        # Paper definition: per-z reconstructed map compared against the
        # FULL integrated real kSZ map, in BOTH real and Fourier space.
        r_cross = pearson_r(ksz_map_full_real, ksz_rec_z)

        k_values, r_k = compute_fourier_correlation_coefficient(
            ksz_map_full_real, ksz_rec_z,
            boxlength=[Ly_crop, Lx_crop],
        )

        # Transverse-tile jackknife (kSZ is a 2D projection so blocks must
        # live in the (x, y) plane only).
        _, r_cross_jk_sigma, _ = jackknife_pearson_r(
            ksz_map_full_real, ksz_rec_z, n_per_side=2
        )
        _, _, r_k_jk_sigma = jackknife_r_k_2d(
            ksz_map_full_real, ksz_rec_z, boxlength=[Ly_crop, Lx_crop],
            n_per_side=2, k_grid=k_values
        )

        cross_corr_results_individual.append({
            'z': z,
            'mean_xhi': mean_xhi,
            'r_cross': r_cross,
            'r_cross_jk_sigma': r_cross_jk_sigma,
            'k_values': k_values,
            'r_k': r_k,
            'r_k_jk_sigma': r_k_jk_sigma,
            'mean_r_k': np.mean(r_k[np.isfinite(r_k)]) if np.any(np.isfinite(r_k)) else np.nan,
        })

        print(f"  z={z:.3f}, xHI={mean_xhi:.3f}: r={r_cross:+.4f} ± {r_cross_jk_sigma:.4f}")

    # =========================================================================
    # ANALYSIS 2: Integrated reconstruction vs integrated real kSZ
    # =========================================================================
    print("\n" + "-"*60)
    print("ANALYSIS 2: Integrated reconstruction vs integrated real kSZ")
    print("-"*60)

    r_integrated = pearson_r(ksz_map_full_real, ksz_map_full_rec)
    k_values_int, r_k_int = compute_fourier_correlation_coefficient(
        ksz_map_full_real, ksz_map_full_rec,
        boxlength=[Ly_crop, Lx_crop],
    )

    print(f"  Real-space correlation: r = {r_integrated:.4f}")
    print(f"  Mean Fourier correlation: r(k) = {np.nanmean(r_k_int):.4f}")

    integrated_result = {
        'r_cross': r_integrated,
        'k_values': k_values_int,
        'r_k': r_k_int,
        'mean_r_k': np.nanmean(r_k_int),
    }

    # Use individual results for plotting
    cross_corr_results = cross_corr_results_individual

    # Select 4 redshifts for plotting based on target neutral fractions
    # Exclude z=6.83 (xHI~0.47), use same selection as velocity plot
    target_xHI_plot = [0.95, 0.55, 0.25, 0.05]
    selected_for_plot = []
    selected_cross_results = []
    for target in target_xHI_plot:
        # Find result with closest neutral fraction
        best_idx = min(range(len(individual_results)), 
                       key=lambda i: abs(individual_results[i]['mean_xhi'] - target))
        if individual_results[best_idx] not in selected_for_plot:
            selected_for_plot.append(individual_results[best_idx])
            selected_cross_results.append(cross_corr_results[best_idx])
    
    print(f"\n  Selected {len(selected_for_plot)} redshifts for plotting:")
    for r in selected_for_plot:
        print(f"    z={r['z']:.3f}, <xHI>={r['mean_xhi']:.3f}")

    # =========================================================================
    # PLOT 1: Correlation vs neutral fraction (real-space and Fourier at fixed ℓ)
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # Extract arrays for ALL redshifts
    xhi_vals_all = np.array([cr['mean_xhi'] for cr in cross_corr_results])
    r_vals_all = np.array([cr['r_cross'] for cr in cross_corr_results])
    z_vals_all = np.array([cr['z'] for cr in cross_corr_results])

    # Left: Real-space correlation r(x) vs neutral fraction
    ax = axes[0]
    ax.plot(xhi_vals_all, r_vals_all, '--', color='gray', lw=2.5, alpha=0.7, zorder=1)
    sc = ax.scatter(xhi_vals_all, r_vals_all, c=z_vals_all, cmap='viridis', s=60, zorder=5)
    cbar = plt.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label('Redshift z', fontsize=20)
    cbar.ax.tick_params(labelsize=18)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax.set_xlabel('Neutral Fraction $\\langle x_{HI} \\rangle$', fontsize=22)
    ax.set_ylabel('r(x)', fontsize=22)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-1.01, 1.01])
    ax.set_xlim([-0.05, 1.05])
    ax.tick_params(axis='both', which='major', labelsize=20)

    # Right: Fourier r(ℓ) at fixed ℓ values vs neutral fraction
    ax = axes[1]
    ell_targets = [2500, 3000, 3500]
    markers = ['o', 's', '^']
    colors_ell = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for ell_t, marker, color in zip(ell_targets, markers, colors_ell):
        r_at_ell = []
        for cr in cross_corr_results:
            chi = comoving_distance(cr['z'])
            ell_values = cr['k_values'] * chi
            # Interpolate r(k) at the k corresponding to this ℓ
            valid = np.isfinite(cr['r_k']) & np.isfinite(ell_values)
            if np.any(valid):
                r_interp = np.interp(ell_t, ell_values[valid], cr['r_k'][valid])
            else:
                r_interp = np.nan
            r_at_ell.append(r_interp)
        r_at_ell = np.array(r_at_ell)
        ax.plot(xhi_vals_all, r_at_ell, '--', color='gray', lw=1.5, alpha=0.4, zorder=1)
        ax.scatter(xhi_vals_all, r_at_ell, c=color, s=50, marker=marker,
                   label=f'$\\ell = {ell_t}$', zorder=5)

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax.set_xlabel('Neutral Fraction $\\langle x_{HI} \\rangle$', fontsize=22)
    ax.set_ylabel('r($\\ell$)', fontsize=22)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-1.01, 1.01])
    ax.set_xlim([-0.05, 1.05])
    ax.legend(fontsize=18, loc='best')
    ax.tick_params(axis='both', which='major', labelsize=20)

    plt.tight_layout()
    outfile = os.path.join(OUTPUT_DIR, 'integrated_real_vs_per_z_reconstruction_v2.png')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outfile}")

    # =========================================================================
    # PLOT 2: Integrated reconstruction vs integrated real kSZ
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Map 1: Integrated real kSZ
    ax = axes[0]
    im = ax.imshow(ksz_map_full_real, origin='lower', cmap='RdBu_r')
    ax.set_xlabel('x', fontsize=22)
    ax.set_ylabel('y', fontsize=22)
    ax.tick_params(axis='both', which='major', labelsize=20)
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=18)

    # Map 2: Integrated reconstructed kSZ
    ax = axes[1]
    im = ax.imshow(ksz_map_full_rec, origin='lower', cmap='RdBu_r')
    ax.set_xlabel('x', fontsize=22)
    ax.set_ylabel('y', fontsize=22)
    ax.tick_params(axis='both', which='major', labelsize=20)
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=18)

    plt.tight_layout()
    outfile = os.path.join(OUTPUT_DIR, 'integrated_real_vs_integrated_reconstruction_v3.png')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outfile}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*80)
    print("SUMMARY: INTEGRATED kSZ ANALYSIS")
    print("="*80)
    
    print("\n--- Per-redshift reconstruction vs integrated real ---")
    print(f"{'z':<8} {'<xHI>':<10} {'r(real)':<12} {'mean r(k)':<12}")
    print("-" * 50)
    for cross_result in cross_corr_results:
        print(f"{cross_result['z']:<8.3f} {cross_result['mean_xhi']:<10.4f} "
              f"{cross_result['r_cross']:<12.4f} {cross_result['mean_r_k']:<12.4f}")
    
    print("\n--- Integrated reconstruction vs integrated real ---")
    print(f"Real-space correlation: r = {r_integrated:.4f}")
    print(f"Mean Fourier correlation: mean r(k) = {integrated_result['mean_r_k']:.4f}")

    del ksz_map_full_real, ksz_map_full_rec
    for result in individual_results:
        del result['ksz_rec']
    gc.collect()

    return cross_corr_results


def plot_cross_correlator_evolution_jk(cross_corr_results, output_dir=OUTPUT_DIR):
    """Fig 4 with jackknife error bars.

    Left panel: real-space cross-correlator r(x) vs ⟨x_HI⟩ between the
    per-redshift reconstructed kSZ and the integrated real kSZ, with σ_jk.
    Right panel: r(ℓ) at ℓ = 2500, 3000, 3500 vs ⟨x_HI⟩, with σ_jk.
    Both use a 2×2 transverse-tile jackknife (kSZ is a 2D projection).
    Saves to <output_dir>/cross_correlator_evolution_jk.{png,pdf}.
    """
    print("\n" + "=" * 80)
    print("Plotting: Cross-correlator evolution (jackknife)")
    print("=" * 80)

    xhi = np.array([cr['mean_xhi'] for cr in cross_corr_results])
    z_arr = np.array([cr['z'] for cr in cross_corr_results])
    r_x = np.array([cr['r_cross'] for cr in cross_corr_results])
    r_x_sigma = np.array([cr.get('r_cross_jk_sigma', np.nan)
                          for cr in cross_corr_results])

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # Left: real-space r(x)
    ax = axes[0]
    ax.plot(xhi, r_x, '--', color='gray', lw=2.5, alpha=0.7, zorder=1)
    ax.errorbar(xhi, r_x, yerr=r_x_sigma, fmt='none', ecolor='gray',
                alpha=0.7, zorder=3)
    sc = ax.scatter(xhi, r_x, c=z_arr, cmap='viridis', s=60, zorder=5)
    cbar = plt.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label('Redshift z', fontsize=20)
    cbar.ax.tick_params(labelsize=18)
    ax.axhline(y=0, color='black', lw=0.8, alpha=0.5)
    ax.set_xlabel('Neutral Fraction $\\langle x_{HI} \\rangle$', fontsize=22)
    ax.set_ylabel('r(x)', fontsize=22)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-1.05, 1.05])
    ax.set_xlim([-0.05, 1.05])
    ax.tick_params(axis='both', which='major', labelsize=20)

    # Right: r(ℓ) at fixed multipoles
    ax = axes[1]
    ell_targets = [2500, 3000, 3500]
    markers = ['o', 's', '^']
    colors_ell = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for ell_t, marker, color in zip(ell_targets, markers, colors_ell):
        r_at_ell = []
        s_at_ell = []
        for cr in cross_corr_results:
            chi = comoving_distance(cr['z'])
            ell_vals = cr['k_values'] * chi
            valid = np.isfinite(cr['r_k']) & np.isfinite(ell_vals)
            if np.any(valid):
                r_interp = np.interp(ell_t, ell_vals[valid], cr['r_k'][valid])
                jk_curve = cr.get('r_k_jk_sigma', None)
                if jk_curve is not None:
                    valid_s = np.isfinite(jk_curve) & np.isfinite(ell_vals)
                    s_interp = (np.interp(ell_t, ell_vals[valid_s],
                                          jk_curve[valid_s])
                                if np.any(valid_s) else np.nan)
                else:
                    s_interp = np.nan
            else:
                r_interp, s_interp = np.nan, np.nan
            r_at_ell.append(r_interp)
            s_at_ell.append(s_interp)
        r_at_ell = np.array(r_at_ell)
        s_at_ell = np.array(s_at_ell)
        ax.plot(xhi, r_at_ell, '--', color='gray', lw=1.5, alpha=0.4,
                zorder=1)
        ax.errorbar(xhi, r_at_ell, yerr=s_at_ell, fmt='none', ecolor=color,
                    alpha=0.4, zorder=3)
        ax.scatter(xhi, r_at_ell, c=color, s=50, marker=marker,
                   label=f'$\\ell = {ell_t}$', zorder=5)

    ax.axhline(y=0, color='black', lw=0.8, alpha=0.5)
    ax.set_xlabel('Neutral Fraction $\\langle x_{HI} \\rangle$', fontsize=22)
    ax.set_ylabel('r($\\ell$)', fontsize=22)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-1.05, 1.05])
    ax.set_xlim([-0.05, 1.05])
    ax.legend(fontsize=18, loc='best')
    ax.tick_params(axis='both', which='major', labelsize=20)

    plt.tight_layout()
    for ext in ('png', 'pdf'):
        out = os.path.join(output_dir, f'cross_correlator_evolution_jk.{ext}')
        plt.savefig(out, dpi=300, bbox_inches='tight')
        print(f"  Saved: {out}")
    plt.close()


def plot_xhi_and_dTb_midpoint(n=600, output_dir=OUTPUT_DIR):
    """Plot x_HI and δTb maps at the midpoint (~50%) of reionization.

    Scans all 51 available redshifts, picks the one closest to
    <x_HI> = 0.5, and produces a two-panel figure:
        Left:  x_HI  slice
        Right: δTb   slice  (∝ x_HI × (1+δ) × H/(dv_r/dr + H))
    """
    all_redshifts = [
        6.056, 6.113, 6.172, 6.231, 6.292, 6.354, 6.418, 6.483, 6.549,
        6.617, 6.686, 6.757, 6.830, 6.905, 6.981, 7.059, 7.139, 7.221,
        7.305, 7.391, 7.480, 7.570, 7.664, 7.760, 7.859, 7.960, 8.064,
        8.172, 8.283, 8.397, 8.515, 8.636, 8.762, 8.892, 9.026, 9.164,
        9.308, 9.457, 9.611, 9.771, 9.938, 10.110, 10.290, 10.478,
        10.673, 10.877, 11.090, 11.313, 11.546, 11.791, 12.048
    ]

    print("\n" + "=" * 80)
    print("Plotting: x_HI and δTb maps at midpoint of reionization")
    print("=" * 80)

    # --- find redshift closest to <x_HI> = 0.5 ---
    best_z = None
    best_xhi_mean = None
    best_diff = np.inf
    for z in all_redshifts:
        zstr = f"{z:.3f}"
        filenameXhi = (f"{DATA_DIR}{zstr}zeta0.389fesc0.389_Mmin0.120E+10"
                       f"_MminX0.120E+10_fx0.100E+03_sed3_al1.200xhi.bin")
        xhi = read_xhi(filenameXhi, n, n, n)
        mean_xhi = float(xhi.mean())
        diff = abs(mean_xhi - 0.5)
        if diff < best_diff:
            best_diff = diff
            best_z = z
            best_xhi_mean = mean_xhi
        del xhi
    print(f"  Midpoint redshift: z = {best_z:.3f}  (<x_HI> = {best_xhi_mean:.4f})")

    # --- load fields at that redshift ---
    zstr = f"{best_z:.3f}"
    filenameDen = f"{DATA_DIR}{zstr}n_all.dat"
    filenameXhi = (f"{DATA_DIR}{zstr}zeta0.389fesc0.389_Mmin0.120E+10"
                   f"_MminX0.120E+10_fx0.100E+03_sed3_al1.200xhi.bin")
    filenameVel = f"{DATA_DIR}{zstr}v_all.dat"

    den = read_den(filenameDen, n, n, n).astype(np.float32)
    xhi = read_xhi(filenameXhi, n, n, n).astype(np.float32)
    vx, vy, vz_field = read_vel(best_z, den, filenameVel, n_cell=n)
    vz_field *= np.float32(1.0 / 1e5)  # cm/s → km/s

    mean_den = den.mean(dtype=np.float64).astype(np.float32)
    delta = den / mean_den  # 1 + δ

    # Velocity gradient term: H / (dv_r/dr + H)
    dz_cell = BOX_MPC_OVER_H / n / LITTLEH  # Mpc per cell
    dvdz = np.gradient(vz_field, dz_cell, axis=2).astype(np.float32)
    H0 = 100.0 * LITTLEH
    Hz = H0 * np.sqrt(0.27 * (1 + best_z)**3 + 0.73)
    velocity_factor = (Hz / (dvdz + Hz)).astype(np.float32)

    # δTb ∝ x_HI × (1+δ) × H/(dv_r/dr + H)
    dTb = xhi * delta * velocity_factor

    # Take a central LOS slice for the 2-D maps
    mid = n // 2
    xhi_slice = xhi[CENTRAL_CROP, CENTRAL_CROP, mid]
    dTb_slice = dTb[CENTRAL_CROP, CENTRAL_CROP, mid]

    # --- plot ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    im = ax.imshow(xhi_slice, origin='lower', cmap='viridis')
    ax.set_title(f'$x_{{HI}}$  (z = {best_z:.2f}, '
                 f'$\\langle x_{{HI}} \\rangle$ = {best_xhi_mean:.2f})',
                 fontsize=16)
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=18)

    ax = axes[1]
    im = ax.imshow(dTb_slice, origin='lower', cmap='RdBu_r')
    ax.set_title(f'$\\delta T_b$  (z = {best_z:.2f}, '
                 f'$\\langle x_{{HI}} \\rangle$ = {best_xhi_mean:.2f})',
                 fontsize=16)
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=18)

    plt.tight_layout()
    outfile = os.path.join(output_dir, 'xhi_dTb_midpoint_reionization.png')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outfile}")

    del den, xhi, vx, vy, vz_field, delta, dvdz, velocity_factor, dTb
    gc.collect()


def plot_mean_brightness_temp_vs_redshift(n=600, output_dir=OUTPUT_DIR):
    """Plot mean 21cm brightness temperature vs redshift from GRIZZLY sims.

    Computes δTb at each cell using:
        δTb = 27 mK × x_HI × (1+δ) × [(1+z)/10]^0.5
              × (Ωb h² / 0.023) × [0.15 / (Ωm h²)]^0.5
              × H(z) / (dv_r/dr + H(z))
    then takes the volume average ⟨δTb⟩ at each snapshot.
    """
    all_redshifts = [
        6.056, 6.113, 6.172, 6.231, 6.292, 6.354, 6.418, 6.483, 6.549,
        6.617, 6.686, 6.757, 6.830, 6.905, 6.981, 7.059, 7.139, 7.221,
        7.305, 7.391, 7.480, 7.570, 7.664, 7.760, 7.859, 7.960, 8.064,
        8.172, 8.283, 8.397, 8.515, 8.636, 8.762, 8.892, 9.026, 9.164,
        9.308, 9.457, 9.611, 9.771, 9.938, 10.110, 10.290, 10.478,
        10.673, 10.877, 11.090, 11.313, 11.546, 11.791, 12.048
    ]

    print("\n" + "=" * 80)
    print("Plotting: Mean brightness temperature vs redshift")
    print("=" * 80)

    # Cosmological parameters (matching the rest of the pipeline)
    h = LITTLEH           # 0.7
    Omega_m = 0.27
    Omega_b = 0.044
    Omega_L = 1.0 - Omega_m
    H0 = 100.0 * h       # km/s/Mpc
    dz_cell_comoving = BOX_MPC_OVER_H / n / h  # comoving Mpc per cell

    # Pre-factor: 27 mK × (Ωb h²/0.023) × (0.15/(Ωm h²))^0.5
    T0 = 27.0  # mK
    cosmo_factor = T0 * (Omega_b * h**2 / 0.023) * np.sqrt(0.15 / (Omega_m * h**2))

    redshifts_out = []
    mean_Tb_out = []

    for z in all_redshifts:
        zstr = f"{z:.3f}"
        filenameDen = f"{DATA_DIR}{zstr}n_all.dat"
        filenameXhi = (f"{DATA_DIR}{zstr}zeta0.389fesc0.389_Mmin0.120E+10"
                       f"_MminX0.120E+10_fx0.100E+03_sed3_al1.200xhi.bin")
        filenameVel = f"{DATA_DIR}{zstr}v_all.dat"

        den = read_den(filenameDen, n, n, n).astype(np.float32)
        xhi = read_xhi(filenameXhi, n, n, n).astype(np.float32)
        vx, vy, vz_field = read_vel(z, den, filenameVel, n_cell=n)
        vz_field *= np.float32(1.0 / 1e5)  # cm/s → km/s

        mean_den = den.mean(dtype=np.float64).astype(np.float32)
        one_plus_delta = den / mean_den  # 1 + δ

        # Velocity gradient term
        dvdz = np.gradient(vz_field, dz_cell_comoving, axis=2).astype(np.float32)
        Hz = H0 * np.sqrt(Omega_m * (1 + z)**3 + Omega_L)
        vel_factor = (Hz / (dvdz + Hz)).astype(np.float32)

        # Full δTb field [mK]
        z_factor = np.sqrt((1.0 + z) / 10.0)
        dTb = cosmo_factor * z_factor * xhi * one_plus_delta * vel_factor
        mean_Tb = float(dTb.mean())

        redshifts_out.append(z)
        mean_Tb_out.append(mean_Tb)
        print(f"  z = {z:7.3f}  <x_HI> = {xhi.mean():.4f}  <δTb> = {mean_Tb:.3f} mK")

        del den, xhi, vx, vy, vz_field, one_plus_delta, dvdz, vel_factor, dTb
        gc.collect()

    redshifts_out = np.array(redshifts_out)
    mean_Tb_out = np.array(mean_Tb_out)

    # --- plot ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(redshifts_out, mean_Tb_out, 'o-', color='C0', markersize=4, lw=1.5)
    ax.set_xlabel('Redshift $z$', fontsize=22)
    ax.set_ylabel('$\\langle \\delta T_b \\rangle$ [mK]', fontsize=22)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()

    plt.tight_layout()
    outfile = os.path.join(output_dir, 'mean_brightness_temp_vs_redshift.png')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outfile}")

    return redshifts_out, mean_Tb_out


def plot_true_vs_reconstructed_ksz_maps(z_high=12.048, z_low=6.905,
                                         output_dir=OUTPUT_DIR):
    """2x2 grid comparing true and reconstructed kSZ maps at two redshifts.

    Top row: z=z_high (pre-reionization, ⟨x_HI⟩~1).
    Bottom row: z=z_low (midpoint, ⟨x_HI⟩~0.5).
    Columns: True kSZ | Reconstructed kSZ.
    Each row has its own shared colorbar; both rows share identical
    Mpc/h tick spacing on x and y.
    """
    from matplotlib.gridspec import GridSpec

    print("\n" + "=" * 80)
    print("Plotting: True vs Reconstructed kSZ maps (2x2)")
    print("=" * 80)

    redshifts = [z_high, z_low]
    panels = []

    for z in redshifts:
        (den, xhi, _, _, vz, _, _, _, _, _, vz_recx, *_) = reconstruct_velocities(z)
        den_c = den[CENTRAL_CROP, CENTRAL_CROP, LOS_CROP]
        xhi_c = xhi[CENTRAL_CROP, CENTRAL_CROP, LOS_CROP]
        vz_c = vz[CENTRAL_CROP, CENTRAL_CROP, LOS_CROP]
        vz_recx_c = vz_recx[CENTRAL_CROP, CENTRAL_CROP, LOS_CROP]
        mean_xhi = float(xhi_c.mean())

        ksz_real = compute_ksz_maps(vz_c, xhi_c, den_c,
                                     z=z, physical_norm=PHYSICAL_NORM)
        ksz_rec = compute_ksz_maps(vz_recx_c, xhi_c, den_c,
                                    z=z, physical_norm=PHYSICAL_NORM)
        r = pearson_r(ksz_real, ksz_rec)

        panels.append({
            'z': z, 'mean_xhi': mean_xhi,
            'ksz_real': ksz_real, 'ksz_rec': ksz_rec, 'r': r,
        })
        print(f"  z={z:6.3f}  ⟨x_HI⟩={mean_xhi:.3f}  r={r:+.3f}")

        del den, xhi, vz, vz_recx, den_c, xhi_c, vz_c, vz_recx_c
        gc.collect()

    # Physical extent of central crop (Mpc/h)
    n_crop = CENTRAL_CROP.stop - CENTRAL_CROP.start
    L = BOX_MPC_OVER_H * n_crop / 600.0
    extent = [0, L, 0, L]
    ticks = [0, 100, 200, 300, 400]

    fig = plt.figure(figsize=(11, 10))
    gs = GridSpec(2, 3, width_ratios=[1, 1, 0.05],
                  hspace=0.18, wspace=0.08)

    for row, d in enumerate(panels):
        vmax = float(np.percentile(
            np.abs(np.concatenate([d['ksz_real'].ravel(),
                                   d['ksz_rec'].ravel()])), 99))

        ax_true = fig.add_subplot(gs[row, 0])
        im = ax_true.imshow(d['ksz_real'], origin='lower', cmap='RdBu_r',
                            vmin=-vmax, vmax=vmax, extent=extent)
        if row == 0:
            ax_true.set_title('True kSZ', fontsize=22)
        ax_true.set_ylabel(
            f"z = {d['z']:.1f}, "
            f"$\\langle x_{{HI}} \\rangle$ = {d['mean_xhi']:.2f}\n"
            "y [Mpc/$h$]",
            fontsize=18,
        )
        ax_true.set_xlabel('x [Mpc/$h$]', fontsize=18)
        ax_true.set_xticks(ticks); ax_true.set_yticks(ticks)
        ax_true.tick_params(axis='both', which='major', labelsize=16)

        ax_rec = fig.add_subplot(gs[row, 1])
        im_rec = ax_rec.imshow(d['ksz_rec'], origin='lower', cmap='RdBu_r',
                                vmin=-vmax, vmax=vmax, extent=extent)
        if row == 0:
            ax_rec.set_title('Reconstructed kSZ', fontsize=22)
        ax_rec.set_xlabel('x [Mpc/$h$]', fontsize=18)
        ax_rec.set_xticks(ticks); ax_rec.set_yticks(ticks)
        ax_rec.set_yticklabels([])
        ax_rec.tick_params(axis='both', which='major', labelsize=16)
        ax_rec.text(0.97, 0.05, f"r = {d['r']:.2f}",
                    transform=ax_rec.transAxes,
                    fontsize=16, ha='right', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.3',
                              facecolor='white', edgecolor='black',
                              alpha=0.9))

        cax = fig.add_subplot(gs[row, 2])
        cbar = plt.colorbar(im_rec, cax=cax)
        cbar.set_label('$\\Delta T_{\\mathrm{kSZ}}$ [$\\mu$K]', fontsize=18)
        cbar.ax.tick_params(labelsize=14)

    outfile = os.path.join(output_dir, 'ksz_maps_true_vs_reconstructed.png')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outfile}")


def plot_mean_brightness_temp_vs_redshift_21cmfast(sim_id=12701, output_dir=OUTPUT_DIR):
    """Plot mean 21cm brightness temperature vs redshift from 21cmFAST sims.

    Uses the pre-computed Tb lightcone (USE_TS_FLUCT=True), so the signal
    includes spin temperature fluctuations (absorption trough at Cosmic
    Dawn, heating transition, etc.).
    """
    data_dir = os.path.join(PROJECT_ROOT, "data_21cmfast")
    Tb_file = os.path.join(data_dir, "Tb", f"{sim_id}_Tb_LC.npy")
    z_file = os.path.join(data_dir, "lightcone_redshifts.npy")

    print("\n" + "=" * 80)
    print(f"Plotting: Mean δTb vs redshift (21cmFAST sim {sim_id})")
    print("=" * 80)

    Tb = np.load(Tb_file, mmap_mode='r')
    z = np.load(z_file)
    print(f"  Tb lightcone shape: {Tb.shape},  z range: [{z.min():.2f}, {z.max():.2f}]")

    # Mean over the transverse (x, y) plane at each LOS slice
    mean_Tb = Tb.reshape(-1, Tb.shape[-1]).mean(axis=0).astype(np.float64)
    print(f"  <δTb> range: [{mean_Tb.min():.2f}, {mean_Tb.max():.2f}] mK")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(z, mean_Tb, '-', color='C1', lw=1.5)
    ax.axhline(0, color='grey', ls='--', lw=0.8, alpha=0.6)
    ax.set_xlabel('Redshift $z$', fontsize=22)
    ax.set_ylabel('$\\langle \\delta T_b \\rangle$ [mK]', fontsize=22)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()

    plt.tight_layout()
    outfile = os.path.join(output_dir, f'mean_brightness_temp_vs_redshift_21cmfast_{sim_id}.png')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outfile}")

    return z, mean_Tb


def _reconstruct_vz_21cmfast_chunk(den_chunk, xhi_chunk, z_ref,
                                   box_mpc_over_h=300.0, littleh=LITTLEH,
                                   include_velocity_term=False,
                                   vz_for_gradient=None):
    """Linear-continuity reconstruction of vz on a 21cmFAST chunk.

    den_chunk: overdensity δ (as stored in 21cmFAST LC files, mean ≈ 0).
    Returns vz_rec in km/s.
    """
    nx, ny, nz = den_chunk.shape
    delta = den_chunk.astype(np.float32, copy=False)
    xhi = xhi_chunk.astype(np.float32, copy=False)

    # Tracer field ∝ x_HI × (1+δ)  (optionally × velocity-gradient factor)
    tracer = (1.0 + delta) * xhi
    if include_velocity_term and vz_for_gradient is not None:
        dz_cell = box_mpc_over_h / nz / littleh  # Mpc per cell
        dvdz = np.gradient(vz_for_gradient.astype(np.float32),
                            dz_cell, axis=2)
        H0 = 100.0 * littleh
        Hz = H0 * np.sqrt(0.27 * (1 + z_ref)**3 + 0.73)
        tracer = tracer * (Hz / (dvdz + Hz)).astype(np.float32)
    tracer = tracer - tracer.mean()

    dlt_r = fft.rfftn(tracer, workers=-1).astype(np.complex64)

    rc = box_mpc_over_h / float(nx) / littleh  # Mpc/cell (physical)
    kx = (2 * np.pi * fft.fftfreq(nx, d=rc)).astype(np.float32)
    ky = (2 * np.pi * fft.fftfreq(ny, d=rc)).astype(np.float32)
    kz = (2 * np.pi * fft.rfftfreq(nz, d=rc)).astype(np.float32)
    tiny = np.finfo(np.float32).tiny
    kx[0] = max(kx[0], tiny); ky[0] = max(ky[0], tiny); kz[0] = max(kz[0], tiny)

    a = 1.0 / (1.0 + z_ref)
    H0 = 100.0 * littleh
    Ha = np.float32(H0 * np.sqrt(0.27 / a**3 + 0.73))
    Omega_m_a = (0.27 / a**3) / (0.27 / a**3 + 0.73)
    f_omega = np.float32(Omega_m_a**0.55)
    factor = np.complex64(Ha * a * f_omega) * 1j

    absk2 = (kx * kx)[:, None, None] + (ky * ky)[None, :, None] + (kz * kz)[None, None, :]
    tmp = dlt_r * factor * kz[None, None, :]
    np.divide(tmp, absk2, out=tmp, where=absk2 != 0)
    vz_rec = safe_real(fft.irfftn(tmp, s=(nx, ny, nz), workers=-1))
    return vz_rec


def run_21cmfast_chunked_reconstruction(sim_id=12701, chunk_size=200,
                                        physical_norm=True,
                                        output_dir=OUTPUT_DIR):
    """Full reconstruction + cross-correlation pipeline for a 21cmFAST LC.

    Chunks the lightcone along the LOS axis and, per chunk, performs:
        - velocity reconstruction from (1+δ)·x_HI via linear continuity
        - kSZ maps from real and reconstructed velocities
        - Pearson r between real and reconstructed kSZ

    Produces:
        - <output_dir>/ksz_reconstruction_21cmfast_<sim_id>_rVsXhi.png
        - <output_dir>/ksz_reconstruction_21cmfast_<sim_id>_integrated_maps.png
    """
    data_dir = os.path.join(PROJECT_ROOT, "data_21cmfast")
    den = np.load(os.path.join(data_dir, "density", f"{sim_id}_density_LC.npy"),
                  mmap_mode='r')
    xhi = np.load(os.path.join(data_dir, "xHI", f"{sim_id}_xHI_LC.npy"),
                  mmap_mode='r')
    vz = np.load(os.path.join(data_dir, "velocity", f"{sim_id}_velocity_z_LC.npy"),
                 mmap_mode='r')
    z_lc = np.load(os.path.join(data_dir, "lightcone_redshifts.npy"))

    nx, ny, nz_lc = den.shape
    print("\n" + "=" * 80)
    print(f"21cmFAST reconstruction pipeline (sim {sim_id})")
    print("=" * 80)
    print(f"  LC shape: {den.shape},  z range: [{z_lc.min():.2f}, {z_lc.max():.2f}]")

    # Physical prefactor (for µK units). vz stored in km/s in py21cmfast LC.
    # (If not, the amplitude will be off, but the correlation r is unchanged.)
    H0 = 100.0 * LITTLEH
    BOX = 300.0  # Mpc/h (21cmFAST box length)

    n_chunks = nz_lc // chunk_size
    print(f"  Chunking along LOS: {n_chunks} chunks of {chunk_size} slices each")

    mean_xhi_per_chunk = []
    mean_z_per_chunk = []
    r_per_chunk = []
    ksz_real_integrated = None
    ksz_rec_integrated = None

    for i in range(n_chunks):
        s, e = i * chunk_size, (i + 1) * chunk_size
        z_chunk = z_lc[s:e]
        z_ref = float(z_chunk.mean())

        # Load chunk into RAM (not mmap'd slice) so FFTs work fast
        den_c = np.ascontiguousarray(den[:, :, s:e])
        xhi_c = np.ascontiguousarray(xhi[:, :, s:e])
        vz_c = np.ascontiguousarray(vz[:, :, s:e])

        mean_xhi = float(xhi_c.mean())
        mean_xhi_per_chunk.append(mean_xhi)
        mean_z_per_chunk.append(z_ref)

        # Skip fully ionised / fully neutral chunks (reconstruction degenerate)
        if mean_xhi < 1e-3 or mean_xhi > 1.0 - 1e-3:
            print(f"  Chunk {i+1:2d}/{n_chunks}: z̄={z_ref:6.3f}  "
                  f"<x_HI>={mean_xhi:.4f}  -> skipped (edge of EoR)")
            r_per_chunk.append(np.nan)
            continue

        # Reconstruct vz from (1+δ)·x_HI
        vz_rec = _reconstruct_vz_21cmfast_chunk(den_c, xhi_c, z_ref,
                                                box_mpc_over_h=BOX)

        # kSZ maps: ΔT ∝ Σ x_e (1+δ) v_r
        # Using the 21cmFAST convention: den = δ (so 1+δ = 1+δ)
        xe_delta = (1.0 - xhi_c) * (1.0 + den_c)

        if physical_norm:
            sigma_T = 6.6524587158e-25
            c_cm_s = 2.99792458e10
            T_CMB = 2.725e6  # µK
            m_p = 1.6726219e-24
            Mpc_to_cm = 3.085677581e24
            Omega_b = 0.044
            H0_cgs = 100.0 * LITTLEH * 1e5 / Mpc_to_cm
            G_cgs = 6.67430e-8
            rho_crit = 3 * H0_cgs**2 / (8 * np.pi * G_cgs)
            n_e0 = Omega_b * rho_crit / m_p * (1.0 - 0.24 / 2.0)
            n_e_z = n_e0 * (1 + z_ref)**3
            dl_proper = (BOX / LITTLEH / (1 + z_ref)) / nx * Mpc_to_cm
            pref = -T_CMB * sigma_T * n_e_z * dl_proper / c_cm_s * 1e5  # v in km/s
        else:
            pref = 1.0

        ksz_real_c = pref * (xe_delta * vz_c).sum(axis=2)
        ksz_rec_c = pref * (xe_delta * vz_rec).sum(axis=2)

        r = pearson_r(ksz_real_c, ksz_rec_c)
        r_per_chunk.append(r)

        if ksz_real_integrated is None:
            ksz_real_integrated = ksz_real_c.copy()
            ksz_rec_integrated = ksz_rec_c.copy()
        else:
            ksz_real_integrated += ksz_real_c
            ksz_rec_integrated += ksz_rec_c

        print(f"  Chunk {i+1:2d}/{n_chunks}: z̄={z_ref:6.3f}  "
              f"<x_HI>={mean_xhi:.4f}  r={r:+.4f}")

        del den_c, xhi_c, vz_c, vz_rec, xe_delta, ksz_real_c, ksz_rec_c
        gc.collect()

    mean_xhi_per_chunk = np.array(mean_xhi_per_chunk)
    mean_z_per_chunk = np.array(mean_z_per_chunk)
    r_per_chunk = np.array(r_per_chunk)

    # Integrated correlation
    r_integrated = pearson_r(ksz_real_integrated, ksz_rec_integrated)
    print(f"\n  Integrated Pearson r = {r_integrated:+.4f}")

    # ------------------------------------------------------------------
    # PLOT 1: r vs <x_HI>, coloured by z
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    valid = np.isfinite(r_per_chunk)
    sc = ax.scatter(mean_xhi_per_chunk[valid], r_per_chunk[valid],
                    c=mean_z_per_chunk[valid], cmap='viridis', s=60, zorder=5)
    ax.plot(mean_xhi_per_chunk[valid], r_per_chunk[valid],
            '--', color='gray', lw=1, alpha=0.5, zorder=1)
    ax.axhline(0, color='black', lw=0.8, alpha=0.5)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Redshift $z$', fontsize=18)
    cbar.ax.tick_params(labelsize=14)
    ax.set_xlabel('Neutral Fraction $\\langle x_{HI} \\rangle$', fontsize=22)
    ax.set_ylabel('Pearson $r$ (real vs rec kSZ)', fontsize=22)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_ylim([-1.05, 1.05])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    outfile = os.path.join(output_dir,
                           f'ksz_reconstruction_21cmfast_{sim_id}_rVsXhi.png')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outfile}")

    # ------------------------------------------------------------------
    # PLOT 2: Integrated real vs reconstructed kSZ maps
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    im = ax.imshow(ksz_real_integrated, origin='lower', cmap='RdBu_r')
    ax.set_title(f'Integrated real kSZ (r$_{{int}}$ = {r_integrated:+.3f})',
                 fontsize=16)
    ax.set_xticks([]); ax.set_yticks([])
    cbar = plt.colorbar(im, ax=ax); cbar.ax.tick_params(labelsize=14)

    ax = axes[1]
    im = ax.imshow(ksz_rec_integrated, origin='lower', cmap='RdBu_r')
    ax.set_title('Integrated reconstructed kSZ', fontsize=16)
    ax.set_xticks([]); ax.set_yticks([])
    cbar = plt.colorbar(im, ax=ax); cbar.ax.tick_params(labelsize=14)

    plt.tight_layout()
    outfile = os.path.join(output_dir,
                           f'ksz_reconstruction_21cmfast_{sim_id}_integrated_maps.png')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outfile}")

    return {
        'mean_xhi': mean_xhi_per_chunk,
        'mean_z': mean_z_per_chunk,
        'r': r_per_chunk,
        'r_integrated': r_integrated,
        'ksz_real_integrated': ksz_real_integrated,
        'ksz_rec_integrated': ksz_rec_integrated,
    }


def comoving_distance(z, H0=70, Om=0.27):
    """Comoving distance to redshift z, in **Mpc/h** (h-unit convention).

    Delegates to ksz_core.cosmology.chi_mpch. Use Mpc/h so that pairing with
    k in h/Mpc gives dimensionally-correct ℓ = k · χ. The old local body
    here returned Mpc and silently dropped an h factor at every call site.
    """
    if H0 == 70 and Om == 0.27:
        c = _PAPER_COSMO
    else:
        c = _KSZ_Constants(H0=H0, Om0=Om, Ob0=0.044, h=H0 / 100.0)
    return _ksz_chi_mpch(z, c)


def plot_ksz_scale_dependence_and_ell3000(all_results, output_dir=OUTPUT_DIR):
    """Plot kSZ power spectrum scale dependence and ℓ=3000 analysis for all redshifts."""
    print("\n" + "="*80)
    print("Plotting: kSZ Scale Dependence and ℓ=3000 Analysis")
    print("="*80)
    
    # Compute power spectra for all results
    print("  Computing power spectra...")
    n_crop = CENTRAL_CROP.stop - CENTRAL_CROP.start
    dx = BOX_MPC_OVER_H / 600
    Lx = dx * n_crop
    
    psd_data = []
    for result in all_results:
        z = result['z']
        (den, xhi, vx, vy, vz, vx_rec, vy_rec, vz_rec, 
         vx_recx, vy_recx, vz_recx, _, _, _, _, _) = reconstruct_velocities(z)
        
        den = den[CENTRAL_CROP, CENTRAL_CROP, LOS_CROP]
        xhi = xhi[CENTRAL_CROP, CENTRAL_CROP, LOS_CROP]
        vz = vz[CENTRAL_CROP, CENTRAL_CROP, LOS_CROP]
        
        ksz_map = compute_ksz_maps(vz, xhi, den, z=z, physical_norm=PHYSICAL_NORM)
        ksz_c = ksz_map - np.mean(ksz_map)
        
        P, k = get_power(ksz_c.astype(np.float32), boxlength=[Lx, Lx], 
                         bins=50, ignore_zero_mode=True)
        
        psd_data.append({
            'z': z,
            'xhi': result['mean_xHI'],
            'k': k.copy(),
            'P': P.copy()
        })
        
        del den, xhi, vx, vy, vz, vx_rec, vy_rec, vz_rec, vx_recx, vy_recx, vz_recx
        del ksz_map, ksz_c
        gc.collect()
        print(f"    z={z:.3f} done")
    
    z_arr = np.array([r['z'] for r in all_results])
    xhi_arr = np.array([r['mean_xHI'] for r in all_results])
    
    # =========================================================================
    # Scale dependence analysis
    # =========================================================================
    k_bins = [0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.2]
    k_labels = ['0.02-0.05', '0.05-0.1', '0.1-0.2', '0.2-0.4', '0.4-0.8', '0.8-1.2']
    scale_power = np.zeros((len(all_results), len(k_bins)-1))
    
    for i, psd in enumerate(psd_data):
        for j in range(len(k_bins)-1):
            mask = (psd['k'] >= k_bins[j]) & (psd['k'] < k_bins[j+1]) & np.isfinite(psd['P'])
            if np.any(mask):
                scale_power[i, j] = np.mean(psd['P'][mask])
    
    high_z_idx = np.argmax(z_arr)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.plasma(np.linspace(0, 1, len(k_labels)))
    
    for j, (lbl, c) in enumerate(zip(k_labels, colors)):
        axes[0].plot(z_arr, scale_power[:, j], '--', color='gray', lw=2, alpha=0.5, zorder=1)
        axes[0].semilogy(z_arr, scale_power[:, j], 'o', color=c, ms=6, label=f'k={lbl}', zorder=5)
        if scale_power[high_z_idx, j] > 0:
            axes[1].plot(z_arr, scale_power[:, j]/scale_power[high_z_idx, j], '--', color='gray', lw=2, alpha=0.5, zorder=1)
            axes[1].plot(z_arr, scale_power[:, j]/scale_power[high_z_idx, j], 'o', color=c, ms=6, label=f'k={lbl}', zorder=5)
    
    axes[0].set_xlabel('Redshift z', fontsize=22)
    axes[0].set_ylabel('P(k)', fontsize=22)
    axes[0].legend(fontsize=14)
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(axis='both', which='major', labelsize=20)
    
    axes[1].axhline(1, color='k', ls='--', alpha=0.5)
    axes[1].set_xlabel('Redshift z', fontsize=22)
    axes[1].set_ylabel(f'P(k)/P(k,z={z_arr[high_z_idx]:.1f})', fontsize=22)
    axes[1].legend(fontsize=14)
    axes[1].grid(True, alpha=0.3)
    axes[1].tick_params(axis='both', which='major', labelsize=20)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/ksz_scale_dependence.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/ksz_scale_dependence.png")
    
    # =========================================================================
    # ℓ=3000 analysis
    # =========================================================================
    ell_target = 3000
    k_ell3000 = []
    power_ell3000 = []
    
    for i, r in enumerate(all_results):
        z = r['z']
        chi = comoving_distance(z)
        k_target = ell_target / chi
        k_ell3000.append(k_target)
        
        psd = psd_data[i]
        valid = np.isfinite(psd['k']) & np.isfinite(psd['P']) & (psd['k'] > 0)
        P_at_k = np.interp(k_target, psd['k'][valid], psd['P'][valid]) if np.any(valid) else np.nan
        power_ell3000.append(P_at_k)
    
    k_ell3000 = np.array(k_ell3000)
    power_ell3000 = np.array(power_ell3000)
    high_xhi_idx = np.argmax(xhi_arr)
    
    # =========================================================================
    # Plot vs Redshift
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    ax = axes[0]
    ax.plot(z_arr, power_ell3000, '--', color='gray', lw=2.5, alpha=0.7, zorder=1)
    sc = ax.scatter(z_arr, power_ell3000, c=xhi_arr, cmap='coolwarm', s=60, zorder=5)
    ax.set_yscale('log')
    ax.set_xlabel('Redshift z', fontsize=22)
    ax.set_ylabel(f'P(k) at $\\ell = {ell_target}$', fontsize=22)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=20)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('$x_{HI}$', fontsize=20)
    cbar.ax.tick_params(labelsize=18)
    
    ax = axes[1]
    power_norm = power_ell3000 / power_ell3000[high_xhi_idx]
    ax.plot(z_arr, power_norm, '--', color='gray', lw=2.5, alpha=0.7, zorder=1)
    sc = ax.scatter(z_arr, power_norm, c=xhi_arr, cmap='coolwarm', s=60, zorder=5)
    ax.axhline(1, color='k', ls='--', alpha=0.5)
    ax.set_xlabel('Redshift z', fontsize=22)
    ax.set_ylabel(f'P / P(z={z_arr[high_xhi_idx]:.1f})', fontsize=22)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=20)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('$x_{HI}$', fontsize=20)
    cbar.ax.tick_params(labelsize=18)
    
    ax = axes[2]
    k_compare = [0.1, 0.2, 0.4]
    colors_compare = ['blue', 'green', 'orange']
    for k_val, col in zip(k_compare, colors_compare):
        power_at_k = []
        for psd in psd_data:
            valid = np.isfinite(psd['k']) & np.isfinite(psd['P']) & (psd['k'] > 0)
            P_at_k = np.interp(k_val, psd['k'][valid], psd['P'][valid]) if np.any(valid) else np.nan
            power_at_k.append(P_at_k)
        power_at_k = np.array(power_at_k)
        ax.plot(z_arr, power_at_k/power_at_k[high_xhi_idx], '--', color='gray', lw=2, alpha=0.4, zorder=1)
        ax.plot(z_arr, power_at_k/power_at_k[high_xhi_idx], 'o', color=col, ms=6, alpha=0.7, 
                label=f'k={k_val} h/Mpc', zorder=5)
    ax.plot(z_arr, power_norm, '--', color='gray', lw=2.5, alpha=0.7, zorder=1)
    ax.scatter(z_arr, power_norm, c='red', s=60, marker='s', label=f'$\\ell$={ell_target}', zorder=5)
    ax.axhline(1, color='k', ls='--', alpha=0.5)
    ax.set_xlabel('Redshift z', fontsize=22)
    ax.set_ylabel(f'P / P(z={z_arr[high_xhi_idx]:.1f})', fontsize=22)
    ax.legend(fontsize=14, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=20)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/ksz_ell3000_analysis_vs_z.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/ksz_ell3000_analysis_vs_z.png")
    
    # =========================================================================
    # Plot vs Neutral Fraction
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    ax = axes[0]
    ax.plot(xhi_arr, power_ell3000, '--', color='gray', lw=2.5, alpha=0.7, zorder=1)
    sc = ax.scatter(xhi_arr, power_ell3000, c=z_arr, cmap='viridis', s=60, zorder=5)
    ax.set_yscale('log')
    ax.set_xlabel('Neutral Fraction $\\langle x_{HI} \\rangle$', fontsize=22)
    ax.set_ylabel(f'P(k) at $\\ell = {ell_target}$', fontsize=22)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=20)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Redshift z', fontsize=20)
    cbar.ax.tick_params(labelsize=18)
    
    ax = axes[1]
    ax.plot(xhi_arr, power_norm, '--', color='gray', lw=2.5, alpha=0.7, zorder=1)
    sc = ax.scatter(xhi_arr, power_norm, c=z_arr, cmap='viridis', s=60, zorder=5)
    ax.axhline(1, color='k', ls='--', alpha=0.5)
    ax.set_xlabel('Neutral Fraction $\\langle x_{HI} \\rangle$', fontsize=22)
    ax.set_ylabel(f'P / P(xHI={xhi_arr[high_xhi_idx]:.2f})', fontsize=22)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=20)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Redshift z', fontsize=20)
    cbar.ax.tick_params(labelsize=18)
    
    ax = axes[2]
    for k_val, col in zip(k_compare, colors_compare):
        power_at_k = []
        for psd in psd_data:
            valid = np.isfinite(psd['k']) & np.isfinite(psd['P']) & (psd['k'] > 0)
            P_at_k = np.interp(k_val, psd['k'][valid], psd['P'][valid]) if np.any(valid) else np.nan
            power_at_k.append(P_at_k)
        power_at_k = np.array(power_at_k)
        ax.plot(xhi_arr, power_at_k/power_at_k[high_xhi_idx], '--', color='gray', lw=2, alpha=0.4, zorder=1)
        ax.plot(xhi_arr, power_at_k/power_at_k[high_xhi_idx], 'o', color=col, ms=6, alpha=0.7, 
                label=f'k={k_val} h/Mpc', zorder=5)
    ax.plot(xhi_arr, power_norm, '--', color='gray', lw=2.5, alpha=0.7, zorder=1)
    ax.scatter(xhi_arr, power_norm, c='red', s=60, marker='s', label=f'$\\ell$={ell_target}', zorder=5)
    ax.axhline(1, color='k', ls='--', alpha=0.5)
    ax.set_xlabel('Neutral Fraction $\\langle x_{HI} \\rangle$', fontsize=22)
    ax.set_ylabel(f'P / P(xHI={xhi_arr[high_xhi_idx]:.2f})', fontsize=22)
    ax.legend(fontsize=14, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=20)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/ksz_ell3000_analysis_vs_xhi.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/ksz_ell3000_analysis_vs_xhi.png")
    
    print(f"\nPeak power at ell={ell_target}: z = {z_arr[np.argmax(power_ell3000)]:.2f}")
    print(f"Growth factor from z={z_arr[high_xhi_idx]:.1f} to z={z_arr[0]:.1f}: {power_ell3000[0]/power_ell3000[high_xhi_idx]:.1f}x")
    print(f"Note: ell=3000 probes k~{k_ell3000.mean():.2f} h/Mpc (varies with z from {k_ell3000.min():.3f} to {k_ell3000.max():.3f})")
    
    # =========================================================================
    # Fractional contribution at ℓ=3000
    # =========================================================================
    print("\n" + "-"*60)
    print(f"Fractional contribution to total kSZ power at ℓ={ell_target}:")
    print("-"*60)
    
    total_power = np.sum(power_ell3000)
    contributions = []
    
    for i, r in enumerate(all_results):
        frac = power_ell3000[i] / total_power * 100
        contributions.append({'z': r['z'], 'xhi': r['mean_xHI'], 'frac': frac, 'power': power_ell3000[i]})
        print(f"  z={r['z']:.3f} (xHI={r['mean_xHI']:.2f}): {frac:.1f}%")
    
    fig, ax = plt.subplots(figsize=(14, 6))
    fracs = [c['frac'] for c in contributions]
    colors = plt.cm.coolwarm(np.array([c['xhi'] for c in contributions]))
    x_pos = np.arange(len(all_results))
    
    bars = ax.bar(x_pos, fracs, color=colors, edgecolor='black', width=0.7)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{r['z']:.1f}" for r in all_results], fontsize=16, rotation=45, ha='right')
    ax.set_xlabel('Redshift z', fontsize=22)
    ax.set_ylabel(f'Contribution to Total Power at $\\ell$={ell_target} (%)', fontsize=22)
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='both', which='major', labelsize=20)
    
    for i, (bar, c) in enumerate(zip(bars, contributions)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                f'{c["xhi"]:.2f}', ha='center', va='bottom', fontsize=9, color='gray')
    
    sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(0, 1))
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label('$x_{HI}$', fontsize=20)
    cbar.ax.tick_params(labelsize=18)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/ksz_ell3000_fractional_contribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/ksz_ell3000_fractional_contribution.png")
    print(f"\nTotal: {sum(fracs):.1f}%")


def plot_ksz_power_paper_figure(all_results, output_dir=OUTPUT_DIR):
    """Create a 2-panel figure for the paper showing kSZ power evolution.
    
    Left: Normalized power at ℓ=3000 vs Neutral Fraction
    Right: Normalized power vs Redshift for multiple ℓ scales
    """
    print("\n" + "="*80)
    print("Plotting: kSZ Power Paper Figure")
    print("="*80)
    
    # Compute power spectra for all results
    print("  Computing power spectra...")
    n_crop = CENTRAL_CROP.stop - CENTRAL_CROP.start
    dx = BOX_MPC_OVER_H / 600
    Lx = dx * n_crop
    
    psd_data = []
    for result in all_results:
        z = result['z']
        (den, xhi, vx, vy, vz, vx_rec, vy_rec, vz_rec, 
         vx_recx, vy_recx, vz_recx, _, _, _, _, _) = reconstruct_velocities(z)
        
        den = den[CENTRAL_CROP, CENTRAL_CROP, LOS_CROP]
        xhi = xhi[CENTRAL_CROP, CENTRAL_CROP, LOS_CROP]
        vz = vz[CENTRAL_CROP, CENTRAL_CROP, LOS_CROP]
        
        ksz_map = compute_ksz_maps(vz, xhi, den, z=z, physical_norm=PHYSICAL_NORM)
        ksz_c = ksz_map - np.mean(ksz_map)
        
        fft2d = np.fft.rfft2(ksz_c)
        P2d = np.abs(fft2d)**2 / (Lx**2)
        
        kx = np.fft.fftfreq(n_crop, d=dx) * 2 * np.pi
        ky = np.fft.rfftfreq(n_crop, d=dx) * 2 * np.pi
        KX, KY = np.meshgrid(kx, ky, indexing='ij')
        K = np.sqrt(KX**2 + KY**2)
        
        k_bins = np.linspace(0, 2.0, 50)
        k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])
        P_binned = np.zeros(len(k_centers))
        for i in range(len(k_centers)):
            mask = (K >= k_bins[i]) & (K < k_bins[i+1])
            if np.any(mask):
                P_binned[i] = np.mean(P2d[mask])
        
        psd_data.append({'k': k_centers, 'P': P_binned, 'z': z})
        
        del vx, vy, vx_rec, vy_rec, vz_rec, vx_recx, vy_recx, vz_recx
        del den, xhi, vz, ksz_map, ksz_c
        gc.collect()
    
    z_arr = np.array([r['z'] for r in all_results])
    xhi_arr = np.array([r['mean_xHI'] for r in all_results])
    
    # Use a representative redshift for k-to-ell conversion (middle of range)
    z_ref = 8.0
    chi_ref = comoving_distance(z_ref)
    
    # ℓ=3000 analysis
    ell_target = 3000
    k_ell3000 = []
    power_ell3000 = []
    
    for i, r in enumerate(all_results):
        z = r['z']
        chi = comoving_distance(z)
        k_target = ell_target / chi
        k_ell3000.append(k_target)
        
        psd = psd_data[i]
        valid = np.isfinite(psd['k']) & np.isfinite(psd['P']) & (psd['k'] > 0)
        P_at_k = np.interp(k_target, psd['k'][valid], psd['P'][valid]) if np.any(valid) else np.nan
        power_ell3000.append(P_at_k)
    
    k_ell3000 = np.array(k_ell3000)
    power_ell3000 = np.array(power_ell3000)
    high_xhi_idx = np.argmax(xhi_arr)
    high_z_idx = np.argmax(z_arr)
    
    # Scale bins with corresponding ℓ values (using z_ref for conversion)
    k_bins = [0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.2]
    # Convert k to ℓ: ℓ = k * χ(z_ref)
    ell_labels = []
    for j in range(len(k_bins)-1):
        k_mid = (k_bins[j] + k_bins[j+1]) / 2
        ell_mid = int(k_mid * chi_ref)
        ell_labels.append(f'$\\ell \\approx {ell_mid}$')
    
    scale_power = np.zeros((len(all_results), len(k_bins)-1))
    for i, psd in enumerate(psd_data):
        for j in range(len(k_bins)-1):
            mask = (psd['k'] >= k_bins[j]) & (psd['k'] < k_bins[j+1]) & np.isfinite(psd['P'])
            if np.any(mask):
                scale_power[i, j] = np.mean(psd['P'][mask])
    
    # Find peak indices for normalization
    peak_power_idx = np.argmax(power_ell3000)  # Index of peak power for left plot
    
    # Create 2-panel figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Normalized power at ℓ=3000 vs Neutral Fraction (normalized at peak)
    ax = axes[0]
    power_norm_xhi = power_ell3000 / power_ell3000[peak_power_idx]
    sc = ax.scatter(xhi_arr, power_norm_xhi, c=z_arr, cmap='viridis', s=60, zorder=5)
    ax.set_xlabel('Neutral Fraction $\\langle x_{HI} \\rangle$', fontsize=22)
    ax.set_ylabel('P / P$_{max}$', fontsize=22)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=20)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Redshift z', fontsize=20)
    cbar.ax.tick_params(labelsize=18)
    
    # Right: Non-normalized power vs Neutral Fraction for multiple ℓ scales
    ax = axes[1]
    colors = plt.cm.plasma(np.linspace(0, 1, len(ell_labels)))
    for j, (lbl, c) in enumerate(zip(ell_labels, colors)):
        ax.semilogy(xhi_arr, scale_power[:, j], 'o', color=c, ms=6, label=lbl, zorder=5)
    ax.set_xlabel('Neutral Fraction $\\langle x_{HI} \\rangle$', fontsize=22)
    ax.set_ylabel('P($\\ell$)', fontsize=22)
    ax.legend(fontsize=14, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=20)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/ksz_power_paper_figure.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/ksz_power_paper_figure.png")


# ============================================================================
# 21cm NORMALIZATION COMPARISON ANALYSIS
# ============================================================================

def compute_regression_metrics(y_true, y_pred):
    """Compute Pearson r, best-fit slope, and RMS scatter.
    
    Returns:
    --------
    r : float
        Pearson correlation coefficient
    slope : float
        Best-fit slope (amplitude bias)
    rms : float
        RMS scatter around the best-fit line
    """
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(mask):
        return np.nan, np.nan, np.nan
    
    y_t = y_true[mask]
    y_p = y_pred[mask]
    
    # Pearson r
    r = np.corrcoef(y_t, y_p)[0, 1] if len(y_t) > 1 else np.nan
    
    # Best-fit slope (linear regression through origin is not appropriate here)
    # Use standard linear regression: y_pred = slope * y_true + intercept
    if len(y_t) > 1:
        slope, intercept = np.polyfit(y_t, y_p, 1)
        residuals = y_p - (slope * y_t + intercept)
        rms = np.sqrt(np.mean(residuals**2))
    else:
        slope, rms = np.nan, np.nan
    
    return float(r), float(slope), float(rms)


def analyze_21cm_normalization_comparison(z, output_dir=OUTPUT_DIR):
    """
    Compare velocity reconstruction using three δTb normalizations.
    
    This function implements the 21cm brightness temperature normalization study:
    (A) Raw δTb [mK] - baseline
    (B) Mean-normalized: (δTb - ⟨δTb⟩) / ⟨δTb⟩ - dimensionless, simulation-only
    (C) Fixed-scale: δTb / T_ref - dimensionless, observable-motivated
    
    Parameters:
    -----------
    z : float
        Redshift to analyze
    output_dir : str
        Output directory for plots
    
    Returns:
    --------
    dict : Results including correlations, slopes, and RMS for each method
    """
    print(f"\n{'='*80}")
    print(f"21cm NORMALIZATION COMPARISON: z = {z:.3f}")
    print(f"{'='*80}")
    
    # Load data with all three normalizations
    (den, xhi, vx, vy, vz, vx_rec, vy_rec, vz_rec, 
     vx_recx, vy_recx, vz_recx, vx_recx_norm, vy_recx_norm, vz_recx_norm,
     vz_rec_mean_norm, vz_rec_fixed_mK) = reconstruct_velocities(z)
    
    # Crop to central region
    vz = vz[CENTRAL_CROP, CENTRAL_CROP, LOS_CROP]
    vz_recx = vz_recx[CENTRAL_CROP, CENTRAL_CROP, LOS_CROP]
    vz_rec_mean_norm = vz_rec_mean_norm[CENTRAL_CROP, CENTRAL_CROP, LOS_CROP]
    vz_rec_fixed_mK = vz_rec_fixed_mK[CENTRAL_CROP, CENTRAL_CROP, LOS_CROP]
    xhi = xhi[CENTRAL_CROP, CENTRAL_CROP, LOS_CROP]
    
    mean_xHI = xhi.mean()
    print(f"  Mean xHI: {mean_xHI:.4f}")
    
    # Exclude boundaries
    boundary = 10
    vz_crop = vz[boundary:-boundary, boundary:-boundary, boundary:-boundary]
    
    # The reconstructions use NEGATIVE sign convention (v_rec = -continuity result)
    vz_raw_crop = (-vz_recx)[boundary:-boundary, boundary:-boundary, boundary:-boundary]
    vz_mean_norm_crop = (-vz_rec_mean_norm)[boundary:-boundary, boundary:-boundary, boundary:-boundary]
    vz_fixed_mK_crop = (-vz_rec_fixed_mK)[boundary:-boundary, boundary:-boundary, boundary:-boundary]
    
    # Flatten for analysis
    vz_true = vz_crop.flatten()
    vz_raw = vz_raw_crop.flatten()
    vz_mean_norm = vz_mean_norm_crop.flatten()
    vz_fixed_mK = vz_fixed_mK_crop.flatten()
    
    # Compute metrics for each method
    r_raw, slope_raw, rms_raw = compute_regression_metrics(vz_true, vz_raw)
    r_mean_norm, slope_mean_norm, rms_mean_norm = compute_regression_metrics(vz_true, vz_mean_norm)
    r_fixed_mK, slope_fixed_mK, rms_fixed_mK = compute_regression_metrics(vz_true, vz_fixed_mK)
    
    print(f"\n  Method A (Raw δTb):        r = {r_raw:.4f}, slope = {slope_raw:.4f}, RMS = {rms_raw:.2f}")
    print(f"  Method B (Mean-norm):      r = {r_mean_norm:.4f}, slope = {slope_mean_norm:.4f}, RMS = {rms_mean_norm:.2f}")
    print(f"  Method C (Fixed {T_REF_MK:.0f}mK):    r = {r_fixed_mK:.4f}, slope = {slope_fixed_mK:.4f}, RMS = {rms_fixed_mK:.2f}")
    
    # =========================================================================
    # PLOT 1: Velocity slice comparison (4 panels)
    # =========================================================================
    slice_idx = vz_crop.shape[2] // 2  # Middle slice
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Get common color limits from true velocity
    vmin, vmax = np.percentile(vz_crop[:, :, slice_idx], [2, 98])
    
    # True velocity
    ax = axes[0, 0]
    im = ax.imshow(vz_crop[:, :, slice_idx], origin='lower', cmap='RdBu_r', 
                   vmin=vmin, vmax=vmax)
    ax.set_title(f'True $v_z$ (z={z:.2f}, xHI={mean_xHI:.2f})', fontsize=14)
    ax.set_xlabel('x [cells]', fontsize=12)
    ax.set_ylabel('y [cells]', fontsize=12)
    plt.colorbar(im, ax=ax, label='km/s')
    
    # Raw δTb reconstruction
    ax = axes[0, 1]
    im = ax.imshow(vz_raw_crop[:, :, slice_idx], origin='lower', cmap='RdBu_r',
                   vmin=vmin, vmax=vmax)
    ax.set_title(f'Raw $\\delta T_b$: r={r_raw:.3f}', fontsize=14)
    ax.set_xlabel('x [cells]', fontsize=12)
    ax.set_ylabel('y [cells]', fontsize=12)
    plt.colorbar(im, ax=ax, label='km/s')
    
    # Mean-normalized reconstruction
    ax = axes[1, 0]
    # Scale to match true velocity amplitude for visualization
    scale_mean = np.std(vz_crop) / np.std(vz_mean_norm_crop) if np.std(vz_mean_norm_crop) > 0 else 1
    im = ax.imshow(vz_mean_norm_crop[:, :, slice_idx] * scale_mean, origin='lower', 
                   cmap='RdBu_r', vmin=vmin, vmax=vmax)
    ax.set_title(f'Mean-norm (scaled): r={r_mean_norm:.3f}', fontsize=14)
    ax.set_xlabel('x [cells]', fontsize=12)
    ax.set_ylabel('y [cells]', fontsize=12)
    plt.colorbar(im, ax=ax, label='km/s (scaled)')
    
    # Fixed-scale reconstruction
    ax = axes[1, 1]
    scale_fixed = np.std(vz_crop) / np.std(vz_fixed_mK_crop) if np.std(vz_fixed_mK_crop) > 0 else 1
    im = ax.imshow(vz_fixed_mK_crop[:, :, slice_idx] * scale_fixed, origin='lower',
                   cmap='RdBu_r', vmin=vmin, vmax=vmax)
    ax.set_title(f'Fixed {T_REF_MK:.0f}mK (scaled): r={r_fixed_mK:.3f}', fontsize=14)
    ax.set_xlabel('x [cells]', fontsize=12)
    ax.set_ylabel('y [cells]', fontsize=12)
    plt.colorbar(im, ax=ax, label='km/s (scaled)')
    
    plt.suptitle(f'Velocity Reconstruction: 21cm Normalization Comparison\n'
                 f'z = {z:.3f}, $\\langle x_{{HI}} \\rangle$ = {mean_xHI:.3f}', fontsize=16)
    plt.tight_layout()
    outfile = os.path.join(output_dir, f'velocity_21cm_norm_slices_z{z:.3f}.png')
    plt.savefig(outfile, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outfile}")
    
    # =========================================================================
    # PLOT 2: Scatter plots (3 panels)
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    sample_size = 50000
    np.random.seed(42)
    n_pts = len(vz_true)
    idx = np.random.choice(n_pts, min(sample_size, n_pts), replace=False)
    
    lim = np.percentile(vz_true, [1, 99])
    
    # Raw δTb
    ax = axes[0]
    ax.scatter(vz_raw[idx], vz_true[idx], alpha=0.3, s=1, c='blue')
    ax.plot([lim[0], lim[1]], [lim[0], lim[1]], 'r--', lw=2, label='y=x')
    ax.set_xlabel('Reconstructed $v_z$ (Raw $\\delta T_b$)', fontsize=14)
    ax.set_ylabel('True $v_z$ [km/s]', fontsize=14)
    ax.set_title(f'Raw $\\delta T_b$\nr={r_raw:.4f}, slope={slope_raw:.3f}', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Mean-normalized
    ax = axes[1]
    ax.scatter(vz_mean_norm[idx], vz_true[idx], alpha=0.3, s=1, c='green')
    lim_mn = np.percentile(vz_mean_norm, [1, 99])
    ax.plot([lim_mn[0], lim_mn[1]], [lim_mn[0], lim_mn[1]], 'r--', lw=2, label='y=x')
    ax.set_xlabel('Reconstructed $v_z$ (Mean-norm)', fontsize=14)
    ax.set_ylabel('True $v_z$ [km/s]', fontsize=14)
    ax.set_title(f'Mean-normalized\nr={r_mean_norm:.4f}, slope={slope_mean_norm:.3f}', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Fixed-scale
    ax = axes[2]
    ax.scatter(vz_fixed_mK[idx], vz_true[idx], alpha=0.3, s=1, c='orange')
    lim_fk = np.percentile(vz_fixed_mK, [1, 99])
    ax.plot([lim_fk[0], lim_fk[1]], [lim_fk[0], lim_fk[1]], 'r--', lw=2, label='y=x')
    ax.set_xlabel(f'Reconstructed $v_z$ (Fixed {T_REF_MK:.0f}mK)', fontsize=14)
    ax.set_ylabel('True $v_z$ [km/s]', fontsize=14)
    ax.set_title(f'Fixed {T_REF_MK:.0f}mK scale\nr={r_fixed_mK:.4f}, slope={slope_fixed_mK:.3f}', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Velocity Scatter: 21cm Normalization Comparison (z={z:.3f})', fontsize=16)
    plt.tight_layout()
    outfile = os.path.join(output_dir, f'velocity_21cm_norm_scatter_z{z:.3f}.png')
    plt.savefig(outfile, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outfile}")
    
    # Clean up
    del den, xhi, vz, vz_recx, vz_rec_mean_norm, vz_rec_fixed_mK
    gc.collect()
    
    return {
        'z': z,
        'mean_xHI': mean_xHI,
        'r_raw': r_raw,
        'r_mean_norm': r_mean_norm,
        'r_fixed_mK': r_fixed_mK,
        'slope_raw': slope_raw,
        'slope_mean_norm': slope_mean_norm,
        'slope_fixed_mK': slope_fixed_mK,
        'rms_raw': rms_raw,
        'rms_mean_norm': rms_mean_norm,
        'rms_fixed_mK': rms_fixed_mK,
    }


def run_21cm_normalization_study(redshifts, output_dir=OUTPUT_DIR):
    """
    Run the full 21cm normalization comparison study across multiple redshifts.
    
    This study compares three normalizations of the 21cm brightness temperature
    for linear continuity velocity reconstruction:
    
    (A) Raw δTb [mK] - baseline, has physical units
    (B) Mean-normalized - dimensionless, theoretically clean but NOT observable
    (C) Fixed-scale (10 mK) - dimensionless, observable-motivated
    
    KEY INSIGHT: All three methods should give IDENTICAL correlation coefficients
    because they differ only by a constant factor. The amplitude (slope) will differ.
    Correlation coefficient is the robust metric for comparing reconstruction quality.
    
    Parameters:
    -----------
    redshifts : list
        List of redshifts to analyze
    output_dir : str
        Output directory for plots
    
    Returns:
    --------
    list : Results for each redshift
    """
    print("\n" + "="*80)
    print("21cm BRIGHTNESS TEMPERATURE NORMALIZATION STUDY")
    print("="*80)
    print(f"Analyzing {len(redshifts)} redshifts")
    print(f"Reference temperature for fixed-scale: T_ref = {T_REF_MK} mK")
    print("="*80)
    
    all_results = []
    for z in redshifts:
        result = analyze_21cm_normalization_comparison(z, output_dir)
        all_results.append(result)
    
    # =========================================================================
    # Summary plot: Correlation vs neutral fraction
    # =========================================================================
    print("\nCreating summary plot...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    xHIs = np.array([r['mean_xHI'] for r in all_results])
    zs = np.array([r['z'] for r in all_results])
    
    # Left: Correlation coefficient
    ax = axes[0]
    ax.plot(xHIs, [r['r_raw'] for r in all_results], 'b-o', 
            label='Raw $\\delta T_b$', markersize=8, linewidth=2)
    ax.plot(xHIs, [r['r_mean_norm'] for r in all_results], 'g-s', 
            label='Mean-normalized', markersize=8, linewidth=2)
    ax.plot(xHIs, [r['r_fixed_mK'] for r in all_results], 'orange', marker='^',
            label=f'Fixed {T_REF_MK:.0f}mK', markersize=8, linewidth=2)
    ax.set_xlabel('Mean Neutral Fraction $\\langle x_{HI} \\rangle$', fontsize=16)
    ax.set_ylabel('Pearson Correlation r', fontsize=16)
    ax.set_title('Correlation Coefficient (Morphology)', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-1.1, 1.1])
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    # Right: Amplitude (slope)
    ax = axes[1]
    ax.semilogy(xHIs, np.abs([r['slope_raw'] for r in all_results]), 'b-o', 
                label='Raw $\\delta T_b$', markersize=8, linewidth=2)
    ax.semilogy(xHIs, np.abs([r['slope_mean_norm'] for r in all_results]), 'g-s', 
                label='Mean-normalized', markersize=8, linewidth=2)
    ax.semilogy(xHIs, np.abs([r['slope_fixed_mK'] for r in all_results]), 'orange', marker='^',
                label=f'Fixed {T_REF_MK:.0f}mK', markersize=8, linewidth=2)
    ax.axhline(y=1.0, color='k', linestyle='--', linewidth=1, alpha=0.5, label='Perfect (slope=1)')
    ax.set_xlabel('Mean Neutral Fraction $\\langle x_{HI} \\rangle$', fontsize=16)
    ax.set_ylabel('|Slope| (Amplitude Bias)', fontsize=16)
    ax.set_title('Amplitude Recovery', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    plt.suptitle('21cm Normalization Study: Linear Continuity Reconstruction', fontsize=18)
    plt.tight_layout()
    outfile = os.path.join(output_dir, 'velocity_21cm_norm_summary.png')
    plt.savefig(outfile, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outfile}")
    
    # =========================================================================
    # Print summary table
    # =========================================================================
    print("\n" + "="*100)
    print("SUMMARY: 21cm Normalization Comparison")
    print("="*100)
    print(f"{'z':>8} {'x_HI':>8} {'r_raw':>10} {'r_mean':>10} {'r_fixed':>10} "
          f"{'slope_raw':>12} {'slope_mean':>12} {'slope_fixed':>12}")
    print("-"*96)
    for r in all_results:
        print(f"{r['z']:8.3f} {r['mean_xHI']:8.3f} {r['r_raw']:10.4f} {r['r_mean_norm']:10.4f} "
              f"{r['r_fixed_mK']:10.4f} {r['slope_raw']:12.4f} {r['slope_mean_norm']:12.4f} "
              f"{r['slope_fixed_mK']:12.4f}")
    print("="*100)
    
    print("\nKEY OBSERVATIONS:")
    print("  - Correlation coefficients should be IDENTICAL across normalizations")
    print("    (they differ only by a constant factor)")
    print("  - Amplitude (slope) varies with normalization choice")
    print("  - Mean-normalized is theoretically clean but NOT observable")
    print("  - Fixed-scale is observable but has arbitrary amplitude")
    print("  - Correlation r is the PRIMARY ROBUST METRIC")
    
    return all_results


# ============================================================================
# MAIN ANALYSIS PIPELINE
# ============================================================================

def analyze_single_redshift(z, n=600):
    """Analyze a single redshift."""
    print(f"\n{'='*80}")
    print(f"Processing z={z:.3f}")
    print(f"{'='*80}")
    
    # Load and reconstruct
    (den, xhi, vx, vy, vz, vx_rec, vy_rec, vz_rec, 
     vx_recx, vy_recx, vz_recx, _, _, _, _, _) = reconstruct_velocities(z, n=n)
    
    # Apply central crop in all 3 directions to avoid boundary effects
    den = den[CENTRAL_CROP, CENTRAL_CROP, LOS_CROP]
    xhi = xhi[CENTRAL_CROP, CENTRAL_CROP, LOS_CROP]
    vz = vz[CENTRAL_CROP, CENTRAL_CROP, LOS_CROP]
    vz_recx = vz_recx[CENTRAL_CROP, CENTRAL_CROP, LOS_CROP]
    
    mean_xHI = xhi.mean()
    print(f"  Mean xHI: {mean_xHI:.4f}")
    
    # Compute box dimensions for cropped region
    n_crop = CENTRAL_CROP.stop - CENTRAL_CROP.start
    dx = BOX_MPC_OVER_H / n
    dy = BOX_MPC_OVER_H / n
    Ly = dy * n_crop
    Lx = dx * n_crop
    smooth_sigma_pixels = SMOOTH_SIGMA / dx
    
    # 3D velocity correlations
    r_3d = pearson_r(vz, -vz_recx)
    print(f"  3D velocity correlation: {r_3d:.4f}")

    # 2D map correlations (LOS already cropped in 3D fields)
    vz_map = np.sum(vz, axis=2)
    vz_rec_map = np.sum(-vz_recx, axis=2)
    r_2d = pearson_r(vz_map, vz_rec_map)
    print(f"  2D map correlation: {r_2d:.4f}")

    # Fourier velocity correlation (2D integrated velocity maps)
    k_values_vel_2d, r_k_vel_2d = compute_fourier_correlation_coefficient(
        vz_map, vz_rec_map, boxlength=[Ly, Lx]
    )

    # Fourier velocity correlation (3D fields)
    box_size_crop = dx * n_crop  # Box size for cropped region in Mpc/h
    k_values_vel_3d, r_k_vel_3d = compute_fourier_correlation_coefficient_3d(
        vz, -vz_recx, boxlength=box_size_crop
    )
    print(f"  3D Fourier r(k=0.1): {np.interp(0.1, k_values_vel_3d, r_k_vel_3d):.4f}")
    print(f"  2D Fourier r(k=0.1): {np.interp(0.1, k_values_vel_2d, r_k_vel_2d):.4f}")

    # ---- Spatial jackknife on velocity correlations ---------------------
    # 3D: 2×2×2 = 8 octants; 2D map: 2×2 = 4 transverse tiles.
    _, r_3d_jk_sigma, _ = jackknife_pearson_r(vz, -vz_recx, n_per_side=2)
    _, r_2d_jk_sigma, _ = jackknife_pearson_r(vz_map, vz_rec_map, n_per_side=2)
    _, _, r_k_vel_3d_jk_sigma = jackknife_r_k_3d(
        vz, -vz_recx, boxlength=box_size_crop,
        n_per_side=2, k_grid=k_values_vel_3d
    )
    _, _, r_k_vel_2d_jk_sigma = jackknife_r_k_2d(
        vz_map, vz_rec_map, boxlength=[Ly, Lx],
        n_per_side=2, k_grid=k_values_vel_2d
    )
    print(f"  Jackknife σ (3D, 2D): {r_3d_jk_sigma:.4f}, {r_2d_jk_sigma:.4f}")
    
    # kSZ maps - unsmoothed
    ksz_map = compute_ksz_maps(vz, xhi, den, z=z, physical_norm=PHYSICAL_NORM)
    ksz_map_rec = compute_ksz_maps(-vz_recx, xhi, den, z=z, physical_norm=PHYSICAL_NORM)
    r_ksz_unsmooth = pearson_r(ksz_map, ksz_map_rec)
    units = "µK" if PHYSICAL_NORM else "arb. units"
    print(f"  kSZ correlation (unsmoothed): {r_ksz_unsmooth:.4f} [{units}]")
    
    # kSZ maps - smoothed
    vz_smooth = gaussian_filter(vz, sigma=smooth_sigma_pixels)
    vz_recx_smooth = gaussian_filter(-vz_recx, sigma=smooth_sigma_pixels)
    ksz_map_smooth = compute_ksz_maps(vz_smooth, xhi, den, z=z, physical_norm=PHYSICAL_NORM)
    ksz_map_rec_smooth = compute_ksz_maps(vz_recx_smooth, xhi, den, z=z, physical_norm=PHYSICAL_NORM)
    r_ksz_smooth = pearson_r(ksz_map_smooth, ksz_map_rec_smooth)
    print(f"  kSZ correlation (smoothed): {r_ksz_smooth:.4f} [{units}]")
    
    # Fourier correlations - unsmoothed
    k_values, r_k = compute_fourier_correlation_coefficient(
        ksz_map, ksz_map_rec, boxlength=[Ly, Lx]
    )
    
    # Fourier correlations - smoothed
    k_values_s, r_k_s = compute_fourier_correlation_coefficient(
        ksz_map_smooth, ksz_map_rec_smooth, boxlength=[Ly, Lx]
    )
    
    # Convert k to ell
    ell_values = k_to_ell(k_values, z)
    ell_values_s = k_to_ell(k_values_s, z)
    
    results = {
        'z': z,
        'mean_xHI': mean_xHI,
        'r_3d': r_3d,
        'r_2d': r_2d,
        'r_3d_jk_sigma': r_3d_jk_sigma,
        'r_2d_jk_sigma': r_2d_jk_sigma,
        'r_ksz_unsmooth': r_ksz_unsmooth,
        'r_ksz_smooth': r_ksz_smooth,
        'k_values': k_values,
        'r_k': r_k,
        'k_values_s': k_values_s,
        'r_k_s': r_k_s,
        'ell_values': ell_values,
        'ell_values_s': ell_values_s,
        'k_values_vel_2d': k_values_vel_2d,
        'r_k_vel_2d': r_k_vel_2d,
        'k_values_vel_3d': k_values_vel_3d,
        'r_k_vel_3d': r_k_vel_3d,
        'r_k_vel_2d_jk_sigma': r_k_vel_2d_jk_sigma,
        'r_k_vel_3d_jk_sigma': r_k_vel_3d_jk_sigma,
    }
    
    # Clean up
    del den, xhi, vz, vz_recx
    del ksz_map, ksz_map_rec, ksz_map_smooth, ksz_map_rec_smooth
    del vz_smooth, vz_recx_smooth
    gc.collect()
    
    return results

def main(redshifts=None):
    """Main analysis pipeline."""
    if redshifts is None:
        # All 51 redshifts with complete data (n_all.dat, v_all.dat, xhi.bin)
        redshifts = [
            6.056, 6.113, 6.172, 6.231, 6.292, 6.354, 6.418, 6.483, 6.549,
            6.617, 6.686, 6.757, 6.830, 6.905, 6.981, 7.059, 7.139, 7.221,
            7.305, 7.391, 7.480, 7.570, 7.664, 7.760, 7.859, 7.960, 8.064,
            8.172, 8.283, 8.397, 8.515, 8.636, 8.762, 8.892, 9.026, 9.164,
            9.308, 9.457, 9.611, 9.771, 9.938, 10.110, 10.290, 10.478,
            10.673, 10.877, 11.090, 11.313, 11.546, 11.791, 12.048
        ]
        # redshifts = [
        #     6.056
        # ]
    
    print("\n" + "="*80)
    print("kSZ RECONSTRUCTION ANALYSIS PIPELINE")
    print("="*80)
    print(f"Number of redshifts: {len(redshifts)}")
    print(f"Redshifts: {redshifts}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Physical normalization: {PHYSICAL_NORM} ({'µK units' if PHYSICAL_NORM else 'arbitrary units'})")
    print("="*80)
    
    # Analyze all redshifts
    all_results = []
    for z in redshifts:
        results = analyze_single_redshift(z)
        all_results.append(results)
    
    # Generate plots
    print("\n" + "="*80)
    print("GENERATING PLOTS")
    print("="*80)
    
    plot_velocity_correlation_vs_neutral_fraction(all_results)
    plot_velocity_correlation_combined(all_results)
    plot_velocity_correlation_jk(all_results)
    
    # 2D velocity scatter plot for a representative redshift (xHI ~ 0.5)
    target_xHI = 0.5
    best_idx = min(range(len(all_results)), key=lambda i: abs(all_results[i]['mean_xHI'] - target_xHI))
    plot_2d_velocity_scatter(all_results[best_idx]['z'])
    
    # Velocity and kSZ comparison (δ vs -δTb reconstruction) at two redshifts
    plot_velocity_and_ksz_comparison(z1=6.231, z2=7.570)
    
    # Compare velocity reconstruction with and without velocity term
    # Use xHI ~ 0.65 to match the reference figure
    target_xHI_compare = 0.65
    compare_idx = min(range(len(all_results)), key=lambda i: abs(all_results[i]['mean_xHI'] - target_xHI_compare))
    plot_velocity_term_comparison(all_results[compare_idx]['z'])
    
    # 4-panel comparison: 3D vs 2D velocity scatter at two neutral fractions
    target_xHI_1, target_xHI_2 = 0.75, 0.25
    idx1 = min(range(len(all_results)), key=lambda i: abs(all_results[i]['mean_xHI'] - target_xHI_1))
    idx2 = min(range(len(all_results)), key=lambda i: abs(all_results[i]['mean_xHI'] - target_xHI_2))
    plot_velocity_scatter_comparison(all_results[idx1]['z'], all_results[idx2]['z'])
    
    # plot_fourier_ksz_correlation(all_results)
    # plot_ksz_correlation_vs_neutral_fraction(all_results)
    # plot_ksz_scale_dependence_and_ell3000(all_results)
    # plot_ksz_power_paper_figure(all_results)
    cross_corr_results = analyze_stitched_full_ksz_vs_individual()
    if cross_corr_results is not None:
        plot_cross_correlator_evolution_jk(cross_corr_results)
    
    # =========================================================================
    # R(k,z) — ionisation-to-density ratio measurement
    # =========================================================================
    # Measures R(k,z) = P_{δx_HI, δ}(k) / [x̄_HI · P_{δδ}(k)]
    # and plots 1+R vs neutral fraction (with redshift colorbar)
    # plus R(k) vs k at selected redshifts.
    R_kz_results = measure_R_kz_all(redshifts)
    plot_one_plus_R_vs_neutral_fraction(R_kz_results)
    plot_R_kz_vs_k(R_kz_results)
    
    # =========================================================================
    # 21cm NORMALIZATION STUDY
    # =========================================================================
    # Run the 21cm brightness temperature normalization comparison
    # This compares three normalizations for linear continuity reconstruction:
    # (A) Raw δTb [mK] - baseline
    # (B) Mean-normalized - dimensionless, simulation-only
    # (C) Fixed-scale (10 mK) - dimensionless, observable-motivated
    #
    # Select a subset of redshifts spanning the EoR for this study
    study_redshifts = [6.231, 6.617, 7.059, 7.570, 8.064, 8.636, 9.308]
    norm_study_results = run_21cm_normalization_study(study_redshifts)
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\n{'z':<8} {'<xHI>':<10} {'r(3D)':<10} {'r(2D)':<10} {'r(kSZ)':<10}")
    print("-" * 50)
    for r in all_results:
        print(f"{r['z']:<8.3f} {r['mean_xHI']:<10.4f} {r['r_3d']:<10.4f} "
              f"{r['r_2d']:<10.4f} {r['r_ksz_unsmooth']:<10.4f}")
    print("="*80)
    
    print("\nAnalysis complete!")
    return all_results

if __name__ == "__main__":
    # Example usage:
    # For 3 redshifts (default)
    results = main()
    
    # For custom redshifts, uncomment and modify:
    # custom_redshifts = [6.483, 6.905, 7.570, 8.0, 8.5, ...]  # Add your 40 redshifts
    # results = main(redshifts=custom_redshifts)
    
    # To enable physical normalization (µK units), set PHYSICAL_NORM = True at the top of the file
    # This applies σ_T * dl_proper / c normalization to convert to temperature units

