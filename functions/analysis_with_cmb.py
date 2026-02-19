#!/usr/bin/env python3
"""
kSZ Reconstruction Analysis with CMB Anisotropies

This module generates flat-sky CMB temperature anisotropy maps using CAMB
and analyzes how CMB contamination affects kSZ reconstruction cross-correlations.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import gc
import camb
from powerbox import get_power

# Import from generate_all_plots
from generate_all_plots import (
    reconstruct_velocities, compute_ksz_maps, pearson_r,
    compute_fourier_correlation_coefficient, k_to_ell,
    BOX_MPC_OVER_H, LITTLEH, CENTRAL_CROP, LOS_CROP,
    K_MIN, K_MAX_PLOT, FOURIER_BINS, OUTPUT_DIR, PHYSICAL_NORM
)

# ============================================================================
# CONFIGURATION
# ============================================================================

# CMB-specific settings
CMB_LENSED = False  # Use lensed CMB power spectrum
CMB_SEED = 42  # Random seed for reproducibility

# Fiducial cosmology (should match CAMB and your simulations)
OMEGA_M0 = 0.27
OMEGA_L0 = 0.73
H0_KM_S_MPC = 70.0

# ============================================================================
# CMB GENERATION FUNCTIONS
# ============================================================================

def get_cmb_cl(lmax=10000, lensed=True):
    """
    Get CMB temperature power spectrum C_ℓ from CAMB.
    
    Parameters:
    -----------
    lmax : int
        Maximum multipole
    lensed : bool
        If True, return lensed C_ℓ; if False, return unlensed
        
    Returns:
    --------
    ells : ndarray
        Multipole values (0 to lmax)
    Cl_TT : ndarray
        Temperature power spectrum in µK² (C_ℓ, not D_ℓ)
    """
    # Set up CAMB parameters with fiducial cosmology
    pars = camb.set_params(
        H0=H0_KM_S_MPC,
        ombh2=0.022,
        omch2=0.12,
        tau=0.06,
        As=2.1e-9,
        ns=0.965,
        lmax=lmax,
        lens_potential_accuracy=1 if lensed else 0
    )
    
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK')
    
    # Get the appropriate spectrum
    if lensed:
        # 'total' includes lensing
        Cl_raw = powers['total'][:, 0]  # TT is first column
    else:
        # 'unlensed_scalar' is unlensed
        Cl_raw = powers['unlensed_scalar'][:, 0]
    
    # CAMB returns D_ℓ = ℓ(ℓ+1)C_ℓ/(2π), convert back to C_ℓ
    ells = np.arange(len(Cl_raw))
    Cl_TT = np.zeros_like(Cl_raw)
    Cl_TT[2:] = Cl_raw[2:] * 2 * np.pi / (ells[2:] * (ells[2:] + 1))
    Cl_TT[0] = 0  # Monopole
    Cl_TT[1] = 0  # Dipole
    
    return ells, Cl_TT


def compute_comoving_distance(z):
    """
    Compute comoving distance to redshift z in Mpc.
    
    Parameters:
    -----------
    z : float
        Redshift
        
    Returns:
    --------
    chi : float
        Comoving distance in Mpc
    """
    c_km_s = 299792.458  # Speed of light in km/s
    
    def E(zp):
        return np.sqrt(OMEGA_M0 * (1 + zp)**3 + OMEGA_L0)
    
    chi_integral, _ = quad(lambda zp: 1.0 / E(zp), 0, z)
    chi_Mpc = (c_km_s / H0_KM_S_MPC) * chi_integral
    
    return chi_Mpc


def generate_flat_sky_cmb(nx, ny, Lx_Mpc, Ly_Mpc, z_mean, lensed=True, seed=None):
    """
    Generate a flat-sky CMB temperature anisotropy map.
    
    Parameters:
    -----------
    nx, ny : int
        Number of pixels in x and y directions
    Lx_Mpc, Ly_Mpc : float
        Physical box size in Mpc (not Mpc/h)
    z_mean : float
        Mean redshift of the observation (for angular diameter distance)
    lensed : bool
        If True, use lensed C_ℓ; if False, use unlensed
    seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    cmb_map : ndarray (ny, nx)
        CMB temperature anisotropy map in µK
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Compute comoving distance to z_mean
    chi_Mpc = compute_comoving_distance(z_mean)
    
    # Angular size of the box in radians
    theta_x = Lx_Mpc / chi_Mpc  # radians
    theta_y = Ly_Mpc / chi_Mpc  # radians
    
    print(f"  CMB generation: χ(z={z_mean:.2f}) = {chi_Mpc:.1f} Mpc")
    print(f"  Angular box size: {np.degrees(theta_x):.2f}° x {np.degrees(theta_y):.2f}°")
    
    # Pixel size in radians
    dx_rad = theta_x / nx
    dy_rad = theta_y / ny
    
    # Get C_ℓ from CAMB
    # Maximum ℓ needed: ℓ_max ~ π / min(dx_rad, dy_rad)
    ell_max = int(np.pi / min(dx_rad, dy_rad)) + 100
    ell_max = min(ell_max, 15000)  # Cap at reasonable value
    print(f"  Using ℓ_max = {ell_max}, lensed = {lensed}")
    
    ells, Cl_TT = get_cmb_cl(lmax=ell_max, lensed=lensed)
    
    # Create 2D k-space grid (in 1/radian units, which equals ℓ)
    # For flat sky: k = ℓ (when distances are in radians)
    # Use fftfreq with pixel size in radians
    kx = 2 * np.pi * np.fft.fftfreq(ny, d=dy_rad)
    ky = 2 * np.pi * np.fft.fftfreq(nx, d=dx_rad)
    KX, KY = np.meshgrid(ky, kx)  # Note: meshgrid order for correct array shape
    ell_2d = np.sqrt(KX**2 + KY**2)
    
    # Interpolate C_ℓ to the ℓ values on our grid
    Cl_2d = np.interp(ell_2d, ells, Cl_TT, left=0, right=0)
    
    # Generate Gaussian random field in Fourier space
    # Area in steradians
    area_sr = theta_x * theta_y
    
    # Power spectrum amplitude for FFT normalization
    # This normalization ensures the measured power spectrum matches the input C_ℓ
    amplitude = np.sqrt(Cl_2d * (2 * np.pi)**2 / area_sr / 2)
    
    # Generate complex Gaussian random field
    real_part = np.random.randn(ny, nx)
    imag_part = np.random.randn(ny, nx)
    fft_field = amplitude * (real_part + 1j * imag_part)
    
    # Take real part of inverse FFT
    cmb_map = np.real(np.fft.ifft2(fft_field)) * nx * ny
    
    # Enforce zero mean
    cmb_map -= cmb_map.mean()
    
    print(f"  CMB map: mean={cmb_map.mean():.2e} µK, std={cmb_map.std():.1f} µK")
    
    return cmb_map.astype(np.float32)


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def analyze_ksz_with_cmb(ksz_map_real, ksz_map_rec, Lx_Mpc, Ly_Mpc, z_mean,
                         lensed=True, seed=42):
    """
    Add CMB anisotropies to kSZ map and compute cross-correlations.
    
    Parameters:
    -----------
    ksz_map_real : ndarray (2D)
        Real kSZ map in µK
    ksz_map_rec : ndarray (2D)
        Reconstructed kSZ map in µK
    Lx_Mpc, Ly_Mpc : float
        Physical box size in Mpc (not Mpc/h)
    z_mean : float
        Mean redshift
    lensed : bool
        Use lensed (True) or unlensed (False) CMB
    seed : int
        Random seed for CMB realization
        
    Returns:
    --------
    results : dict
        Contains CMB map, observed map, and correlation statistics
    """
    ny, nx = ksz_map_real.shape
    
    print(f"\n  Generating {'lensed' if lensed else 'unlensed'} CMB realization...")
    cmb_map = generate_flat_sky_cmb(nx, ny, Lx_Mpc, Ly_Mpc, z_mean, 
                                     lensed=lensed, seed=seed)
    
    # "Observed" map = kSZ + CMB
    observed_map = ksz_map_real + cmb_map
    
    # Box length in Mpc/h for Fourier analysis
    boxlength = [Ly_Mpc * LITTLEH, Lx_Mpc * LITTLEH]
    
    # Correlations without CMB (baseline)
    r_no_cmb = pearson_r(ksz_map_real, ksz_map_rec)
    k_vals, r_k_no_cmb = compute_fourier_correlation_coefficient(
        ksz_map_real, ksz_map_rec, boxlength=boxlength
    )
    
    # Correlations with CMB
    r_with_cmb = pearson_r(observed_map, ksz_map_rec)
    _, r_k_with_cmb = compute_fourier_correlation_coefficient(
        observed_map, ksz_map_rec, boxlength=boxlength
    )
    
    print(f"  Real-space correlation: without CMB = {r_no_cmb:.4f}, with CMB = {r_with_cmb:.4f}")
    print(f"  kSZ std = {ksz_map_real.std():.2f} µK, CMB std = {cmb_map.std():.2f} µK")
    print(f"  Signal-to-noise (kSZ/CMB) = {ksz_map_real.std() / cmb_map.std():.3f}")
    
    return {
        'cmb_map': cmb_map,
        'observed_map': observed_map,
        'r_no_cmb': r_no_cmb,
        'r_with_cmb': r_with_cmb,
        'k_values': k_vals,
        'r_k_no_cmb': r_k_no_cmb,
        'r_k_with_cmb': r_k_with_cmb,
        'ksz_std': ksz_map_real.std(),
        'cmb_std': cmb_map.std(),
        'lensed': lensed,
        'z_mean': z_mean
    }


def plot_cmb_analysis(cmb_results, ksz_map_real, ksz_map_rec, z_info,
                      output_dir=OUTPUT_DIR):
    """
    Plot CMB analysis results.
    
    Parameters:
    -----------
    cmb_results : dict
        Output from analyze_ksz_with_cmb
    ksz_map_real : ndarray
        Real kSZ map
    ksz_map_rec : ndarray
        Reconstructed kSZ map
    z_info : str
        Redshift info string for title
    output_dir : str
        Output directory for plots
    """
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    cmb_map = cmb_results['cmb_map']
    observed_map = cmb_results['observed_map']
    
    # Determine color scale for kSZ maps
    vmax_ksz = max(abs(ksz_map_real.min()), abs(ksz_map_real.max()))
    vmax_cmb = max(abs(cmb_map.min()), abs(cmb_map.max()))
    vmax_obs = max(abs(observed_map.min()), abs(observed_map.max()))
    
    # Row 1: Maps
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(ksz_map_real, origin='lower', cmap='RdBu_r', 
                   vmin=-vmax_ksz, vmax=vmax_ksz)
    ax.set_title(f'Real kSZ (σ={ksz_map_real.std():.1f} µK)', fontweight='bold')
    plt.colorbar(im, ax=ax, label='µK')
    
    ax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(cmb_map, origin='lower', cmap='RdBu_r',
                   vmin=-vmax_cmb, vmax=vmax_cmb)
    lensed_str = 'lensed' if cmb_results['lensed'] else 'unlensed'
    ax.set_title(f'CMB ({lensed_str}, σ={cmb_map.std():.1f} µK)', fontweight='bold')
    plt.colorbar(im, ax=ax, label='µK')
    
    ax = fig.add_subplot(gs[0, 2])
    im = ax.imshow(observed_map, origin='lower', cmap='RdBu_r',
                   vmin=-vmax_obs, vmax=vmax_obs)
    ax.set_title(f'Observed (kSZ + CMB)', fontweight='bold')
    plt.colorbar(im, ax=ax, label='µK')
    
    # Row 2: Reconstructed and scatter plots
    ax = fig.add_subplot(gs[1, 0])
    vmax_rec = max(abs(ksz_map_rec.min()), abs(ksz_map_rec.max()))
    im = ax.imshow(ksz_map_rec, origin='lower', cmap='RdBu_r',
                   vmin=-vmax_rec, vmax=vmax_rec)
    ax.set_title(f'Reconstructed kSZ', fontweight='bold')
    plt.colorbar(im, ax=ax, label='µK')
    
    # Scatter: Real vs Reconstructed (no CMB)
    ax = fig.add_subplot(gs[1, 1])
    n_sample = 5000
    rng = np.random.RandomState(42)
    idx = rng.choice(ksz_map_real.size, size=min(n_sample, ksz_map_real.size), replace=False)
    ax.scatter(ksz_map_real.ravel()[idx], ksz_map_rec.ravel()[idx], 
               s=1, alpha=0.3, c='blue')
    ax.set_xlabel('Real kSZ [µK]')
    ax.set_ylabel('Reconstructed kSZ [µK]')
    ax.set_title(f'No CMB: r = {cmb_results["r_no_cmb"]:.4f}', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    # Scatter: Observed vs Reconstructed (with CMB)
    ax = fig.add_subplot(gs[1, 2])
    ax.scatter(observed_map.ravel()[idx], ksz_map_rec.ravel()[idx],
               s=1, alpha=0.3, c='red')
    ax.set_xlabel('Observed (kSZ + CMB) [µK]')
    ax.set_ylabel('Reconstructed kSZ [µK]')
    ax.set_title(f'With CMB: r = {cmb_results["r_with_cmb"]:.4f}', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    # Row 3: Fourier correlations
    ax = fig.add_subplot(gs[2, :])
    k_vals = cmb_results['k_values']
    k_mask = k_vals <= K_MAX_PLOT
    
    ax.plot(k_vals[k_mask], cmb_results['r_k_no_cmb'][k_mask], 
            'b-', linewidth=2, label=f'Without CMB (mean r={np.nanmean(cmb_results["r_k_no_cmb"][k_mask]):.3f})')
    ax.plot(k_vals[k_mask], cmb_results['r_k_with_cmb'][k_mask],
            'r-', linewidth=2, label=f'With CMB (mean r={np.nanmean(cmb_results["r_k_with_cmb"][k_mask]):.3f})')
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax.axhline(y=1, color='gray', linestyle='--', linewidth=1, alpha=0.3)
    ax.set_xlabel('k [h/Mpc]', fontsize=14)
    ax.set_ylabel('Correlation coefficient r(k)', fontsize=14)
    ax.set_title('Fourier-space correlation: kSZ reconstruction vs real/observed', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.2, 1.1])
    ax.set_xlim([0, K_MAX_PLOT])
    ax.legend(fontsize=12, loc='lower right')
    
    plt.suptitle(f'kSZ Reconstruction with CMB Contamination\n{z_info}',
                 fontsize=16, fontweight='bold')
    
    outfile = os.path.join(output_dir, 'ksz_with_cmb_analysis.png')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outfile}")


def run_cmb_analysis(redshifts=None, lensed=True, seed=42):
    """
    Run full CMB contamination analysis on stitched kSZ maps.
    
    Parameters:
    -----------
    redshifts : list, optional
        List of redshifts to use. If None, uses a subset of available redshifts.
    lensed : bool
        Use lensed (True) or unlensed (False) CMB
    seed : int
        Random seed for CMB realization
    """
    if redshifts is None:
        # Use a subset of redshifts for efficiency
        redshifts = [6.113, 6.757, 7.305, 7.859, 8.515]
    
    print("\n" + "="*80)
    print("kSZ RECONSTRUCTION ANALYSIS WITH CMB CONTAMINATION")
    print("="*80)
    print(f"Redshifts: {redshifts}")
    print(f"CMB: {'lensed' if lensed else 'unlensed'}")
    print("="*80)
    
    # Initialize accumulated kSZ maps (will sum 2D maps instead of stitching 3D)
    ksz_map_real = None
    ksz_map_rec = None
    individual_results = []  # Store per-redshift reconstructions
    n_full = None
    
    print("\nComputing kSZ maps per redshift and summing (memory efficient)...")
    
    for z in redshifts:
        print(f"\nLoading z={z}...")
        den, xhi, vx, vy, vz, vx_rec, vy_rec, vz_rec, vx_recx, vy_recx, vz_recx = \
            reconstruct_velocities(z)
        
        if n_full is None:
            n_full = vz.shape[0]
        
        # Apply central crop in all 3 directions
        den = den[CENTRAL_CROP, CENTRAL_CROP, LOS_CROP]
        xhi = xhi[CENTRAL_CROP, CENTRAL_CROP, LOS_CROP]
        vz = vz[CENTRAL_CROP, CENTRAL_CROP, LOS_CROP]
        vz_recx = vz_recx[CENTRAL_CROP, CENTRAL_CROP, LOS_CROP]
        
        mean_xhi = xhi.mean()
        print(f"  Loaded z={z}, shape: {vz.shape}, mean xHI: {mean_xhi:.4f}")
        
        # Compute kSZ maps for this redshift (physical units µK)
        ksz_real_z = compute_ksz_maps(vz, xhi, den, z=z, physical_norm=True)
        ksz_rec_z = compute_ksz_maps(-vz_recx, xhi, den, z=z, physical_norm=True)
        
        # Store per-redshift reconstruction
        individual_results.append({
            'z': z,
            'mean_xhi': mean_xhi,
            'ksz_rec': ksz_rec_z.copy(),
        })
        
        # Accumulate (sum 2D maps)
        if ksz_map_real is None:
            ksz_map_real = ksz_real_z.copy()
            ksz_map_rec = ksz_rec_z.copy()
        else:
            ksz_map_real += ksz_real_z
            ksz_map_rec += ksz_rec_z
        
        print(f"  kSZ contribution: real std={ksz_real_z.std():.4f} µK, rec std={ksz_rec_z.std():.4f} µK")
        
        # Clean up - only keep 2D maps, free 3D arrays
        del den, xhi, vz, vz_recx, ksz_real_z, ksz_rec_z
        del vx, vy, vx_rec, vy_rec, vz_rec, vx_recx, vy_recx
        gc.collect()
    
    z_mean = np.mean(redshifts)
    
    print(f"\nTotal kSZ maps (summed over {len(redshifts)} redshifts):")
    print(f"  Real kSZ map: mean={ksz_map_real.mean():.2e}, std={ksz_map_real.std():.4f} µK")
    print(f"  Reconstructed kSZ map: mean={ksz_map_rec.mean():.2e}, std={ksz_map_rec.std():.4f} µK")
    
    # Physical box size in Mpc (not Mpc/h)
    n_crop = CENTRAL_CROP.stop - CENTRAL_CROP.start
    dx = BOX_MPC_OVER_H / n_full
    Lx_Mpc_h = dx * n_crop  # Mpc/h
    Lx_Mpc = Lx_Mpc_h / LITTLEH  # Mpc
    Ly_Mpc = Lx_Mpc  # Square box
    
    print(f"\nBox size: {Lx_Mpc:.1f} Mpc x {Ly_Mpc:.1f} Mpc")
    
    # Box length in Mpc/h for Fourier analysis
    boxlength = [Ly_Mpc * LITTLEH, Lx_Mpc * LITTLEH]
    
    # =========================================================================
    # Generate CMB realization
    # =========================================================================
    print("\n" + "="*80)
    print("GENERATING CMB REALIZATION")
    print("="*80)
    
    cmb_map = generate_flat_sky_cmb(
        ksz_map_real.shape[1], ksz_map_real.shape[0],
        Lx_Mpc, Ly_Mpc, z_mean,
        lensed=lensed, seed=seed
    )
    
    # "Observed" map = integrated real kSZ + CMB
    observed_map = ksz_map_real + cmb_map
    
    print(f"\nkSZ std = {ksz_map_real.std():.4f} µK, CMB std = {cmb_map.std():.2f} µK")
    print(f"Signal-to-noise (kSZ/CMB) = {ksz_map_real.std() / cmb_map.std():.6f}")
    
    # =========================================================================
    # ANALYSIS 1: Per-redshift reconstruction vs integrated real kSZ (with CMB)
    # =========================================================================
    print("\n" + "="*80)
    print("ANALYSIS 1: Per-redshift reconstruction vs integrated real kSZ")
    print("="*80)
    
    per_z_results = []
    for result in individual_results:
        z = result['z']
        mean_xhi = result['mean_xhi']
        ksz_rec_z = result['ksz_rec']
        
        # Without CMB
        r_no_cmb = pearson_r(ksz_map_real, ksz_rec_z)
        k_vals, r_k_no_cmb = compute_fourier_correlation_coefficient(
            ksz_map_real, ksz_rec_z, boxlength=boxlength
        )
        
        # With CMB
        r_with_cmb = pearson_r(observed_map, ksz_rec_z)
        _, r_k_with_cmb = compute_fourier_correlation_coefficient(
            observed_map, ksz_rec_z, boxlength=boxlength
        )
        
        per_z_results.append({
            'z': z,
            'mean_xhi': mean_xhi,
            'r_no_cmb': r_no_cmb,
            'r_with_cmb': r_with_cmb,
            'k_values': k_vals,
            'r_k_no_cmb': r_k_no_cmb,
            'r_k_with_cmb': r_k_with_cmb,
        })
        
        print(f"  z={z:.3f}, xHI={mean_xhi:.3f}: r(no CMB)={r_no_cmb:.4f}, r(with CMB)={r_with_cmb:.4f}")
    
    # =========================================================================
    # ANALYSIS 2: Integrated reconstruction vs integrated real kSZ (with CMB)
    # =========================================================================
    print("\n" + "="*80)
    print("ANALYSIS 2: Integrated reconstruction vs integrated real kSZ")
    print("="*80)
    
    # Without CMB
    r_int_no_cmb = pearson_r(ksz_map_real, ksz_map_rec)
    k_vals_int, r_k_int_no_cmb = compute_fourier_correlation_coefficient(
        ksz_map_real, ksz_map_rec, boxlength=boxlength
    )
    
    # With CMB
    r_int_with_cmb = pearson_r(observed_map, ksz_map_rec)
    _, r_k_int_with_cmb = compute_fourier_correlation_coefficient(
        observed_map, ksz_map_rec, boxlength=boxlength
    )
    
    print(f"  Without CMB: r = {r_int_no_cmb:.4f}, mean r(k) = {np.nanmean(r_k_int_no_cmb):.4f}")
    print(f"  With CMB:    r = {r_int_with_cmb:.4f}, mean r(k) = {np.nanmean(r_k_int_with_cmb):.4f}")
    
    integrated_results = {
        'r_no_cmb': r_int_no_cmb,
        'r_with_cmb': r_int_with_cmb,
        'k_values': k_vals_int,
        'r_k_no_cmb': r_k_int_no_cmb,
        'r_k_with_cmb': r_k_int_with_cmb,
    }
    
    # =========================================================================
    # Plotting
    # =========================================================================
    z_info = f"z = {min(redshifts):.2f} - {max(redshifts):.2f}"
    lensed_str = "lensed" if lensed else "unlensed"
    
    # Store results for plotting function
    cmb_results = {
        'cmb_map': cmb_map,
        'observed_map': observed_map,
        'r_no_cmb': r_int_no_cmb,
        'r_with_cmb': r_int_with_cmb,
        'k_values': k_vals_int,
        'r_k_no_cmb': r_k_int_no_cmb,
        'r_k_with_cmb': r_k_int_with_cmb,
        'ksz_std': ksz_map_real.std(),
        'cmb_std': cmb_map.std(),
        'lensed': lensed,
        'z_mean': z_mean
    }
    
    # -------------------------------------------------------------------------
    # PLOT 1: Main CMB analysis (maps + correlations)
    # -------------------------------------------------------------------------
    plot_cmb_analysis(cmb_results, ksz_map_real, ksz_map_rec, z_info)
    
    # -------------------------------------------------------------------------
    # PLOT 2: Correlation vs neutral fraction (with and without CMB)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    xhi_vals = [r['mean_xhi'] for r in per_z_results]
    r_no_cmb_vals = [r['r_no_cmb'] for r in per_z_results]
    r_with_cmb_vals = [r['r_with_cmb'] for r in per_z_results]
    
    ax = axes[0]
    ax.plot(xhi_vals, r_no_cmb_vals, 'bo-', markersize=10, linewidth=2, label='Without CMB')
    ax.plot(xhi_vals, r_with_cmb_vals, 'rs-', markersize=10, linewidth=2, label='With CMB')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Neutral Fraction $\\langle x_{HI} \\rangle$', fontsize=14)
    ax.set_ylabel('Correlation coefficient r', fontsize=14)
    ax.set_title('Per-z Reconstruction vs Integrated Real kSZ', fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.5, 1.0])
    
    # Add integrated result as horizontal lines
    ax = axes[1]
    ax.axhline(y=r_int_no_cmb, color='blue', linestyle='-', linewidth=2, label=f'Without CMB: r={r_int_no_cmb:.3f}')
    ax.axhline(y=r_int_with_cmb, color='red', linestyle='-', linewidth=2, label=f'With CMB: r={r_int_with_cmb:.3f}')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Neutral Fraction $\\langle x_{HI} \\rangle$', fontsize=14)
    ax.set_ylabel('Correlation coefficient r', fontsize=14)
    ax.set_title('Integrated Reconstruction vs Integrated Real kSZ', fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.5, 1.0])
    ax.set_xlim([0, 1])
    
    plt.suptitle(f'kSZ Reconstruction Correlation ({lensed_str} CMB)\n{z_info}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    outfile = os.path.join(OUTPUT_DIR, 'ksz_correlation_vs_xhi_with_cmb.png')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outfile}")
    
    # -------------------------------------------------------------------------
    # PLOT 3: Fourier correlation comparison (per-z and integrated)
    # -------------------------------------------------------------------------
    k_max_plot = 1.2  # Updated cutoff
    
    # Select 6 redshifts based on neutral fraction coverage
    xhi_targets = [0.1, 0.3, 0.5, 0.7, 0.85, 0.95]  # Target neutral fractions
    selected_indices = []
    for xhi_target in xhi_targets:
        xhi_vals = np.array([r['mean_xhi'] for r in per_z_results])
        idx = np.argmin(np.abs(xhi_vals - xhi_target))
        if idx not in selected_indices:
            selected_indices.append(idx)
    # Ensure we have 6 unique indices
    if len(selected_indices) < 6:
        for idx in np.linspace(0, len(per_z_results)-1, 6, dtype=int):
            if idx not in selected_indices:
                selected_indices.append(idx)
            if len(selected_indices) >= 6:
                break
    selected_indices = sorted(selected_indices[:6])
    selected_per_z = [per_z_results[i] for i in selected_indices]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Colors for selected per-z results
    n_selected = len(selected_per_z)
    cmap = plt.cm.viridis
    colors = [cmap(i / max(n_selected - 1, 1)) for i in range(n_selected)]
    
    # Left: Per-z Fourier correlations (without CMB) - 6 selected redshifts
    ax = axes[0]
    for i, res in enumerate(selected_per_z):
        k_mask = res['k_values'] <= k_max_plot
        ax.plot(res['k_values'][k_mask], res['r_k_no_cmb'][k_mask],
                color=colors[i], linewidth=2, linestyle='-',
                label=f"z={res['z']:.2f}, xHI={res['mean_xhi']:.2f}")
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax.set_xlabel('k [h/Mpc]', fontsize=14)
    ax.set_ylabel('r(k)', fontsize=14)
    ax.set_title('Per-z Reconstruction (no CMB)', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-1.01, 1.01])
    ax.set_xlim([0, k_max_plot])
    ax.legend(fontsize=10, loc='lower right')
    
    # Right: Integrated Fourier correlations (with and without CMB)
    ax = axes[1]
    k_mask = integrated_results['k_values'] <= k_max_plot
    ax.plot(integrated_results['k_values'][k_mask], integrated_results['r_k_no_cmb'][k_mask],
            'b-', linewidth=2.5, label=f"Without CMB (mean={np.nanmean(integrated_results['r_k_no_cmb'][k_mask]):.3f})")
    ax.plot(integrated_results['k_values'][k_mask], integrated_results['r_k_with_cmb'][k_mask],
            'r-', linewidth=2.5, label=f"With CMB (mean={np.nanmean(integrated_results['r_k_with_cmb'][k_mask]):.3f})")
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax.set_xlabel('k [h/Mpc]', fontsize=14)
    ax.set_ylabel('r(k)', fontsize=14)
    ax.set_title('Integrated Reconstruction', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-1.01, 1.01])
    ax.set_xlim([0, k_max_plot])
    ax.legend(fontsize=12, loc='lower right')
    
    plt.suptitle(f'Fourier-Space Correlation r(k) ({lensed_str} CMB)\n{z_info}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    outfile = os.path.join(OUTPUT_DIR, 'fourier_correlation_with_cmb.png')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outfile}")
    
    # -------------------------------------------------------------------------
    # PLOT 3b: Per-z r(k) for 6 representative redshifts (with and without CMB)
    # -------------------------------------------------------------------------
    # Uses selected_indices from PLOT 3 above
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for plot_idx, res_idx in enumerate(selected_indices):
        res = per_z_results[res_idx]
        ax = axes[plot_idx]
        
        k_mask = res['k_values'] <= k_max_plot
        k_vals = res['k_values'][k_mask]
        
        # Plot without CMB
        ax.plot(k_vals, res['r_k_no_cmb'][k_mask], 'b-', linewidth=2, 
                label=f'No CMB (mean={np.nanmean(res["r_k_no_cmb"][k_mask]):.3f})')
        
        # Plot with CMB
        ax.plot(k_vals, res['r_k_with_cmb'][k_mask], 'r--', linewidth=2,
                label=f'With CMB (mean={np.nanmean(res["r_k_with_cmb"][k_mask]):.3f})')
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
        ax.set_xlabel('k [h/Mpc]', fontsize=12)
        ax.set_ylabel('r(k)', fontsize=12)
        ax.set_title(f'z = {res["z"]:.2f}, $\\langle x_{{HI}} \\rangle$ = {res["mean_xhi"]:.2f}', 
                     fontweight='bold', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-1.01, 1.01])
        ax.set_xlim([0, k_max_plot])
        ax.legend(fontsize=9, loc='lower right')
    
    plt.suptitle(f'Per-z Fourier Correlation r(k): Per-z Rec vs Integrated Real kSZ\n'
                 f'({lensed_str} CMB, {z_info})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    outfile = os.path.join(OUTPUT_DIR, 'per_z_fourier_correlation_6panels.png')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outfile}")
    
    # -------------------------------------------------------------------------
    # PLOT 4: Power spectra comparison (kSZ vs CMB)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Compute power spectra
    from powerbox import get_power
    
    # kSZ power spectrum
    p_ksz, k_ksz = get_power(ksz_map_real, boxlength[0], bins=50)
    p_rec, k_rec = get_power(ksz_map_rec, boxlength[0], bins=50)
    p_cmb, k_cmb = get_power(cmb_map, boxlength[0], bins=50)
    p_obs, k_obs = get_power(observed_map, boxlength[0], bins=50)
    
    ax = axes[0]
    k_mask = (k_ksz >= 0) & (k_ksz <= k_max_plot)
    ax.semilogy(k_ksz[k_mask], p_ksz[k_mask], 'b-', linewidth=2, label='Real kSZ')
    ax.semilogy(k_rec[k_mask], p_rec[k_mask], 'g--', linewidth=2, label='Reconstructed kSZ')
    ax.semilogy(k_cmb[k_mask], p_cmb[k_mask], 'r-', linewidth=2, label=f'CMB ({lensed_str})')
    ax.semilogy(k_obs[k_mask], p_obs[k_mask], 'k:', linewidth=2, label='Observed (kSZ+CMB)')
    ax.set_xlabel('k [h/Mpc]', fontsize=14)
    ax.set_ylabel('P(k) [µK² (Mpc/h)²]', fontsize=14)
    ax.set_title('Power Spectra Comparison', fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, k_max_plot])
    
    # Signal-to-noise ratio vs k
    ax = axes[1]
    # S/N = P_kSZ / P_CMB
    k_common = k_ksz[1:]  # Skip k=0
    p_ksz_interp = np.interp(k_common, k_ksz, p_ksz)
    p_cmb_interp = np.interp(k_common, k_cmb, p_cmb)
    snr = np.sqrt(p_ksz_interp / p_cmb_interp)
    
    k_mask_snr = (k_common >= 0) & (k_common <= k_max_plot)
    ax.plot(k_common[k_mask_snr], snr[k_mask_snr], 'b-', linewidth=2)
    ax.axhline(y=1, color='red', linestyle='--', linewidth=1, label='S/N = 1')
    ax.axhline(y=0.1, color='orange', linestyle='--', linewidth=1, label='S/N = 0.1')
    ax.set_xlabel('k [h/Mpc]', fontsize=14)
    ax.set_ylabel('Signal-to-Noise (kSZ/CMB)', fontsize=14)
    ax.set_title('kSZ Signal-to-Noise vs Scale', fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, k_max_plot])
    ax.set_ylim([0, max(0.5, snr[k_mask_snr].max() * 1.1)])
    
    plt.suptitle(f'Power Spectrum Analysis ({lensed_str} CMB)\n{z_info}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    outfile = os.path.join(OUTPUT_DIR, 'power_spectra_ksz_cmb.png')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outfile}")
    
    # -------------------------------------------------------------------------
    # PLOT 5: Histograms of pixel values
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    ax = axes[0]
    ax.hist(ksz_map_real.ravel(), bins=100, alpha=0.7, color='blue', label=f'Real kSZ (σ={ksz_map_real.std():.2f} µK)')
    ax.hist(ksz_map_rec.ravel(), bins=100, alpha=0.7, color='green', label=f'Rec kSZ (σ={ksz_map_rec.std():.2f} µK)')
    ax.set_xlabel('Temperature [µK]', fontsize=12)
    ax.set_ylabel('Pixel count', fontsize=12)
    ax.set_title('kSZ Maps', fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_xlim([-50, 50])
    
    ax = axes[1]
    ax.hist(cmb_map.ravel(), bins=100, alpha=0.7, color='red', label=f'CMB (σ={cmb_map.std():.1f} µK)')
    ax.set_xlabel('Temperature [µK]', fontsize=12)
    ax.set_ylabel('Pixel count', fontsize=12)
    ax.set_title(f'CMB ({lensed_str})', fontweight='bold')
    ax.legend(fontsize=10)
    
    ax = axes[2]
    ax.hist(observed_map.ravel(), bins=100, alpha=0.7, color='purple', label=f'Observed (σ={observed_map.std():.1f} µK)')
    ax.set_xlabel('Temperature [µK]', fontsize=12)
    ax.set_ylabel('Pixel count', fontsize=12)
    ax.set_title('Observed (kSZ + CMB)', fontweight='bold')
    ax.legend(fontsize=10)
    
    plt.suptitle(f'Pixel Value Distributions\n{z_info}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    outfile = os.path.join(OUTPUT_DIR, 'pixel_histograms_ksz_cmb.png')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outfile}")
    
    # -------------------------------------------------------------------------
    # PLOT 6: All maps side by side
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Determine color scales
    vmax_ksz = np.percentile(np.abs(ksz_map_real), 99)
    vmax_cmb = np.percentile(np.abs(cmb_map), 99)
    
    # Row 1: kSZ maps
    ax = axes[0, 0]
    im = ax.imshow(ksz_map_real, origin='lower', cmap='RdBu_r', vmin=-vmax_ksz, vmax=vmax_ksz)
    ax.set_title(f'Real kSZ (σ={ksz_map_real.std():.2f} µK)', fontweight='bold')
    plt.colorbar(im, ax=ax, label='µK')
    
    ax = axes[0, 1]
    im = ax.imshow(ksz_map_rec, origin='lower', cmap='RdBu_r', vmin=-vmax_ksz, vmax=vmax_ksz)
    ax.set_title(f'Reconstructed kSZ (σ={ksz_map_rec.std():.2f} µK)', fontweight='bold')
    plt.colorbar(im, ax=ax, label='µK')
    
    ax = axes[0, 2]
    im = ax.imshow(ksz_map_real - ksz_map_rec, origin='lower', cmap='RdBu_r')
    ax.set_title('Residual (Real - Rec)', fontweight='bold')
    plt.colorbar(im, ax=ax, label='µK')
    
    # Row 2: CMB and observed
    ax = axes[1, 0]
    im = ax.imshow(cmb_map, origin='lower', cmap='RdBu_r', vmin=-vmax_cmb, vmax=vmax_cmb)
    ax.set_title(f'CMB ({lensed_str}, σ={cmb_map.std():.1f} µK)', fontweight='bold')
    plt.colorbar(im, ax=ax, label='µK')
    
    ax = axes[1, 1]
    im = ax.imshow(observed_map, origin='lower', cmap='RdBu_r', vmin=-vmax_cmb, vmax=vmax_cmb)
    ax.set_title(f'Observed (kSZ+CMB)', fontweight='bold')
    plt.colorbar(im, ax=ax, label='µK')
    
    ax = axes[1, 2]
    # Cross-correlation map (local correlation)
    im = ax.imshow(ksz_map_real * ksz_map_rec, origin='lower', cmap='RdBu_r')
    ax.set_title('Real × Rec (correlation map)', fontweight='bold')
    plt.colorbar(im, ax=ax, label='µK²')
    
    plt.suptitle(f'kSZ and CMB Maps Comparison\n{z_info}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    outfile = os.path.join(OUTPUT_DIR, 'all_maps_ksz_cmb.png')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outfile}")
    
    # -------------------------------------------------------------------------
    # PLOT 7: Cross-Power Spectra
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Compute auto and cross power spectra using powerbox
    # get_power with two fields computes cross-power
    P_real, k_real = get_power(ksz_map_real, boxlength[0], bins=50)
    P_rec, k_rec = get_power(ksz_map_rec, boxlength[0], bins=50)
    P_cmb, k_cmb = get_power(cmb_map, boxlength[0], bins=50)
    P_obs, k_obs = get_power(observed_map, boxlength[0], bins=50)
    
    # Cross-power spectra
    P_cross_real_rec, k_cross = get_power(ksz_map_real, boxlength[0], bins=50, 
                                           deltax2=ksz_map_rec)
    P_cross_obs_rec, _ = get_power(observed_map, boxlength[0], bins=50,
                                    deltax2=ksz_map_rec)
    P_cross_real_cmb, _ = get_power(ksz_map_real, boxlength[0], bins=50,
                                     deltax2=cmb_map)
    
    # Per-z cross-power spectra with integrated real kSZ
    per_z_cross_powers = []
    for i, res in enumerate(per_z_results):
        ksz_rec_z = individual_results[i]['ksz_rec']
        P_cross_z, k_z = get_power(ksz_map_real, boxlength[0], bins=50,
                                    deltax2=ksz_rec_z)
        per_z_cross_powers.append({
            'z': res['z'],
            'mean_xhi': res['mean_xhi'],
            'k': k_z,
            'P_cross': P_cross_z
        })
    
    # Panel (0,0): Auto-power spectra
    ax = axes[0, 0]
    k_mask = (k_real >= 0) & (k_real <= k_max_plot)
    ax.semilogy(k_real[k_mask], P_real[k_mask], 'b-', linewidth=2, label='Real kSZ')
    ax.semilogy(k_rec[k_mask], P_rec[k_mask], 'g--', linewidth=2, label='Reconstructed kSZ')
    ax.semilogy(k_cmb[k_mask], P_cmb[k_mask], 'r-', linewidth=2, label=f'CMB ({lensed_str})')
    ax.semilogy(k_obs[k_mask], P_obs[k_mask], 'k:', linewidth=2, label='Observed (kSZ+CMB)')
    ax.set_xlabel('k [h/Mpc]', fontsize=12)
    ax.set_ylabel('P(k) [µK² (Mpc/h)²]', fontsize=12)
    ax.set_title('Auto-Power Spectra', fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, k_max_plot])
    
    # Panel (0,1): Cross-power spectra (integrated)
    ax = axes[0, 1]
    k_mask_cross = (k_cross >= 0) & (k_cross <= k_max_plot)
    ax.plot(k_cross[k_mask_cross], P_cross_real_rec[k_mask_cross], 'b-', linewidth=2, 
            label='Real × Rec (no CMB)')
    ax.plot(k_cross[k_mask_cross], P_cross_obs_rec[k_mask_cross], 'r--', linewidth=2,
            label='Observed × Rec (with CMB)')
    ax.plot(k_cross[k_mask_cross], P_cross_real_cmb[k_mask_cross], 'gray', linewidth=1.5,
            linestyle=':', label='Real × CMB (should be ~0)')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.set_xlabel('k [h/Mpc]', fontsize=12)
    ax.set_ylabel('P_cross(k) [µK² (Mpc/h)²]', fontsize=12)
    ax.set_title('Cross-Power Spectra (Integrated)', fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, k_max_plot])
    
    # Panel (1,0): Per-z cross-power spectra
    ax = axes[1, 0]
    n_z = len(per_z_cross_powers)
    cmap_colors = plt.cm.viridis
    colors = [cmap_colors(i / max(n_z - 1, 1)) for i in range(n_z)]
    
    for i, pz in enumerate(per_z_cross_powers):
        k_mask_z = (pz['k'] >= 0) & (pz['k'] <= k_max_plot)
        ax.plot(pz['k'][k_mask_z], pz['P_cross'][k_mask_z], 
                color=colors[i], linewidth=1.5,
                label=f"z={pz['z']:.2f}, xHI={pz['mean_xhi']:.2f}")
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.set_xlabel('k [h/Mpc]', fontsize=12)
    ax.set_ylabel('P_cross(k) [µK² (Mpc/h)²]', fontsize=12)
    ax.set_title('Cross-Power: Per-z Rec × Integrated Real', fontweight='bold')
    ax.legend(fontsize=8, loc='upper right', ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, k_max_plot])
    
    # Panel (1,1): Normalized cross-power (cross-correlation coefficient in Fourier space)
    ax = axes[1, 1]
    # r(k) = P_cross / sqrt(P_11 * P_22)
    r_k_real_rec = P_cross_real_rec / np.sqrt(P_real * P_rec)
    r_k_obs_rec = P_cross_obs_rec / np.sqrt(P_obs * P_rec)
    
    ax.plot(k_cross[k_mask_cross], r_k_real_rec[k_mask_cross], 'b-', linewidth=2,
            label='Real × Rec (no CMB)')
    ax.plot(k_cross[k_mask_cross], r_k_obs_rec[k_mask_cross], 'r--', linewidth=2,
            label='Observed × Rec (with CMB)')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.axhline(y=1, color='gray', linestyle='--', linewidth=1, alpha=0.3)
    ax.set_xlabel('k [h/Mpc]', fontsize=12)
    ax.set_ylabel('r(k) = P_cross / √(P₁P₂)', fontsize=12)
    ax.set_title('Cross-Correlation Coefficient r(k)', fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, k_max_plot])
    ax.set_ylim([-1.1, 1.1])
    
    plt.suptitle(f'Cross-Power Spectrum Analysis ({lensed_str} CMB)\n{z_info}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    outfile = os.path.join(OUTPUT_DIR, 'cross_power_spectra_ksz_cmb.png')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outfile}")
    
    # -------------------------------------------------------------------------
    # PLOT 8: Analysis at ℓ = 3000 (where kSZ is typically measured)
    # -------------------------------------------------------------------------
    # Convert k to ℓ using ℓ = k * χ(z)
    chi_mean = compute_comoving_distance(z_mean) * LITTLEH  # Mpc/h
    ell_from_k = k_cross * chi_mean  # ℓ = k * χ
    
    # Get actual CAMB D_ℓ for comparison (primary CMB only)
    ells_camb, Cl_camb = get_cmb_cl(lmax=10000, lensed=lensed)
    Dl_camb = ells_camb * (ells_camb + 1) * Cl_camb / (2 * np.pi)  # D_ℓ in µK²
    
    # Target multipoles for analysis
    ell_targets = [1000, 2000, 3000, 4000, 5000]
    
    print("\n" + "="*80)
    print("ANALYSIS AT SPECIFIC MULTIPOLES (ℓ)")
    print("="*80)
    print(f"Using χ(z={z_mean:.2f}) = {chi_mean:.1f} Mpc/h for k-to-ℓ conversion")
    print("\nNote: Primary CMB is heavily Silk-damped at ℓ > 2000!")
    print("D_ℓ(CMB) from CAMB:")
    for ell_check in [1000, 2000, 3000, 4000, 5000]:
        if ell_check < len(Dl_camb):
            print(f"  D_ℓ(ℓ={ell_check}) = {Dl_camb[ell_check]:.2f} µK²")
    
    # Find r(k) at specific ℓ values
    ell_results = []
    for ell_target in ell_targets:
        # Find k corresponding to this ℓ
        k_target = ell_target / chi_mean
        
        # Find nearest k bin
        idx = np.argmin(np.abs(k_cross - k_target))
        k_actual = k_cross[idx]
        ell_actual = ell_from_k[idx]
        
        # Get r(k) values at this k
        r_no_cmb_at_ell = r_k_real_rec[idx]
        r_with_cmb_at_ell = r_k_obs_rec[idx]
        
        # Get power values from our maps
        P_real_at_ell = P_real[idx]
        P_cmb_at_ell = P_cmb[idx]
        P_cross_at_ell = P_cross_real_rec[idx]
        
        # Get actual CAMB D_ℓ at this multipole
        ell_int = int(round(ell_actual))
        Dl_cmb_actual = Dl_camb[ell_int] if ell_int < len(Dl_camb) else 0
        
        ell_results.append({
            'ell_target': ell_target,
            'ell_actual': ell_actual,
            'k': k_actual,
            'r_no_cmb': r_no_cmb_at_ell,
            'r_with_cmb': r_with_cmb_at_ell,
            'P_ksz': P_real_at_ell,
            'P_cmb': P_cmb_at_ell,
            'P_cross': P_cross_at_ell,
            'Dl_cmb_camb': Dl_cmb_actual,
            'snr': np.sqrt(P_real_at_ell / P_cmb_at_ell) if P_cmb_at_ell > 0 else 0
        })
        
        print(f"\n  ℓ ≈ {ell_actual:.0f} (k={k_actual:.3f} h/Mpc):")
        print(f"    r(no CMB) = {r_no_cmb_at_ell:.4f}")
        print(f"    r(with CMB) = {r_with_cmb_at_ell:.4f}")
        print(f"    D_ℓ(CMB, CAMB) = {Dl_cmb_actual:.2f} µK²")
    
    # Create plot for ℓ-space analysis
    # ell_max corresponds to k_max_plot = 1.2 h/Mpc
    ell_max_plot = k_max_plot * chi_mean  # ~7600 for k=1.2
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Panel (0,0): Power spectra vs ℓ
    ax = axes[0, 0]
    ell_mask = (ell_from_k >= 0) & (ell_from_k <= ell_max_plot)
    ax.semilogy(ell_from_k[ell_mask], P_real[ell_mask], 'b-', linewidth=2, label='Real kSZ')
    ax.semilogy(ell_from_k[ell_mask], P_rec[ell_mask], 'g--', linewidth=2, label='Reconstructed kSZ')
    ax.semilogy(ell_from_k[ell_mask], P_cmb[ell_mask], 'r-', linewidth=2, label=f'CMB ({lensed_str})')
    # Mark ℓ = 3000
    ax.axvline(x=3000, color='purple', linestyle='--', linewidth=2, alpha=0.7, label='ℓ = 3000')
    ax.set_xlabel('Multipole ℓ', fontsize=12)
    ax.set_ylabel('P(ℓ) [µK² (Mpc/h)²]', fontsize=12)
    ax.set_title('Power Spectra vs Multipole', fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, ell_max_plot])
    
    # Panel (0,1): Cross-correlation r(ℓ)
    ax = axes[0, 1]
    ax.plot(ell_from_k[ell_mask], r_k_real_rec[ell_mask], 'b-', linewidth=2, 
            label='Real × Rec (no CMB)')
    ax.plot(ell_from_k[ell_mask], r_k_obs_rec[ell_mask], 'r--', linewidth=2,
            label='Observed × Rec (with CMB)')
    ax.axvline(x=3000, color='purple', linestyle='--', linewidth=2, alpha=0.7, label='ℓ = 3000')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.set_xlabel('Multipole ℓ', fontsize=12)
    ax.set_ylabel('r(ℓ)', fontsize=12)
    ax.set_title('Cross-Correlation Coefficient vs Multipole', fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, ell_max_plot])
    ax.set_ylim([-1.1, 1.1])
    
    # Panel (1,0): Bar chart of r at specific ℓ values
    ax = axes[1, 0]
    ell_labels = [f"ℓ={r['ell_target']}" for r in ell_results]
    x_pos = np.arange(len(ell_results))
    width = 0.35
    
    r_no_cmb_vals = [r['r_no_cmb'] for r in ell_results]
    r_with_cmb_vals = [r['r_with_cmb'] for r in ell_results]
    
    bars1 = ax.bar(x_pos - width/2, r_no_cmb_vals, width, label='Without CMB', color='blue', alpha=0.7)
    bars2 = ax.bar(x_pos + width/2, r_with_cmb_vals, width, label='With CMB', color='red', alpha=0.7)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Multipole', fontsize=12)
    ax.set_ylabel('Correlation coefficient r', fontsize=12)
    ax.set_title('Correlation at Specific Multipoles', fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(ell_labels)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([-0.5, 1.0])
    
    # Highlight ℓ = 3000
    ell_3000_idx = ell_targets.index(3000)
    ax.get_children()[ell_3000_idx].set_edgecolor('purple')
    ax.get_children()[ell_3000_idx].set_linewidth(2)
    ax.get_children()[ell_3000_idx + len(ell_results)].set_edgecolor('purple')
    ax.get_children()[ell_3000_idx + len(ell_results)].set_linewidth(2)
    
    # Panel (1,1): Signal-to-noise vs ℓ
    ax = axes[1, 1]
    snr_vs_ell = np.sqrt(P_real / P_cmb)
    ax.plot(ell_from_k[ell_mask], snr_vs_ell[ell_mask], 'b-', linewidth=2)
    ax.axvline(x=3000, color='purple', linestyle='--', linewidth=2, alpha=0.7, label='ℓ = 3000')
    ax.axhline(y=1, color='red', linestyle='--', linewidth=1, alpha=0.5, label='S/N = 1')
    
    # Mark S/N at ℓ = 3000
    ell_3000_result = ell_results[ell_3000_idx]
    ax.scatter([3000], [ell_3000_result['snr']], s=100, c='purple', zorder=5, 
               label=f"S/N(ℓ=3000) = {ell_3000_result['snr']:.4f}")
    
    ax.set_xlabel('Multipole ℓ', fontsize=12)
    ax.set_ylabel('Signal-to-Noise (kSZ/CMB)', fontsize=12)
    ax.set_title('kSZ Signal-to-Noise vs Multipole', fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, ell_max_plot])
    
    plt.suptitle(f'Analysis at ℓ = 3000 ({lensed_str} CMB)\n{z_info}, χ = {chi_mean:.0f} Mpc/h', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    outfile = os.path.join(OUTPUT_DIR, 'ksz_analysis_ell_3000.png')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outfile}")
    
    # Print focused summary for ℓ = 3000
    print("\n" + "-"*60)
    print("FOCUS: ℓ = 3000 (typical kSZ measurement scale)")
    print("-"*60)
    print(f"  k = {ell_3000_result['k']:.4f} h/Mpc")
    print(f"  D_ℓ(CMB, CAMB) = {ell_3000_result['Dl_cmb_camb']:.2f} µK² (primary CMB is damped!)")
    print(f"  r(ℓ=3000) without CMB = {ell_3000_result['r_no_cmb']:.4f}")
    print(f"  r(ℓ=3000) with CMB    = {ell_3000_result['r_with_cmb']:.4f}")
    print("-"*60)
    
    # -------------------------------------------------------------------------
    # Bandpass-filtered analysis around ℓ = 3000
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("BANDPASS-FILTERED ANALYSIS (ℓ = 2500-3500)")
    print("="*80)
    print("Filtering maps to isolate modes around ℓ = 3000...")
    
    # Get angular scales for filtering
    chi_Mpc = compute_comoving_distance(z_mean)  # Mpc (not Mpc/h)
    theta_x = boxlength[0] / LITTLEH / chi_Mpc  # radians
    theta_y = boxlength[1] / LITTLEH / chi_Mpc  # radians
    dx_rad = theta_x / ksz_map_real.shape[1]
    dy_rad = theta_y / ksz_map_real.shape[0]
    
    # Create ℓ grid
    kx = 2 * np.pi * np.fft.fftfreq(ksz_map_real.shape[0], d=dy_rad)
    ky = 2 * np.pi * np.fft.fftfreq(ksz_map_real.shape[1], d=dx_rad)
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing='ij')
    ell_grid = np.sqrt(kx_grid**2 + ky_grid**2)
    
    # Bandpass filter: ℓ = 2500-3500
    ell_min, ell_max = 2500, 3500
    bandpass = (ell_grid >= ell_min) & (ell_grid <= ell_max)
    
    # Apply bandpass filter to maps
    def apply_bandpass(map_2d, bandpass_mask):
        fft_map = np.fft.fft2(map_2d)
        fft_filtered = fft_map * bandpass_mask
        return np.real(np.fft.ifft2(fft_filtered))
    
    ksz_real_bp = apply_bandpass(ksz_map_real, bandpass)
    ksz_rec_bp = apply_bandpass(ksz_map_rec, bandpass)
    cmb_bp = apply_bandpass(cmb_map, bandpass)
    observed_bp = apply_bandpass(observed_map, bandpass)
    
    # Compute correlations on bandpass-filtered maps
    r_bp_no_cmb = pearson_r(ksz_real_bp, ksz_rec_bp)
    r_bp_with_cmb = pearson_r(observed_bp, ksz_rec_bp)
    
    # Compute std of filtered maps
    std_ksz_bp = ksz_real_bp.std()
    std_cmb_bp = cmb_bp.std()
    snr_bp = std_ksz_bp / std_cmb_bp if std_cmb_bp > 0 else np.inf
    
    print(f"\nBandpass ℓ = {ell_min}-{ell_max}:")
    print(f"  kSZ std (filtered): {std_ksz_bp:.4f} µK")
    print(f"  CMB std (filtered): {std_cmb_bp:.4f} µK")
    print(f"  S/N (kSZ/CMB) at ℓ~3000: {snr_bp:.4f}")
    print(f"\n  r(no CMB) at ℓ~3000: {r_bp_no_cmb:.4f}")
    print(f"  r(with CMB) at ℓ~3000: {r_bp_with_cmb:.4f}")
    print(f"\nNote: Primary CMB D_ℓ(3000) = {Dl_camb[3000]:.2f} µK² (heavily damped)")
    print("At ℓ=3000, kSZ should dominate over primary CMB!")
    print("-"*60)
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*80)
    print("SUMMARY: kSZ RECONSTRUCTION WITH CMB CONTAMINATION")
    print("="*80)
    
    print(f"\nkSZ signal std:  {ksz_map_real.std():.4f} µK")
    print(f"CMB noise std:   {cmb_map.std():.2f} µK")
    print(f"Signal-to-noise: {ksz_map_real.std() / cmb_map.std():.6f}")
    
    print("\n--- Per-redshift reconstruction vs integrated real ---")
    print(f"{'z':<8} {'<xHI>':<10} {'r(no CMB)':<12} {'r(with CMB)':<12}")
    print("-" * 50)
    for res in per_z_results:
        print(f"{res['z']:<8.3f} {res['mean_xhi']:<10.4f} "
              f"{res['r_no_cmb']:<12.4f} {res['r_with_cmb']:<12.4f}")
    
    print("\n--- Integrated reconstruction vs integrated real ---")
    print(f"Without CMB: r = {r_int_no_cmb:.4f}")
    print(f"With CMB:    r = {r_int_with_cmb:.4f}")
    print("="*80)
    
    # Clean up
    for result in individual_results:
        del result['ksz_rec']
    gc.collect()
    
    return {
        'cmb_results': cmb_results,
        'per_z_results': per_z_results,
        'integrated_results': integrated_results,
    }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Run analysis with lensed CMB
    results_lensed = run_cmb_analysis(
        redshifts=[
        6.056, 6.113, 6.172, 6.231, 6.292, 6.354, 6.418, 6.483, 6.549,
        6.617, 6.686, 6.757, 6.830, 6.905, 6.981, 7.059, 7.139, 7.221,
        7.305, 7.391, 7.480, 7.570, 7.664, 7.760, 7.859, 7.960, 8.064,
        8.172, 8.283, 8.397, 8.515, 8.636, 8.762, 8.892, 9.026, 9.164,
        9.308, 9.457, 9.611, 9.771, 9.938, 10.110, 10.290, 10.478,
        10.673, 10.877, 11.090, 11.313, 11.546, 11.791, 12.048
        ],
        lensed=True,
        seed=42
    )
    
    print("\nAnalysis complete!")
