# ============================================================================
# KSZ-21CM CROSS-CORRELATION EVOLUTION WITH NEUTRAL FRACTION
# Analyzes how correlation changes with mean_xHI by chunking in redshift
# ============================================================================

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import fft
import warnings
from powerbox.tools import get_power

# ============================================================================
# CELL 1: Load 21cmFAST Data
# ============================================================================

print("Loading 21cmFAST data...")

sim_ids = list(range(12701, 12731))  # 30 simulations
n_sims = len(sim_ids)

den = np.load("data_21cmfast/density/12701_density_LC.npy")
xhi = np.load("data_21cmfast/xHI/12701_xHI_LC.npy")
vz = np.load("data_21cmfast/velocity/12701_velocity_z_LC.npy")
redshifts = np.load("data_21cmfast/lightcone_redshifts.npy")

nx, ny, nz = den.shape
print(f"Shape (nx, ny, nz): {den.shape}")
print(f"Redshifts: {len(redshifts)} values, range [{redshifts.min():.3f}, {redshifts.max():.3f}]")

# ============================================================================
# CELL 2: Velocity Reconstruction Function
# ============================================================================

def safe_real(a):
    """Extract real part safely."""
    return np.real(a).astype(np.float32, copy=False)

def reconstruct_velocity_single_z_method(
    den_xyz, xhi_xyz=None, vz_for_gradient=None, *,
    weight="deltaXhi",
    z_ref=None,
    littleh=0.7,
    box_mpc_over_h=300.0,
    include_velocity_term=False,
    dtype=np.float32
):
    """Reconstruct velocity using single-z method with rfftn.
    
    Parameters:
        den_xyz: Density field (or delta field)
        xhi_xyz: Neutral fraction field
        vz_for_gradient: Velocity field for computing gradient term (optional)
        weight: Weighting scheme - 'delta', 'delta_xhi', or 'deltaXhi'
        include_velocity_term: If True, include (1 - dv/dr / aH) term in tracer field
    """
    d = np.asarray(den_xyz, dtype=dtype)
    nx, ny, nz = d.shape
    
    # Density contrast
    mean_den = d.mean(dtype=np.float64).astype(dtype)
    delta = d
    
    # Cosmology parameters
    if z_ref is None:
        z_ref = float(np.mean(redshifts))
    
    a = 1.0 / (1.0 + z_ref)
    H0 = 100.0 * littleh
    omega_l0 = 0.73
    omega_m0 = 1.0 - omega_l0
    Hz = H0 * np.sqrt(omega_m0 * (1 + z_ref)**3 + omega_l0)
    aH = a * Hz  # km/s/Mpc
    
    # Weighting field
    if weight == "delta":
        field = delta - 1.0
    elif weight == "delta_xhi":
        if xhi_xyz is None:
            raise ValueError("xhi_xyz required")
        field = (delta - 1.0) * xhi_xyz.astype(dtype, copy=False)
    elif weight == "deltaXhi":
        if xhi_xyz is None:
            raise ValueError("xhi_xyz required")
        field = delta * xhi_xyz.astype(dtype, copy=False)
    else:
        raise ValueError("weight must be 'delta' | 'delta_xhi' | 'deltaXhi'")
    
    # Add velocity gradient term if requested
    # δTb ∝ xHI × (1 + δ) × H / (dv_r/dr + H)
    if include_velocity_term and vz_for_gradient is not None:
        dz_cell = box_mpc_over_h / nz / littleh  # Cell size in Mpc
        vz_kms = vz_for_gradient / 1e5
        dvdz = np.gradient(vz_kms, dz_cell, axis=2).astype(dtype)  # km/s/Mpc
        velocity_factor = Hz / (dvdz + Hz)
        field = field * velocity_factor.astype(dtype)
    
    # Mean-subtract tracer field to remove DC offset
    field = field - field.mean()
    
    # Real FFT
    dlt_r = fft.rfftn(field, workers=-1).astype(np.complex64, copy=False)
    
    # k-space grids
    rc = box_mpc_over_h / float(nx) / littleh
    kx = (2.0 * np.pi * fft.fftfreq(nx, d=rc)).astype(dtype)
    ky = (2.0 * np.pi * fft.fftfreq(ny, d=rc)).astype(dtype)
    kz = (2.0 * np.pi * fft.rfftfreq(nz, d=rc)).astype(dtype)
    
    # Avoid division by zero
    tiny = np.finfo(dtype).tiny
    if kz.size: kz[0] = max(kz[0], tiny)
    if kx.size: kx[0] = max(kx[0], tiny)
    if ky.size: ky[0] = max(ky[0], tiny)
    
    # Cosmology for reconstruction factor
    Ha = dtype(H0 * np.sqrt(omega_m0 / a**3 + omega_l0))
    Omega_m_a = (omega_m0 / a**3) / (omega_m0 / a**3 + omega_l0)
    f_omega = dtype(Omega_m_a**0.55)
    factor = np.complex64(Ha * a * f_omega) * 1j
    
    # Precompute k^2
    kx2 = (kx * kx).astype(dtype, copy=False)
    ky2 = (ky * ky).astype(dtype, copy=False)
    kz2 = (kz * kz).astype(dtype, copy=False)
    
    # Reconstruct vz
    tmp = dlt_r.astype(np.complex64, copy=True)
    np.multiply(tmp, factor, out=tmp)
    np.multiply(tmp, kz[None, None, :], out=tmp)
    
    absk2 = kx2[:, None, None] + ky2[None, :, None] + kz2[None, None, :]
    np.divide(tmp, absk2, out=tmp, where=absk2 != 0)
    
    vz_rec = safe_real(fft.irfftn(tmp, s=(nx, ny, nz), workers=-1))
    
    # Convert km/s to cm/s
    vz_rec *= dtype(1e5)
    
    return vz_rec

# ============================================================================
# CELL 3: Create kSZ Maps for a Chunk
# ============================================================================

def make_ksz_maps_chunk(
    den_xyz, xhi_xyz, vz_real, vz_rec, redshifts_z, *,
    littleh=0.7,
    box_mpc_over_h=300.0,
    xhi_eps=0.5,
    physical_norm=False,
    include_velocity_term=True,
    dtype=np.float32
):
    """
    Create kSZ and 21cm maps for a redshift chunk.
    
    The 21cm brightness temperature includes the velocity gradient term:
    δTb ∝ xHI × (1 + δ) × H / (dv_r/dr + H)
    
    Parameters:
        include_velocity_term: If True, include the H/(dv/dr + H) term in T_b
    
    Returns: ksz_map, ksz_map_abs, t21_map, mean_xhi, mean_z
    """
    nx, ny, nz = den_xyz.shape
    mean_den = np.mean(den_xyz, dtype=np.float64).astype(dtype)
    z_mean = float(np.mean(redshifts_z))
    
    # Initialize maps
    ksz_map = np.zeros((nx, ny), dtype=dtype)
    ksz_map_abs = np.zeros((nx, ny), dtype=dtype)
    t21_map = np.zeros((nx, ny), dtype=dtype)
    
    # Physical prefactor
    if physical_norm:
        sigma_T = 6.6524587158e-25  # cm^2
        c_cm_s = 2.99792458e10
        dl_proper = (box_mpc_over_h / littleh / (1.0 + z_mean)) / nz * 3.085677581e24
        pref = dtype(sigma_T * dl_proper / c_cm_s)
    else:
        pref = dtype(1.0)
    
    # Compute velocity gradient dv_z/dr_z if needed
    if include_velocity_term:
        # Cell size in Mpc (physical)
        dz_cell = box_mpc_over_h / nz / littleh
        # vz_real is in cm/s, convert to km/s
        vz_kms = vz_real / 1e5
        dvdz = np.gradient(vz_kms, dz_cell, axis=2)  # km/s/Mpc
        
        # Hubble parameter at z_mean: H(z) = H0 * E(z)
        H0 = 100.0 * littleh  # km/s/Mpc
        omega_m0 = 0.27
        omega_l0 = 0.73
        Hz = H0 * np.sqrt(omega_m0 * (1 + z_mean)**3 + omega_l0)  # km/s/Mpc
    
    # Integrate along z-axis
    for k in range(nz):
        d = den_xyz[:, :, k].astype(dtype, copy=False)
        x = xhi_xyz[:, :, k].astype(dtype, copy=False)
        vr = vz_real[:, :, k].astype(dtype, copy=False)
        
        delta = d 
        ne = (dtype(1.0)-x) * (dtype(1.0) + delta)
        
        # 21cm brightness temperature with velocity term
        # δTb ∝ xHI × (1 + δ) × H / (dv_r/dr + H)
        if include_velocity_term:
            dvdz_slice = dvdz[:, :, k].astype(dtype, copy=False)
            velocity_factor = Hz / (dvdz_slice + Hz)
            t21 = x * (dtype(1.0) + delta) * velocity_factor.astype(dtype)
        else:
            t21 = x * (dtype(1.0) + delta)
        
        ksz_map += pref * ne * vr
        ksz_map_abs += pref * ne * np.abs(vr)
        t21_map += t21
    
    # Compute mean xHI for this chunk
    mean_xhi = float(np.mean(xhi_xyz))
    
    return ksz_map, ksz_map_abs, t21_map, mean_xhi, z_mean

# ============================================================================
# CELL 4: Pearson Correlation Function
# ============================================================================

def pearson_r(a, b):
    """Pearson correlation coefficient."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    m = np.isfinite(a) & np.isfinite(b)
    if not np.any(m):
        return np.nan
    a = a[m] - a[m].mean()
    b = b[m] - b[m].mean()
    den = np.sqrt((a*a).sum() * (b*b).sum())
    return float((a*b).sum() / den) if den > 0 else np.nan

# ============================================================================
# CELL 5: k to ell conversion
# ============================================================================

def k_to_ell(k, z, littleh=0.7):
    """
    Convert k [h/Mpc] to multipole ell.
    ell = k * chi(z) where chi(z) is comoving distance
    """
    from scipy.integrate import quad
    omega_m0 = 0.27
    omega_l0 = 0.73
    
    def E(zp):
        return np.sqrt(omega_m0 * (1+zp)**3 + omega_l0)
    
    chi, _ = quad(lambda zp: 1.0/E(zp), 0, z)
    chi *= 3000.0 / littleh  # c/H0 in Mpc/h
    
    ell = k * chi
    return ell

# ============================================================================
# CELL 6: Chunk Analysis
# ============================================================================

# Define number of chunks
n_chunks = 20  # Adjust this to control resolution
chunk_size = 200

print(f"\nAnalyzing {n_chunks} redshift chunks...")
print(f"Chunk size: {chunk_size} z-slices")

# Storage for results
results = {
    'mean_xhi': [],
    'mean_z': [],
    'z_min': [],
    'z_max': [],
    'correlation': [],
    'correlation_rec': [],
    # Real velocity
    'P_k_cross': [],
    'l_s_cross': [],
    'err_cross': [],
    'ell_cross': [],
    'P_k_cross_sq': [],
    'l_s_cross_sq': [],
    'err_cross_sq': [],
    'ell_cross_sq': [],
    'r_k': [],
    'r_k_sq': [],
    # Reconstructed velocity
    'P_k_cross_rec': [],
    'err_cross_rec': [],
    'ell_cross_rec': [],
    'P_k_cross_sq_rec': [],
    'err_cross_sq_rec': [],
    'ell_cross_sq_rec': [],
    'r_k_rec': [],
    'r_k_sq_rec': [],
    # 21cm and maps
    'P_k_21cm': [],
    'err_21cm': [],
    'ell_21cm': [],
    'ksz_maps': [],
    'ksz_abs_maps': [],
    't21_maps': []
}

# Process each chunk
for i in range(n_chunks):
    z_start = i * chunk_size
    z_end = min((i + 1) * chunk_size, nz)
    
    if z_end - z_start < 10:  # Skip very small chunks
        continue
    
    print(f"\nChunk {i+1}/{n_chunks}: z-slices [{z_start}:{z_end}]")
    
    # Extract chunk
    den_chunk = den[:, :, z_start:z_end]
    xhi_chunk = xhi[:, :, z_start:z_end]
    vz_chunk = vz[:, :, z_start:z_end]
    redshifts_chunk = redshifts[z_start:z_end]
    
    # Reconstruct velocity for this chunk
    z_ref_chunk = float(np.mean(redshifts_chunk))
    vz_rec_chunk = reconstruct_velocity_single_z_method(
        den_chunk, xhi_xyz=xhi_chunk,
        weight="deltaXhi",
        z_ref=z_ref_chunk,
        littleh=0.7,
        box_mpc_over_h=300.0
    )
    
    # Create kSZ maps (real velocity)
    ksz_real, ksz_abs_real, t21, mean_xhi, mean_z = make_ksz_maps_chunk(
        den_chunk, xhi_chunk, vz_chunk, vz_rec_chunk, redshifts_chunk,
        littleh=0.7,
        box_mpc_over_h=300.0,
        physical_norm=False
    )
    
    # Create kSZ maps (reconstructed velocity)
    ksz_rec, ksz_abs_rec, _, _, _ = make_ksz_maps_chunk(
        den_chunk, xhi_chunk, vz_rec_chunk, vz_rec_chunk, redshifts_chunk,
        littleh=0.7,
        box_mpc_over_h=300.0,
        physical_norm=False
    )
    
    # Compute correlations
    r_real = pearson_r(ksz_abs_real, t21)
    r_rec = pearson_r(ksz_abs_rec, t21)
    
    # Compute power spectra
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        # === REAL VELOCITY ===
        # Auto-power: |kSZ| (real)
        P_k_ksz_real, l_s_ksz_real, err_ksz_real = get_power(
            ksz_abs_real,
            300,
            bins=100,
            log_bins=True,
            get_variance=True,
        )
        P_k_ksz_real = P_k_ksz_real * l_s_ksz_real ** 2
        err_ksz_real = np.sqrt(err_ksz_real) * l_s_ksz_real ** 2
        
        # Cross-power: |kSZ| x 21cm (real)
        P_k_cross_real, l_s_cross_real, err_cross_real = get_power(
            deltax=ksz_abs_real,
            deltax2=t21,
            boxlength=300,
            bins=100,
            log_bins=True,
            get_variance=True,
        )
        P_k_cross_real = P_k_cross_real * l_s_cross_real ** 2
        err_cross_real = np.sqrt(err_cross_real) * l_s_cross_real ** 2
        
        # Cross-power: kSZ^2 x 21cm (real) - use signed kSZ squared
        P_k_cross_sq_real, l_s_cross_sq_real, err_cross_sq_real = get_power(
            deltax=ksz_real**2,
            deltax2=t21,
            boxlength=300,
            bins=100,
            log_bins=True,
            get_variance=True,
        )
        P_k_cross_sq_real = P_k_cross_sq_real * l_s_cross_sq_real ** 2
        err_cross_sq_real = np.sqrt(err_cross_sq_real) * l_s_cross_sq_real ** 2
        
        # Auto-power: kSZ^2 (real)
        P_k_ksz_sq_real, l_s_ksz_sq_real, err_ksz_sq_real = get_power(
            ksz_real**2,
            300,
            bins=100,
            log_bins=True,
            get_variance=True,
        )
        P_k_ksz_sq_real = P_k_ksz_sq_real * l_s_ksz_sq_real ** 2
        
        # === RECONSTRUCTED VELOCITY ===
        # Auto-power: |kSZ| (reconstructed)
        P_k_ksz_rec, l_s_ksz_rec, err_ksz_rec = get_power(
            ksz_abs_rec,
            300,
            bins=100,
            log_bins=True,
            get_variance=True,
        )
        P_k_ksz_rec = P_k_ksz_rec * l_s_ksz_rec ** 2
        err_ksz_rec = np.sqrt(err_ksz_rec) * l_s_ksz_rec ** 2
        
        # Cross-power: |kSZ| x 21cm (reconstructed)
        P_k_cross_rec, l_s_cross_rec, err_cross_rec = get_power(
            deltax=ksz_abs_rec,
            deltax2=t21,
            boxlength=300,
            bins=100,
            log_bins=True,
            get_variance=True,
        )
        P_k_cross_rec = P_k_cross_rec * l_s_cross_rec ** 2
        err_cross_rec = np.sqrt(err_cross_rec) * l_s_cross_rec ** 2
        
        # Cross-power: kSZ^2 x 21cm (reconstructed) - use signed kSZ squared
        P_k_cross_sq_rec, l_s_cross_sq_rec, err_cross_sq_rec = get_power(
            deltax=ksz_rec**2,
            deltax2=t21,
            boxlength=300,
            bins=100,
            log_bins=True,
            get_variance=True,
        )
        P_k_cross_sq_rec = P_k_cross_sq_rec * l_s_cross_sq_rec ** 2
        err_cross_sq_rec = np.sqrt(err_cross_sq_rec) * l_s_cross_sq_rec ** 2
        
        # Auto-power: kSZ^2 (reconstructed)
        P_k_ksz_sq_rec, l_s_ksz_sq_rec, err_ksz_sq_rec = get_power(
            ksz_rec**2,
            300,
            bins=100,
            log_bins=True,
            get_variance=True,
        )
        P_k_ksz_sq_rec = P_k_ksz_sq_rec * l_s_ksz_sq_rec ** 2
        
        # Auto-power: 21cm (same for both)
        P_k_21cm, l_s_21cm, err_21cm = get_power(
            t21,
            300,
            bins=100,
            log_bins=True,
            get_variance=True,
        )
        P_k_21cm = P_k_21cm * l_s_21cm ** 2
        err_21cm = np.sqrt(err_21cm) * l_s_21cm ** 2
    
    # Compute correlation coefficient r(k) = P_cross / sqrt(P_ksz * P_21cm)
    # Real velocity
    r_k_real = P_k_cross_real / np.sqrt(P_k_21cm * P_k_ksz_real)
    r_k_sq_real = P_k_cross_sq_real / np.sqrt(P_k_21cm * P_k_ksz_sq_real)
    
    # Reconstructed velocity
    r_k_rec = P_k_cross_rec / np.sqrt(P_k_21cm * P_k_ksz_rec)
    r_k_sq_rec = P_k_cross_sq_rec / np.sqrt(P_k_21cm * P_k_ksz_sq_rec)
    # print(P_k_cross_sq_real,P_k_cross_sq_rec)
    # Convert k to ell
    ell_cross_real = k_to_ell(l_s_cross_real, mean_z)
    ell_cross_sq_real = k_to_ell(l_s_cross_sq_real, mean_z)
    ell_cross_rec = k_to_ell(l_s_cross_rec, mean_z)
    ell_cross_sq_rec = k_to_ell(l_s_cross_sq_rec, mean_z)
    ell_21cm = k_to_ell(l_s_21cm, mean_z)
    
    # Store results
    results['mean_xhi'].append(mean_xhi)
    results['mean_z'].append(mean_z)
    results['z_min'].append(redshifts_chunk.min())
    results['z_max'].append(redshifts_chunk.max())
    results['correlation'].append(r_real)
    results['correlation_rec'].append(r_rec)
    
    # Real velocity
    results['P_k_cross'].append(P_k_cross_real)
    results['l_s_cross'].append(l_s_cross_real)
    results['err_cross'].append(err_cross_real)
    results['ell_cross'].append(ell_cross_real)
    results['P_k_cross_sq'].append(P_k_cross_sq_real)
    results['l_s_cross_sq'].append(l_s_cross_sq_real)
    results['err_cross_sq'].append(err_cross_sq_real)
    results['ell_cross_sq'].append(ell_cross_sq_real)
    results['r_k'].append(r_k_real)
    results['r_k_sq'].append(r_k_sq_real)
    
    # Reconstructed velocity
    results['P_k_cross_rec'].append(P_k_cross_rec)
    results['err_cross_rec'].append(err_cross_rec)
    results['ell_cross_rec'].append(ell_cross_rec)
    results['P_k_cross_sq_rec'].append(P_k_cross_sq_rec)
    results['err_cross_sq_rec'].append(err_cross_sq_rec)
    results['ell_cross_sq_rec'].append(ell_cross_sq_rec)
    results['r_k_rec'].append(r_k_rec)
    results['r_k_sq_rec'].append(r_k_sq_rec)
    
    # 21cm and maps
    results['P_k_21cm'].append(P_k_21cm)
    results['err_21cm'].append(err_21cm)
    results['ell_21cm'].append(ell_21cm)
    results['ksz_maps'].append(ksz_real)
    results['ksz_abs_maps'].append(ksz_abs_real)
    results['t21_maps'].append(t21)
    
    print(f"  mean_xHI = {mean_xhi:.4f}, mean_z = {mean_z:.3f}")
    print(f"  r(|kSZ_real|, T21) = {r_real:+.4f}")
    print(f"  r(|kSZ_rec|, T21)  = {r_rec:+.4f}")

# Convert to arrays
for key in results:
    results[key] = np.array(results[key])

print("\nChunk analysis complete!")

# ============================================================================
# CELL 7: Visualization - Cross-Power at Specific ell Values vs xHI
# ============================================================================

# First, find what ell values are actually available
# Use a valid chunk (skip first if it has NaN)
valid_idx = 0
for idx in range(len(results['ell_cross'])):
    if np.any(np.isfinite(results['ell_cross'][idx])):
        valid_idx = idx
        break

sample_ell = results['ell_cross'][valid_idx]
valid_ell = sample_ell[np.isfinite(sample_ell)]

if len(valid_ell) > 0:
    print(f"\nAvailable ℓ range: [{valid_ell.min():.1f}, {valid_ell.max():.1f}]")
    print(f"Number of ℓ bins: {len(valid_ell)}")
    
    # Choose 4 specific ell values to track
    ell_targets_desired = [1000, 2000, 3000, 4000]
    
    # Single plot with all ell values
    fig1 = plt.figure(figsize=(12, 8))
    ax = fig1.add_subplot(111)
    
    colors_ell = plt.cm.plasma(np.linspace(0, 1, len(ell_targets_desired)))
    
    for color_idx, ell_target in enumerate(ell_targets_desired):
        # Find closest available ell value
        idx_closest = np.argmin(np.abs(valid_ell - ell_target))
        ell_actual = valid_ell[idx_closest]
        
        # Extract cross-power at this ell for each chunk
        P_at_ell = []
        err_at_ell = []
        xhi_vals = []
        
        for i in range(len(results['mean_xhi'])):
            ell_arr = results['ell_cross'][i]
            P_arr = results['P_k_cross'][i]
            err_arr = results['err_cross'][i]
            
            # Find closest ell value in this chunk
            valid_mask = np.isfinite(ell_arr)
            if not np.any(valid_mask):
                continue
                
            idx_ell = np.argmin(np.abs(ell_arr[valid_mask] - ell_actual))
            actual_idx = np.where(valid_mask)[0][idx_ell]
            
            # Only include if we have valid data
            if np.isfinite(P_arr[actual_idx]) and np.isfinite(err_arr[actual_idx]):
                P_at_ell.append(P_arr[actual_idx])
                err_at_ell.append(err_arr[actual_idx])
                xhi_vals.append(results['mean_xhi'][i])
        
        P_at_ell = np.array(P_at_ell)
        err_at_ell = np.array(err_at_ell)
        xhi_vals = np.array(xhi_vals)
        
        # Plot
        if len(P_at_ell) > 0:
            ax.errorbar(xhi_vals, P_at_ell, yerr=err_at_ell,
                        fmt='o-', linewidth=2, markersize=6, capsize=4, alpha=0.7,
                        color=colors_ell[color_idx], label=f'ℓ = {ell_actual:.0f}')
    
    ax.set_xlabel('Mean Neutral Fraction (xHI)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Cross-Power P(ℓ)', fontsize=14, fontweight='bold')
    ax.set_title('|kSZ| × 21cm Cross-Power at Fixed ℓ vs Neutral Fraction', 
                 fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    
    plt.savefig('plots/21cmfast_cross_power_vs_xhi.png', dpi=300, bbox_inches='tight')
    plt.show()
else:
    print("\nWarning: No valid ell data found!")

# ============================================================================
# CELL 8: Visualization - Cross-Power Spectra for Specific xHI Values
# ============================================================================

# Choose 4-5 representative xHI values (epochs)
n_epochs = min(5, len(results['mean_xhi']))
indices_to_plot = np.linspace(0, len(results['mean_xhi'])-1, n_epochs, dtype=int)

fig2 = plt.figure(figsize=(20, 10))
gs2 = fig2.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

# Plot 1: Full spectra for different epochs
ax1 = fig2.add_subplot(gs2[0, :])
colors = plt.cm.viridis(np.linspace(0, 1, n_epochs))

for idx, i in enumerate(indices_to_plot):
    ell = results['ell_cross'][i]
    P_k = results['P_k_cross'][i]
    err = results['err_cross'][i]
    xhi = results['mean_xhi'][i]
    z = results['mean_z'][i]
    
    # Only plot up to ell=5000
    valid = np.isfinite(ell) & np.isfinite(P_k) & (ell <= 5000)
    if np.any(valid):
        ax1.errorbar(ell[valid], P_k[valid], yerr=err[valid], fmt='-', linewidth=2, alpha=0.7,
                     color=colors[idx], label=f'xHI={xhi:.2f}, z={z:.1f}')

ax1.set_xlabel('Multipole ℓ', fontsize=14, fontweight='bold')
ax1.set_ylabel('Cross-Power P(ℓ)', fontsize=14, fontweight='bold')
ax1.set_title('|kSZ| × 21cm Cross-Power Evolution', fontsize=16, fontweight='bold')
ax1.legend(fontsize=10, ncol=2, loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 5000)

# Add redshift axis on top
ax1_top = ax1.twiny()
ax1_top.set_xlim(ax1.get_xlim())
# Get redshift values for the plotted epochs
z_values = [results['mean_z'][i] for i in indices_to_plot]
z_labels = [f'{z:.1f}' for z in z_values]
ax1_top.set_xlabel('Redshift', fontsize=14, fontweight='bold')
# This is approximate - just showing the redshift range
ax1_top.set_xticks([])
ax1_top.text(0.5, 1.05, f'Redshift range: z={results["mean_z"].max():.1f} → {results["mean_z"].min():.1f}', 
             transform=ax1.transAxes, ha='center', fontsize=11)

# Plot 2: Zoomed to ell=[2000,4000]
ax2 = fig2.add_subplot(gs2[1, 0])
for idx, i in enumerate(indices_to_plot):
    ell = results['ell_cross'][i]
    P_k = results['P_k_cross'][i]
    err = results['err_cross'][i]
    xhi = results['mean_xhi'][i]
    
    mask = (ell >= 2000) & (ell <= 4000)
    ax2.errorbar(ell[mask], P_k[mask], yerr=err[mask], fmt='o-', linewidth=2,
                 markersize=6, alpha=0.7, color=colors[idx], label=f'xHI={xhi:.3f}')

ax2.set_xlabel('Multipole ℓ', fontsize=12, fontweight='bold')
ax2.set_ylabel('Cross-Power P(ℓ)', fontsize=12, fontweight='bold')
ax2.set_title('Zoomed: ℓ=[2000,4000]', fontsize=14, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# Plot 3: kSZ² × 21cm cross-power (not 21cm²)
ax3 = fig2.add_subplot(gs2[1, 1])
for idx, i in enumerate(indices_to_plot):
    ell = results['ell_cross_sq'][i]
    P_k = results['P_k_cross_sq'][i]
    xhi = results['mean_xhi'][i]
    
    # Only plot up to ell=5000
    valid = np.isfinite(ell) & np.isfinite(P_k) & (ell <= 5000)
    if np.any(valid):
        ax3.plot(ell[valid], P_k[valid], '-', linewidth=2, alpha=0.7,
                 color=colors[idx], label=f'xHI={xhi:.2f}')

ax3.set_xlabel('Multipole ℓ', fontsize=12, fontweight='bold')
ax3.set_ylabel('Cross-Power P(ℓ)', fontsize=12, fontweight='bold')
ax3.set_title('kSZ² × 21cm Cross-Power', fontsize=14, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)
ax3.set_xlim(2000, 4000)

plt.suptitle('Cross-Power Spectra at Different Reionization Epochs', 
             fontsize=18, fontweight='bold')
plt.savefig('plots/21cmfast_cross_power_spectra.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# CELL 9: Additional Correlation Plots
# ============================================================================

fig3 = plt.figure(figsize=(18, 6))
gs3 = fig3.add_gridspec(1, 3, hspace=0.3, wspace=0.3)

# Plot 1: Correlation vs xHI
ax1 = fig3.add_subplot(gs3[0, 0])
ax1.plot(results['mean_xhi'], results['correlation'], 'bo-', 
         label='Real velocity', linewidth=2, markersize=8, alpha=0.7)
ax1.plot(results['mean_xhi'], results['correlation_rec'], 'rs--', 
         label='Reconstructed velocity', linewidth=2, markersize=8, alpha=0.7)
ax1.axhline(0, color='black', linewidth=0.8, linestyle=':')
ax1.set_xlabel('Mean Neutral Fraction (mean_xHI)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Correlation r(|kSZ|, T21)', fontsize=14, fontweight='bold')
ax1.set_title('Cross-Correlation vs Neutral Fraction', fontsize=16, fontweight='bold')
ax1.legend(fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 1)

# Plot 2: Reionization history
ax2 = fig3.add_subplot(gs3[0, 1])
ax2.plot(results['mean_z'], results['mean_xhi'], 'go-', 
         linewidth=2, markersize=8, alpha=0.7)
ax2.set_xlabel('Mean Redshift', fontsize=14, fontweight='bold')
ax2.set_ylabel('Mean Neutral Fraction (mean_xHI)', fontsize=14, fontweight='bold')
ax2.set_title('Reionization History', fontsize=16, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.invert_xaxis()
ax2.set_ylim(0, 1)

# Plot 3: Scatter with redshift color-coding
ax3 = fig3.add_subplot(gs3[0, 2])
scatter = ax3.scatter(results['mean_xhi'], results['correlation'], 
                      c=results['mean_z'], s=150, cmap='viridis', 
                      edgecolors='black', linewidth=1.5, alpha=0.8)
cbar = plt.colorbar(scatter, ax=ax3, label='Redshift')
ax3.axhline(0, color='black', linewidth=0.8, linestyle=':')
ax3.set_xlabel('Mean Neutral Fraction (mean_xHI)', fontsize=14, fontweight='bold')
ax3.set_ylabel('Correlation r(|kSZ|, T21)', fontsize=14, fontweight='bold')
ax3.set_title('Correlation vs xHI (color = redshift)', fontsize=16, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.set_xlim(0, 1)

plt.suptitle('kSZ-21cm Cross-Correlation Evolution', 
             fontsize=18, fontweight='bold')
plt.savefig('plots/21cmfast_correlation_evolution.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# CELL 10: Correlation Coefficient r(ℓ) Analysis
# ============================================================================

# Choose 5 representative xHI values (epochs)
n_epochs_plot = min(5, len(results['mean_xhi']))
indices_to_plot_r = np.linspace(0, len(results['mean_xhi'])-1, n_epochs_plot, dtype=int)
colors_r = plt.cm.viridis(np.linspace(0, 1, n_epochs_plot))

fig4 = plt.figure(figsize=(16, 12))
gs4 = fig4.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# Plot 1: r(ℓ) for |kSZ| × 21cm - REAL velocity
ax1 = fig4.add_subplot(gs4[0, 0])
for idx, i in enumerate(indices_to_plot_r):
    ell = results['ell_cross'][i]
    r_k = results['r_k'][i]
    xhi = results['mean_xhi'][i]
    z = results['mean_z'][i]
    
    valid = np.isfinite(ell) & np.isfinite(r_k) & (ell <= 10000)
    if np.any(valid):
        ax1.plot(ell[valid], r_k[valid], '-', linewidth=2.5, alpha=0.8,
                 color=colors_r[idx], label=f'xHI={xhi:.2f}, z={z:.1f}')

ax1.axhline(0, color='black', linewidth=0.8, linestyle=':')
ax1.set_xlabel('Multipole ℓ', fontsize=12, fontweight='bold')
ax1.set_ylabel('Correlation Coefficient r(ℓ)', fontsize=12, fontweight='bold')
ax1.set_title('r(ℓ) for |kSZ| × 21cm - Real Velocity', fontsize=14, fontweight='bold')
ax1.legend(fontsize=9, loc='best')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(-1, 1)
ax1.set_xlim(0, 10000)

# Plot 2: r(ℓ) for |kSZ| × 21cm - RECONSTRUCTED velocity
ax2 = fig4.add_subplot(gs4[0, 1])
for idx, i in enumerate(indices_to_plot_r):
    ell = results['ell_cross_rec'][i]
    r_k = results['r_k_rec'][i]
    xhi = results['mean_xhi'][i]
    z = results['mean_z'][i]
    
    valid = np.isfinite(ell) & np.isfinite(r_k) & (ell <= 10000)
    if np.any(valid):
        ax2.plot(ell[valid], r_k[valid], '-', linewidth=2.5, alpha=0.8,
                 color=colors_r[idx], label=f'xHI={xhi:.2f}, z={z:.1f}')

ax2.axhline(0, color='black', linewidth=0.8, linestyle=':')
ax2.set_xlabel('Multipole ℓ', fontsize=12, fontweight='bold')
ax2.set_ylabel('Correlation Coefficient r(ℓ)', fontsize=12, fontweight='bold')
ax2.set_title('r(ℓ) for |kSZ| × 21cm - Reconstructed Velocity', fontsize=14, fontweight='bold')
ax2.legend(fontsize=9, loc='best')
ax2.grid(True, alpha=0.3)
ax2.set_ylim(-1, 1)
ax2.set_xlim(0, 10000)

# Plot 3: r(ℓ) for kSZ² × 21cm - REAL velocity
ax3 = fig4.add_subplot(gs4[1, 0])
for idx, i in enumerate(indices_to_plot_r):
    ell = results['ell_cross_sq'][i]
    r_k_sq = results['r_k_sq'][i]
    xhi = results['mean_xhi'][i]
    z = results['mean_z'][i]
    
    valid = np.isfinite(ell) & np.isfinite(r_k_sq) & (ell <= 10000)
    if np.any(valid):
        ax3.plot(ell[valid], r_k_sq[valid], '-', linewidth=2.5, alpha=0.8,
                 color=colors_r[idx], label=f'xHI={xhi:.2f}, z={z:.1f}')

ax3.axhline(0, color='black', linewidth=0.8, linestyle=':')
ax3.set_xlabel('Multipole ℓ', fontsize=12, fontweight='bold')
ax3.set_ylabel('Correlation Coefficient r(ℓ)', fontsize=12, fontweight='bold')
ax3.set_title('r(ℓ) for kSZ² × 21cm - Real Velocity', fontsize=14, fontweight='bold')
ax3.legend(fontsize=9, loc='best')
ax3.grid(True, alpha=0.3)
ax3.set_ylim(-1, 1)
ax3.set_xlim(0, 10000)

# Plot 4: r(ℓ) for kSZ² × 21cm - RECONSTRUCTED velocity
ax4 = fig4.add_subplot(gs4[1, 1])
for idx, i in enumerate(indices_to_plot_r):
    ell = results['ell_cross_sq_rec'][i]
    r_k_sq = results['r_k_sq_rec'][i]
    xhi = results['mean_xhi'][i]
    z = results['mean_z'][i]
    
    valid = np.isfinite(ell) & np.isfinite(r_k_sq) & (ell <= 10000)
    if np.any(valid):
        ax4.plot(ell[valid], r_k_sq[valid], '-', linewidth=2.5, alpha=0.8,
                 color=colors_r[idx], label=f'xHI={xhi:.2f}, z={z:.1f}')

ax4.axhline(0, color='black', linewidth=0.8, linestyle=':')
ax4.set_xlabel('Multipole ℓ', fontsize=12, fontweight='bold')
ax4.set_ylabel('Correlation Coefficient r(ℓ)', fontsize=12, fontweight='bold')
ax4.set_title('r(ℓ) for kSZ² × 21cm - Reconstructed Velocity', fontsize=14, fontweight='bold')
ax4.legend(fontsize=9, loc='best')
ax4.grid(True, alpha=0.3)
ax4.set_ylim(-1, 1)
ax4.set_xlim(0, 10000)

plt.suptitle('Correlation Coefficient r(ℓ) = P_cross / √(P_kSZ × P_21cm)', 
             fontsize=18, fontweight='bold')
plt.savefig('plots/21cmfast_r_ell_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# CELL 11: Summary Statistics
# ============================================================================

print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

# Find peak correlation (ignoring NaN values)
valid_mask_real = np.isfinite(results['correlation'])
valid_mask_rec = np.isfinite(results['correlation_rec'])

if np.any(valid_mask_real):
    idx_max_real = np.nanargmax(results['correlation'])
    print(f"\nPeak correlation (real velocity):")
    print(f"  r_max = {results['correlation'][idx_max_real]:+.4f}")
    print(f"  at mean_xHI = {results['mean_xhi'][idx_max_real]:.4f}")
    print(f"  at mean_z = {results['mean_z'][idx_max_real]:.3f}")
else:
    print("\nNo valid correlation data (real velocity)")

if np.any(valid_mask_rec):
    idx_max_rec = np.nanargmax(results['correlation_rec'])
    print(f"\nPeak correlation (reconstructed velocity):")
    print(f"  r_max = {results['correlation_rec'][idx_max_rec]:+.4f}")
    print(f"  at mean_xHI = {results['mean_xhi'][idx_max_rec]:.4f}")
    print(f"  at mean_z = {results['mean_z'][idx_max_rec]:.3f}")
else:
    print("\nNo valid correlation data (reconstructed velocity)")

# Correlation in different xHI regimes
high_xhi = (results['mean_xhi'] > 0.5) & valid_mask_real
low_xhi = (results['mean_xhi'] < 0.5) & valid_mask_real

if np.any(high_xhi):
    print(f"\nHigh neutral fraction (xHI > 0.5):")
    print(f"  Mean correlation (real): {np.nanmean(results['correlation'][high_xhi]):+.4f}")
    print(f"  Mean correlation (rec):  {np.nanmean(results['correlation_rec'][high_xhi]):+.4f}")

if np.any(low_xhi):
    print(f"\nLow neutral fraction (xHI < 0.5):")
    print(f"  Mean correlation (real): {np.nanmean(results['correlation'][low_xhi]):+.4f}")
    print(f"  Mean correlation (rec):  {np.nanmean(results['correlation_rec'][low_xhi]):+.4f}")

print("="*80)

# Save results to file
import os
os.makedirs('results', exist_ok=True)
np.savez('results/xhi_evolution_results.npz', **results)
print("\nResults saved to: results/xhi_evolution_results.npz")

print("\nAnalysis complete!")
