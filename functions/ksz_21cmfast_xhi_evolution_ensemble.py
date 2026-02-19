# ============================================================================
# KSZ-21CM CROSS-CORRELATION EVOLUTION - ENSEMBLE AVERAGE OVER 30 SIMULATIONS
# Analyzes how correlation changes with mean_xHI by chunking in redshift
# Processes each simulation separately and averages results at the end
# ============================================================================

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import fft
import warnings
from powerbox.tools import get_power

# ============================================================================
# CELL 1: Setup and Load Metadata
# ============================================================================

print("Setting up ensemble analysis for 30 simulations...")

sim_ids = list(range(12701, 12731))  # 30 simulations
n_sims = len(sim_ids)

# Load redshifts (same for all simulations)
redshifts = np.load("data_21cmfast/lightcone_redshifts.npy")

# Load first simulation to get dimensions
den_sample = np.load("data_21cmfast/density/12701_density_LC.npy")
nx, ny, nz = den_sample.shape
del den_sample  # Free memory

print(f"Number of simulations: {n_sims}")
print(f"Shape (nx, ny, nz): ({nx}, {ny}, {nz})")
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
        field = delta 
    elif weight == "delta_xhi":
        if xhi_xyz is None:
            raise ValueError("xhi_xyz required")
        field = (delta) * xhi_xyz.astype(dtype, copy=False)
    elif weight == "deltaXhi":
        if xhi_xyz is None:
            raise ValueError("xhi_xyz required")
        field = (1+delta) * xhi_xyz.astype(dtype, copy=False)
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
    T_b ∝ x_HI × (1 + δ) × (1 - (1/aH) × dv_∥/dr_∥)
    
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
# CELL 6: Process Single Chunk Function
# ============================================================================

def process_single_chunk(den_chunk, xhi_chunk, vz_chunk, redshifts_chunk):
    """
    Process a single chunk and return all relevant statistics.
    
    Returns: dict with all chunk results
    """
    # Reconstruct velocity for this chunk
    z_ref_chunk = float(np.mean(redshifts_chunk))
    
    vz_rec_chunk_minus = reconstruct_velocity_single_z_method(
        den_chunk, xhi_xyz=xhi_chunk,
        weight="deltaXhi",
        z_ref=z_ref_chunk,
        littleh=0.7,
        box_mpc_over_h=300.0
    )

    vz_rec_chunk = -vz_rec_chunk_minus
    
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
            ksz_abs_real, 300, bins=100, log_bins=True, get_variance=True)
        P_k_ksz_real = P_k_ksz_real * l_s_ksz_real ** 2
        
        # Cross-power: |kSZ| x 21cm (real)
        P_k_cross_real, l_s_cross_real, err_cross_real = get_power(
            deltax=ksz_abs_real, deltax2=t21, boxlength=300,
            bins=100, log_bins=True, get_variance=True)
        P_k_cross_real = P_k_cross_real * l_s_cross_real ** 2
        err_cross_real = np.sqrt(err_cross_real) * l_s_cross_real ** 2
        
        # Cross-power: kSZ^2 x 21cm (real)
        P_k_cross_sq_real, l_s_cross_sq_real, err_cross_sq_real = get_power(
            deltax=ksz_real**2, deltax2=t21, boxlength=300,
            bins=100, log_bins=True, get_variance=True)
        P_k_cross_sq_real = P_k_cross_sq_real * l_s_cross_sq_real ** 2
        err_cross_sq_real = np.sqrt(err_cross_sq_real) * l_s_cross_sq_real ** 2
        
        # Auto-power: kSZ^2 (real)
        P_k_ksz_sq_real, l_s_ksz_sq_real, err_ksz_sq_real = get_power(
            ksz_real**2, 300, bins=100, log_bins=True, get_variance=True)
        P_k_ksz_sq_real = P_k_ksz_sq_real * l_s_ksz_sq_real ** 2
        
        # === RECONSTRUCTED VELOCITY ===
        # Auto-power: |kSZ| (reconstructed)
        P_k_ksz_rec, l_s_ksz_rec, err_ksz_rec = get_power(
            ksz_abs_rec, 300, bins=100, log_bins=True, get_variance=True)
        P_k_ksz_rec = P_k_ksz_rec * l_s_ksz_rec ** 2
        
        # Cross-power: |kSZ| x 21cm (reconstructed)
        P_k_cross_rec, l_s_cross_rec, err_cross_rec = get_power(
            deltax=ksz_abs_rec, deltax2=t21, boxlength=300,
            bins=100, log_bins=True, get_variance=True)
        P_k_cross_rec = P_k_cross_rec * l_s_cross_rec ** 2
        err_cross_rec = np.sqrt(err_cross_rec) * l_s_cross_rec ** 2
        
        # Cross-power: kSZ^2 x 21cm (reconstructed)
        P_k_cross_sq_rec, l_s_cross_sq_rec, err_cross_sq_rec = get_power(
            deltax=ksz_rec**2, deltax2=t21, boxlength=300,
            bins=100, log_bins=True, get_variance=True)
        P_k_cross_sq_rec = P_k_cross_sq_rec * l_s_cross_sq_rec ** 2
        err_cross_sq_rec = np.sqrt(err_cross_sq_rec) * l_s_cross_sq_rec ** 2
        
        # Auto-power: kSZ^2 (reconstructed)
        P_k_ksz_sq_rec, l_s_ksz_sq_rec, err_ksz_sq_rec = get_power(
            ksz_rec**2, 300, bins=100, log_bins=True, get_variance=True)
        P_k_ksz_sq_rec = P_k_ksz_sq_rec * l_s_ksz_sq_rec ** 2
        
        # Auto-power: 21cm
        P_k_21cm, l_s_21cm, err_21cm = get_power(
            t21, 300, bins=100, log_bins=True, get_variance=True)
        P_k_21cm = P_k_21cm * l_s_21cm ** 2
        err_21cm = np.sqrt(err_21cm) * l_s_21cm ** 2
        
        # === NEW: CROSS-CORRELATIONS WITH 21cm^2 ===
        # Auto-power: 21cm^2
        P_k_21cm_sq, l_s_21cm_sq, err_21cm_sq = get_power(
            t21**2, 300, bins=100, log_bins=True, get_variance=True)
        P_k_21cm_sq = P_k_21cm_sq * l_s_21cm_sq ** 2
        
        # Cross-power: |kSZ| x 21cm^2 (real)
        P_k_cross_t21sq_real, l_s_cross_t21sq_real, err_cross_t21sq_real = get_power(
            deltax=ksz_abs_real, deltax2=t21**2, boxlength=300,
            bins=100, log_bins=True, get_variance=True)
        P_k_cross_t21sq_real = P_k_cross_t21sq_real * l_s_cross_t21sq_real ** 2
        err_cross_t21sq_real = np.sqrt(err_cross_t21sq_real) * l_s_cross_t21sq_real ** 2
        
        # Cross-power: kSZ^2 x 21cm^2 (real)
        P_k_cross_sq_t21sq_real, l_s_cross_sq_t21sq_real, err_cross_sq_t21sq_real = get_power(
            deltax=ksz_real**2, deltax2=t21**2, boxlength=300,
            bins=100, log_bins=True, get_variance=True)
        P_k_cross_sq_t21sq_real = P_k_cross_sq_t21sq_real * l_s_cross_sq_t21sq_real ** 2
        err_cross_sq_t21sq_real = np.sqrt(err_cross_sq_t21sq_real) * l_s_cross_sq_t21sq_real ** 2
        
        # Cross-power: |kSZ| x 21cm^2 (reconstructed)
        P_k_cross_t21sq_rec, l_s_cross_t21sq_rec, err_cross_t21sq_rec = get_power(
            deltax=ksz_abs_rec, deltax2=t21**2, boxlength=300,
            bins=100, log_bins=True, get_variance=True)
        P_k_cross_t21sq_rec = P_k_cross_t21sq_rec * l_s_cross_t21sq_rec ** 2
        err_cross_t21sq_rec = np.sqrt(err_cross_t21sq_rec) * l_s_cross_t21sq_rec ** 2
        
        # Cross-power: kSZ^2 x 21cm^2 (reconstructed)
        P_k_cross_sq_t21sq_rec, l_s_cross_sq_t21sq_rec, err_cross_sq_t21sq_rec = get_power(
            deltax=ksz_rec**2, deltax2=t21**2, boxlength=300,
            bins=100, log_bins=True, get_variance=True)
        P_k_cross_sq_t21sq_rec = P_k_cross_sq_t21sq_rec * l_s_cross_sq_t21sq_rec ** 2
        err_cross_sq_t21sq_rec = np.sqrt(err_cross_sq_t21sq_rec) * l_s_cross_sq_t21sq_rec ** 2
    
    # Compute correlation coefficients r(k)
    r_k_real = P_k_cross_real / np.sqrt(P_k_21cm * P_k_ksz_real)
    r_k_sq_real = P_k_cross_sq_real / np.sqrt(P_k_21cm * P_k_ksz_sq_real)
    r_k_rec = P_k_cross_rec / np.sqrt(P_k_21cm * P_k_ksz_rec)
    r_k_sq_rec = P_k_cross_sq_rec / np.sqrt(P_k_21cm * P_k_ksz_sq_rec)
    
    # NEW: Correlation coefficients with 21cm^2
    r_k_t21sq_real = P_k_cross_t21sq_real / np.sqrt(P_k_21cm_sq * P_k_ksz_real)
    r_k_sq_t21sq_real = P_k_cross_sq_t21sq_real / np.sqrt(P_k_21cm_sq * P_k_ksz_sq_real)
    r_k_t21sq_rec = P_k_cross_t21sq_rec / np.sqrt(P_k_21cm_sq * P_k_ksz_rec)
    r_k_sq_t21sq_rec = P_k_cross_sq_t21sq_rec / np.sqrt(P_k_21cm_sq * P_k_ksz_sq_rec)
    
    # Convert k to ell
    ell_cross_real = k_to_ell(l_s_cross_real, mean_z)
    ell_cross_sq_real = k_to_ell(l_s_cross_sq_real, mean_z)
    ell_cross_rec = k_to_ell(l_s_cross_rec, mean_z)
    ell_cross_sq_rec = k_to_ell(l_s_cross_sq_rec, mean_z)
    ell_21cm = k_to_ell(l_s_21cm, mean_z)
    
    # NEW: ell for 21cm^2 cross-correlations
    ell_cross_t21sq_real = k_to_ell(l_s_cross_t21sq_real, mean_z)
    ell_cross_sq_t21sq_real = k_to_ell(l_s_cross_sq_t21sq_real, mean_z)
    ell_cross_t21sq_rec = k_to_ell(l_s_cross_t21sq_rec, mean_z)
    ell_cross_sq_t21sq_rec = k_to_ell(l_s_cross_sq_t21sq_rec, mean_z)
    
    # Return all results as a dictionary
    return {
        'mean_xhi': mean_xhi,
        'mean_z': mean_z,
        'z_min': redshifts_chunk.min(),
        'z_max': redshifts_chunk.max(),
        'correlation': r_real,
        'correlation_rec': r_rec,
        # Real velocity
        'P_k_cross': P_k_cross_real,
        'l_s_cross': l_s_cross_real,
        'err_cross': err_cross_real,
        'ell_cross': ell_cross_real,
        'P_k_cross_sq': P_k_cross_sq_real,
        'l_s_cross_sq': l_s_cross_sq_real,
        'err_cross_sq': err_cross_sq_real,
        'ell_cross_sq': ell_cross_sq_real,
        'r_k': r_k_real,
        'r_k_sq': r_k_sq_real,
        # Reconstructed velocity
        'P_k_cross_rec': P_k_cross_rec,
        'err_cross_rec': err_cross_rec,
        'ell_cross_rec': ell_cross_rec,
        'P_k_cross_sq_rec': P_k_cross_sq_rec,
        'err_cross_sq_rec': err_cross_sq_rec,
        'ell_cross_sq_rec': ell_cross_sq_rec,
        'r_k_rec': r_k_rec,
        'r_k_sq_rec': r_k_sq_rec,
        # 21cm
        'P_k_21cm': P_k_21cm,
        'err_21cm': err_21cm,
        'ell_21cm': ell_21cm,
        # NEW: 21cm^2 cross-correlations
        'P_k_cross_t21sq': P_k_cross_t21sq_real,
        'err_cross_t21sq': err_cross_t21sq_real,
        'ell_cross_t21sq': ell_cross_t21sq_real,
        'P_k_cross_sq_t21sq': P_k_cross_sq_t21sq_real,
        'err_cross_sq_t21sq': err_cross_sq_t21sq_real,
        'ell_cross_sq_t21sq': ell_cross_sq_t21sq_real,
        'r_k_t21sq': r_k_t21sq_real,
        'r_k_sq_t21sq': r_k_sq_t21sq_real,
        # Reconstructed with 21cm^2
        'P_k_cross_t21sq_rec': P_k_cross_t21sq_rec,
        'err_cross_t21sq_rec': err_cross_t21sq_rec,
        'ell_cross_t21sq_rec': ell_cross_t21sq_rec,
        'P_k_cross_sq_t21sq_rec': P_k_cross_sq_t21sq_rec,
        'err_cross_sq_t21sq_rec': err_cross_sq_t21sq_rec,
        'ell_cross_sq_t21sq_rec': ell_cross_sq_t21sq_rec,
        'r_k_t21sq_rec': r_k_t21sq_rec,
        'r_k_sq_t21sq_rec': r_k_sq_t21sq_rec,
        # kSZ maps for reconstruction comparison
        'ksz_maps_real': ksz_real,
        'ksz_abs_maps_real': ksz_abs_real,
        'ksz_maps_rec': ksz_rec,
        'ksz_abs_maps_rec': ksz_abs_rec,
        # Velocity fields
        'vz_real': vz_chunk,
        'vz_rec': vz_rec_chunk,
    }

print("\nAll functions defined. Ready to process simulations.")

# ============================================================================
# CELL 7: Main Processing Loop - Iterate Over All Simulations
# ============================================================================

# Define chunking parameters
n_chunks = 20
chunk_size = 200

print(f"\nProcessing {n_sims} simulations with {n_chunks} chunks each...")
print(f"Chunk size: {chunk_size} z-slices")

# Storage for ensemble results - list of lists
# Each outer list element corresponds to a chunk
# Each inner list contains results from all simulations for that chunk
ensemble_storage = {
    'mean_xhi': [[] for _ in range(n_chunks)],
    'mean_z': [[] for _ in range(n_chunks)],
    'z_min': [[] for _ in range(n_chunks)],
    'z_max': [[] for _ in range(n_chunks)],
    'correlation': [[] for _ in range(n_chunks)],
    'correlation_rec': [[] for _ in range(n_chunks)],
    # Real velocity
    'P_k_cross': [[] for _ in range(n_chunks)],
    'l_s_cross': [[] for _ in range(n_chunks)],
    'err_cross': [[] for _ in range(n_chunks)],
    'ell_cross': [[] for _ in range(n_chunks)],
    'P_k_cross_sq': [[] for _ in range(n_chunks)],
    'l_s_cross_sq': [[] for _ in range(n_chunks)],
    'err_cross_sq': [[] for _ in range(n_chunks)],
    'ell_cross_sq': [[] for _ in range(n_chunks)],
    'r_k': [[] for _ in range(n_chunks)],
    'r_k_sq': [[] for _ in range(n_chunks)],
    # Reconstructed velocity
    'P_k_cross_rec': [[] for _ in range(n_chunks)],
    'err_cross_rec': [[] for _ in range(n_chunks)],
    'ell_cross_rec': [[] for _ in range(n_chunks)],
    'P_k_cross_sq_rec': [[] for _ in range(n_chunks)],
    'err_cross_sq_rec': [[] for _ in range(n_chunks)],
    'ell_cross_sq_rec': [[] for _ in range(n_chunks)],
    'r_k_rec': [[] for _ in range(n_chunks)],
    'r_k_sq_rec': [[] for _ in range(n_chunks)],
    # 21cm
    'P_k_21cm': [[] for _ in range(n_chunks)],
    'err_21cm': [[] for _ in range(n_chunks)],
    'ell_21cm': [[] for _ in range(n_chunks)],
    # NEW: 21cm^2 cross-correlations
    'P_k_cross_t21sq': [[] for _ in range(n_chunks)],
    'err_cross_t21sq': [[] for _ in range(n_chunks)],
    'ell_cross_t21sq': [[] for _ in range(n_chunks)],
    'P_k_cross_sq_t21sq': [[] for _ in range(n_chunks)],
    'err_cross_sq_t21sq': [[] for _ in range(n_chunks)],
    'ell_cross_sq_t21sq': [[] for _ in range(n_chunks)],
    'r_k_t21sq': [[] for _ in range(n_chunks)],
    'r_k_sq_t21sq': [[] for _ in range(n_chunks)],
    'P_k_cross_t21sq_rec': [[] for _ in range(n_chunks)],
    'err_cross_t21sq_rec': [[] for _ in range(n_chunks)],
    'ell_cross_t21sq_rec': [[] for _ in range(n_chunks)],
    'P_k_cross_sq_t21sq_rec': [[] for _ in range(n_chunks)],
    'err_cross_sq_t21sq_rec': [[] for _ in range(n_chunks)],
    'ell_cross_sq_t21sq_rec': [[] for _ in range(n_chunks)],
    'r_k_t21sq_rec': [[] for _ in range(n_chunks)],
    'r_k_sq_t21sq_rec': [[] for _ in range(n_chunks)],
    # kSZ maps for reconstruction comparison
    'ksz_maps_real': [[] for _ in range(n_chunks)],
    'ksz_abs_maps_real': [[] for _ in range(n_chunks)],
    'ksz_maps_rec': [[] for _ in range(n_chunks)],
    'ksz_abs_maps_rec': [[] for _ in range(n_chunks)],
    # Velocity fields
    'vz_real': [[] for _ in range(n_chunks)],
    'vz_rec': [[] for _ in range(n_chunks)],
}

# Process each simulation
for sim_idx, sim_id in enumerate(sim_ids):
    print(f"\n{'='*80}")
    print(f"SIMULATION {sim_idx+1}/{n_sims}: ID={sim_id}")
    print(f"{'='*80}")
    
    # Load data for this simulation
    print(f"Loading simulation {sim_id}...")
    den = np.load(f"data_21cmfast/density/{sim_id}_density_LC.npy")
    xhi = np.load(f"data_21cmfast/xHI/{sim_id}_xHI_LC.npy")
    vz = np.load(f"data_21cmfast/velocity/{sim_id}_velocity_z_LC.npy")
    
    # Process each chunk for this simulation
    for i in range(n_chunks):
        z_start = i * chunk_size
        z_end = min((i + 1) * chunk_size, nz)
        
        if z_end - z_start < 10:  # Skip very small chunks
            continue
        
        print(f"  Chunk {i+1}/{n_chunks}: z-slices [{z_start}:{z_end}]", end='')
        
        # Extract chunk
        den_chunk = den[:, :, z_start:z_end]
        xhi_chunk = xhi[:, :, z_start:z_end]
        vz_chunk = vz[:, :, z_start:z_end]
        redshifts_chunk = redshifts[z_start:z_end]
        
        # Process this chunk
        chunk_results = process_single_chunk(den_chunk, xhi_chunk, vz_chunk, redshifts_chunk)
        
        # Store results in ensemble storage
        for key in ensemble_storage.keys():
            ensemble_storage[key][i].append(chunk_results[key])
        
        print(f" - xHI={chunk_results['mean_xhi']:.4f}, z={chunk_results['mean_z']:.3f}, r={chunk_results['correlation']:+.4f}")
    
    # Free memory
    del den, xhi, vz
    print(f"Simulation {sim_id} complete!")

print(f"\n{'='*80}")
print("All simulations processed!")
print(f"{'='*80}")

# ============================================================================
# CELL 8: Compute Ensemble Averages
# ============================================================================

print("\nComputing ensemble averages across simulations...")

import warnings

# Final results storage
results = {
    'mean_xhi': [],
    'mean_z': [],
    'z_min': [],
    'z_max': [],
    'correlation': [],
    'correlation_rec': [],
    'correlation_std': [],
    'correlation_rec_std': [],
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
    # 21cm
    'P_k_21cm': [],
    'err_21cm': [],
    'ell_21cm': [],
    # NEW: 21cm^2 cross-correlations
    'P_k_cross_t21sq': [],
    'err_cross_t21sq': [],
    'ell_cross_t21sq': [],
    'P_k_cross_sq_t21sq': [],
    'err_cross_sq_t21sq': [],
    'ell_cross_sq_t21sq': [],
    'r_k_t21sq': [],
    'r_k_sq_t21sq': [],
    'P_k_cross_t21sq_rec': [],
    'err_cross_t21sq_rec': [],
    'ell_cross_t21sq_rec': [],
    'P_k_cross_sq_t21sq_rec': [],
    'err_cross_sq_t21sq_rec': [],
    'ell_cross_sq_t21sq_rec': [],
    'r_k_t21sq_rec': [],
    'r_k_sq_t21sq_rec': [],
    # kSZ maps for reconstruction comparison (ensemble averaged)
    'ksz_maps_real': [],
    'ksz_abs_maps_real': [],
    'ksz_maps_rec': [],
    'ksz_abs_maps_rec': [],
    # Velocity fields (ensemble averaged)
    'vz_real': [],
    'vz_rec': [],
}

# Average over simulations for each chunk
# Suppress warnings about empty slices (some chunks may not have data from all sims)
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=RuntimeWarning, message='Mean of empty slice')
    warnings.filterwarnings('ignore', category=RuntimeWarning, message='Degrees of freedom')
    
    for i in range(n_chunks):
        # Skip if no data for this chunk
        if len(ensemble_storage['mean_xhi'][i]) == 0:
            continue
        
        # Scalars - simple mean
        results['mean_xhi'].append(np.nanmean(ensemble_storage['mean_xhi'][i]))
        results['mean_z'].append(np.nanmean(ensemble_storage['mean_z'][i]))
        results['z_min'].append(np.nanmean(ensemble_storage['z_min'][i]))
        results['z_max'].append(np.nanmean(ensemble_storage['z_max'][i]))
        results['correlation'].append(np.nanmean(ensemble_storage['correlation'][i]))
        results['correlation_rec'].append(np.nanmean(ensemble_storage['correlation_rec'][i]))
        results['correlation_std'].append(np.nanstd(ensemble_storage['correlation'][i]))
        results['correlation_rec_std'].append(np.nanstd(ensemble_storage['correlation_rec'][i]))
        
        # Arrays - mean over simulations
        results['P_k_cross'].append(np.nanmean(ensemble_storage['P_k_cross'][i], axis=0))
        results['l_s_cross'].append(ensemble_storage['l_s_cross'][i][0])  # Same for all sims
        results['err_cross'].append(np.nanstd(ensemble_storage['P_k_cross'][i], axis=0) / np.sqrt(n_sims))
        results['ell_cross'].append(ensemble_storage['ell_cross'][i][0])  # Same for all sims
        
        results['P_k_cross_sq'].append(np.nanmean(ensemble_storage['P_k_cross_sq'][i], axis=0))
        results['l_s_cross_sq'].append(ensemble_storage['l_s_cross_sq'][i][0])
        results['err_cross_sq'].append(np.nanstd(ensemble_storage['P_k_cross_sq'][i], axis=0) / np.sqrt(n_sims))
        results['ell_cross_sq'].append(ensemble_storage['ell_cross_sq'][i][0])
        
        results['r_k'].append(np.nanmean(ensemble_storage['r_k'][i], axis=0))
        results['r_k_sq'].append(np.nanmean(ensemble_storage['r_k_sq'][i], axis=0))
        
        # Reconstructed velocity
        results['P_k_cross_rec'].append(np.nanmean(ensemble_storage['P_k_cross_rec'][i], axis=0))
        results['err_cross_rec'].append(np.nanstd(ensemble_storage['P_k_cross_rec'][i], axis=0) / np.sqrt(n_sims))
        results['ell_cross_rec'].append(ensemble_storage['ell_cross_rec'][i][0])
        
        results['P_k_cross_sq_rec'].append(np.nanmean(ensemble_storage['P_k_cross_sq_rec'][i], axis=0))
        results['err_cross_sq_rec'].append(np.nanstd(ensemble_storage['P_k_cross_sq_rec'][i], axis=0) / np.sqrt(n_sims))
        results['ell_cross_sq_rec'].append(ensemble_storage['ell_cross_sq_rec'][i][0])
        
        results['r_k_rec'].append(np.nanmean(ensemble_storage['r_k_rec'][i], axis=0))
        results['r_k_sq_rec'].append(np.nanmean(ensemble_storage['r_k_sq_rec'][i], axis=0))
        
        # 21cm
        results['P_k_21cm'].append(np.nanmean(ensemble_storage['P_k_21cm'][i], axis=0))
        results['err_21cm'].append(np.nanstd(ensemble_storage['P_k_21cm'][i], axis=0) / np.sqrt(n_sims))
        results['ell_21cm'].append(ensemble_storage['ell_21cm'][i][0])
        
        # NEW: 21cm^2 cross-correlations
        results['P_k_cross_t21sq'].append(np.nanmean(ensemble_storage['P_k_cross_t21sq'][i], axis=0))
        results['err_cross_t21sq'].append(np.nanstd(ensemble_storage['P_k_cross_t21sq'][i], axis=0) / np.sqrt(n_sims))
        results['ell_cross_t21sq'].append(ensemble_storage['ell_cross_t21sq'][i][0])
        
        results['P_k_cross_sq_t21sq'].append(np.nanmean(ensemble_storage['P_k_cross_sq_t21sq'][i], axis=0))
        results['err_cross_sq_t21sq'].append(np.nanstd(ensemble_storage['P_k_cross_sq_t21sq'][i], axis=0) / np.sqrt(n_sims))
        results['ell_cross_sq_t21sq'].append(ensemble_storage['ell_cross_sq_t21sq'][i][0])
        
        results['r_k_t21sq'].append(np.nanmean(ensemble_storage['r_k_t21sq'][i], axis=0))
        results['r_k_sq_t21sq'].append(np.nanmean(ensemble_storage['r_k_sq_t21sq'][i], axis=0))
        
        results['P_k_cross_t21sq_rec'].append(np.nanmean(ensemble_storage['P_k_cross_t21sq_rec'][i], axis=0))
        results['err_cross_t21sq_rec'].append(np.nanstd(ensemble_storage['P_k_cross_t21sq_rec'][i], axis=0) / np.sqrt(n_sims))
        results['ell_cross_t21sq_rec'].append(ensemble_storage['ell_cross_t21sq_rec'][i][0])
        
        results['P_k_cross_sq_t21sq_rec'].append(np.nanmean(ensemble_storage['P_k_cross_sq_t21sq_rec'][i], axis=0))
        results['err_cross_sq_t21sq_rec'].append(np.nanstd(ensemble_storage['P_k_cross_sq_t21sq_rec'][i], axis=0) / np.sqrt(n_sims))
        results['ell_cross_sq_t21sq_rec'].append(ensemble_storage['ell_cross_sq_t21sq_rec'][i][0])
        
        results['r_k_t21sq_rec'].append(np.nanmean(ensemble_storage['r_k_t21sq_rec'][i], axis=0))
        results['r_k_sq_t21sq_rec'].append(np.nanmean(ensemble_storage['r_k_sq_t21sq_rec'][i], axis=0))
        
        # kSZ maps - average over simulations
        results['ksz_maps_real'].append(np.nanmean(ensemble_storage['ksz_maps_real'][i], axis=0))
        results['ksz_abs_maps_real'].append(np.nanmean(ensemble_storage['ksz_abs_maps_real'][i], axis=0))
        results['ksz_maps_rec'].append(np.nanmean(ensemble_storage['ksz_maps_rec'][i], axis=0))
        results['ksz_abs_maps_rec'].append(np.nanmean(ensemble_storage['ksz_abs_maps_rec'][i], axis=0))
        
        # Velocity fields - average over simulations
        results['vz_real'].append(np.nanmean(ensemble_storage['vz_real'][i], axis=0))
        results['vz_rec'].append(np.nanmean(ensemble_storage['vz_rec'][i], axis=0))

# Store ensemble_storage temporarily for computing r_k std
ensemble_storage_r_k = ensemble_storage['r_k']
ensemble_storage_r_k_sq = ensemble_storage['r_k_sq']
ensemble_storage_r_k_rec = ensemble_storage['r_k_rec']
ensemble_storage_r_k_sq_rec = ensemble_storage['r_k_sq_rec']
# NEW: Store 21cm^2 r_k for std computation
ensemble_storage_r_k_t21sq = ensemble_storage['r_k_t21sq']
ensemble_storage_r_k_sq_t21sq = ensemble_storage['r_k_sq_t21sq']
ensemble_storage_r_k_t21sq_rec = ensemble_storage['r_k_t21sq_rec']
ensemble_storage_r_k_sq_t21sq_rec = ensemble_storage['r_k_sq_t21sq_rec']

# Convert to arrays (except velocity fields which have variable z-dimension)
for key in results:
    if key not in ['vz_real', 'vz_rec']:
        results[key] = np.array(results[key])

print(f"Ensemble averaging complete! {len(results['mean_xhi'])} chunks with data.")

# Compute std for r_k arrays
results['r_k_std'] = []
results['r_k_sq_std'] = []
results['r_k_rec_std'] = []
results['r_k_sq_rec_std'] = []
# NEW: 21cm^2 std
results['r_k_t21sq_std'] = []
results['r_k_sq_t21sq_std'] = []
results['r_k_t21sq_rec_std'] = []
results['r_k_sq_t21sq_rec_std'] = []

for i in range(len(results['mean_xhi'])):
    # Find original chunk index
    chunk_idx = i
    results['r_k_std'].append(np.nanstd(ensemble_storage_r_k[chunk_idx], axis=0))
    results['r_k_sq_std'].append(np.nanstd(ensemble_storage_r_k_sq[chunk_idx], axis=0))
    results['r_k_rec_std'].append(np.nanstd(ensemble_storage_r_k_rec[chunk_idx], axis=0))
    results['r_k_sq_rec_std'].append(np.nanstd(ensemble_storage_r_k_sq_rec[chunk_idx], axis=0))
    # NEW: 21cm^2 std
    results['r_k_t21sq_std'].append(np.nanstd(ensemble_storage_r_k_t21sq[chunk_idx], axis=0))
    results['r_k_sq_t21sq_std'].append(np.nanstd(ensemble_storage_r_k_sq_t21sq[chunk_idx], axis=0))
    results['r_k_t21sq_rec_std'].append(np.nanstd(ensemble_storage_r_k_t21sq_rec[chunk_idx], axis=0))
    results['r_k_sq_t21sq_rec_std'].append(np.nanstd(ensemble_storage_r_k_sq_t21sq_rec[chunk_idx], axis=0))

results['r_k_std'] = np.array(results['r_k_std'])
results['r_k_sq_std'] = np.array(results['r_k_sq_std'])
results['r_k_rec_std'] = np.array(results['r_k_rec_std'])
results['r_k_sq_rec_std'] = np.array(results['r_k_sq_rec_std'])
# NEW: 21cm^2 std
results['r_k_t21sq_std'] = np.array(results['r_k_t21sq_std'])
results['r_k_sq_t21sq_std'] = np.array(results['r_k_sq_t21sq_std'])
results['r_k_t21sq_rec_std'] = np.array(results['r_k_t21sq_rec_std'])
results['r_k_sq_t21sq_rec_std'] = np.array(results['r_k_sq_t21sq_rec_std'])

# Free memory
del ensemble_storage
del ensemble_storage_r_k, ensemble_storage_r_k_sq, ensemble_storage_r_k_rec, ensemble_storage_r_k_sq_rec
del ensemble_storage_r_k_t21sq, ensemble_storage_r_k_sq_t21sq, ensemble_storage_r_k_t21sq_rec, ensemble_storage_r_k_sq_t21sq_rec

print("\nEnsemble analysis complete!")

# ============================================================================
# CELL 9: Create Plots Directory
# ============================================================================

os.makedirs('plots', exist_ok=True)

# ============================================================================
# CELL 10: Figure 1 - Cross-Power vs xHI at Fixed Multipoles
# ============================================================================

# Find available ell range
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
    
    fig1 = plt.figure(figsize=(12, 8))
    ax = fig1.add_subplot(111)
    
    colors_ell = plt.cm.plasma(np.linspace(0, 1, len(ell_targets_desired)))
    
    for color_idx, ell_target in enumerate(ell_targets_desired):
        idx_closest = np.argmin(np.abs(valid_ell - ell_target))
        ell_actual = valid_ell[idx_closest]
        
        P_at_ell = []
        err_at_ell = []
        xhi_vals = []
        
        for i in range(len(results['mean_xhi'])):
            ell_arr = results['ell_cross'][i]
            P_arr = results['P_k_cross'][i]
            err_arr = results['err_cross'][i]
            
            valid_mask = np.isfinite(ell_arr)
            if not np.any(valid_mask):
                continue
                
            idx_ell = np.argmin(np.abs(ell_arr[valid_mask] - ell_actual))
            actual_idx = np.where(valid_mask)[0][idx_ell]
            
            if np.isfinite(P_arr[actual_idx]) and np.isfinite(err_arr[actual_idx]):
                P_at_ell.append(P_arr[actual_idx])
                err_at_ell.append(err_arr[actual_idx])
                xhi_vals.append(results['mean_xhi'][i])
        
        P_at_ell = np.array(P_at_ell)
        err_at_ell = np.array(err_at_ell)
        xhi_vals = np.array(xhi_vals)
        
        if len(P_at_ell) > 0:
            # Plot line with shaded error region
            ax.plot(xhi_vals, P_at_ell, 'o-', linewidth=2, markersize=6, alpha=0.8,
                    color=colors_ell[color_idx], label=f'ℓ = {ell_actual:.0f}')
            ax.fill_between(xhi_vals, P_at_ell - err_at_ell, P_at_ell + err_at_ell,
                           alpha=0.2, color=colors_ell[color_idx])
    
    ax.set_xlabel('Mean Neutral Fraction (xHI)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Cross-Power P(ℓ)', fontsize=14, fontweight='bold')
    ax.set_title('|kSZ| × 21cm Cross-Power at Fixed ℓ vs Neutral Fraction\n(Ensemble Average over 30 Simulations)', 
                 fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    
    os.makedirs('plots/plots_multiple_sims', exist_ok=True)
    plt.savefig('plots/plots_multiple_sims/cross_power_vs_xhi_ensemble.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Figure 1 saved!")
else:
    print("\nWarning: No valid ell data found!")

# ============================================================================
# CELL 11: Figure 2 - Cross-Power Spectra at Different Epochs
# ============================================================================

# Select specific xHI values to plot: ~0.03, ~0.22, ~0.50, ~0.79, ~0.96
target_xhi_values = [0.03, 0.22, 0.50, 0.79, 0.96]
indices_to_plot = []

for target_xhi in target_xhi_values:
    # Find closest chunk to target xHI
    idx = np.argmin(np.abs(results['mean_xhi'] - target_xhi))
    if idx not in indices_to_plot:  # Avoid duplicates
        indices_to_plot.append(idx)

indices_to_plot = np.array(indices_to_plot)
n_epochs = len(indices_to_plot)

fig2 = plt.figure(figsize=(20, 10))
gs2 = fig2.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

# Plot 1: Full spectra
ax1 = fig2.add_subplot(gs2[0, :])
colors = plt.cm.viridis(np.linspace(0, 1, n_epochs))

for idx, i in enumerate(indices_to_plot):
    ell = results['ell_cross'][i]
    P_k = results['P_k_cross'][i]
    err = results['err_cross'][i]
    xhi = results['mean_xhi'][i]
    z = results['mean_z'][i]
    
    valid = np.isfinite(ell) & np.isfinite(P_k) & (ell <= 5000)
    if np.any(valid):
        # Plot line with shaded error region
        ax1.plot(ell[valid], P_k[valid], '-', linewidth=2, alpha=0.8,
                 color=colors[idx], label=f'xHI={xhi:.2f}, z={z:.1f}')
        ax1.fill_between(ell[valid], P_k[valid] - err[valid], P_k[valid] + err[valid],
                        alpha=0.2, color=colors[idx])

ax1.set_xlabel('Multipole ℓ', fontsize=14, fontweight='bold')
ax1.set_ylabel('Cross-Power P(ℓ)', fontsize=14, fontweight='bold')
ax1.set_title('|kSZ| × 21cm Cross-Power Evolution (Ensemble Average)', fontsize=16, fontweight='bold')
ax1.legend(fontsize=10, ncol=2, loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 5000)

ax1_top = ax1.twiny()
ax1_top.set_xlim(ax1.get_xlim())
ax1_top.set_xlabel('Redshift', fontsize=14, fontweight='bold')
ax1_top.set_xticks([])
ax1_top.text(0.5, 1.05, f'Redshift range: z={results["mean_z"].max():.1f} → {results["mean_z"].min():.1f}', 
             transform=ax1.transAxes, ha='center', fontsize=11)

# Plot 2: Zoomed
ax2 = fig2.add_subplot(gs2[1, 0])
for idx, i in enumerate(indices_to_plot):
    ell = results['ell_cross'][i]
    P_k = results['P_k_cross'][i]
    err = results['err_cross'][i]
    xhi = results['mean_xhi'][i]
    
    mask = (ell >= 2000) & (ell <= 4000)
    # Plot line with shaded error region
    ax2.plot(ell[mask], P_k[mask], 'o-', linewidth=2, markersize=6, alpha=0.8,
             color=colors[idx], label=f'xHI={xhi:.3f}')
    ax2.fill_between(ell[mask], P_k[mask] - err[mask], P_k[mask] + err[mask],
                    alpha=0.2, color=colors[idx])

ax2.set_xlabel('Multipole ℓ', fontsize=12, fontweight='bold')
ax2.set_ylabel('Cross-Power P(ℓ)', fontsize=12, fontweight='bold')
ax2.set_title('Zoomed: ℓ=[2000,4000]', fontsize=14, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# Plot 3: kSZ² × 21cm
ax3 = fig2.add_subplot(gs2[1, 1])
for idx, i in enumerate(indices_to_plot):
    ell = results['ell_cross_sq'][i]
    P_k = results['P_k_cross_sq'][i]
    xhi = results['mean_xhi'][i]
    
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

# Plot 4: kSZ² × 21cm² (NEW)
ax4 = fig2.add_subplot(gs2[1, 2])
for idx, i in enumerate(indices_to_plot):
    ell = results['ell_cross_sq_t21sq'][i]
    P_k = results['P_k_cross_sq_t21sq'][i]
    err = results['err_cross_sq_t21sq'][i]
    xhi = results['mean_xhi'][i]
    
    valid = np.isfinite(ell) & np.isfinite(P_k) & (ell <= 5000)
    if np.any(valid):
        ax4.plot(ell[valid], P_k[valid], '-', linewidth=2, alpha=0.7,
                 color=colors[idx], label=f'xHI={xhi:.2f}')
        ax4.fill_between(ell[valid], P_k[valid] - err[valid], P_k[valid] + err[valid],
                        alpha=0.2, color=colors[idx])

ax4.set_xlabel('Multipole ℓ', fontsize=12, fontweight='bold')
ax4.set_ylabel('Cross-Power P(ℓ)', fontsize=12, fontweight='bold')
ax4.set_title('kSZ² × 21cm² Cross-Power', fontsize=14, fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)
ax4.set_xlim(2000, 4000)

plt.suptitle('Cross-Power Spectra at Different Reionization Epochs\n(Ensemble Average over 30 Simulations)', 
             fontsize=18, fontweight='bold')
plt.savefig('plots/plots_multiple_sims/cross_power_spectra_all_ensemble.png', dpi=300, bbox_inches='tight')
plt.show()
print("Figure 2 saved!")

# ============================================================================
# CELL 12: Figure 3 - Correlation Evolution
# ============================================================================

fig3 = plt.figure(figsize=(18, 6))
gs3 = fig3.add_gridspec(1, 3, hspace=0.3, wspace=0.3)

# Plot 1: Correlation vs xHI with error bars
ax1 = fig3.add_subplot(gs3[0, 0])
ax1.errorbar(results['mean_xhi'], results['correlation'], 
             yerr=results['correlation_std']/np.sqrt(n_sims),
             fmt='bo-', label='Real velocity', linewidth=2, markersize=8, 
             alpha=0.7, capsize=4)
ax1.errorbar(results['mean_xhi'], results['correlation_rec'], 
             yerr=results['correlation_rec_std']/np.sqrt(n_sims),
             fmt='rs--', label='Reconstructed velocity', linewidth=2, markersize=8, 
             alpha=0.7, capsize=4)
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

plt.suptitle('kSZ-21cm Cross-Correlation Evolution (Ensemble Average over 30 Simulations)', 
             fontsize=18, fontweight='bold')
plt.savefig('plots/plots_multiple_sims/correlation_evolution_ensemble.png', dpi=300, bbox_inches='tight')
plt.show()
print("Figure 3 saved!")

# ============================================================================
# CELL 13: Figure 4 - r(ℓ) Analysis
# ============================================================================

# Select specific xHI values to plot: ~0.03, ~0.22, ~0.50, ~0.79, ~0.96
target_xhi_values_r = [0.03, 0.22, 0.50, 0.79, 0.96]
indices_to_plot_r = []

for target_xhi in target_xhi_values_r:
    # Find closest chunk to target xHI
    idx = np.argmin(np.abs(results['mean_xhi'] - target_xhi))
    if idx not in indices_to_plot_r:  # Avoid duplicates
        indices_to_plot_r.append(idx)

indices_to_plot_r = np.array(indices_to_plot_r)
n_epochs_plot = len(indices_to_plot_r)
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
    
    r_k_std = results['r_k_std'][i]
    
    valid = np.isfinite(ell) & np.isfinite(r_k) & (ell <= 10000)
    if np.any(valid):
        ax1.plot(ell[valid], r_k[valid], '-', linewidth=2.5, alpha=0.8,
                 color=colors_r[idx], label=f'xHI={xhi:.2f}, z={z:.1f}')
        # Add shaded region using std across simulations
        ax1.fill_between(ell[valid], r_k[valid] - r_k_std[valid], r_k[valid] + r_k_std[valid],
                        alpha=0.2, color=colors_r[idx])

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
    
    r_k_std = results['r_k_rec_std'][i]
    
    valid = np.isfinite(ell) & np.isfinite(r_k) & (ell <= 10000)
    if np.any(valid):
        ax2.plot(ell[valid], r_k[valid], '-', linewidth=2.5, alpha=0.8,
                 color=colors_r[idx], label=f'xHI={xhi:.2f}, z={z:.1f}')
        ax2.fill_between(ell[valid], r_k[valid] - r_k_std[valid], r_k[valid] + r_k_std[valid],
                        alpha=0.2, color=colors_r[idx])

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
    
    r_k_sq_std = results['r_k_sq_std'][i]
    
    valid = np.isfinite(ell) & np.isfinite(r_k_sq) & (ell <= 10000)
    if np.any(valid):
        ax3.plot(ell[valid], r_k_sq[valid], '-', linewidth=2.5, alpha=0.8,
                 color=colors_r[idx], label=f'xHI={xhi:.2f}, z={z:.1f}')
        ax3.fill_between(ell[valid], r_k_sq[valid] - r_k_sq_std[valid], r_k_sq[valid] + r_k_sq_std[valid],
                        alpha=0.2, color=colors_r[idx])

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
    
    r_k_sq_std = results['r_k_sq_rec_std'][i]
    
    valid = np.isfinite(ell) & np.isfinite(r_k_sq) & (ell <= 10000)
    if np.any(valid):
        ax4.plot(ell[valid], r_k_sq[valid], '-', linewidth=2.5, alpha=0.8,
                 color=colors_r[idx], label=f'xHI={xhi:.2f}, z={z:.1f}')
        ax4.fill_between(ell[valid], r_k_sq[valid] - r_k_sq_std[valid], r_k_sq[valid] + r_k_sq_std[valid],
                        alpha=0.2, color=colors_r[idx])

ax4.axhline(0, color='black', linewidth=0.8, linestyle=':')
ax4.set_xlabel('Multipole ℓ', fontsize=12, fontweight='bold')
ax4.set_ylabel('Correlation Coefficient r(ℓ)', fontsize=12, fontweight='bold')
ax4.set_title('r(ℓ) for kSZ² × 21cm - Reconstructed Velocity', fontsize=14, fontweight='bold')
ax4.legend(fontsize=9, loc='best')
ax4.grid(True, alpha=0.3)
ax4.set_ylim(-1, 1)
ax4.set_xlim(0, 10000)

plt.suptitle('Correlation Coefficient r(ℓ) = P_cross / √(P_kSZ × P_21cm)\n(Ensemble Average over 30 Simulations)', 
             fontsize=18, fontweight='bold')
plt.savefig('plots/plots_multiple_sims/r_ell_analysis_all_ensemble.png', dpi=300, bbox_inches='tight')
plt.show()
print("Figure 4 saved!")

# ============================================================================
# CELL 13b: Figure 5 - r(ℓ) Analysis with 21cm² (NEW)
# ============================================================================

fig5 = plt.figure(figsize=(16, 12))
gs5 = fig5.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# Plot 1: r(ℓ) for |kSZ| × 21cm² - REAL velocity
ax1 = fig5.add_subplot(gs5[0, 0])
for idx, i in enumerate(indices_to_plot_r):
    ell = results['ell_cross_t21sq'][i]
    r_k = results['r_k_t21sq'][i]
    xhi = results['mean_xhi'][i]
    z = results['mean_z'][i]
    
    r_k_std = results['r_k_t21sq_std'][i]
    
    valid = np.isfinite(ell) & np.isfinite(r_k) & (ell <= 10000)
    if np.any(valid):
        ax1.plot(ell[valid], r_k[valid], '-', linewidth=2.5, alpha=0.8,
                 color=colors_r[idx], label=f'xHI={xhi:.2f}, z={z:.1f}')
        ax1.fill_between(ell[valid], r_k[valid] - r_k_std[valid], r_k[valid] + r_k_std[valid],
                        alpha=0.2, color=colors_r[idx])

ax1.axhline(0, color='black', linewidth=0.8, linestyle=':')
ax1.set_xlabel('Multipole ℓ', fontsize=12, fontweight='bold')
ax1.set_ylabel('Correlation Coefficient r(ℓ)', fontsize=12, fontweight='bold')
ax1.set_title('r(ℓ) for |kSZ| × 21cm² - Real Velocity', fontsize=14, fontweight='bold')
ax1.legend(fontsize=9, loc='best')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(-1, 1)
ax1.set_xlim(0, 10000)

# Plot 2: r(ℓ) for |kSZ| × 21cm² - RECONSTRUCTED velocity
ax2 = fig5.add_subplot(gs5[0, 1])
for idx, i in enumerate(indices_to_plot_r):
    ell = results['ell_cross_t21sq_rec'][i]
    r_k = results['r_k_t21sq_rec'][i]
    xhi = results['mean_xhi'][i]
    z = results['mean_z'][i]
    
    r_k_std = results['r_k_t21sq_rec_std'][i]
    
    valid = np.isfinite(ell) & np.isfinite(r_k) & (ell <= 10000)
    if np.any(valid):
        ax2.plot(ell[valid], r_k[valid], '-', linewidth=2.5, alpha=0.8,
                 color=colors_r[idx], label=f'xHI={xhi:.2f}, z={z:.1f}')
        ax2.fill_between(ell[valid], r_k[valid] - r_k_std[valid], r_k[valid] + r_k_std[valid],
                        alpha=0.2, color=colors_r[idx])

ax2.axhline(0, color='black', linewidth=0.8, linestyle=':')
ax2.set_xlabel('Multipole ℓ', fontsize=12, fontweight='bold')
ax2.set_ylabel('Correlation Coefficient r(ℓ)', fontsize=12, fontweight='bold')
ax2.set_title('r(ℓ) for |kSZ| × 21cm² - Reconstructed Velocity', fontsize=14, fontweight='bold')
ax2.legend(fontsize=9, loc='best')
ax2.grid(True, alpha=0.3)
ax2.set_ylim(-1, 1)
ax2.set_xlim(0, 10000)

# Plot 3: r(ℓ) for kSZ² × 21cm² - REAL velocity
ax3 = fig5.add_subplot(gs5[1, 0])
for idx, i in enumerate(indices_to_plot_r):
    ell = results['ell_cross_sq_t21sq'][i]
    r_k_sq = results['r_k_sq_t21sq'][i]
    xhi = results['mean_xhi'][i]
    z = results['mean_z'][i]
    
    r_k_sq_std = results['r_k_sq_t21sq_std'][i]
    
    valid = np.isfinite(ell) & np.isfinite(r_k_sq) & (ell <= 10000)
    if np.any(valid):
        ax3.plot(ell[valid], r_k_sq[valid], '-', linewidth=2.5, alpha=0.8,
                 color=colors_r[idx], label=f'xHI={xhi:.2f}, z={z:.1f}')
        ax3.fill_between(ell[valid], r_k_sq[valid] - r_k_sq_std[valid], r_k_sq[valid] + r_k_sq_std[valid],
                        alpha=0.2, color=colors_r[idx])

ax3.axhline(0, color='black', linewidth=0.8, linestyle=':')
ax3.set_xlabel('Multipole ℓ', fontsize=12, fontweight='bold')
ax3.set_ylabel('Correlation Coefficient r(ℓ)', fontsize=12, fontweight='bold')
ax3.set_title('r(ℓ) for kSZ² × 21cm² - Real Velocity', fontsize=14, fontweight='bold')
ax3.legend(fontsize=9, loc='best')
ax3.grid(True, alpha=0.3)
ax3.set_ylim(-1, 1)
ax3.set_xlim(0, 10000)

# Plot 4: r(ℓ) for kSZ² × 21cm² - RECONSTRUCTED velocity
ax4 = fig5.add_subplot(gs5[1, 1])
for idx, i in enumerate(indices_to_plot_r):
    ell = results['ell_cross_sq_t21sq_rec'][i]
    r_k_sq = results['r_k_sq_t21sq_rec'][i]
    xhi = results['mean_xhi'][i]
    z = results['mean_z'][i]
    
    r_k_sq_std = results['r_k_sq_t21sq_rec_std'][i]
    
    valid = np.isfinite(ell) & np.isfinite(r_k_sq) & (ell <= 10000)
    if np.any(valid):
        ax4.plot(ell[valid], r_k_sq[valid], '-', linewidth=2.5, alpha=0.8,
                 color=colors_r[idx], label=f'xHI={xhi:.2f}, z={z:.1f}')
        ax4.fill_between(ell[valid], r_k_sq[valid] - r_k_sq_std[valid], r_k_sq[valid] + r_k_sq_std[valid],
                        alpha=0.2, color=colors_r[idx])

ax4.axhline(0, color='black', linewidth=0.8, linestyle=':')
ax4.set_xlabel('Multipole ℓ', fontsize=12, fontweight='bold')
ax4.set_ylabel('Correlation Coefficient r(ℓ)', fontsize=12, fontweight='bold')
ax4.set_title('r(ℓ) for kSZ² × 21cm² - Reconstructed Velocity', fontsize=14, fontweight='bold')
ax4.legend(fontsize=9, loc='best')
ax4.grid(True, alpha=0.3)
ax4.set_ylim(-1, 1)
ax4.set_xlim(0, 10000)

plt.suptitle('Correlation Coefficient r(ℓ) with 21cm² = P_cross / √(P_kSZ × P_21cm²)\n(Ensemble Average over 30 Simulations)', 
             fontsize=18, fontweight='bold')
plt.savefig('plots/plots_multiple_sims/r_ell_analysis_21cm_squared_ensemble.png', dpi=300, bbox_inches='tight')
plt.show()
print("Figure 5 saved!")

# ============================================================================
# CELL 15: kSZ Reconstruction vs Real kSZ Comparison
# ============================================================================

print("\nCreating kSZ reconstruction comparison plots...")

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

# Compute kSZ and velocity correlations for each chunk (using ensemble-averaged maps)
ksz_correlations = {
    'mean_xhi': [],
    'mean_z': [],
    'r_ksz_signed': [],      # correlation between kSZ_real and kSZ_rec (signed)
    'r_ksz_abs': [],         # correlation between |kSZ_real| and |kSZ_rec| (absolute)
    'r_vz': [],              # correlation between vz_real and vz_rec
    'chunk_names': []
}

print("Computing kSZ correlations for each chunk (ensemble average)...")
for i in range(len(results['mean_xhi'])):
    if i < len(results['ksz_maps_real']) and i < len(results['ksz_maps_rec']):
        ksz_real = results['ksz_maps_real'][i]
        ksz_rec = results['ksz_maps_rec'][i]
        ksz_abs_real = results['ksz_abs_maps_real'][i]
        ksz_abs_rec = results['ksz_abs_maps_rec'][i]
        
        # Compute correlations
        r_signed = pearson_r(ksz_real, ksz_rec)
        r_abs = pearson_r(ksz_abs_real, ksz_abs_rec)
        
        # Compute velocity correlation
        vz_real = results['vz_real'][i]
        vz_rec = results['vz_rec'][i]
        r_vz = pearson_r(vz_real, vz_rec)
        
        ksz_correlations['mean_xhi'].append(results['mean_xhi'][i])
        ksz_correlations['mean_z'].append(results['mean_z'][i])
        ksz_correlations['r_ksz_signed'].append(r_signed)
        ksz_correlations['r_ksz_abs'].append(r_abs)
        ksz_correlations['r_vz'].append(r_vz)
        ksz_correlations['chunk_names'].append(f"Chunk {i+1}")
        
        print(f"  Chunk {i+1}: xHI={results['mean_xhi'][i]:.3f}, "
              f"r(kSZ_signed)={r_signed:+.3f}, r(|kSZ|)={r_abs:+.3f}, r(vz)={r_vz:+.3f}")

# Convert to arrays
for key in ksz_correlations:
    if key != 'chunk_names':
        ksz_correlations[key] = np.array(ksz_correlations[key])

# Create comprehensive kSZ comparison plot
fig_ksz = plt.figure(figsize=(20, 5))
gs_ksz = fig_ksz.add_gridspec(1, 3, hspace=0.3, wspace=0.3)

# Plot 1: kSZ correlations vs xHI
ax_ksz1 = fig_ksz.add_subplot(gs_ksz[0, 0])
ax_ksz1.plot(ksz_correlations['mean_xhi'], ksz_correlations['r_ksz_signed'], 'bo-',
             label='kSZ (signed)', linewidth=2, markersize=8, alpha=0.7)
ax_ksz1.plot(ksz_correlations['mean_xhi'], ksz_correlations['r_ksz_abs'], 'rs-',
             label='|kSZ| (absolute)', linewidth=2, markersize=8, alpha=0.7)
ax_ksz1.axhline(0, color='black', linewidth=0.8, linestyle=':')
ax_ksz1.set_xlabel('Mean Neutral Fraction (xHI)', fontsize=12, fontweight='bold')
ax_ksz1.set_ylabel('Correlation r(kSZ_real, kSZ_rec)', fontsize=12, fontweight='bold')
ax_ksz1.set_title('kSZ Reconstruction Quality vs xHI', fontsize=14, fontweight='bold')
ax_ksz1.legend(fontsize=10)
ax_ksz1.grid(True, alpha=0.3)
ax_ksz1.set_xlim(0, 1)
ax_ksz1.set_ylim(-1, 1)

# Plot 2: kSZ correlations vs redshift
ax_ksz2 = fig_ksz.add_subplot(gs_ksz[0, 1])
ax_ksz2.plot(ksz_correlations['mean_z'], ksz_correlations['r_ksz_signed'], 'bo-',
             label='kSZ (signed)', linewidth=2, markersize=8, alpha=0.7)
ax_ksz2.plot(ksz_correlations['mean_z'], ksz_correlations['r_ksz_abs'], 'rs-',
             label='|kSZ| (absolute)', linewidth=2, markersize=8, alpha=0.7)
ax_ksz2.axhline(0, color='black', linewidth=0.8, linestyle=':')
ax_ksz2.set_xlabel('Mean Redshift', fontsize=12, fontweight='bold')
ax_ksz2.set_ylabel('Correlation r(kSZ_real, kSZ_rec)', fontsize=12, fontweight='bold')
ax_ksz2.set_title('kSZ Reconstruction Quality vs Redshift', fontsize=14, fontweight='bold')
ax_ksz2.legend(fontsize=10)
ax_ksz2.grid(True, alpha=0.3)
ax_ksz2.invert_xaxis()
ax_ksz2.set_ylim(-1, 1)

# Plot 3: Velocity field correlation
ax_ksz3 = fig_ksz.add_subplot(gs_ksz[0, 2])
ax_ksz3.plot(ksz_correlations['mean_xhi'], ksz_correlations['r_vz'], 'go-',
             label='Velocity field', linewidth=2, markersize=8, alpha=0.7)
ax_ksz3.axhline(0, color='black', linewidth=0.8, linestyle=':')
ax_ksz3.set_xlabel('Mean Neutral Fraction (xHI)', fontsize=12, fontweight='bold')
ax_ksz3.set_ylabel('Correlation r(vz_real, vz_rec)', fontsize=12, fontweight='bold')
ax_ksz3.set_title('Velocity Reconstruction Quality vs xHI', fontsize=14, fontweight='bold')
ax_ksz3.legend(fontsize=10)
ax_ksz3.grid(True, alpha=0.3)
ax_ksz3.set_xlim(0, 1)
ax_ksz3.set_ylim(-1, 1)

plt.suptitle('kSZ Reconstruction Quality: Real vs Reconstructed Comparison\n(Ensemble Average over 30 Simulations)', 
             fontsize=18, fontweight='bold')
plt.savefig('plots/plots_multiple_sims/ksz_reconstruction_comparison_ensemble.png', dpi=300, bbox_inches='tight')
plt.show()
print("Figure 6 saved!")

# Print kSZ reconstruction summary
print("\n" + "="*80)
print("kSZ RECONSTRUCTION SUMMARY (ENSEMBLE AVERAGE)")
print("="*80)

print(f"\nSigned kSZ correlation:")
print(f"  Best: r = {np.nanmax(ksz_correlations['r_ksz_signed']):.3f} at xHI = {ksz_correlations['mean_xhi'][np.nanargmax(ksz_correlations['r_ksz_signed'])]:.3f}")
print(f"  Mean: r = {np.nanmean(ksz_correlations['r_ksz_signed']):.3f} ± {np.nanstd(ksz_correlations['r_ksz_signed']):.3f}")

print(f"\nAbsolute |kSZ| correlation:")
print(f"  Best: r = {np.nanmax(ksz_correlations['r_ksz_abs']):.3f} at xHI = {ksz_correlations['mean_xhi'][np.nanargmax(ksz_correlations['r_ksz_abs'])]:.3f}")
print(f"  Mean: r = {np.nanmean(ksz_correlations['r_ksz_abs']):.3f} ± {np.nanstd(ksz_correlations['r_ksz_abs']):.3f}")

# Add kSZ correlations to results
results['r_ksz_signed'] = ksz_correlations['r_ksz_signed']
results['r_ksz_abs'] = ksz_correlations['r_ksz_abs']

# ============================================================================
# CELL 16: Summary Statistics (Updated xHI Thresholds)
# ============================================================================

print("\n" + "="*80)
print("SUMMARY STATISTICS (ENSEMBLE AVERAGE OVER 30 SIMULATIONS)")
print("="*80)

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

# Updated xHI regimes: 0.50 ≤ xHI < 0.99 and xHI < 0.50
high_xhi = (results['mean_xhi'] >= 0.50) & (results['mean_xhi'] < 0.99) & valid_mask_real
low_xhi = (results['mean_xhi'] < 0.50) & valid_mask_real

if np.any(high_xhi):
    print(f"\nHigh neutral fraction (0.50 ≤ xHI < 0.99):")
    print(f"  Mean correlation (real): {np.nanmean(results['correlation'][high_xhi]):+.4f} ± {np.nanstd(results['correlation'][high_xhi]):.4f}")
    print(f"  Mean correlation (rec):  {np.nanmean(results['correlation_rec'][high_xhi]):+.4f} ± {np.nanstd(results['correlation_rec'][high_xhi]):.4f}")

if np.any(low_xhi):
    print(f"\nLow neutral fraction (xHI < 0.50):")
    print(f"  Mean correlation (real): {np.nanmean(results['correlation'][low_xhi]):+.4f} ± {np.nanstd(results['correlation'][low_xhi]):.4f}")
    print(f"  Mean correlation (rec):  {np.nanmean(results['correlation_rec'][low_xhi]):+.4f} ± {np.nanstd(results['correlation_rec'][low_xhi]):.4f}")

print("="*80)

# Save results (exclude velocity fields which have variable shapes)
os.makedirs('results', exist_ok=True)
results_to_save = {k: v for k, v in results.items() if k not in ['vz_real', 'vz_rec']}
np.savez('results/xhi_evolution_results_ensemble.npz', **results_to_save)
print("\nResults saved to: results/xhi_evolution_results_ensemble.npz")

print("\n" + "="*80)
print("ENSEMBLE ANALYSIS COMPLETE!")
print(f"Processed {n_sims} simulations with {len(results['mean_xhi'])} redshift chunks")
print("="*80)
