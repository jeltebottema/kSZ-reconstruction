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
import tools21cm    
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


# Clean the data - Apply wedge filter to density and velocity
# NOTE: Do NOT filter xHI - it's a physical field, not an observable
# den_filtered = tools21cm.foreground_model.rolling_wedge_removal_lightcone(den, redshifts, cell_size=None, chunk_length=None, OMm=None, buffer_threshold=1e-10)
# vz_filtered = tools21cm.foreground_model.rolling_wedge_removal_lightcone(vz, redshifts, cell_size=None, chunk_length=None, OMm=None, buffer_threshold=1e-10)
# xhi_filtered = xhi  # Keep xHI unfiltered - it's used for weighting and analysis

den_filtered = den
vz_filtered = vz
xhi_filtered = xhi

print("data filtered (density and velocity only)")

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
                        If None and include_velocity_term=True, uses iterative approach
        weight: Weighting scheme - 'delta', 'delta_xhi', or 'deltaXhi'
        include_velocity_term: If True, include (1 - dv/dr / aH) term in tracer field
    """
    d = np.asarray(den_xyz, dtype=dtype)
    nx, ny, nz = d.shape
    
    # Density contrast
    mean_den = d.mean(dtype=np.float64).astype(dtype)
    delta = d
    
    # Cosmology parameters (needed for velocity term)
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
        # Cell size in Mpc (physical)
        dz_cell = box_mpc_over_h / nz / littleh
        # vz is in cm/s, convert to km/s
        vz_kms = vz_for_gradient / 1e5
        dvdz = np.gradient(vz_kms, dz_cell, axis=2).astype(dtype)  # km/s/Mpc
        # Velocity factor: H / (dv_r/dr + H)
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
        include_velocity_term: If True, include the (1 - dv/dr / aH) term in T_b
    
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
    
    # Pixel-space correlations
    'correlation_21cm': [],  # |kSZ_real| × 21cm
    'correlation_21cm_rec': [],  # |kSZ_rec| × 21cm
    'correlation_21cm_sq': [],  # |kSZ_real| × 21cm²
    'correlation_21cm_sq_rec': [],  # |kSZ_rec| × 21cm²
    
    # Real velocity - |kSZ| × 21cm
    'P_k_cross_21cm': [],
    'l_s_cross_21cm': [],
    'err_cross_21cm': [],
    'ell_cross_21cm': [],
    'r_k_21cm': [],
    
    # Real velocity - |kSZ| × 21cm²
    'P_k_cross_21cm_sq': [],
    'l_s_cross_21cm_sq': [],
    'err_cross_21cm_sq': [],
    'ell_cross_21cm_sq': [],
    'r_k_21cm_sq': [],
    
    # Real velocity - kSZ² × 21cm
    'P_k_cross_ksz_sq_21cm': [],
    'l_s_cross_ksz_sq_21cm': [],
    'err_cross_ksz_sq_21cm': [],
    'ell_cross_ksz_sq_21cm': [],
    'r_k_ksz_sq_21cm': [],
    
    # Real velocity - kSZ² × 21cm²
    'P_k_cross_ksz_sq_21cm_sq': [],
    'l_s_cross_ksz_sq_21cm_sq': [],
    'err_cross_ksz_sq_21cm_sq': [],
    'ell_cross_ksz_sq_21cm_sq': [],
    'r_k_ksz_sq_21cm_sq': [],
    
    # Reconstructed velocity - |kSZ| × 21cm
    'P_k_cross_21cm_rec': [],
    'err_cross_21cm_rec': [],
    'ell_cross_21cm_rec': [],
    'r_k_21cm_rec': [],
    
    # Reconstructed velocity - |kSZ| × 21cm²
    'P_k_cross_21cm_sq_rec': [],
    'err_cross_21cm_sq_rec': [],
    'ell_cross_21cm_sq_rec': [],
    'r_k_21cm_sq_rec': [],
    
    # Reconstructed velocity - kSZ² × 21cm
    'P_k_cross_ksz_sq_21cm_rec': [],
    'err_cross_ksz_sq_21cm_rec': [],
    'ell_cross_ksz_sq_21cm_rec': [],
    'r_k_ksz_sq_21cm_rec': [],
    
    # Reconstructed velocity - kSZ² × 21cm²
    'P_k_cross_ksz_sq_21cm_sq_rec': [],
    'err_cross_ksz_sq_21cm_sq_rec': [],
    'ell_cross_ksz_sq_21cm_sq_rec': [],
    'r_k_ksz_sq_21cm_sq_rec': [],
    
    # Auto-power spectra
    'P_k_21cm': [],
    'err_21cm': [],
    'ell_21cm': [],
    'P_k_21cm_sq': [],
    'err_21cm_sq': [],
    'ell_21cm_sq': [],
    
    # Maps
    'ksz_maps_real': [],
    'ksz_abs_maps_real': [],
    'ksz_maps_rec': [],
    'ksz_abs_maps_rec': [],
    't21_maps': [],
    't21_sq_maps': [],
    
    # Velocity fields
    'vz_real': [],
    'vz_rec': []
}

# Process each chunk
for i in range(n_chunks):
    z_start = i * chunk_size
    z_end = min((i + 1) * chunk_size, nz)
    
    if z_end - z_start < 10:  # Skip very small chunks
        continue
    
    print(f"\nChunk {i+1}/{n_chunks}: z-slices [{z_start}:{z_end}]")
    
    # Extract chunk
    den_chunk = den_filtered[:, :, z_start:z_end]
    xhi_chunk = xhi_filtered[:, :, z_start:z_end]
    vz_chunk = vz_filtered[:, :, z_start:z_end]
    redshifts_chunk = redshifts[z_start:z_end]

    # den_chunk = den[:, :, z_start:z_end]
    # xhi_chunk = xhi[:, :, z_start:z_end]
    # vz_chunk = vz[:, :, z_start:z_end]
    # redshifts_chunk = redshifts[z_start:z_end]
    
    # Reconstruct velocity for this chunk
    z_ref_chunk = float(np.mean(redshifts_chunk))
    vz_rec_chunk_minus = reconstruct_velocity_single_z_method(
        den_chunk, xhi_xyz=xhi_chunk,
        weight="deltaXhi",
        z_ref=z_ref_chunk,
        littleh=0.7,
        box_mpc_over_h=300.0
    )

    vz_rec_chunk = vz_rec_chunk_minus
    
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
    
    # Create squared maps
    ksz_sq_real = ksz_real**2
    ksz_sq_rec = ksz_rec**2
    t21_sq = t21**2
    
    # Compute pixel-space correlations
    r_real_21cm = pearson_r(ksz_abs_real, t21)
    r_rec_21cm = pearson_r(ksz_abs_rec, t21)
    r_real_21cm_sq = pearson_r(ksz_abs_real, t21_sq)
    r_rec_21cm_sq = pearson_r(ksz_abs_rec, t21_sq)
    
    # Compute power spectra
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        # === AUTO-POWER SPECTRA ===
        # Auto-power: |kSZ| (real)
        P_k_ksz_abs_real, l_s_ksz_abs_real, err_ksz_abs_real = get_power(
            ksz_abs_real, 300, bins=100, log_bins=True, get_variance=True)
        P_k_ksz_abs_real = P_k_ksz_abs_real * l_s_ksz_abs_real ** 2
        
        # Auto-power: |kSZ| (reconstructed)
        P_k_ksz_abs_rec, l_s_ksz_abs_rec, err_ksz_abs_rec = get_power(
            ksz_abs_rec, 300, bins=100, log_bins=True, get_variance=True)
        P_k_ksz_abs_rec = P_k_ksz_abs_rec * l_s_ksz_abs_rec ** 2
        
        # Auto-power: kSZ² (real)
        P_k_ksz_sq_real, l_s_ksz_sq_real, err_ksz_sq_real = get_power(
            ksz_sq_real, 300, bins=100, log_bins=True, get_variance=True)
        P_k_ksz_sq_real = P_k_ksz_sq_real * l_s_ksz_sq_real ** 2
        
        # Auto-power: kSZ² (reconstructed)
        P_k_ksz_sq_rec, l_s_ksz_sq_rec, err_ksz_sq_rec = get_power(
            ksz_sq_rec, 300, bins=100, log_bins=True, get_variance=True)
        P_k_ksz_sq_rec = P_k_ksz_sq_rec * l_s_ksz_sq_rec ** 2
        
        # Auto-power: 21cm
        P_k_21cm, l_s_21cm, err_21cm = get_power(
            t21, 300, bins=100, log_bins=True, get_variance=True)
        P_k_21cm = P_k_21cm * l_s_21cm ** 2
        err_21cm = np.sqrt(err_21cm) * l_s_21cm ** 2
        
        # Auto-power: 21cm²
        P_k_21cm_sq, l_s_21cm_sq, err_21cm_sq = get_power(
            t21_sq, 300, bins=100, log_bins=True, get_variance=True)
        P_k_21cm_sq = P_k_21cm_sq * l_s_21cm_sq ** 2
        err_21cm_sq = np.sqrt(err_21cm_sq) * l_s_21cm_sq ** 2
        
        # === REAL VELOCITY CROSS-POWERS ===
        # |kSZ| × 21cm
        P_k_cross_21cm_real, l_s_cross_21cm_real, err_cross_21cm_real = get_power(
            deltax=ksz_abs_real, deltax2=t21, boxlength=300, bins=100, 
            log_bins=True, get_variance=True)
        P_k_cross_21cm_real = P_k_cross_21cm_real * l_s_cross_21cm_real ** 2
        err_cross_21cm_real = np.sqrt(err_cross_21cm_real) * l_s_cross_21cm_real ** 2
        
        # |kSZ| × 21cm²
        P_k_cross_21cm_sq_real, l_s_cross_21cm_sq_real, err_cross_21cm_sq_real = get_power(
            deltax=ksz_abs_real, deltax2=t21_sq, boxlength=300, bins=100,
            log_bins=True, get_variance=True)
        P_k_cross_21cm_sq_real = P_k_cross_21cm_sq_real * l_s_cross_21cm_sq_real ** 2
        err_cross_21cm_sq_real = np.sqrt(err_cross_21cm_sq_real) * l_s_cross_21cm_sq_real ** 2
        
        # kSZ² × 21cm
        P_k_cross_ksz_sq_21cm_real, l_s_cross_ksz_sq_21cm_real, err_cross_ksz_sq_21cm_real = get_power(
            deltax=ksz_sq_real, deltax2=t21, boxlength=300, bins=100,
            log_bins=True, get_variance=True)
        P_k_cross_ksz_sq_21cm_real = P_k_cross_ksz_sq_21cm_real * l_s_cross_ksz_sq_21cm_real ** 2
        err_cross_ksz_sq_21cm_real = np.sqrt(err_cross_ksz_sq_21cm_real) * l_s_cross_ksz_sq_21cm_real ** 2
        
        # kSZ² × 21cm²
        P_k_cross_ksz_sq_21cm_sq_real, l_s_cross_ksz_sq_21cm_sq_real, err_cross_ksz_sq_21cm_sq_real = get_power(
            deltax=ksz_sq_real, deltax2=t21_sq, boxlength=300, bins=100,
            log_bins=True, get_variance=True)
        P_k_cross_ksz_sq_21cm_sq_real = P_k_cross_ksz_sq_21cm_sq_real * l_s_cross_ksz_sq_21cm_sq_real ** 2
        err_cross_ksz_sq_21cm_sq_real = np.sqrt(err_cross_ksz_sq_21cm_sq_real) * l_s_cross_ksz_sq_21cm_sq_real ** 2
        
        # === RECONSTRUCTED VELOCITY CROSS-POWERS ===
        # |kSZ| × 21cm
        P_k_cross_21cm_rec, l_s_cross_21cm_rec, err_cross_21cm_rec = get_power(
            deltax=ksz_abs_rec, deltax2=t21, boxlength=300, bins=100,
            log_bins=True, get_variance=True)
        P_k_cross_21cm_rec = P_k_cross_21cm_rec * l_s_cross_21cm_rec ** 2
        err_cross_21cm_rec = np.sqrt(err_cross_21cm_rec) * l_s_cross_21cm_rec ** 2
        
        # |kSZ| × 21cm²
        P_k_cross_21cm_sq_rec, l_s_cross_21cm_sq_rec, err_cross_21cm_sq_rec = get_power(
            deltax=ksz_abs_rec, deltax2=t21_sq, boxlength=300, bins=100,
            log_bins=True, get_variance=True)
        P_k_cross_21cm_sq_rec = P_k_cross_21cm_sq_rec * l_s_cross_21cm_sq_rec ** 2
        err_cross_21cm_sq_rec = np.sqrt(err_cross_21cm_sq_rec) * l_s_cross_21cm_sq_rec ** 2
        
        # kSZ² × 21cm
        P_k_cross_ksz_sq_21cm_rec, l_s_cross_ksz_sq_21cm_rec, err_cross_ksz_sq_21cm_rec = get_power(
            deltax=ksz_sq_rec, deltax2=t21, boxlength=300, bins=100,
            log_bins=True, get_variance=True)
        P_k_cross_ksz_sq_21cm_rec = P_k_cross_ksz_sq_21cm_rec * l_s_cross_ksz_sq_21cm_rec ** 2
        err_cross_ksz_sq_21cm_rec = np.sqrt(err_cross_ksz_sq_21cm_rec) * l_s_cross_ksz_sq_21cm_rec ** 2
        
        # kSZ² × 21cm²
        P_k_cross_ksz_sq_21cm_sq_rec, l_s_cross_ksz_sq_21cm_sq_rec, err_cross_ksz_sq_21cm_sq_rec = get_power(
            deltax=ksz_sq_rec, deltax2=t21_sq, boxlength=300, bins=100,
            log_bins=True, get_variance=True)
        P_k_cross_ksz_sq_21cm_sq_rec = P_k_cross_ksz_sq_21cm_sq_rec * l_s_cross_ksz_sq_21cm_sq_rec ** 2
        err_cross_ksz_sq_21cm_sq_rec = np.sqrt(err_cross_ksz_sq_21cm_sq_rec) * l_s_cross_ksz_sq_21cm_sq_rec ** 2
    
    # === COMPUTE CORRELATION COEFFICIENTS r(k) ===
    # Real velocity
    r_k_21cm_real = P_k_cross_21cm_real / np.sqrt(P_k_21cm * P_k_ksz_abs_real)
    r_k_21cm_sq_real = P_k_cross_21cm_sq_real / np.sqrt(P_k_21cm_sq * P_k_ksz_abs_real)
    r_k_ksz_sq_21cm_real = P_k_cross_ksz_sq_21cm_real / np.sqrt(P_k_21cm * P_k_ksz_sq_real)
    r_k_ksz_sq_21cm_sq_real = P_k_cross_ksz_sq_21cm_sq_real / np.sqrt(P_k_21cm_sq * P_k_ksz_sq_real)
    
    # Reconstructed velocity
    r_k_21cm_rec = P_k_cross_21cm_rec / np.sqrt(P_k_21cm * P_k_ksz_abs_rec)
    r_k_21cm_sq_rec = P_k_cross_21cm_sq_rec / np.sqrt(P_k_21cm_sq * P_k_ksz_abs_rec)
    r_k_ksz_sq_21cm_rec = P_k_cross_ksz_sq_21cm_rec / np.sqrt(P_k_21cm * P_k_ksz_sq_rec)
    r_k_ksz_sq_21cm_sq_rec = P_k_cross_ksz_sq_21cm_sq_rec / np.sqrt(P_k_21cm_sq * P_k_ksz_sq_rec)
    
    # === CONVERT k TO ell ===
    ell_cross_21cm_real = k_to_ell(l_s_cross_21cm_real, mean_z)
    ell_cross_21cm_sq_real = k_to_ell(l_s_cross_21cm_sq_real, mean_z)
    ell_cross_ksz_sq_21cm_real = k_to_ell(l_s_cross_ksz_sq_21cm_real, mean_z)
    ell_cross_ksz_sq_21cm_sq_real = k_to_ell(l_s_cross_ksz_sq_21cm_sq_real, mean_z)
    
    ell_cross_21cm_rec = k_to_ell(l_s_cross_21cm_rec, mean_z)
    ell_cross_21cm_sq_rec = k_to_ell(l_s_cross_21cm_sq_rec, mean_z)
    ell_cross_ksz_sq_21cm_rec = k_to_ell(l_s_cross_ksz_sq_21cm_rec, mean_z)
    ell_cross_ksz_sq_21cm_sq_rec = k_to_ell(l_s_cross_ksz_sq_21cm_sq_rec, mean_z)
    
    ell_21cm = k_to_ell(l_s_21cm, mean_z)
    ell_21cm_sq = k_to_ell(l_s_21cm_sq, mean_z)
    
    # Store results
    results['mean_xhi'].append(mean_xhi)
    results['mean_z'].append(mean_z)
    results['z_min'].append(redshifts_chunk.min())
    results['z_max'].append(redshifts_chunk.max())
    
    # Pixel-space correlations
    results['correlation_21cm'].append(r_real_21cm)
    results['correlation_21cm_rec'].append(r_rec_21cm)
    results['correlation_21cm_sq'].append(r_real_21cm_sq)
    results['correlation_21cm_sq_rec'].append(r_rec_21cm_sq)
    
    # Real velocity - |kSZ| × 21cm
    results['P_k_cross_21cm'].append(P_k_cross_21cm_real)
    results['l_s_cross_21cm'].append(l_s_cross_21cm_real)
    results['err_cross_21cm'].append(err_cross_21cm_real)
    results['ell_cross_21cm'].append(ell_cross_21cm_real)
    results['r_k_21cm'].append(r_k_21cm_real)
    
    # Real velocity - |kSZ| × 21cm²
    results['P_k_cross_21cm_sq'].append(P_k_cross_21cm_sq_real)
    results['l_s_cross_21cm_sq'].append(l_s_cross_21cm_sq_real)
    results['err_cross_21cm_sq'].append(err_cross_21cm_sq_real)
    results['ell_cross_21cm_sq'].append(ell_cross_21cm_sq_real)
    results['r_k_21cm_sq'].append(r_k_21cm_sq_real)
    
    # Real velocity - kSZ² × 21cm
    results['P_k_cross_ksz_sq_21cm'].append(P_k_cross_ksz_sq_21cm_real)
    results['l_s_cross_ksz_sq_21cm'].append(l_s_cross_ksz_sq_21cm_real)
    results['err_cross_ksz_sq_21cm'].append(err_cross_ksz_sq_21cm_real)
    results['ell_cross_ksz_sq_21cm'].append(ell_cross_ksz_sq_21cm_real)
    results['r_k_ksz_sq_21cm'].append(r_k_ksz_sq_21cm_real)
    
    # Real velocity - kSZ² × 21cm²
    results['P_k_cross_ksz_sq_21cm_sq'].append(P_k_cross_ksz_sq_21cm_sq_real)
    results['l_s_cross_ksz_sq_21cm_sq'].append(l_s_cross_ksz_sq_21cm_sq_real)
    results['err_cross_ksz_sq_21cm_sq'].append(err_cross_ksz_sq_21cm_sq_real)
    results['ell_cross_ksz_sq_21cm_sq'].append(ell_cross_ksz_sq_21cm_sq_real)
    results['r_k_ksz_sq_21cm_sq'].append(r_k_ksz_sq_21cm_sq_real)
    
    # Reconstructed velocity - |kSZ| × 21cm
    results['P_k_cross_21cm_rec'].append(P_k_cross_21cm_rec)
    results['err_cross_21cm_rec'].append(err_cross_21cm_rec)
    results['ell_cross_21cm_rec'].append(ell_cross_21cm_rec)
    results['r_k_21cm_rec'].append(r_k_21cm_rec)
    
    # Reconstructed velocity - |kSZ| × 21cm²
    results['P_k_cross_21cm_sq_rec'].append(P_k_cross_21cm_sq_rec)
    results['err_cross_21cm_sq_rec'].append(err_cross_21cm_sq_rec)
    results['ell_cross_21cm_sq_rec'].append(ell_cross_21cm_sq_rec)
    results['r_k_21cm_sq_rec'].append(r_k_21cm_sq_rec)
    
    # Reconstructed velocity - kSZ² × 21cm
    results['P_k_cross_ksz_sq_21cm_rec'].append(P_k_cross_ksz_sq_21cm_rec)
    results['err_cross_ksz_sq_21cm_rec'].append(err_cross_ksz_sq_21cm_rec)
    results['ell_cross_ksz_sq_21cm_rec'].append(ell_cross_ksz_sq_21cm_rec)
    results['r_k_ksz_sq_21cm_rec'].append(r_k_ksz_sq_21cm_rec)
    
    # Reconstructed velocity - kSZ² × 21cm²
    results['P_k_cross_ksz_sq_21cm_sq_rec'].append(P_k_cross_ksz_sq_21cm_sq_rec)
    results['err_cross_ksz_sq_21cm_sq_rec'].append(err_cross_ksz_sq_21cm_sq_rec)
    results['ell_cross_ksz_sq_21cm_sq_rec'].append(ell_cross_ksz_sq_21cm_sq_rec)
    results['r_k_ksz_sq_21cm_sq_rec'].append(r_k_ksz_sq_21cm_sq_rec)
    
    # Auto-power spectra
    results['P_k_21cm'].append(P_k_21cm)
    results['err_21cm'].append(err_21cm)
    results['ell_21cm'].append(ell_21cm)
    results['P_k_21cm_sq'].append(P_k_21cm_sq)
    results['err_21cm_sq'].append(err_21cm_sq)
    results['ell_21cm_sq'].append(ell_21cm_sq)
    
    # Maps
    results['ksz_maps_real'].append(ksz_real)
    results['ksz_abs_maps_real'].append(ksz_abs_real)
    results['ksz_maps_rec'].append(ksz_rec)
    results['ksz_abs_maps_rec'].append(ksz_abs_rec)
    results['t21_maps'].append(t21)
    results['t21_sq_maps'].append(t21_sq)
    results['vz_real'].append(vz_chunk)
    results['vz_rec'].append(vz_rec_chunk)
    
    print(f"  mean_xHI = {mean_xhi:.4f}, mean_z = {mean_z:.3f}")
    print(f"  r(|kSZ_real| × 21cm) = {r_real_21cm:+.4f}")
    print(f"  r(|kSZ_rec| × 21cm)  = {r_rec_21cm:+.4f}")
    print(f"  r(|kSZ_real| × 21cm²) = {r_real_21cm_sq:+.4f}")
    print(f"  r(|kSZ_rec| × 21cm²)  = {r_rec_21cm_sq:+.4f}")

# Convert to arrays (except velocity fields which have variable z-dimension)
for key in results:
    if key not in ['vz_real', 'vz_rec']:
        results[key] = np.array(results[key])

print("\nChunk analysis complete!")

# ============================================================================
# CELL 7: Visualization - Cross-Power at Specific ell Values vs xHI
# ============================================================================

# First, find what ell values are actually available
# Use a valid chunk (skip first if it has NaN)
valid_idx = 0
for idx in range(len(results['ell_cross_21cm'])):
    if np.any(np.isfinite(results['ell_cross_21cm'][idx])):
        valid_idx = idx
        break

sample_ell = results['ell_cross_21cm'][valid_idx]
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
            ell_arr = results['ell_cross_21cm'][i]
            P_arr = results['P_k_cross_21cm'][i]
            err_arr = results['err_cross_21cm'][i]
            
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
    
    os.makedirs('plots/plots_single_sim', exist_ok=True)
    plt.savefig('plots/plots_single_sim/cross_power_vs_xhi.png', dpi=300, bbox_inches='tight')
    plt.show()
else:
    print("\nWarning: No valid ell data found!")

# ============================================================================
# CELL 8: Visualization - Cross-Power Spectra for Specific xHI Values
# ============================================================================

# Choose 4-5 representative xHI values (epochs)
n_epochs = min(5, len(results['mean_xhi']))
indices_to_plot = np.linspace(0, len(results['mean_xhi'])-1, n_epochs, dtype=int)

fig2 = plt.figure(figsize=(24, 16))
gs2 = fig2.add_gridspec(4, 2, hspace=0.35, wspace=0.3)

colors = plt.cm.viridis(np.linspace(0, 1, n_epochs))

# Plot 1: |kSZ| × 21cm (Real velocity)
ax1 = fig2.add_subplot(gs2[0, 0])
for idx, i in enumerate(indices_to_plot):
    ell = results['ell_cross_21cm'][i]
    P_k = results['P_k_cross_21cm'][i]
    xhi = results['mean_xhi'][i]
    z = results['mean_z'][i]
    valid = np.isfinite(ell) & np.isfinite(P_k) & (ell <= 5000)
    if np.any(valid):
        ax1.plot(ell[valid], P_k[valid], '-', linewidth=2, alpha=0.7,
                 color=colors[idx], label=f'xHI={xhi:.2f}, z={z:.1f}')
ax1.set_xlabel('Multipole ℓ', fontsize=12, fontweight='bold')
ax1.set_ylabel('Cross-Power P(ℓ)', fontsize=12, fontweight='bold')
ax1.set_title('|kSZ| × 21cm (Real Velocity)', fontsize=14, fontweight='bold')
ax1.legend(fontsize=9, ncol=1, loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 5000)

# Plot 2: |kSZ| × 21cm (Reconstructed velocity)
ax2 = fig2.add_subplot(gs2[0, 1])
for idx, i in enumerate(indices_to_plot):
    ell = results['ell_cross_21cm_rec'][i]
    P_k = results['P_k_cross_21cm_rec'][i]
    xhi = results['mean_xhi'][i]
    z = results['mean_z'][i]
    valid = np.isfinite(ell) & np.isfinite(P_k) & (ell <= 5000)
    if np.any(valid):
        ax2.plot(ell[valid], P_k[valid], '-', linewidth=2, alpha=0.7,
                 color=colors[idx], label=f'xHI={xhi:.2f}, z={z:.1f}')
ax2.set_xlabel('Multipole ℓ', fontsize=12, fontweight='bold')
ax2.set_ylabel('Cross-Power P(ℓ)', fontsize=12, fontweight='bold')
ax2.set_title('|kSZ| × 21cm (Reconstructed Velocity)', fontsize=14, fontweight='bold')
ax2.legend(fontsize=9, ncol=1, loc='upper right')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 5000)

# Plot 3: |kSZ| × 21cm² (Real velocity)
ax3 = fig2.add_subplot(gs2[1, 0])
for idx, i in enumerate(indices_to_plot):
    ell = results['ell_cross_21cm_sq'][i]
    P_k = results['P_k_cross_21cm_sq'][i]
    xhi = results['mean_xhi'][i]
    valid = np.isfinite(ell) & np.isfinite(P_k) & (ell <= 5000)
    if np.any(valid):
        ax3.plot(ell[valid], P_k[valid], '-', linewidth=2, alpha=0.7,
                 color=colors[idx], label=f'xHI={xhi:.2f}')
ax3.set_xlabel('Multipole ℓ', fontsize=12, fontweight='bold')
ax3.set_ylabel('Cross-Power P(ℓ)', fontsize=12, fontweight='bold')
ax3.set_title('|kSZ| × 21cm² (Real Velocity)', fontsize=14, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)
ax3.set_xlim(0, 5000)

# Plot 4: |kSZ| × 21cm² (Reconstructed velocity)
ax4 = fig2.add_subplot(gs2[1, 1])
for idx, i in enumerate(indices_to_plot):
    ell = results['ell_cross_21cm_sq_rec'][i]
    P_k = results['P_k_cross_21cm_sq_rec'][i]
    xhi = results['mean_xhi'][i]
    valid = np.isfinite(ell) & np.isfinite(P_k) & (ell <= 5000)
    if np.any(valid):
        ax4.plot(ell[valid], P_k[valid], '-', linewidth=2, alpha=0.7,
                 color=colors[idx], label=f'xHI={xhi:.2f}')
ax4.set_xlabel('Multipole ℓ', fontsize=12, fontweight='bold')
ax4.set_ylabel('Cross-Power P(ℓ)', fontsize=12, fontweight='bold')
ax4.set_title('|kSZ| × 21cm² (Reconstructed Velocity)', fontsize=14, fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)
ax4.set_xlim(0, 5000)

# Plot 5: kSZ² × 21cm (Real velocity)
ax5 = fig2.add_subplot(gs2[2, 0])
for idx, i in enumerate(indices_to_plot):
    ell = results['ell_cross_ksz_sq_21cm'][i]
    P_k = results['P_k_cross_ksz_sq_21cm'][i]
    xhi = results['mean_xhi'][i]
    valid = np.isfinite(ell) & np.isfinite(P_k) & (ell <= 5000)
    if np.any(valid):
        ax5.plot(ell[valid], P_k[valid], '-', linewidth=2, alpha=0.7,
                 color=colors[idx], label=f'xHI={xhi:.2f}')
ax5.set_xlabel('Multipole ℓ', fontsize=12, fontweight='bold')
ax5.set_ylabel('Cross-Power P(ℓ)', fontsize=12, fontweight='bold')
ax5.set_title('kSZ² × 21cm (Real Velocity)', fontsize=14, fontweight='bold')
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.3)
ax5.set_xlim(0, 5000)

# Plot 6: kSZ² × 21cm (Reconstructed velocity)
ax6 = fig2.add_subplot(gs2[2, 1])
for idx, i in enumerate(indices_to_plot):
    ell = results['ell_cross_ksz_sq_21cm_rec'][i]
    P_k = results['P_k_cross_ksz_sq_21cm_rec'][i]
    xhi = results['mean_xhi'][i]
    valid = np.isfinite(ell) & np.isfinite(P_k) & (ell <= 5000)
    if np.any(valid):
        ax6.plot(ell[valid], P_k[valid], '-', linewidth=2, alpha=0.7,
                 color=colors[idx], label=f'xHI={xhi:.2f}')
ax6.set_xlabel('Multipole ℓ', fontsize=12, fontweight='bold')
ax6.set_ylabel('Cross-Power P(ℓ)', fontsize=12, fontweight='bold')
ax6.set_title('kSZ² × 21cm (Reconstructed Velocity)', fontsize=14, fontweight='bold')
ax6.legend(fontsize=9)
ax6.grid(True, alpha=0.3)
ax6.set_xlim(0, 5000)

# Plot 7: kSZ² × 21cm² (Real velocity)
ax7 = fig2.add_subplot(gs2[3, 0])
for idx, i in enumerate(indices_to_plot):
    ell = results['ell_cross_ksz_sq_21cm_sq'][i]
    P_k = results['P_k_cross_ksz_sq_21cm_sq'][i]
    xhi = results['mean_xhi'][i]
    valid = np.isfinite(ell) & np.isfinite(P_k) & (ell <= 5000)
    if np.any(valid):
        ax7.plot(ell[valid], P_k[valid], '-', linewidth=2, alpha=0.7,
                 color=colors[idx], label=f'xHI={xhi:.2f}')
ax7.set_xlabel('Multipole ℓ', fontsize=12, fontweight='bold')
ax7.set_ylabel('Cross-Power P(ℓ)', fontsize=12, fontweight='bold')
ax7.set_title('kSZ² × 21cm² (Real Velocity)', fontsize=14, fontweight='bold')
ax7.legend(fontsize=9)
ax7.grid(True, alpha=0.3)
ax7.set_xlim(0, 5000)

# Plot 8: kSZ² × 21cm² (Reconstructed velocity)
ax8 = fig2.add_subplot(gs2[3, 1])
for idx, i in enumerate(indices_to_plot):
    ell = results['ell_cross_ksz_sq_21cm_sq_rec'][i]
    P_k = results['P_k_cross_ksz_sq_21cm_sq_rec'][i]
    xhi = results['mean_xhi'][i]
    valid = np.isfinite(ell) & np.isfinite(P_k) & (ell <= 5000)
    if np.any(valid):
        ax8.plot(ell[valid], P_k[valid], '-', linewidth=2, alpha=0.7,
                 color=colors[idx], label=f'xHI={xhi:.2f}')
ax8.set_xlabel('Multipole ℓ', fontsize=12, fontweight='bold')
ax8.set_ylabel('Cross-Power P(ℓ)', fontsize=12, fontweight='bold')
ax8.set_title('kSZ² × 21cm² (Reconstructed Velocity)', fontsize=14, fontweight='bold')
ax8.legend(fontsize=9)
ax8.grid(True, alpha=0.3)
ax8.set_xlim(0, 5000)

plt.suptitle('All Cross-Power Spectra at Different Reionization Epochs', 
             fontsize=18, fontweight='bold')
plt.savefig('plots/plots_single_sim/cross_power_spectra_all.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# CELL 9: Additional Correlation Plots
# ============================================================================

fig3 = plt.figure(figsize=(24, 12))
gs3 = fig3.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# Plot 1: |kSZ| × 21cm Correlation vs xHI
ax1 = fig3.add_subplot(gs3[0, 0])
ax1.plot(results['mean_xhi'], results['correlation_21cm'], 'bo-', 
         label='Real velocity', linewidth=2, markersize=8, alpha=0.7)
ax1.plot(results['mean_xhi'], results['correlation_21cm_rec'], 'rs--', 
         label='Reconstructed velocity', linewidth=2, markersize=8, alpha=0.7)
ax1.axhline(0, color='black', linewidth=0.8, linestyle=':')
ax1.set_xlabel('Mean Neutral Fraction (mean_xHI)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Correlation r(|kSZ|, 21cm)', fontsize=14, fontweight='bold')
ax1.set_title('|kSZ| × 21cm Correlation vs Neutral Fraction', fontsize=16, fontweight='bold')
ax1.legend(fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 1)

# Plot 2: |kSZ| × 21cm² Correlation vs xHI
ax2 = fig3.add_subplot(gs3[0, 1])
ax2.plot(results['mean_xhi'], results['correlation_21cm_sq'], 'go-', 
         label='Real velocity', linewidth=2, markersize=8, alpha=0.7)
ax2.plot(results['mean_xhi'], results['correlation_21cm_sq_rec'], 'ms--', 
         label='Reconstructed velocity', linewidth=2, markersize=8, alpha=0.7)
ax2.axhline(0, color='black', linewidth=0.8, linestyle=':')
ax2.set_xlabel('Mean Neutral Fraction (mean_xHI)', fontsize=14, fontweight='bold')
ax2.set_ylabel('Correlation r(|kSZ|, 21cm²)', fontsize=14, fontweight='bold')
ax2.set_title('|kSZ| × 21cm² Correlation vs Neutral Fraction', fontsize=16, fontweight='bold')
ax2.legend(fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 1)

# Plot 3: Reionization history
ax3 = fig3.add_subplot(gs3[1, 0])
ax3.plot(results['mean_z'], results['mean_xhi'], 'ko-', 
         linewidth=2, markersize=8, alpha=0.7)
ax3.set_xlabel('Mean Redshift', fontsize=14, fontweight='bold')
ax3.set_ylabel('Mean Neutral Fraction (mean_xHI)', fontsize=14, fontweight='bold')
ax3.set_title('Reionization History', fontsize=16, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.invert_xaxis()
ax3.set_ylim(0, 1)

# Plot 4: Comparison of all correlations
ax4 = fig3.add_subplot(gs3[1, 1])
ax4.plot(results['mean_xhi'], results['correlation_21cm'], 'b-', 
         label='|kSZ| × 21cm (Real)', linewidth=2, alpha=0.7)
ax4.plot(results['mean_xhi'], results['correlation_21cm_rec'], 'b--', 
         label='|kSZ| × 21cm (Rec)', linewidth=2, alpha=0.7)
ax4.plot(results['mean_xhi'], results['correlation_21cm_sq'], 'r-', 
         label='|kSZ| × 21cm² (Real)', linewidth=2, alpha=0.7)
ax4.plot(results['mean_xhi'], results['correlation_21cm_sq_rec'], 'r--', 
         label='|kSZ| × 21cm² (Rec)', linewidth=2, alpha=0.7)
ax4.axhline(0, color='black', linewidth=0.8, linestyle=':')
ax4.set_xlabel('Mean Neutral Fraction (mean_xHI)', fontsize=14, fontweight='bold')
ax4.set_ylabel('Correlation Coefficient', fontsize=14, fontweight='bold')
ax4.set_title('All Correlations Comparison', fontsize=16, fontweight='bold')
ax4.legend(fontsize=10, loc='best')
ax4.grid(True, alpha=0.3)
ax4.set_xlim(0, 1)

plt.suptitle('kSZ-21cm Cross-Correlation Evolution', 
             fontsize=18, fontweight='bold')
plt.savefig('plots/plots_single_sim/correlation_evolution.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# CELL 10: Correlation Coefficient r(ℓ) Analysis
# ============================================================================

# Choose 5 representative xHI values (epochs)
n_epochs_plot = min(5, len(results['mean_xhi']))
indices_to_plot_r = np.linspace(0, len(results['mean_xhi'])-1, n_epochs_plot, dtype=int)
colors_r = plt.cm.viridis(np.linspace(0, 1, n_epochs_plot))

fig4 = plt.figure(figsize=(24, 24))
gs4 = fig4.add_gridspec(4, 4, hspace=0.35, wspace=0.3)

# Row 1: |kSZ| × 21cm
ax1 = fig4.add_subplot(gs4[0, 0])
for idx, i in enumerate(indices_to_plot_r):
    ell = results['ell_cross_21cm'][i]
    r_k = results['r_k_21cm'][i]
    xhi = results['mean_xhi'][i]
    z = results['mean_z'][i]
    valid = np.isfinite(ell) & np.isfinite(r_k) & (ell <= 10000)
    if np.any(valid):
        ax1.plot(ell[valid], r_k[valid], '-', linewidth=2, alpha=0.8,
                 color=colors_r[idx], label=f'xHI={xhi:.2f}')
ax1.axhline(0, color='black', linewidth=0.8, linestyle=':')
ax1.set_xlabel('Multipole ℓ', fontsize=10, fontweight='bold')
ax1.set_ylabel('r(ℓ)', fontsize=10, fontweight='bold')
ax1.set_title('|kSZ| × 21cm (Real)', fontsize=12, fontweight='bold')
ax1.legend(fontsize=8, loc='best')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(-1, 1)
ax1.set_xlim(0, 10000)

ax2 = fig4.add_subplot(gs4[0, 1])
for idx, i in enumerate(indices_to_plot_r):
    ell = results['ell_cross_21cm_rec'][i]
    r_k = results['r_k_21cm_rec'][i]
    xhi = results['mean_xhi'][i]
    valid = np.isfinite(ell) & np.isfinite(r_k) & (ell <= 10000)
    if np.any(valid):
        ax2.plot(ell[valid], r_k[valid], '-', linewidth=2, alpha=0.8,
                 color=colors_r[idx], label=f'xHI={xhi:.2f}')
ax2.axhline(0, color='black', linewidth=0.8, linestyle=':')
ax2.set_xlabel('Multipole ℓ', fontsize=10, fontweight='bold')
ax2.set_ylabel('r(ℓ)', fontsize=10, fontweight='bold')
ax2.set_title('|kSZ| × 21cm (Rec)', fontsize=12, fontweight='bold')
ax2.legend(fontsize=8, loc='best')
ax2.grid(True, alpha=0.3)
ax2.set_ylim(-1, 1)
ax2.set_xlim(0, 10000)

# Row 1 continued: |kSZ| × 21cm²
ax3 = fig4.add_subplot(gs4[0, 2])
for idx, i in enumerate(indices_to_plot_r):
    ell = results['ell_cross_21cm_sq'][i]
    r_k = results['r_k_21cm_sq'][i]
    xhi = results['mean_xhi'][i]
    valid = np.isfinite(ell) & np.isfinite(r_k) & (ell <= 10000)
    if np.any(valid):
        ax3.plot(ell[valid], r_k[valid], '-', linewidth=2, alpha=0.8,
                 color=colors_r[idx], label=f'xHI={xhi:.2f}')
ax3.axhline(0, color='black', linewidth=0.8, linestyle=':')
ax3.set_xlabel('Multipole ℓ', fontsize=10, fontweight='bold')
ax3.set_ylabel('r(ℓ)', fontsize=10, fontweight='bold')
ax3.set_title('|kSZ| × 21cm² (Real)', fontsize=12, fontweight='bold')
ax3.legend(fontsize=8, loc='best')
ax3.grid(True, alpha=0.3)
ax3.set_ylim(-1, 1)
ax3.set_xlim(0, 10000)

ax4 = fig4.add_subplot(gs4[0, 3])
for idx, i in enumerate(indices_to_plot_r):
    ell = results['ell_cross_21cm_sq_rec'][i]
    r_k = results['r_k_21cm_sq_rec'][i]
    xhi = results['mean_xhi'][i]
    valid = np.isfinite(ell) & np.isfinite(r_k) & (ell <= 10000)
    if np.any(valid):
        ax4.plot(ell[valid], r_k[valid], '-', linewidth=2, alpha=0.8,
                 color=colors_r[idx], label=f'xHI={xhi:.2f}')
ax4.axhline(0, color='black', linewidth=0.8, linestyle=':')
ax4.set_xlabel('Multipole ℓ', fontsize=10, fontweight='bold')
ax4.set_ylabel('r(ℓ)', fontsize=10, fontweight='bold')
ax4.set_title('|kSZ| × 21cm² (Rec)', fontsize=12, fontweight='bold')
ax4.legend(fontsize=8, loc='best')
ax4.grid(True, alpha=0.3)
ax4.set_ylim(-1, 1)
ax4.set_xlim(0, 10000)

# Row 2: kSZ² × 21cm
ax5 = fig4.add_subplot(gs4[1, 0])
for idx, i in enumerate(indices_to_plot_r):
    ell = results['ell_cross_ksz_sq_21cm'][i]
    r_k = results['r_k_ksz_sq_21cm'][i]
    xhi = results['mean_xhi'][i]
    valid = np.isfinite(ell) & np.isfinite(r_k) & (ell <= 10000)
    if np.any(valid):
        ax5.plot(ell[valid], r_k[valid], '-', linewidth=2, alpha=0.8,
                 color=colors_r[idx], label=f'xHI={xhi:.2f}')
ax5.axhline(0, color='black', linewidth=0.8, linestyle=':')
ax5.set_xlabel('Multipole ℓ', fontsize=10, fontweight='bold')
ax5.set_ylabel('r(ℓ)', fontsize=10, fontweight='bold')
ax5.set_title('kSZ² × 21cm (Real)', fontsize=12, fontweight='bold')
ax5.legend(fontsize=8, loc='best')
ax5.grid(True, alpha=0.3)
ax5.set_ylim(-1, 1)
ax5.set_xlim(0, 10000)

ax6 = fig4.add_subplot(gs4[1, 1])
for idx, i in enumerate(indices_to_plot_r):
    ell = results['ell_cross_ksz_sq_21cm_rec'][i]
    r_k = results['r_k_ksz_sq_21cm_rec'][i]
    xhi = results['mean_xhi'][i]
    valid = np.isfinite(ell) & np.isfinite(r_k) & (ell <= 10000)
    if np.any(valid):
        ax6.plot(ell[valid], r_k[valid], '-', linewidth=2, alpha=0.8,
                 color=colors_r[idx], label=f'xHI={xhi:.2f}')
ax6.axhline(0, color='black', linewidth=0.8, linestyle=':')
ax6.set_xlabel('Multipole ℓ', fontsize=10, fontweight='bold')
ax6.set_ylabel('r(ℓ)', fontsize=10, fontweight='bold')
ax6.set_title('kSZ² × 21cm (Rec)', fontsize=12, fontweight='bold')
ax6.legend(fontsize=8, loc='best')
ax6.grid(True, alpha=0.3)
ax6.set_ylim(-1, 1)
ax6.set_xlim(0, 10000)

# Row 2 continued: kSZ² × 21cm²
ax7 = fig4.add_subplot(gs4[1, 2])
for idx, i in enumerate(indices_to_plot_r):
    ell = results['ell_cross_ksz_sq_21cm_sq'][i]
    r_k = results['r_k_ksz_sq_21cm_sq'][i]
    xhi = results['mean_xhi'][i]
    valid = np.isfinite(ell) & np.isfinite(r_k) & (ell <= 10000)
    if np.any(valid):
        ax7.plot(ell[valid], r_k[valid], '-', linewidth=2, alpha=0.8,
                 color=colors_r[idx], label=f'xHI={xhi:.2f}')
ax7.axhline(0, color='black', linewidth=0.8, linestyle=':')
ax7.set_xlabel('Multipole ℓ', fontsize=10, fontweight='bold')
ax7.set_ylabel('r(ℓ)', fontsize=10, fontweight='bold')
ax7.set_title('kSZ² × 21cm² (Real)', fontsize=12, fontweight='bold')
ax7.legend(fontsize=8, loc='best')
ax7.grid(True, alpha=0.3)
ax7.set_ylim(-1, 1)
ax7.set_xlim(0, 10000)

ax8 = fig4.add_subplot(gs4[1, 3])
for idx, i in enumerate(indices_to_plot_r):
    ell = results['ell_cross_ksz_sq_21cm_sq_rec'][i]
    r_k = results['r_k_ksz_sq_21cm_sq_rec'][i]
    xhi = results['mean_xhi'][i]
    valid = np.isfinite(ell) & np.isfinite(r_k) & (ell <= 10000)
    if np.any(valid):
        ax8.plot(ell[valid], r_k[valid], '-', linewidth=2, alpha=0.8,
                 color=colors_r[idx], label=f'xHI={xhi:.2f}')
ax8.axhline(0, color='black', linewidth=0.8, linestyle=':')
ax8.set_xlabel('Multipole ℓ', fontsize=10, fontweight='bold')
ax8.set_ylabel('r(ℓ)', fontsize=10, fontweight='bold')
ax8.set_title('kSZ² × 21cm² (Rec)', fontsize=12, fontweight='bold')
ax8.legend(fontsize=8, loc='best')
ax8.grid(True, alpha=0.3)
ax8.set_ylim(-1, 1)
ax8.set_xlim(0, 10000)

plt.suptitle('Correlation Coefficient r(ℓ) = P_cross / √(P_kSZ × P_21cm) - All Combinations', 
             fontsize=18, fontweight='bold')
plt.savefig('plots/plots_single_sim/r_ell_analysis_all.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# CELL 11: kSZ Reconstruction vs Real kSZ Comparison
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

# Compute kSZ and velocity correlations for each chunk
ksz_correlations = {
    'mean_xhi': [],
    'mean_z': [],
    'r_ksz_signed': [],      # correlation between kSZ_real and kSZ_rec (signed)
    'r_ksz_abs': [],         # correlation between |kSZ_real| and |kSZ_rec| (absolute)
    'r_vz': [],              # correlation between vz_real and vz_rec
    'chunk_names': []
}

print("Computing kSZ correlations for each chunk...")
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

plt.suptitle('kSZ Reconstruction Quality: Real vs Reconstructed Comparison', 
             fontsize=18, fontweight='bold')
plt.savefig('plots/plots_single_sim/ksz_reconstruction_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Print kSZ reconstruction summary
print("\n" + "="*80)
print("kSZ RECONSTRUCTION SUMMARY")
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
# CELL 12: Summary Statistics
# ============================================================================

print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

# Find peak correlations for different combinations
print("\n=== PEAK CORRELATIONS ===")

# |kSZ| × 21cm
valid_mask = np.isfinite(results['correlation_21cm'])
if np.any(valid_mask):
    idx_max = np.nanargmax(results['correlation_21cm'])
    print(f"\n|kSZ| × 21cm (Real velocity):")
    print(f"  r_max = {results['correlation_21cm'][idx_max]:+.4f}")
    print(f"  at mean_xHI = {results['mean_xhi'][idx_max]:.4f}, z = {results['mean_z'][idx_max]:.3f}")

valid_mask = np.isfinite(results['correlation_21cm_rec'])
if np.any(valid_mask):
    idx_max = np.nanargmax(results['correlation_21cm_rec'])
    print(f"\n|kSZ| × 21cm (Reconstructed velocity):")
    print(f"  r_max = {results['correlation_21cm_rec'][idx_max]:+.4f}")
    print(f"  at mean_xHI = {results['mean_xhi'][idx_max]:.4f}, z = {results['mean_z'][idx_max]:.3f}")

# |kSZ| × 21cm²
valid_mask = np.isfinite(results['correlation_21cm_sq'])
if np.any(valid_mask):
    idx_max = np.nanargmax(results['correlation_21cm_sq'])
    print(f"\n|kSZ| × 21cm² (Real velocity):")
    print(f"  r_max = {results['correlation_21cm_sq'][idx_max]:+.4f}")
    print(f"  at mean_xHI = {results['mean_xhi'][idx_max]:.4f}, z = {results['mean_z'][idx_max]:.3f}")

valid_mask = np.isfinite(results['correlation_21cm_sq_rec'])
if np.any(valid_mask):
    idx_max = np.nanargmax(results['correlation_21cm_sq_rec'])
    print(f"\n|kSZ| × 21cm² (Reconstructed velocity):")
    print(f"  r_max = {results['correlation_21cm_sq_rec'][idx_max]:+.4f}")
    print(f"  at mean_xHI = {results['mean_xhi'][idx_max]:.4f}, z = {results['mean_z'][idx_max]:.3f}")

# Correlation in different xHI regimes
print("\n=== CORRELATIONS BY REIONIZATION PHASE ===")
high_xhi = results['mean_xhi'] > 0.5
low_xhi = results['mean_xhi'] < 0.5

if np.any(high_xhi):
    print(f"\nHigh neutral fraction (xHI > 0.5):")
    print(f"  |kSZ| × 21cm (Real):  {np.nanmean(results['correlation_21cm'][high_xhi]):+.4f}")
    print(f"  |kSZ| × 21cm (Rec):   {np.nanmean(results['correlation_21cm_rec'][high_xhi]):+.4f}")
    print(f"  |kSZ| × 21cm² (Real): {np.nanmean(results['correlation_21cm_sq'][high_xhi]):+.4f}")
    print(f"  |kSZ| × 21cm² (Rec):  {np.nanmean(results['correlation_21cm_sq_rec'][high_xhi]):+.4f}")

if np.any(low_xhi):
    print(f"\nLow neutral fraction (xHI < 0.5):")
    print(f"  |kSZ| × 21cm (Real):  {np.nanmean(results['correlation_21cm'][low_xhi]):+.4f}")
    print(f"  |kSZ| × 21cm (Rec):   {np.nanmean(results['correlation_21cm_rec'][low_xhi]):+.4f}")
    print(f"  |kSZ| × 21cm² (Real): {np.nanmean(results['correlation_21cm_sq'][low_xhi]):+.4f}")
    print(f"  |kSZ| × 21cm² (Rec):  {np.nanmean(results['correlation_21cm_sq_rec'][low_xhi]):+.4f}")

print("="*80)

# Save results to file (exclude velocity fields which have variable shapes)
import os
os.makedirs('results', exist_ok=True)
results_to_save = {k: v for k, v in results.items() if k not in ['vz_real', 'vz_rec']}
np.savez('results/xhi_evolution_results.npz', **results_to_save)
print("\nResults saved to: results/xhi_evolution_results.npz")

print("\nAnalysis complete!")
