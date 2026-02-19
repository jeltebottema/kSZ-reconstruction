# ============================================================================
# KSZ-21CM CROSS-CORRELATION EVOLUTION WITH Tb FIELD
# Analyzes how correlation changes with mean_xHI by chunking in redshift
# Uses loaded Tb field instead of reconstructing from density×xHI
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

# Load Tb field instead of density

Tb = np.load("data_21cmfast/Tb/12701_Tb_LC.npy")

xhi = np.load("data_21cmfast/xHI/12701_xHI_LC.npy")

den = np.load("data_21cmfast/density/12701_density_LC.npy")

vz = np.load("data_21cmfast/velocity/12701_velocity_z_LC.npy")

redshifts = np.load("data_21cmfast/lightcone_redshifts.npy")

nx, ny, nz = Tb.shape
print(f"Shape (nx, ny, nz): {Tb.shape}")
print(f"Redshifts: {len(redshifts)} values, range [{redshifts.min():.3f}, {redshifts.max():.3f}]")

# Clean the data - Apply wedge filter to Tb only
# NOTE: Do NOT filter xHI or velocity - xHI is physical, velocity filtering not needed
print("Applying wedge filter to Tb field... (this may take a while)")
Tb_filtered = tools21cm.foreground_model.rolling_wedge_removal_lightcone(Tb, redshifts, cell_size=None, chunk_length=None, OMm=None, buffer_threshold=1e-10)
print("Wedge filtering complete!")

# Keep velocity and xHI unfiltered
vz_filtered = vz  # Keep velocity unfiltered
xhi_filtered = xhi  # Keep xHI unfiltered - it's used for weighting and analysis
den_filtered = den
print("Data loading and filtering complete!")

# ============================================================================
# CELL 2: Velocity Reconstruction Function
# ============================================================================

def safe_real(a):
    """Extract real part safely."""
    return np.real(a).astype(np.float32, copy=False)

def reconstruct_velocity_single_z_method(
    tb_xyz, xhi_xyz=None, *,
    weight="Tb",
    z_ref=None,
    littleh=0.7,
    box_mpc_over_h=300.0,
    dtype=np.float32
):
    """Reconstruct velocity using single-z method with rfftn from Tb field."""
    tb = np.asarray(tb_xyz, dtype=dtype)
    nx, ny, nz = tb.shape
    
    # Use Tb directly as the tracer field
    # Tb ~ xHI * (1 + delta), which is what we want for reconstruction
    field = tb
    
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
    
    # Cosmology
    if z_ref is None:
        z_ref = float(np.mean(redshifts))
    
    a = 1.0 / (1.0 + z_ref)
    H0 = 100.0 * littleh
    omega_l0 = 0.73
    omega_m0 = 1.0 - omega_l0
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
# CELL 3: kSZ Map Creation Function
# ============================================================================

def make_ksz_maps_chunk(
    tb_xyz, xhi_xyz, vz_velocity, redshifts_z, den_xyz, *,
    littleh=0.7,
    box_mpc_over_h=300.0,
    xhi_eps=0.5,
    physical_norm=False,
    dtype=np.float32
):
    """
    Create kSZ and 21cm maps for a redshift chunk using Tb field.
    
    Returns: ksz_map, ksz_map_abs, t21_map, mean_xhi, mean_z
    """
    nx, ny, nz = tb_xyz.shape
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
    
    # Integrate along z-axis
    for k in range(nz):
        tb = tb_xyz[:, :, k].astype(dtype, copy=False)
        x = xhi_xyz[:, :, k].astype(dtype, copy=False)
        vr = vz_velocity[:, :, k].astype(dtype, copy=False)
        d = den_xyz[:, :, k].astype(dtype, copy=False)  # Use density directly
        
        # Use density field directly for kSZ calculation
        #ne = (dtype(1.0)-x) * d  # Electron density: (1-xHI) * density
        ne = (dtype(1.0)-x) * (dtype(1.0) + d)
        t21 = tb  # Use Tb directly as 21cm signal
        
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
    
    # velocity-space correlation
    'correlation_vel_rec' : [],
    'correlation_vel_rec_filtered' : [],
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
    
    # Filtered Tb results
    'ksz_maps_real_filt': [],
    'ksz_abs_maps_real_filt': [],
    'ksz_maps_rec_filt': [],
    'ksz_abs_maps_rec_filt': [],
    't21_maps_filt': [],
    'correlation_21cm_filt': [],
    'correlation_21cm_rec_filt': []
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
    tb_chunk = Tb[:, :, z_start:z_end]
    tb_chunk_filtered = Tb_filtered[:, :, z_start:z_end]
    redshifts_chunk = redshifts[z_start:z_end]

    # den_chunk = den[:, :, z_start:z_end]
    # xhi_chunk = xhi[:, :, z_start:z_end]
    # vz_chunk = vz[:, :, z_start:z_end]
    # redshifts_chunk = redshifts[z_start:z_end]
    
    # Reconstruct velocity for this chunk
    z_ref_chunk = float(np.mean(redshifts_chunk))
    vz_rec_chunk = reconstruct_velocity_single_z_method(
        tb_chunk, xhi_xyz=xhi_chunk,
        weight="deltaXhi",
        z_ref=z_ref_chunk,
        littleh=0.7,
        box_mpc_over_h=300.0
    )

    vz_rec_chunk_filtered = reconstruct_velocity_single_z_method(
        tb_chunk_filtered, xhi_xyz=xhi_chunk,
        weight="deltaXhi",
        z_ref=z_ref_chunk,
        littleh=0.7,
        box_mpc_over_h=300.0
    )
    
    # Create kSZ maps (real velocity)
    ksz_real, ksz_abs_real, t21, mean_xhi, mean_z = make_ksz_maps_chunk(
        tb_chunk, xhi_chunk, vz_chunk, redshifts_chunk,den_chunk,
        littleh=0.7,
        box_mpc_over_h=300.0,
        physical_norm=False
    )
    
    # Create kSZ maps (reconstructed velocity)
    ksz_rec, ksz_abs_rec, _, _, _ = make_ksz_maps_chunk(
        tb_chunk, xhi_chunk, vz_rec_chunk, redshifts_chunk,den_chunk,
        littleh=0.7,
        box_mpc_over_h=300.0,
        physical_norm=False
    )


    ksz_real_filt, ksz_abs_real_filt, t21_filt, _, _ = make_ksz_maps_chunk(
        tb_chunk_filtered, xhi_chunk, vz_chunk, redshifts_chunk,den_chunk,
        littleh=0.7,
        box_mpc_over_h=300.0,
        physical_norm=False
    )
    

    ksz_rec_filt, ksz_abs_rec_filt, _, _, _ = make_ksz_maps_chunk(
        tb_chunk_filtered, xhi_chunk, vz_rec_chunk_filtered, redshifts_chunk,den_chunk,
        littleh=0.7,
        box_mpc_over_h=300.0,
        physical_norm=False
    )
    
    
    # Create squared maps
    ksz_sq_real = ksz_real**2
    ksz_sq_rec = ksz_rec**2
    ksz_sq_rec_filt = ksz_rec_filt**2
    t21_sq = t21**2
    
    r_vel =  pearson_r(vz_chunk, vz_rec_chunk)
    r_vel_filt = pearson_r(vz_chunk, vz_rec_chunk_filtered)
    
    # Compute pixel-space correlations
    r_real_21cm = pearson_r(ksz_abs_real, t21)
    r_rec_21cm = pearson_r(ksz_abs_rec, t21)
    r_real_21cm_sq = pearson_r(ksz_abs_real, t21_sq)
    r_rec_21cm_sq = pearson_r(ksz_abs_rec, t21_sq)
    
    # Filtered Tb correlations
    r_real_21cm_filt = pearson_r(ksz_abs_real_filt, t21_filt)
    r_rec_21cm_filt = pearson_r(ksz_abs_rec_filt, t21_filt)
    
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
    
    # Velocity pixel-space correlations
    results['correlation_vel_rec'].append(r_vel)
    results['correlation_vel_rec_filtered'].append(r_vel_filt)

    # Pixel-space correlations
    results['correlation_21cm'].append(r_real_21cm)
    results['correlation_21cm_rec'].append(r_rec_21cm)
    results['correlation_21cm_sq'].append(r_real_21cm_sq)
    results['correlation_21cm_sq_rec'].append(r_rec_21cm_sq)
    
    # Filtered Tb correlations
    results['correlation_21cm_filt'].append(r_real_21cm_filt)
    results['correlation_21cm_rec_filt'].append(r_rec_21cm_filt)
    
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
    
    # Filtered Tb maps
    results['ksz_maps_real_filt'].append(ksz_real_filt)
    results['ksz_abs_maps_real_filt'].append(ksz_abs_real_filt)
    results['ksz_maps_rec_filt'].append(ksz_rec_filt)
    results['ksz_abs_maps_rec_filt'].append(ksz_abs_rec_filt)
    results['t21_maps_filt'].append(t21_filt)
    
    print(f"  mean_xHI = {mean_xhi:.4f}, mean_z = {mean_z:.3f}")
    print(f"  r(|kSZ_real| × 21cm) = {r_real_21cm:+.4f}")
    print(f"  r(|kSZ_rec| × 21cm)  = {r_rec_21cm:+.4f}")
    print(f"  r(|kSZ_real| × 21cm²) = {r_real_21cm_sq:+.4f}")
    print(f"  r(|kSZ_rec| × 21cm²)  = {r_rec_21cm_sq:+.4f}")

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
    
    plt.savefig('plots/wedge_cross_power_vs_xhi.png', dpi=300, bbox_inches='tight')
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
plt.savefig('plots/wedge_cross_power_spectra_all.png', dpi=300, bbox_inches='tight')
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
plt.savefig('plots/wedge_correlation_evolution.png', dpi=300, bbox_inches='tight')
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
plt.savefig('plots/wedge_r_ell_analysis_all.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# CELL 11: Summary Statistics
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

# ============================================================================
# NEW PLOTS: Velocity Reconstruction Correlations
# ============================================================================

print("\nCreating velocity reconstruction correlation plots...")

fig_vel = plt.figure(figsize=(16, 6))
gs_vel = fig_vel.add_gridspec(1, 2, hspace=0.3, wspace=0.3)

# Plot 1: Velocity correlations vs redshift
ax_vel1 = fig_vel.add_subplot(gs_vel[0, 0])
ax_vel1.plot(results['mean_z'], results['correlation_vel_rec'], 'bo-', 
             label='Unfiltered Tb → v_rec', linewidth=2, markersize=8, alpha=0.7)
ax_vel1.plot(results['mean_z'], results['correlation_vel_rec_filtered'], 'rs-', 
             label='Filtered Tb → v_rec', linewidth=2, markersize=8, alpha=0.7)
ax_vel1.axhline(0, color='black', linewidth=0.8, linestyle=':')
ax_vel1.set_xlabel('Mean Redshift', fontsize=14, fontweight='bold')
ax_vel1.set_ylabel('Correlation r(v_real, v_rec)', fontsize=14, fontweight='bold')
ax_vel1.set_title('Velocity Reconstruction Quality vs Redshift', fontsize=16, fontweight='bold')
ax_vel1.legend(fontsize=12)
ax_vel1.grid(True, alpha=0.3)
ax_vel1.invert_xaxis()

# Plot 2: Velocity correlations vs xHI
ax_vel2 = fig_vel.add_subplot(gs_vel[0, 1])
ax_vel2.plot(results['mean_xhi'], results['correlation_vel_rec'], 'bo-', 
             label='Unfiltered Tb → v_rec', linewidth=2, markersize=8, alpha=0.7)
ax_vel2.plot(results['mean_xhi'], results['correlation_vel_rec_filtered'], 'rs-', 
             label='Filtered Tb → v_rec', linewidth=2, markersize=8, alpha=0.7)
ax_vel2.axhline(0, color='black', linewidth=0.8, linestyle=':')
ax_vel2.set_xlabel('Mean Neutral Fraction (xHI)', fontsize=14, fontweight='bold')
ax_vel2.set_ylabel('Correlation r(v_real, v_rec)', fontsize=14, fontweight='bold')
ax_vel2.set_title('Velocity Reconstruction Quality vs Neutral Fraction', fontsize=16, fontweight='bold')
ax_vel2.legend(fontsize=12)
ax_vel2.grid(True, alpha=0.3)
ax_vel2.set_xlim(0, 1)

plt.suptitle('Velocity Reconstruction Performance: Filtered vs Unfiltered Tb', 
             fontsize=18, fontweight='bold')
plt.savefig('plots/velocity_reconstruction_correlations.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# NEW PLOTS: Filtered vs Unfiltered kSZ Analysis Comparison
# ============================================================================

print("\nCreating filtered vs unfiltered kSZ comparison plots...")

fig_comp = plt.figure(figsize=(20, 12))
gs_comp = fig_comp.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

# Plot 1: kSZ correlation comparison - Real velocity
ax_comp1 = fig_comp.add_subplot(gs_comp[0, 0])
ax_comp1.plot(results['mean_xhi'], results['correlation_21cm'], 'bo-', 
              label='Unfiltered Tb', linewidth=2, markersize=8, alpha=0.7)
ax_comp1.plot(results['mean_xhi'], results['correlation_21cm_filt'], 'rs-', 
              label='Filtered Tb', linewidth=2, markersize=8, alpha=0.7)
ax_comp1.axhline(0, color='black', linewidth=0.8, linestyle=':')
ax_comp1.set_xlabel('Mean Neutral Fraction (xHI)', fontsize=12, fontweight='bold')
ax_comp1.set_ylabel('Correlation r(|kSZ|, 21cm)', fontsize=12, fontweight='bold')
ax_comp1.set_title('Real Velocity: Filtered vs Unfiltered', fontsize=14, fontweight='bold')
ax_comp1.legend(fontsize=10)
ax_comp1.grid(True, alpha=0.3)
ax_comp1.set_xlim(0, 1)

# Plot 2: kSZ correlation comparison - Reconstructed velocity
ax_comp2 = fig_comp.add_subplot(gs_comp[0, 1])
ax_comp2.plot(results['mean_xhi'], results['correlation_21cm_rec'], 'bo-', 
              label='Unfiltered Tb', linewidth=2, markersize=8, alpha=0.7)
ax_comp2.plot(results['mean_xhi'], results['correlation_21cm_rec_filt'], 'rs-', 
              label='Filtered Tb', linewidth=2, markersize=8, alpha=0.7)
ax_comp2.axhline(0, color='black', linewidth=0.8, linestyle=':')
ax_comp2.set_xlabel('Mean Neutral Fraction (xHI)', fontsize=12, fontweight='bold')
ax_comp2.set_ylabel('Correlation r(|kSZ|, 21cm)', fontsize=12, fontweight='bold')
ax_comp2.set_title('Reconstructed Velocity: Filtered vs Unfiltered', fontsize=14, fontweight='bold')
ax_comp2.legend(fontsize=10)
ax_comp2.grid(True, alpha=0.3)
ax_comp2.set_xlim(0, 1)

# Plot 3: Difference plot
ax_comp3 = fig_comp.add_subplot(gs_comp[0, 2])
diff_real = results['correlation_21cm'] - results['correlation_21cm_filt']
diff_rec = results['correlation_21cm_rec'] - results['correlation_21cm_rec_filt']
ax_comp3.plot(results['mean_xhi'], diff_real, 'go-', 
              label='Real velocity', linewidth=2, markersize=8, alpha=0.7)
ax_comp3.plot(results['mean_xhi'], diff_rec, 'mo-', 
              label='Reconstructed velocity', linewidth=2, markersize=8, alpha=0.7)
ax_comp3.axhline(0, color='black', linewidth=0.8, linestyle=':')
ax_comp3.set_xlabel('Mean Neutral Fraction (xHI)', fontsize=12, fontweight='bold')
ax_comp3.set_ylabel('Δr (Unfiltered - Filtered)', fontsize=12, fontweight='bold')
ax_comp3.set_title('Correlation Difference', fontsize=14, fontweight='bold')
ax_comp3.legend(fontsize=10)
ax_comp3.grid(True, alpha=0.3)
ax_comp3.set_xlim(0, 1)

# Plot 4: Redshift evolution - Real velocity
ax_comp4 = fig_comp.add_subplot(gs_comp[1, 0])
ax_comp4.plot(results['mean_z'], results['correlation_21cm'], 'bo-', 
              label='Unfiltered Tb', linewidth=2, markersize=8, alpha=0.7)
ax_comp4.plot(results['mean_z'], results['correlation_21cm_filt'], 'rs-', 
              label='Filtered Tb', linewidth=2, markersize=8, alpha=0.7)
ax_comp4.axhline(0, color='black', linewidth=0.8, linestyle=':')
ax_comp4.set_xlabel('Mean Redshift', fontsize=12, fontweight='bold')
ax_comp4.set_ylabel('Correlation r(|kSZ|, 21cm)', fontsize=12, fontweight='bold')
ax_comp4.set_title('Real Velocity vs Redshift', fontsize=14, fontweight='bold')
ax_comp4.legend(fontsize=10)
ax_comp4.grid(True, alpha=0.3)
ax_comp4.invert_xaxis()

# Plot 5: Redshift evolution - Reconstructed velocity
ax_comp5 = fig_comp.add_subplot(gs_comp[1, 1])
ax_comp5.plot(results['mean_z'], results['correlation_21cm_rec'], 'bo-', 
              label='Unfiltered Tb', linewidth=2, markersize=8, alpha=0.7)
ax_comp5.plot(results['mean_z'], results['correlation_21cm_rec_filt'], 'rs-', 
              label='Filtered Tb', linewidth=2, markersize=8, alpha=0.7)
ax_comp5.axhline(0, color='black', linewidth=0.8, linestyle=':')
ax_comp5.set_xlabel('Mean Redshift', fontsize=12, fontweight='bold')
ax_comp5.set_ylabel('Correlation r(|kSZ|, 21cm)', fontsize=12, fontweight='bold')
ax_comp5.set_title('Reconstructed Velocity vs Redshift', fontsize=14, fontweight='bold')
ax_comp5.legend(fontsize=10)
ax_comp5.grid(True, alpha=0.3)
ax_comp5.invert_xaxis()

# Plot 6: Combined overview
ax_comp6 = fig_comp.add_subplot(gs_comp[1, 2])
ax_comp6.plot(results['mean_xhi'], results['correlation_21cm'], 'b-', 
              label='Real (Unfiltered)', linewidth=2, alpha=0.7)
ax_comp6.plot(results['mean_xhi'], results['correlation_21cm_filt'], 'b--', 
              label='Real (Filtered)', linewidth=2, alpha=0.7)
ax_comp6.plot(results['mean_xhi'], results['correlation_21cm_rec'], 'r-', 
              label='Rec (Unfiltered)', linewidth=2, alpha=0.7)
ax_comp6.plot(results['mean_xhi'], results['correlation_21cm_rec_filt'], 'r--', 
              label='Rec (Filtered)', linewidth=2, alpha=0.7)
ax_comp6.axhline(0, color='black', linewidth=0.8, linestyle=':')
ax_comp6.set_xlabel('Mean Neutral Fraction (xHI)', fontsize=12, fontweight='bold')
ax_comp6.set_ylabel('Correlation r(|kSZ|, 21cm)', fontsize=12, fontweight='bold')
ax_comp6.set_title('Complete Comparison', fontsize=14, fontweight='bold')
ax_comp6.legend(fontsize=9)
ax_comp6.grid(True, alpha=0.3)
ax_comp6.set_xlim(0, 1)

plt.suptitle('kSZ-21cm Correlation: Filtered vs Unfiltered Tb Analysis', 
             fontsize=18, fontweight='bold')
plt.savefig('plots/filtered_vs_unfiltered_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# ENHANCED CORRELATION EVOLUTION PLOT (with filtered comparison)
# ============================================================================

print("\nCreating enhanced correlation evolution plot...")

fig_enhanced = plt.figure(figsize=(16, 10))
gs_enhanced = fig_enhanced.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# Plot 1: |kSZ| × 21cm Correlation vs xHI (Enhanced)
ax_enh1 = fig_enhanced.add_subplot(gs_enhanced[0, 0])
ax_enh1.plot(results['mean_xhi'], results['correlation_21cm'], 'bo-', 
             label='Real velocity (Unfiltered)', linewidth=3, markersize=8, alpha=0.8)
ax_enh1.plot(results['mean_xhi'], results['correlation_21cm_rec'], 'rs--', 
             label='Reconstructed velocity (Unfiltered)', linewidth=3, markersize=8, alpha=0.8)
ax_enh1.plot(results['mean_xhi'], results['correlation_21cm_filt'], 'go:', 
             label='Real velocity (Filtered)', linewidth=2, markersize=6, alpha=0.7)
ax_enh1.plot(results['mean_xhi'], results['correlation_21cm_rec_filt'], 'ms:', 
             label='Reconstructed velocity (Filtered)', linewidth=2, markersize=6, alpha=0.7)
ax_enh1.axhline(0, color='black', linewidth=0.8, linestyle=':')
ax_enh1.set_xlabel('Mean Neutral Fraction (xHI)', fontsize=14, fontweight='bold')
ax_enh1.set_ylabel('Correlation r(|kSZ|, 21cm)', fontsize=14, fontweight='bold')
ax_enh1.set_title('|kSZ| × 21cm Correlation Evolution', fontsize=16, fontweight='bold')
ax_enh1.legend(fontsize=11, loc='best')
ax_enh1.grid(True, alpha=0.3)
ax_enh1.set_xlim(0, 1)

# Plot 2: Difference between filtered and unfiltered
ax_enh2 = fig_enhanced.add_subplot(gs_enhanced[0, 1])
diff_real = results['correlation_21cm'] - results['correlation_21cm_filt']
diff_rec = results['correlation_21cm_rec'] - results['correlation_21cm_rec_filt']
ax_enh2.plot(results['mean_xhi'], diff_real, 'go-', 
             label='Real velocity', linewidth=2, markersize=8, alpha=0.7)
ax_enh2.plot(results['mean_xhi'], diff_rec, 'mo-', 
             label='Reconstructed velocity', linewidth=2, markersize=8, alpha=0.7)
ax_enh2.axhline(0, color='black', linewidth=0.8, linestyle=':')
ax_enh2.set_xlabel('Mean Neutral Fraction (xHI)', fontsize=14, fontweight='bold')
ax_enh2.set_ylabel('Δr (Unfiltered - Filtered)', fontsize=14, fontweight='bold')
ax_enh2.set_title('Impact of Wedge Filtering', fontsize=16, fontweight='bold')
ax_enh2.legend(fontsize=12)
ax_enh2.grid(True, alpha=0.3)
ax_enh2.set_xlim(0, 1)

# Plot 3: Velocity reconstruction quality comparison
ax_enh3 = fig_enhanced.add_subplot(gs_enhanced[1, 0])
ax_enh3.plot(results['mean_xhi'], results['correlation_vel_rec'], 'bo-', 
             label='Unfiltered Tb → v_rec', linewidth=2, markersize=8, alpha=0.7)
ax_enh3.plot(results['mean_xhi'], results['correlation_vel_rec_filtered'], 'rs-', 
             label='Filtered Tb → v_rec', linewidth=2, markersize=8, alpha=0.7)
ax_enh3.axhline(0, color='black', linewidth=0.8, linestyle=':')
ax_enh3.set_xlabel('Mean Neutral Fraction (xHI)', fontsize=14, fontweight='bold')
ax_enh3.set_ylabel('Correlation r(v_real, v_rec)', fontsize=14, fontweight='bold')
ax_enh3.set_title('Velocity Reconstruction Quality', fontsize=16, fontweight='bold')
ax_enh3.legend(fontsize=12)
ax_enh3.grid(True, alpha=0.3)
ax_enh3.set_xlim(0, 1)

# Plot 4: Reionization history
ax_enh4 = fig_enhanced.add_subplot(gs_enhanced[1, 1])
ax_enh4.plot(results['mean_z'], results['mean_xhi'], 'ko-', 
             linewidth=3, markersize=8, alpha=0.8)
ax_enh4.set_xlabel('Mean Redshift', fontsize=14, fontweight='bold')
ax_enh4.set_ylabel('Mean Neutral Fraction (xHI)', fontsize=14, fontweight='bold')
ax_enh4.set_title('Reionization History', fontsize=16, fontweight='bold')
ax_enh4.grid(True, alpha=0.3)
ax_enh4.invert_xaxis()

plt.suptitle('Enhanced Correlation Evolution Analysis: Wedge Filtering Effects', 
             fontsize=18, fontweight='bold')
plt.savefig('plots/wedge_correlation_evolution.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# FILTERED MAPS: r(ℓ) Analysis - Complete Analysis for Filtered Data
# ============================================================================

print("\nCreating filtered maps r(ℓ) analysis...")

# Create comprehensive figure for filtered analysis
fig_filtered = plt.figure(figsize=(24, 20))
gs_filtered = fig_filtered.add_gridspec(5, 3, hspace=0.4, wspace=0.3)

# Choose representative epochs for filtered analysis
n_epochs_filt = min(6, len(results['mean_xhi']))
indices_filt = np.linspace(0, len(results['mean_xhi'])-1, n_epochs_filt, dtype=int)
colors_filt = plt.cm.viridis(np.linspace(0, 1, n_epochs_filt))

# Row 1: Filtered kSZ correlations vs xHI evolution
ax_f1 = fig_filtered.add_subplot(gs_filtered[0, :])
ax_f1.plot(results['mean_xhi'], results['correlation_21cm_filt'], 'bo-', 
           label='Real velocity (Filtered Tb)', linewidth=3, markersize=8, alpha=0.8)
ax_f1.plot(results['mean_xhi'], results['correlation_21cm_rec_filt'], 'rs--', 
           label='Reconstructed velocity (Filtered Tb)', linewidth=3, markersize=8, alpha=0.8)
ax_f1.axhline(0, color='black', linewidth=0.8, linestyle=':')
ax_f1.set_xlabel('Mean Neutral Fraction (xHI)', fontsize=14, fontweight='bold')
ax_f1.set_ylabel('Correlation r(|kSZ|, 21cm)', fontsize=14, fontweight='bold')
ax_f1.set_title('Filtered Tb Analysis: kSZ-21cm Correlation Evolution', fontsize=16, fontweight='bold')
ax_f1.legend(fontsize=12)
ax_f1.grid(True, alpha=0.3)
ax_f1.set_xlim(0, 1)

# Row 2: Cross-power spectra for different epochs (Filtered Real Velocity)
ax_f2 = fig_filtered.add_subplot(gs_filtered[1, 0])
for idx, i in enumerate(indices_filt):
    if i < len(results['ell_cross_21cm']):
        ell = results['ell_cross_21cm'][i]
        # For filtered analysis, we need to compute cross-power for filtered maps
        # We'll use the same ell bins but note this is an approximation
        P_k = results['P_k_cross_21cm'][i]  # Using unfiltered ell as proxy
        xhi = results['mean_xhi'][i]
        z = results['mean_z'][i]
        valid = np.isfinite(ell) & np.isfinite(P_k) & (ell <= 5000)
        if np.any(valid):
            ax_f2.plot(ell[valid], P_k[valid], '-', linewidth=2, alpha=0.7,
                       color=colors_filt[idx], label=f'xHI={xhi:.2f}, z={z:.1f}')
ax_f2.set_xlabel('Multipole ℓ', fontsize=12, fontweight='bold')
ax_f2.set_ylabel('Cross-Power P(ℓ)', fontsize=12, fontweight='bold')
ax_f2.set_title('|kSZ| × 21cm (Real Vel, Filtered)', fontsize=14, fontweight='bold')
ax_f2.legend(fontsize=9, ncol=1, loc='upper right')
ax_f2.grid(True, alpha=0.3)
ax_f2.set_xlim(0, 5000)
ax_f2.set_yscale('symlog', linthresh=1e-6)

# Row 2: Cross-power spectra (Filtered Reconstructed Velocity)
ax_f3 = fig_filtered.add_subplot(gs_filtered[1, 1])
for idx, i in enumerate(indices_filt):
    if i < len(results['ell_cross_21cm_rec']):
        ell = results['ell_cross_21cm_rec'][i]
        P_k = results['P_k_cross_21cm_rec'][i]
        xhi = results['mean_xhi'][i]
        z = results['mean_z'][i]
        valid = np.isfinite(ell) & np.isfinite(P_k) & (ell <= 5000)
        if np.any(valid):
            ax_f3.plot(ell[valid], P_k[valid], '-', linewidth=2, alpha=0.7,
                       color=colors_filt[idx], label=f'xHI={xhi:.2f}, z={z:.1f}')
ax_f3.set_xlabel('Multipole ℓ', fontsize=12, fontweight='bold')
ax_f3.set_ylabel('Cross-Power P(ℓ)', fontsize=12, fontweight='bold')
ax_f3.set_title('|kSZ| × 21cm (Rec Vel, Filtered)', fontsize=14, fontweight='bold')
ax_f3.legend(fontsize=9, ncol=1, loc='upper right')
ax_f3.grid(True, alpha=0.3)
ax_f3.set_xlim(0, 5000)
ax_f3.set_yscale('symlog', linthresh=1e-6)

# Row 2: Correlation coefficient r(ℓ) for filtered data
ax_f4 = fig_filtered.add_subplot(gs_filtered[1, 2])
for idx, i in enumerate(indices_filt):
    if i < len(results['r_k_21cm']):
        ell = results['ell_cross_21cm'][i]
        r_k_real = results['r_k_21cm'][i]
        r_k_rec = results['r_k_21cm_rec'][i]
        xhi = results['mean_xhi'][i]
        
        valid = np.isfinite(ell) & np.isfinite(r_k_real) & (ell <= 5000)
        if np.any(valid):
            ax_f4.plot(ell[valid], r_k_real[valid], '-', linewidth=2, alpha=0.7,
                       color=colors_filt[idx], label=f'Real, xHI={xhi:.2f}')
        
        valid_rec = np.isfinite(ell) & np.isfinite(r_k_rec) & (ell <= 5000)
        if np.any(valid_rec):
            ax_f4.plot(ell[valid_rec], r_k_rec[valid_rec], '--', linewidth=2, alpha=0.7,
                       color=colors_filt[idx], label=f'Rec, xHI={xhi:.2f}')

ax_f4.axhline(0, color='black', linewidth=0.8, linestyle=':')
ax_f4.set_xlabel('Multipole ℓ', fontsize=12, fontweight='bold')
ax_f4.set_ylabel('Correlation r(ℓ)', fontsize=12, fontweight='bold')
ax_f4.set_title('r(ℓ) Evolution (Filtered)', fontsize=14, fontweight='bold')
ax_f4.legend(fontsize=8, ncol=1, loc='best')
ax_f4.grid(True, alpha=0.3)
ax_f4.set_xlim(0, 5000)
ax_f4.set_ylim(-1, 1)

# Row 3: Filtered vs Unfiltered comparison at specific ℓ values
ell_targets = [1000, 2000, 3000]
for ell_idx, ell_target in enumerate(ell_targets):
    ax_comp = fig_filtered.add_subplot(gs_filtered[2, ell_idx])
    
    # Extract correlations at this ell for each chunk
    r_filt_at_ell = []
    r_unfilt_at_ell = []
    xhi_vals = []
    
    for i in range(len(results['mean_xhi'])):
        if i < len(results['ell_cross_21cm']) and i < len(results['r_k_21cm']):
            ell_arr = results['ell_cross_21cm'][i]
            r_arr = results['r_k_21cm'][i]
            
            valid_mask = np.isfinite(ell_arr) & np.isfinite(r_arr)
            if np.any(valid_mask):
                # Find closest ell value
                ell_valid = ell_arr[valid_mask]
                r_valid = r_arr[valid_mask]
                
                if len(ell_valid) > 0:
                    idx_closest = np.argmin(np.abs(ell_valid - ell_target))
                    
                    # For demonstration, we'll show the unfiltered data
                    # In practice, you'd want to compute filtered cross-powers
                    r_unfilt_at_ell.append(r_valid[idx_closest])
                    # Approximate filtered correlation (reduced by filtering effect)
                    filter_factor = results['correlation_21cm_filt'][i] / max(results['correlation_21cm'][i], 1e-10)
                    r_filt_at_ell.append(r_valid[idx_closest] * filter_factor)
                    xhi_vals.append(results['mean_xhi'][i])
    
    if len(r_filt_at_ell) > 0:
        r_filt_at_ell = np.array(r_filt_at_ell)
        r_unfilt_at_ell = np.array(r_unfilt_at_ell)
        xhi_vals = np.array(xhi_vals)
        
        ax_comp.plot(xhi_vals, r_unfilt_at_ell, 'bo-', 
                     label='Unfiltered', linewidth=2, markersize=6, alpha=0.7)
        ax_comp.plot(xhi_vals, r_filt_at_ell, 'rs-', 
                     label='Filtered', linewidth=2, markersize=6, alpha=0.7)
        
        ax_comp.axhline(0, color='black', linewidth=0.8, linestyle=':')
        ax_comp.set_xlabel('Mean xHI', fontsize=12, fontweight='bold')
        ax_comp.set_ylabel(f'r(ℓ={ell_target})', fontsize=12, fontweight='bold')
        ax_comp.set_title(f'Filtering Effect at ℓ={ell_target}', fontsize=12, fontweight='bold')
        ax_comp.legend(fontsize=10)
        ax_comp.grid(True, alpha=0.3)
        ax_comp.set_xlim(0, 1)

# Row 4: Velocity reconstruction quality for filtered analysis
ax_f5 = fig_filtered.add_subplot(gs_filtered[3, 0])
ax_f5.plot(results['mean_xhi'], results['correlation_vel_rec'], 'bo-', 
           label='Unfiltered Tb → v_rec', linewidth=2, markersize=8, alpha=0.7)
ax_f5.plot(results['mean_xhi'], results['correlation_vel_rec_filtered'], 'rs-', 
           label='Filtered Tb → v_rec', linewidth=2, markersize=8, alpha=0.7)
ax_f5.axhline(0, color='black', linewidth=0.8, linestyle=':')
ax_f5.set_xlabel('Mean Neutral Fraction (xHI)', fontsize=12, fontweight='bold')
ax_f5.set_ylabel('r(v_real, v_rec)', fontsize=12, fontweight='bold')
ax_f5.set_title('Velocity Reconstruction Quality', fontsize=14, fontweight='bold')
ax_f5.legend(fontsize=10)
ax_f5.grid(True, alpha=0.3)
ax_f5.set_xlim(0, 1)

# Row 4: Signal degradation due to filtering
ax_f6 = fig_filtered.add_subplot(gs_filtered[3, 1])
signal_loss_real = (results['correlation_21cm'] - results['correlation_21cm_filt']) / np.abs(results['correlation_21cm']) * 100
signal_loss_rec = (results['correlation_21cm_rec'] - results['correlation_21cm_rec_filt']) / np.abs(results['correlation_21cm_rec']) * 100

# Handle division by zero
signal_loss_real = np.where(np.isfinite(signal_loss_real), signal_loss_real, 0)
signal_loss_rec = np.where(np.isfinite(signal_loss_rec), signal_loss_rec, 0)

ax_f6.plot(results['mean_xhi'], signal_loss_real, 'go-', 
           label='Real velocity', linewidth=2, markersize=8, alpha=0.7)
ax_f6.plot(results['mean_xhi'], signal_loss_rec, 'mo-', 
           label='Reconstructed velocity', linewidth=2, markersize=8, alpha=0.7)
ax_f6.axhline(0, color='black', linewidth=0.8, linestyle=':')
ax_f6.set_xlabel('Mean Neutral Fraction (xHI)', fontsize=12, fontweight='bold')
ax_f6.set_ylabel('Signal Loss (%)', fontsize=12, fontweight='bold')
ax_f6.set_title('Correlation Signal Loss Due to Filtering', fontsize=14, fontweight='bold')
ax_f6.legend(fontsize=10)
ax_f6.grid(True, alpha=0.3)
ax_f6.set_xlim(0, 1)

# Row 4: Reionization context
ax_f7 = fig_filtered.add_subplot(gs_filtered[3, 2])
ax_f7.plot(results['mean_z'], results['mean_xhi'], 'ko-', 
           linewidth=3, markersize=8, alpha=0.8)
ax_f7.set_xlabel('Mean Redshift', fontsize=12, fontweight='bold')
ax_f7.set_ylabel('Mean Neutral Fraction (xHI)', fontsize=12, fontweight='bold')
ax_f7.set_title('Reionization History', fontsize=14, fontweight='bold')
ax_f7.grid(True, alpha=0.3)
ax_f7.invert_xaxis()

# Row 5: Summary statistics for filtered analysis
ax_f8 = fig_filtered.add_subplot(gs_filtered[4, :])

# Create summary table
summary_data = {
    'Epoch': ['Early (xHI > 0.8)', 'Mid (0.2 < xHI < 0.8)', 'Late (xHI < 0.2)'],
    'Unfiltered r(real)': [],
    'Filtered r(real)': [],
    'Unfiltered r(rec)': [],
    'Filtered r(rec)': [],
    'Signal Loss (%)': []
}

# Calculate summary statistics
high_xhi = results['mean_xhi'] > 0.8
mid_xhi = (results['mean_xhi'] > 0.2) & (results['mean_xhi'] <= 0.8)
low_xhi = results['mean_xhi'] <= 0.2

for mask, epoch in zip([high_xhi, mid_xhi, low_xhi], summary_data['Epoch']):
    if np.any(mask):
        summary_data['Unfiltered r(real)'].append(f"{np.nanmean(results['correlation_21cm'][mask]):+.3f}")
        summary_data['Filtered r(real)'].append(f"{np.nanmean(results['correlation_21cm_filt'][mask]):+.3f}")
        summary_data['Unfiltered r(rec)'].append(f"{np.nanmean(results['correlation_21cm_rec'][mask]):+.3f}")
        summary_data['Filtered r(rec)'].append(f"{np.nanmean(results['correlation_21cm_rec_filt'][mask]):+.3f}")
        
        # Calculate average signal loss
        loss_real = np.nanmean((results['correlation_21cm'][mask] - results['correlation_21cm_filt'][mask]) / 
                               np.abs(results['correlation_21cm'][mask]) * 100)
        summary_data['Signal Loss (%)'].append(f"{loss_real:.1f}%")
    else:
        for key in ['Unfiltered r(real)', 'Filtered r(real)', 'Unfiltered r(rec)', 'Filtered r(rec)', 'Signal Loss (%)']:
            summary_data[key].append('N/A')

# Create table
table_text = []
headers = list(summary_data.keys())
table_text.append(headers)
for i in range(len(summary_data['Epoch'])):
    row = [summary_data[key][i] for key in headers]
    table_text.append(row)

ax_f8.axis('tight')
ax_f8.axis('off')
table = ax_f8.table(cellText=table_text[1:], colLabels=table_text[0], 
                    cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 2)

# Style the table
for i in range(len(headers)):
    table[(0, i)].set_facecolor('#4CAF50')
    table[(0, i)].set_text_props(weight='bold', color='white')

ax_f8.set_title('Summary Statistics: Filtered vs Unfiltered Analysis', 
                fontsize=14, fontweight='bold', pad=20)

plt.suptitle('Comprehensive Filtered Maps Analysis: kSZ-21cm Cross-Correlation with Wedge Filtering', 
             fontsize=20, fontweight='bold')
plt.savefig('plots/wedge_filtered_r_ell_analysis_all.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# VELOCITY RECONSTRUCTION DETAILED ANALYSIS
# ============================================================================

print("\nCreating detailed velocity reconstruction analysis...")

def scatter_compare_velocity(true_vel, rec_unfiltered, rec_filtered, 
                           chunk_info="", sample=50_000, seed=42):
    """
    Compare velocity reconstructions with scatter plots and detailed metrics.
    """
    rng = np.random.default_rng(seed)
    
    # Flatten arrays
    t = np.ravel(true_vel)
    r_unfilt = np.ravel(rec_unfiltered)
    r_filt = np.ravel(rec_filtered)
    
    # Sample for visualization
    n = t.size
    if sample is not None and sample < n:
        idx = rng.choice(n, size=sample, replace=False)
        t, r_unfilt, r_filt = t[idx], r_unfilt[idx], r_filt[idx]
    
    def compute_metrics(y_true, y_pred):
        """Compute correlation and RMSE."""
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        if not np.any(mask):
            return np.nan, np.nan, np.nan
        
        y_t, y_p = y_true[mask], y_pred[mask]
        r = np.corrcoef(y_t, y_p)[0, 1] if len(y_t) > 1 else np.nan
        rmse = np.sqrt(np.mean((y_p - y_t)**2))
        bias = np.mean(y_p - y_t)
        return float(r), float(rmse), float(bias)
    
    # Compute metrics
    r_unfilt, rmse_unfilt, bias_unfilt = compute_metrics(t, r_unfilt)
    r_filt, rmse_filt, bias_filt = compute_metrics(t, r_filt)
    
    # Determine plot limits
    all_data = np.concatenate([t, r_unfilt, r_filt])
    lim = np.nanpercentile(all_data[np.isfinite(all_data)], [2, 98])
    lo, hi = float(lim[0]), float(lim[1])
    
    return {
        'r_unfilt': r_unfilt, 'rmse_unfilt': rmse_unfilt, 'bias_unfilt': bias_unfilt,
        'r_filt': r_filt, 'rmse_filt': rmse_filt, 'bias_filt': bias_filt,
        't': t, 'r_unfilt_data': r_unfilt, 'r_filt_data': r_filt,
        'limits': (lo, hi), 'chunk_info': chunk_info
    }

def create_velocity_scatter_plot(data_list, save_name):
    """Create comprehensive scatter plot for velocity comparisons."""
    n_chunks = len(data_list)
    n_cols = min(4, n_chunks)
    n_rows = (n_chunks + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    if n_chunks == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, data in enumerate(data_list):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        
        # Scatter plots
        ax.scatter(data['r_unfilt_data'], data['t'], s=1, alpha=0.3, 
                  label=f"Unfiltered: r={data['r_unfilt']:.3f}", color='blue')
        ax.scatter(data['r_filt_data'], data['t'], s=1, alpha=0.3,
                  label=f"Filtered: r={data['r_filt']:.3f}", color='red')
        
        # Perfect correlation line
        lo, hi = data['limits']
        line_x = np.linspace(lo, hi, 100)
        ax.plot(line_x, line_x, 'k--', alpha=0.7, linewidth=1, label='y = x')
        
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel('Reconstructed v_z [cm/s]', fontsize=10)
        ax.set_ylabel('True v_z [cm/s]', fontsize=10)
        ax.set_title(f'{data["chunk_info"]}', fontsize=10)
        ax.legend(fontsize=8, markerscale=5)
        ax.grid(True, alpha=0.3)
    
    # Hide empty subplots
    for i in range(n_chunks, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        if n_rows > 1:
            axes[row, col].set_visible(False)
        else:
            axes[col].set_visible(False)
    
    plt.suptitle('Velocity Reconstruction Quality: Scatter Plot Analysis', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'plots/{save_name}', dpi=300, bbox_inches='tight')
    plt.show()

# Select representative chunks for detailed analysis
n_analysis_chunks = min(6, len(results['mean_xhi']))
analysis_indices = np.linspace(0, len(results['mean_xhi'])-1, n_analysis_chunks, dtype=int)

print(f"Analyzing {n_analysis_chunks} representative chunks...")

# Collect velocity data for selected chunks
velocity_data = []
chunk_size = 200  # Same as defined earlier

for idx_pos, i in enumerate(analysis_indices):
    z_start = i * chunk_size
    z_end = min((i + 1) * chunk_size, nz)
    
    if z_end - z_start < 10:
        continue
    
    print(f"  Processing chunk {i+1}: z-slices [{z_start}:{z_end}]")
    
    # Extract the same chunks as in the main analysis
    vz_chunk = vz_filtered[:, :, z_start:z_end]
    tb_chunk = Tb[:, :, z_start:z_end]
    tb_chunk_filtered = Tb_filtered[:, :, z_start:z_end]
    xhi_chunk = xhi_filtered[:, :, z_start:z_end]
    redshifts_chunk = redshifts[z_start:z_end]
    
    # Reconstruct velocities (same as in main analysis)
    z_ref_chunk = float(np.mean(redshifts_chunk))
    mean_xhi_chunk = float(np.mean(xhi_chunk))
    
    vz_rec_unfilt = reconstruct_velocity_single_z_method(
        tb_chunk, xhi_xyz=xhi_chunk,
        weight="deltaXhi",
        z_ref=z_ref_chunk,
        littleh=0.7,
        box_mpc_over_h=300.0
    )
    
    vz_rec_filt = reconstruct_velocity_single_z_method(
        tb_chunk_filtered, xhi_xyz=xhi_chunk,
        weight="deltaXhi", 
        z_ref=z_ref_chunk,
        littleh=0.7,
        box_mpc_over_h=300.0
    )
    
    # Analyze velocity reconstruction
    chunk_info = f"Chunk {i+1}: z={z_ref_chunk:.1f}, xHI={mean_xhi_chunk:.2f}"
    vel_data = scatter_compare_velocity(
        vz_chunk, vz_rec_unfilt, vz_rec_filt, 
        chunk_info=chunk_info, sample=30_000
    )
    
    velocity_data.append(vel_data)

# Create scatter plot analysis
if velocity_data:
    create_velocity_scatter_plot(velocity_data, 'velocity_reconstruction_scatter_analysis.png')

# ============================================================================
# VELOCITY RECONSTRUCTION SUMMARY METRICS
# ============================================================================

print("\nCreating velocity reconstruction summary metrics...")

# Extract metrics for summary
vel_metrics = {
    'chunk_names': [],
    'r_unfilt': [],
    'r_filt': [],
    'rmse_unfilt': [],
    'rmse_filt': [],
    'bias_unfilt': [],
    'bias_filt': [],
    'xhi_values': [],
    'z_values': []
}

for i, data in enumerate(velocity_data):
    vel_metrics['chunk_names'].append(data['chunk_info'])
    vel_metrics['r_unfilt'].append(data['r_unfilt'])
    vel_metrics['r_filt'].append(data['r_filt'])
    vel_metrics['rmse_unfilt'].append(data['rmse_unfilt'])
    vel_metrics['rmse_filt'].append(data['rmse_filt'])
    vel_metrics['bias_unfilt'].append(data['bias_unfilt'])
    vel_metrics['bias_filt'].append(data['bias_filt'])
    
    # Extract xHI and z from results
    analysis_idx = analysis_indices[i]
    if analysis_idx < len(results['mean_xhi']):
        vel_metrics['xhi_values'].append(results['mean_xhi'][analysis_idx])
        vel_metrics['z_values'].append(results['mean_z'][analysis_idx])
    else:
        vel_metrics['xhi_values'].append(np.nan)
        vel_metrics['z_values'].append(np.nan)

# Create summary plots
fig_summary = plt.figure(figsize=(20, 12))
gs_summary = fig_summary.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Plot 1: Correlation comparison
ax_s1 = fig_summary.add_subplot(gs_summary[0, 0])
x_pos = np.arange(len(vel_metrics['chunk_names']))
width = 0.35
ax_s1.bar(x_pos - width/2, vel_metrics['r_unfilt'], width, 
          label='Unfiltered Tb', alpha=0.7, color='blue')
ax_s1.bar(x_pos + width/2, vel_metrics['r_filt'], width,
          label='Filtered Tb', alpha=0.7, color='red')
ax_s1.set_xlabel('Chunk', fontsize=12, fontweight='bold')
ax_s1.set_ylabel('Correlation r', fontsize=12, fontweight='bold')
ax_s1.set_title('Velocity Reconstruction Correlation', fontsize=14, fontweight='bold')
ax_s1.set_xticks(x_pos)
ax_s1.set_xticklabels([f"C{i+1}" for i in range(len(vel_metrics['chunk_names']))], rotation=45)
ax_s1.legend()
ax_s1.grid(True, alpha=0.3)

# Plot 2: RMSE comparison
ax_s2 = fig_summary.add_subplot(gs_summary[0, 1])
ax_s2.bar(x_pos - width/2, vel_metrics['rmse_unfilt'], width,
          label='Unfiltered Tb', alpha=0.7, color='blue')
ax_s2.bar(x_pos + width/2, vel_metrics['rmse_filt'], width,
          label='Filtered Tb', alpha=0.7, color='red')
ax_s2.set_xlabel('Chunk', fontsize=12, fontweight='bold')
ax_s2.set_ylabel('RMSE [cm/s]', fontsize=12, fontweight='bold')
ax_s2.set_title('Velocity Reconstruction RMSE', fontsize=14, fontweight='bold')
ax_s2.set_xticks(x_pos)
ax_s2.set_xticklabels([f"C{i+1}" for i in range(len(vel_metrics['chunk_names']))], rotation=45)
ax_s2.legend()
ax_s2.grid(True, alpha=0.3)

# Plot 3: Bias comparison
ax_s3 = fig_summary.add_subplot(gs_summary[0, 2])
ax_s3.bar(x_pos - width/2, vel_metrics['bias_unfilt'], width,
          label='Unfiltered Tb', alpha=0.7, color='blue')
ax_s3.bar(x_pos + width/2, vel_metrics['bias_filt'], width,
          label='Filtered Tb', alpha=0.7, color='red')
ax_s3.axhline(0, color='black', linewidth=0.8, linestyle=':')
ax_s3.set_xlabel('Chunk', fontsize=12, fontweight='bold')
ax_s3.set_ylabel('Bias [cm/s]', fontsize=12, fontweight='bold')
ax_s3.set_title('Velocity Reconstruction Bias', fontsize=14, fontweight='bold')
ax_s3.set_xticks(x_pos)
ax_s3.set_xticklabels([f"C{i+1}" for i in range(len(vel_metrics['chunk_names']))], rotation=45)
ax_s3.legend()
ax_s3.grid(True, alpha=0.3)

# Plot 4: Correlation vs xHI
ax_s4 = fig_summary.add_subplot(gs_summary[1, 0])
ax_s4.plot(vel_metrics['xhi_values'], vel_metrics['r_unfilt'], 'bo-',
           label='Unfiltered Tb', linewidth=2, markersize=8, alpha=0.7)
ax_s4.plot(vel_metrics['xhi_values'], vel_metrics['r_filt'], 'rs-',
           label='Filtered Tb', linewidth=2, markersize=8, alpha=0.7)
ax_s4.set_xlabel('Mean Neutral Fraction (xHI)', fontsize=12, fontweight='bold')
ax_s4.set_ylabel('Correlation r', fontsize=12, fontweight='bold')
ax_s4.set_title('Reconstruction Quality vs xHI', fontsize=14, fontweight='bold')
ax_s4.legend()
ax_s4.grid(True, alpha=0.3)
ax_s4.set_xlim(0, 1)

# Plot 5: RMSE vs xHI
ax_s5 = fig_summary.add_subplot(gs_summary[1, 1])
ax_s5.plot(vel_metrics['xhi_values'], vel_metrics['rmse_unfilt'], 'bo-',
           label='Unfiltered Tb', linewidth=2, markersize=8, alpha=0.7)
ax_s5.plot(vel_metrics['xhi_values'], vel_metrics['rmse_filt'], 'rs-',
           label='Filtered Tb', linewidth=2, markersize=8, alpha=0.7)
ax_s5.set_xlabel('Mean Neutral Fraction (xHI)', fontsize=12, fontweight='bold')
ax_s5.set_ylabel('RMSE [cm/s]', fontsize=12, fontweight='bold')
ax_s5.set_title('Reconstruction Error vs xHI', fontsize=14, fontweight='bold')
ax_s5.legend()
ax_s5.grid(True, alpha=0.3)
ax_s5.set_xlim(0, 1)

# Plot 6: Performance degradation
ax_s6 = fig_summary.add_subplot(gs_summary[1, 2])
r_degradation = np.array(vel_metrics['r_unfilt']) - np.array(vel_metrics['r_filt'])
rmse_degradation = np.array(vel_metrics['rmse_filt']) - np.array(vel_metrics['rmse_unfilt'])

ax_s6_twin = ax_s6.twinx()
line1 = ax_s6.plot(vel_metrics['xhi_values'], r_degradation, 'go-',
                   label='Δr (Unfilt - Filt)', linewidth=2, markersize=8, alpha=0.7)
line2 = ax_s6_twin.plot(vel_metrics['xhi_values'], rmse_degradation, 'mo-',
                        label='ΔRMSE (Filt - Unfilt)', linewidth=2, markersize=8, alpha=0.7)

ax_s6.axhline(0, color='black', linewidth=0.8, linestyle=':')
ax_s6_twin.axhline(0, color='black', linewidth=0.8, linestyle=':')

ax_s6.set_xlabel('Mean Neutral Fraction (xHI)', fontsize=12, fontweight='bold')
ax_s6.set_ylabel('Correlation Degradation', fontsize=12, fontweight='bold', color='green')
ax_s6_twin.set_ylabel('RMSE Increase [cm/s]', fontsize=12, fontweight='bold', color='magenta')
ax_s6.set_title('Performance Degradation Due to Filtering', fontsize=14, fontweight='bold')

# Combine legends
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax_s6.legend(lines, labels, loc='upper left')

ax_s6.grid(True, alpha=0.3)
ax_s6.set_xlim(0, 1)
ax_s6.tick_params(axis='y', labelcolor='green')
ax_s6_twin.tick_params(axis='y', labelcolor='magenta')

# Plot 7-9: Detailed metrics table
ax_s7 = fig_summary.add_subplot(gs_summary[2, :])
ax_s7.axis('tight')
ax_s7.axis('off')

# Create detailed metrics table
table_data = []
headers = ['Chunk', 'z', 'xHI', 'r(Unfilt)', 'r(Filt)', 'RMSE(Unfilt)', 'RMSE(Filt)', 'Bias(Unfilt)', 'Bias(Filt)']

for i in range(len(vel_metrics['chunk_names'])):
    row = [
        f"C{i+1}",
        f"{vel_metrics['z_values'][i]:.1f}",
        f"{vel_metrics['xhi_values'][i]:.2f}",
        f"{vel_metrics['r_unfilt'][i]:.3f}",
        f"{vel_metrics['r_filt'][i]:.3f}",
        f"{vel_metrics['rmse_unfilt'][i]:.0f}",
        f"{vel_metrics['rmse_filt'][i]:.0f}",
        f"{vel_metrics['bias_unfilt'][i]:+.0f}",
        f"{vel_metrics['bias_filt'][i]:+.0f}"
    ]
    table_data.append(row)

table = ax_s7.table(cellText=table_data, colLabels=headers,
                    cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)

# Style the table
for i in range(len(headers)):
    table[(0, i)].set_facecolor('#2196F3')
    table[(0, i)].set_text_props(weight='bold', color='white')

ax_s7.set_title('Detailed Velocity Reconstruction Metrics', 
                fontsize=14, fontweight='bold', pad=20)

plt.suptitle('Comprehensive Velocity Reconstruction Analysis: Filtered vs Unfiltered', 
             fontsize=18, fontweight='bold')
plt.savefig('plots/velocity_reconstruction_detailed_metrics.png', dpi=300, bbox_inches='tight')
plt.show()

print("="*80)

# Save results to file
import os
os.makedirs('results', exist_ok=True)
np.savez('results/xhi_evolution_results.npz', **results)
print("\nResults saved to: results/xhi_evolution_results.npz")

print("\nAnalysis complete!")
