"""
ksz_reconstruction_lightcone_analysis.py

Analyze kSZ reconstruction quality using 21cmFAST lightcone data.
Compares real kSZ with reconstructed kSZ across different redshift chunks.
"""

import numpy as np
from scipy import fft
from scipy.ndimage import gaussian_filter
from scipy.integrate import quad
import matplotlib.pyplot as plt
from powerbox import get_power
import gc

# ============================================================================
# PARAMETERS
# ============================================================================

DENSITY_FILE = "data_21cmfast/density/12701_density_LC.npy"
VELOCITY_FILE = "data_21cmfast/velocity/12701_velocity_z_LC.npy"
XHI_FILE = "data_21cmfast/xHI/12701_xHI_LC.npy"

BOX_MPC_OVER_H = 500.0
LITTLEH = 0.7
N_CHUNKS = 10  # Number of redshift chunks to analyze
CHUNK_SIZE = 200  # Size of each chunk in z-slices
SMOOTH_SIGMA = 12.0  # Mpc/h smoothing scale

# Redshift parameters for lightcone
Z_MIN = 5.0  # Starting redshift (adjust based on your simulation)
Z_MAX = 32.0  # Ending redshift (adjust based on your simulation)

# ============================================================================
# HELPER FUNCTIONS
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

def k_to_ell(k, z, littleh=0.7):
    """Convert k [h/Mpc] to multipole ell."""
    omega_m0 = 0.27
    omega_l0 = 0.73
    
    def E(zp):
        return np.sqrt(omega_m0 * (1+zp)**3 + omega_l0)
    
    chi, _ = quad(lambda zp: 1.0/E(zp), 0, z)
    chi *= 3000.0 / littleh  # c/H0 in Mpc/h
    
    ell = k * chi
    return ell

def safe_real(a):
    return np.real(a).astype(np.float32, copy=False)

def kspace_rfft(n, rc, dtype=np.float32):
    kx = (2.0 * np.pi * np.fft.fftfreq(n, d=rc)).astype(dtype)
    ky = (2.0 * np.pi * np.fft.fftfreq(n, d=rc)).astype(dtype)
    kz = (2.0 * np.pi * np.fft.rfftfreq(n, d=rc)).astype(dtype)
    tiny = np.finfo(dtype).tiny
    if kz.size: kz[0] = max(kz[0], tiny)
    if kx.size: kx[0] = max(kx[0], tiny)
    if ky.size: ky[0] = max(ky[0], tiny)
    return kx, ky, kz

def compute_fourier_correlation_coefficient(field_X, field_Y, boxlength, bins=100):
    """
    Compute correlation coefficient in Fourier space:
    r(k) = P_XY(k) / sqrt(P_XX(k) * P_YY(k))
    """
    X = field_X.astype(np.float32) - np.mean(field_X, dtype=np.float64)
    Y = field_Y.astype(np.float32) - np.mean(field_Y, dtype=np.float64)
    
    P_XX, k_X = get_power(deltax=X, boxlength=boxlength, bins=bins, ignore_zero_mode=True)
    P_YY, k_Y = get_power(deltax=Y, boxlength=boxlength, bins=bins, ignore_zero_mode=True)
    P_XY, k_XY = get_power(deltax=X, boxlength=boxlength, deltax2=Y, bins=bins, ignore_zero_mode=True)
    
    r_k = P_XY / np.sqrt(P_XX * P_YY)
    
    mask = np.isfinite(k_XY) & np.isfinite(r_k) & (k_XY > 0.01) & (k_XY < 0.5)
    
    return k_XY[mask], r_k[mask]

# ============================================================================
# VELOCITY RECONSTRUCTION
# ============================================================================

def reconstruct_velocities_from_density(den, xhi, z, n=600):
    """Reconstruct velocities from density field."""
    mean_den = den.mean(dtype=np.float64).astype(np.float32)
    delta = den
    
    dltk = fft.rfftn(delta, workers=-1).astype(np.complex64, copy=False)
    dltXhk = fft.rfftn((1.0 + delta) * xhi, workers=-1).astype(np.complex64, copy=False)
    
    rc = BOX_MPC_OVER_H / float(n) / LITTLEH
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
    
    vz_rec = reconstruct_one("z", dltk)
    vz_recx = reconstruct_one("z", dltXhk)
    
    return vz_rec, vz_recx

def compute_ksz_map(vz, xhi, den, xhi_eps=1e-8):
    """Compute kSZ map by integrating along line of sight."""
    weighted_vz = (1.0 - np.clip(xhi, 0, 1 - xhi_eps))*den * vz
    ksz_map = np.sum(weighted_vz, axis=2)
    return ksz_map

# ============================================================================
# LOAD AND PROCESS LIGHTCONE DATA
# ============================================================================

# Load full lightcone arrays once at module level (memory-mapped for efficiency)
_density_full = None
_xhi_full = None
_velocity_full = None
_redshifts_full = None

def load_full_lightcones():
    """Load the full lightcone arrays (memory-mapped)."""
    global _density_full, _xhi_full, _velocity_full, _redshifts_full
    
    if _density_full is None:
        print("Loading lightcone data files...")
        _density_full = np.load(DENSITY_FILE, mmap_mode='r')
        _xhi_full = np.load(XHI_FILE, mmap_mode='r')
        _velocity_full = np.load(VELOCITY_FILE, mmap_mode='r')
        
        # Generate redshift array assuming linear spacing
        n_slices = _density_full.shape[2]
        _redshifts_full = np.linspace(Z_MIN, Z_MAX, n_slices)
        
        print(f"  Density shape: {_density_full.shape}")
        print(f"  xHI shape: {_xhi_full.shape}")
        print(f"  Velocity shape: {_velocity_full.shape}")
        print(f"  Redshift range: {Z_MIN:.3f} - {Z_MAX:.3f}")

def load_lightcone_chunk(start_idx, end_idx):
    """Load a chunk of the lightcone data."""
    load_full_lightcones()
    
    # Load chunks (copy to avoid memory-map issues with FFT)
    den = np.array(_density_full[:, :, start_idx:end_idx], dtype=np.float32)
    xhi = np.array(_xhi_full[:, :, start_idx:end_idx], dtype=np.float32)
    vz = np.array(_velocity_full[:, :, start_idx:end_idx], dtype=np.float32)
    redshifts = _redshifts_full[start_idx:end_idx]
    
    return den, xhi, vz, redshifts

# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def main():
    print("="*80)
    print("kSZ RECONSTRUCTION QUALITY ANALYSIS - LIGHTCONE CHUNKS")
    print("="*80)
    
    # Storage for results
    chunk_results = []
    
    # Load data and get dimensions
    load_full_lightcones()
    n_slices = _density_full.shape[2]
    n_xy = _density_full.shape[0]
    
    print(f"\nLightcone dimensions: {n_xy} x {n_xy} x {n_slices}")
    print(f"Processing {N_CHUNKS} chunks of size {CHUNK_SIZE}")
    
    # Compute box dimensions for 2D maps
    dx = BOX_MPC_OVER_H / n_xy
    dy = BOX_MPC_OVER_H / n_xy
    Ly = dy * n_xy
    Lx = dx * n_xy
    
    # Process each chunk
    for i_chunk in range(N_CHUNKS):
        start_idx = i_chunk * CHUNK_SIZE
        end_idx = start_idx + CHUNK_SIZE
        
        if end_idx > n_slices:
            break
        
        print(f"\n{'='*80}")
        print(f"Processing chunk {i_chunk+1}/{N_CHUNKS} (slices {start_idx}-{end_idx})")
        print(f"{'='*80}")
        
        # Load chunk
        den, xhi, vz_real, redshifts = load_lightcone_chunk(start_idx, end_idx)
        
        mean_z = redshifts.mean()
        mean_xhi = xhi.mean()
        
        print(f"  Mean redshift: {mean_z:.3f}")
        print(f"  Mean xHI: {mean_xhi:.4f}")
        print(f"  Redshift range: {redshifts.min():.3f} - {redshifts.max():.3f}")
        
        # Reconstruct velocities
        print("  Reconstructing velocities...")
        vz_rec, vz_recx = reconstruct_velocities_from_density(den, xhi, mean_z, n=n_xy)
        
        # Compute kSZ maps
        print("  Computing kSZ maps...")
        ksz_map_real = compute_ksz_map(vz_real, xhi, den)
        ksz_map_rec = compute_ksz_map(-vz_recx, xhi, den)
        
        # Real-space correlation
        r_real = pearson_r(ksz_map_real, ksz_map_rec)
        print(f"  Real-space r(kSZ, kSZ_rec): {r_real:.4f}")
        
        # Fourier correlation
        print("  Computing Fourier correlation...")
        k_values, r_k = compute_fourier_correlation_coefficient(
            ksz_map_real, ksz_map_rec,
            boxlength=[Ly, Lx], bins=300
        )
        
        # Convert k to ell
        ell_values = k_to_ell(k_values, mean_z)
        
        # Store results
        chunk_results.append({
            'chunk_idx': i_chunk,
            'mean_z': mean_z,
            'z_min': redshifts.min(),
            'z_max': redshifts.max(),
            'mean_xhi': mean_xhi,
            'r_real': r_real,
            'k_values': k_values,
            'r_k': r_k,
            'ell_values': ell_values,
            'ksz_map_real': ksz_map_real.copy(),
            'ksz_map_rec': ksz_map_rec.copy()
        })
        
        # Clean up
        del den, xhi, vz_real, vz_rec, vz_recx, ksz_map_real, ksz_map_rec
        gc.collect()
    
    # ========================================================================
    # FULL LIGHTCONE INTEGRATION
    # ========================================================================
    
    print(f"\n{'='*80}")
    print("FULL LIGHTCONE INTEGRATION")
    print(f"{'='*80}")
    
    # Sum all chunk kSZ maps to get integrated effect
    ksz_full_real = np.sum([r['ksz_map_real'] for r in chunk_results], axis=0)
    ksz_full_rec = np.sum([r['ksz_map_rec'] for r in chunk_results], axis=0)
    
    r_full = pearson_r(ksz_full_real, ksz_full_rec)
    print(f"Full lightcone real-space r(kSZ, kSZ_rec): {r_full:.4f}")
    
    # Fourier correlation for full lightcone
    k_full, r_k_full = compute_fourier_correlation_coefficient(
        ksz_full_real, ksz_full_rec,
        boxlength=[Ly, Lx], bins=300
    )
    
    # Use average redshift for ell conversion
    mean_z_all = np.mean([r['mean_z'] for r in chunk_results])
    ell_full = k_to_ell(k_full, mean_z_all)
    
    # ========================================================================
    # PLOTTING
    # ========================================================================
    
    print(f"\n{'='*80}")
    print("GENERATING PLOTS")
    print(f"{'='*80}")
    
    # Plot 1: r(k) for each chunk
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # Top left: r(k) vs k for all chunks
    ax1 = axes[0, 0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(chunk_results)))
    
    for idx, result in enumerate(chunk_results):
        ax1.plot(result['k_values'], result['r_k'],
                color=colors[idx], linewidth=2,
                label=f'z={result["mean_z"]:.2f}, xHI={result["mean_xhi"]:.2f}')
    
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax1.axhline(y=1, color='gray', linestyle='--', linewidth=1, alpha=0.3)
    ax1.set_xlabel('k [h/Mpc]', fontsize=13)
    ax1.set_ylabel('Correlation Coefficient r(k)', fontsize=13)
    ax1.set_title('kSZ Correlation by Redshift Chunk', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([-0.2, 1.15])
    ax1.legend(fontsize=8, ncol=2, loc='lower right')
    
    # Top right: r(ℓ) vs ℓ for all chunks
    ax2 = axes[0, 1]
    
    for idx, result in enumerate(chunk_results):
        ax2.plot(result['ell_values'], result['r_k'],
                color=colors[idx], linewidth=2,
                label=f'z={result["mean_z"]:.2f}')
    
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax2.axhline(y=1, color='gray', linestyle='--', linewidth=1, alpha=0.3)
    ax2.set_xlabel('Multipole ℓ', fontsize=13)
    ax2.set_ylabel('Correlation Coefficient r(ℓ)', fontsize=13)
    ax2.set_title('kSZ Correlation (Multipole Space)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([-0.2, 1.15])
    ax2.legend(fontsize=8, ncol=2, loc='lower right')
    
    # Bottom left: r(ℓ) focused on ℓ = 2000-4000
    ax3 = axes[1, 0]
    
    for idx, result in enumerate(chunk_results):
        mask = (result['ell_values'] >= 2000) & (result['ell_values'] <= 4000)
        if np.any(mask):
            ax3.plot(result['ell_values'][mask], result['r_k'][mask],
                    color=colors[idx], linewidth=2,
                    label=f'z={result["mean_z"]:.2f}, xHI={result["mean_xhi"]:.2f}')
    
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax3.axhline(y=1, color='gray', linestyle='--', linewidth=1, alpha=0.3)
    ax3.set_xlabel('Multipole ℓ', fontsize=13)
    ax3.set_ylabel('Correlation Coefficient r(ℓ)', fontsize=13)
    ax3.set_title('kSZ Correlation (ℓ = 2000-4000)', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([-0.2, 1.15])
    ax3.set_xlim([2000, 4000])
    ax3.legend(fontsize=8, ncol=2, loc='lower right')
    
    # Bottom right: Full lightcone integration
    ax4 = axes[1, 1]
    
    ax4.plot(k_full, r_k_full, 'r-', linewidth=3,
            label=f'Full lightcone integration\nr={r_full:.3f}')
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax4.axhline(y=1, color='gray', linestyle='--', linewidth=1, alpha=0.3)
    ax4.set_xlabel('k [h/Mpc]', fontsize=13)
    ax4.set_ylabel('Correlation Coefficient r(k)', fontsize=13)
    ax4.set_title('Full Lightcone Integration', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([-0.2, 1.15])
    ax4.legend(fontsize=11, loc='lower right')
    
    plt.suptitle('kSZ Reconstruction Quality - Lightcone Analysis',
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plots/ksz_lightcone_reconstruction_analysis.png',
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 2: Evolution with redshift/neutral fraction
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    mean_zs = [r['mean_z'] for r in chunk_results]
    mean_xhis = [r['mean_xhi'] for r in chunk_results]
    r_reals = [r['r_real'] for r in chunk_results]
    mean_r_ks = [np.mean(r['r_k']) for r in chunk_results]
    
    # Real-space correlation vs z
    ax1 = axes[0]
    ax1.plot(mean_zs, r_reals, 'o-', markersize=8, linewidth=2, color='blue')
    ax1.set_xlabel('Mean Redshift', fontsize=13)
    ax1.set_ylabel('Real-space r(kSZ, kSZ_rec)', fontsize=13)
    ax1.set_title('Correlation vs Redshift', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])
    
    # Real-space correlation vs xHI
    ax2 = axes[1]
    ax2.plot(mean_xhis, r_reals, 'o-', markersize=8, linewidth=2, color='green')
    ax2.set_xlabel('Mean Neutral Fraction <xHI>', fontsize=13)
    ax2.set_ylabel('Real-space r(kSZ, kSZ_rec)', fontsize=13)
    ax2.set_title('Correlation vs Neutral Fraction', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    # Mean r(k) vs xHI
    ax3 = axes[2]
    ax3.plot(mean_xhis, mean_r_ks, 'o-', markersize=8, linewidth=2, color='red')
    ax3.set_xlabel('Mean Neutral Fraction <xHI>', fontsize=13)
    ax3.set_ylabel('Mean r(k)', fontsize=13)
    ax3.set_title('Fourier Correlation vs Neutral Fraction', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1])
    
    plt.suptitle('kSZ Reconstruction Quality Evolution',
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plots/ksz_reconstruction_evolution.png',
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # ========================================================================
    # SUMMARY STATISTICS
    # ========================================================================
    
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")
    
    print("\nPer-chunk results:")
    print(f"{'Chunk':<8} {'z_mean':<8} {'<xHI>':<8} {'r_real':<8} {'mean r(k)':<10}")
    print("-" * 50)
    for result in chunk_results:
        print(f"{result['chunk_idx']:<8} "
              f"{result['mean_z']:<8.3f} "
              f"{result['mean_xhi']:<8.4f} "
              f"{result['r_real']:<8.4f} "
              f"{np.mean(result['r_k']):<10.4f}")
    
    print(f"\nFull lightcone integration:")
    print(f"  Real-space r: {r_full:.4f}")
    print(f"  Mean r(k): {np.mean(r_k_full):.4f}")
    print(f"  Max r(k): {np.max(r_k_full):.4f}")
    
    print(f"\n{'='*80}")
    print("Analysis complete!")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
