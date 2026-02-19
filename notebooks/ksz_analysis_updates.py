"""
Updated cells for ksz_analysis.ipynb

Copy these into the corresponding cells in your notebook.
Key changes:
1. Uses optical depth formulation (dτ/dz weighting)
2. Normalizes power spectra to PEAK instead of high-z value
"""

# =============================================================================
# CELL 0 (header) - Updated markdown
# =============================================================================
"""
# kSZ Signal Evolution Over Redshift

Using optical depth formulation from Jelić et al.:
$$\Delta T_{kSZ}/T_{CMB} = -\int \frac{v_r}{c} x_e (1+\delta) d\tau$$

where $d\tau/dz = c \sigma_T n_e (1+z)^2 / H(z)$
"""

# =============================================================================
# CELL 1 (imports) - Add optical depth imports
# =============================================================================
CELL_1_IMPORTS = '''
import numpy as np
import matplotlib.pyplot as plt
from powerbox.tools import get_power
from scipy.integrate import quad
import gc
import sys
sys.path.insert(0, '../functions')

# Import optical depth functions from generate_all_plots
from generate_all_plots import compute_dtau_dz, compute_tau_0_to_z, compute_tau_6_to_z, TAU_0_6

DATA_DIR = "../data_raghu/"
BOX_MPC_OVER_H = 500.0
LITTLEH = 0.7
N_CELLS = 600

SELECTED_REDSHIFTS = [
    6.056, 6.113, 6.172, 6.231, 6.292, 6.354, 6.418, 6.483, 6.549, 6.617,
    6.686, 6.757, 6.830, 6.905, 6.981, 7.059, 7.139, 7.221, 7.305, 7.391,
    7.480, 7.570, 7.664, 7.760, 7.859, 7.960, 8.064, 8.172, 8.283, 8.397,
    8.515, 8.636, 8.762, 8.892, 9.026, 9.164, 9.308, 9.457, 9.611, 9.771,
    9.938, 10.110, 10.290, 10.478, 10.673, 10.877, 11.090, 11.313, 11.546, 11.791, 12.048
]
print(f"Selected {len(SELECTED_REDSHIFTS)} redshifts")
print(f"τ̄₀₆ (z=0 to 6) = {TAU_0_6:.4f}")
'''

# =============================================================================
# CELL 3 (ksz_functions) - Updated with optical depth formulation
# =============================================================================
CELL_3_KSZ_FUNCTIONS = '''
def compute_ksz_map_optical_depth(den, xhi, vlos, z, axis=2):
    """
    Compute kSZ map using optical depth formulation.
    
    ΔT_kSZ/T_CMB = -∫ (v_r/c) * x_e * (1+δ) * dτ
    where dτ = (dτ/dz) * dz
    """
    den_L = np.moveaxis(den, axis, -1)
    xhi_L = np.moveaxis(xhi, axis, -1)
    vlos_L = np.moveaxis(vlos, axis, -1)
    nz = den_L.shape[-1]
    
    # Density contrast
    delta = den_L / np.mean(den_L) - 1.0
    
    # Electron density weight: x_e * (1 + δ)
    xe_delta = (1.0 - xhi_L) * (1.0 + delta)
    
    # Physical constants
    c_km_s = 2.99792458e5       # Speed of light [km/s]
    T_CMB = 2.725e6             # CMB temperature [µK]
    
    # Cosmological parameters
    H0 = 70.0  # km/s/Mpc
    Omega_m = 0.27
    Omega_L = 1.0 - Omega_m
    
    # Hubble parameter at z
    E_z = np.sqrt(Omega_m * (1 + z)**3 + Omega_L)
    H_z = H0 * E_z  # km/s/Mpc
    
    # Box size in proper Mpc
    box_proper_Mpc = BOX_MPC_OVER_H / LITTLEH / (1.0 + z)
    dl_proper = box_proper_Mpc / nz  # Mpc per cell
    
    # dz per cell: dz = H(z)/c * (1+z) * dl_proper
    dz_per_cell = H_z / c_km_s * (1 + z) * dl_proper
    
    # dτ/dz at this redshift (for mean ionized universe)
    dtau_dz = compute_dtau_dz(z)
    
    # Prefactor: -T_CMB * (v/c) * dτ
    # where dτ = (dτ/dz) * dz_per_cell for each cell
    # v is already in km/s, c in km/s, so v/c is dimensionless
    pref = -T_CMB * dtau_dz * dz_per_cell / c_km_s  # [µK]
    
    # Sum along LOS
    ksz_map = pref * np.sum(xe_delta * vlos_L, axis=-1, dtype=np.float64).astype(np.float32)
    
    del den_L, xhi_L, vlos_L, delta, xe_delta
    return ksz_map

def analyze_redshift(z):
    print(f"  z={z:.3f}...", end=" ")
    den, xhi, vz = load_data(z)
    mean_xhi = float(np.mean(xhi))
    xe = 1.0 - xhi
    xe_fluct = float(np.std(xe))
    delta = den / np.mean(den) - 1.0
    delta_rms = float(np.std(delta))
    v_rms = float(np.std(vz))
    ne = xe * (1.0 + delta)
    ne_mean = float(np.mean(ne))
    ne_std = float(np.std(ne))
    ne_fluct_rel = ne_std / ne_mean if ne_mean > 0 else 0
    
    # Compute dτ/dz at this redshift
    dtau_dz = compute_dtau_dz(z)
    
    del delta, ne
    gc.collect()
    
    # Use optical depth formulation
    ksz_map = compute_ksz_map_optical_depth(den, xhi, vz, z)
    ksz_rms = float(np.std(ksz_map))
    print(f"xHI={mean_xhi:.3f}, kSZ_rms={ksz_rms:.2e} µK, dτ/dz={dtau_dz:.4e}")
    
    del den, xhi, vz, xe
    gc.collect()
    return {'z': z, 'mean_xhi': mean_xhi, 'xe_fluct': xe_fluct, 'delta_rms': delta_rms,
            'v_rms': v_rms, 'ne_std': ne_std, 'ne_fluct_rel': ne_fluct_rel, 
            'ksz_rms': ksz_rms, 'ksz_map': ksz_map, 'dtau_dz': dtau_dz}

print("Analysis functions defined (using optical depth formulation).")
'''

# =============================================================================
# CELL 11 (ell3000) - Updated to normalize to PEAK
# =============================================================================
CELL_11_ELL3000 = '''
ell_target = 3000
k_ell3000, power_ell3000 = [], []
for i, r in enumerate(results):
    chi = comoving_distance(r['z'])
    k_target = ell_target / chi
    k_ell3000.append(k_target)
    psd = psd_data[i]
    valid = np.isfinite(psd['k']) & np.isfinite(psd['P']) & (psd['k'] > 0)
    P_at_k = np.interp(k_target, psd['k'][valid], psd['P'][valid]) if np.any(valid) else np.nan
    power_ell3000.append(P_at_k)
k_ell3000 = np.array(k_ell3000)
power_ell3000 = np.array(power_ell3000)

# Find PEAK power instead of high-z
peak_idx = np.argmax(power_ell3000)
peak_z = z_arr[peak_idx]
peak_xhi = xhi_arr[peak_idx]
print(f"Peak kSZ power at ℓ={ell_target}: z={peak_z:.2f}, xHI={peak_xhi:.2f}")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

ax = axes[0]
ax.plot(z_arr, power_ell3000, '--', color='gray', lw=2.5, alpha=0.7, zorder=1)
sc = ax.scatter(z_arr, power_ell3000, c=xhi_arr, cmap='coolwarm', s=60, zorder=5)
ax.axvline(peak_z, color='green', ls=':', lw=2, alpha=0.7, label=f'Peak z={peak_z:.2f}')
ax.set_yscale('log'); ax.set_xlabel('Redshift z', fontsize=12); ax.set_ylabel(f'P(k) at $\\ell$={ell_target} [µK²]', fontsize=12)
ax.set_title(f'kSZ Power at $\\ell$={ell_target}', fontweight='bold'); ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)
cbar = plt.colorbar(sc, ax=ax); cbar.set_label('$x_{HI}$')

ax = axes[1]
# Normalize to PEAK instead of high-z
power_norm = power_ell3000 / power_ell3000[peak_idx]
ax.plot(z_arr, power_norm, '--', color='gray', lw=2.5, alpha=0.7, zorder=1)
sc = ax.scatter(z_arr, power_norm, c=xhi_arr, cmap='coolwarm', s=60, zorder=5)
ax.axhline(1, color='green', ls=':', lw=2, alpha=0.7)
ax.axvline(peak_z, color='green', ls=':', lw=2, alpha=0.7)
ax.set_xlabel('Redshift z', fontsize=12)
ax.set_ylabel(f'P / P$_{{peak}}$ (z={peak_z:.1f}, xHI={peak_xhi:.2f})', fontsize=12)
ax.set_title('Normalized to Peak', fontweight='bold')
ax.grid(True, alpha=0.3); cbar = plt.colorbar(sc, ax=ax); cbar.set_label('$x_{HI}$')

ax = axes[2]
k_compare = [0.1, 0.2, 0.4]
colors_c = ['blue', 'green', 'orange']
for k_val, col in zip(k_compare, colors_c):
    power_at_k = [np.interp(k_val, psd['k'][np.isfinite(psd['P'])], psd['P'][np.isfinite(psd['P'])]) for psd in psd_data]
    power_at_k = np.array(power_at_k)
    peak_k_idx = np.argmax(power_at_k)
    ax.plot(z_arr, power_at_k/power_at_k[peak_k_idx], '--', color='gray', lw=2, alpha=0.4, zorder=1)
    ax.plot(z_arr, power_at_k/power_at_k[peak_k_idx], 'o', color=col, ms=6, alpha=0.7, label=f'k={k_val}', zorder=5)
ax.plot(z_arr, power_norm, '--', color='gray', lw=2.5, alpha=0.7, zorder=1)
ax.scatter(z_arr, power_norm, c='red', s=60, marker='s', label=f'$\\ell$={ell_target}', zorder=5)
ax.axhline(1, color='k', ls='--', alpha=0.5); ax.set_xlabel('Redshift z', fontsize=12)
ax.set_ylabel('P / P$_{peak}$', fontsize=12); ax.set_title('Scale Comparison (normalized to peak)', fontweight='bold')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
plt.suptitle(f'kSZ at $\\ell$={ell_target} vs Redshift (Optical Depth Formulation)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/ksz_ell3000_vs_z_peak_norm.png', dpi=200, bbox_inches='tight')
plt.show()
'''

# =============================================================================
# CELL 12 (ell3000_xhi) - Updated to normalize to PEAK
# =============================================================================
CELL_12_ELL3000_XHI = '''
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.plot(xhi_arr, power_ell3000, '--', color='gray', lw=2.5, alpha=0.7, zorder=1)
sc = ax.scatter(xhi_arr, power_ell3000, c=z_arr, cmap='viridis', s=60, zorder=5)
ax.axvline(peak_xhi, color='red', ls=':', lw=2, alpha=0.7, label=f'Peak xHI={peak_xhi:.2f}')
ax.set_yscale('log'); ax.set_xlabel('$\\langle x_{HI} \\rangle$', fontsize=12)
ax.set_ylabel(f'P(k) at $\\ell$={ell_target} [µK²]', fontsize=12)
ax.set_title(f'kSZ Power at $\\ell$={ell_target}', fontweight='bold'); ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)
cbar = plt.colorbar(sc, ax=ax); cbar.set_label('Redshift z')

ax = axes[1]
ax.plot(xhi_arr, power_norm, '--', color='gray', lw=2.5, alpha=0.7, zorder=1)
sc = ax.scatter(xhi_arr, power_norm, c=z_arr, cmap='viridis', s=60, zorder=5)
ax.axhline(1, color='red', ls=':', lw=2, alpha=0.7)
ax.axvline(peak_xhi, color='red', ls=':', lw=2, alpha=0.7)
ax.set_xlabel('$\\langle x_{HI} \\rangle$', fontsize=12)
ax.set_ylabel(f'P / P$_{{peak}}$ (xHI={peak_xhi:.2f})', fontsize=12)
ax.set_title('Normalized to Peak', fontweight='bold')
ax.grid(True, alpha=0.3); cbar = plt.colorbar(sc, ax=ax); cbar.set_label('Redshift z')

plt.suptitle(f'kSZ at $\\ell$={ell_target} vs Neutral Fraction (Peak Normalization)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/ksz_ell3000_vs_xhi_peak_norm.png', dpi=200, bbox_inches='tight')
plt.show()

print(f"\\nPeak kSZ power occurs at:")
print(f"  Redshift z = {peak_z:.3f}")
print(f"  Neutral fraction xHI = {peak_xhi:.3f}")
print(f"  This corresponds to ~{(1-peak_xhi)*100:.0f}% ionization")
'''

# =============================================================================
# CELL 13 (fractional) - Updated to show peak
# =============================================================================
CELL_13_FRACTIONAL = '''
total_power = np.sum(power_ell3000)
fracs = power_ell3000 / total_power * 100

fig, ax = plt.subplots(figsize=(14, 5))
colors = plt.cm.coolwarm(xhi_arr)
bars = ax.bar(range(len(results)), fracs, color=colors, edgecolor='black', width=0.8)

# Highlight peak
bars[peak_idx].set_edgecolor('lime')
bars[peak_idx].set_linewidth(3)

ax.set_xticks(range(0, len(results), 5))
ax.set_xticklabels([f"{results[i]['z']:.1f}" for i in range(0, len(results), 5)], rotation=45, ha='right')
ax.set_xlabel('Redshift z', fontsize=12)
ax.set_ylabel(f'Contribution to $\\ell$={ell_target} Power (%)', fontsize=12)
ax.set_title(f'Fractional Contribution to kSZ at $\\ell$={ell_target} (Peak at z={peak_z:.2f}, xHI={peak_xhi:.2f})', fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Add annotation for peak
ax.annotate(f'Peak\\nz={peak_z:.1f}', xy=(peak_idx, fracs[peak_idx]), 
            xytext=(peak_idx+5, fracs[peak_idx]+0.5),
            arrowprops=dict(arrowstyle='->', color='green'), fontsize=10, color='green')

sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(0, 1))
cbar = plt.colorbar(sm, ax=ax); cbar.set_label('$x_{HI}$')
plt.tight_layout()
plt.savefig('plots/ksz_ell3000_fractional_peak.png', dpi=200, bbox_inches='tight')
plt.show()

print(f"\\nPeak contribution: {fracs[peak_idx]:.1f}% at z={peak_z:.2f}")
'''

print("="*70)
print("INSTRUCTIONS:")
print("="*70)
print("""
Copy the code from the following variables into your notebook cells:

1. CELL 0 (markdown header): Update with optical depth formula
2. CELL 1 (imports): Use CELL_1_IMPORTS - adds optical depth imports
3. CELL 3 (ksz_functions): Use CELL_3_KSZ_FUNCTIONS - optical depth kSZ computation
4. CELL 11 (ell3000): Use CELL_11_ELL3000 - normalize to PEAK
5. CELL 12 (ell3000_xhi): Use CELL_12_ELL3000_XHI - normalize to PEAK
6. CELL 13 (fractional): Use CELL_13_FRACTIONAL - highlight peak

Key changes:
- kSZ maps now computed with dτ/dz weighting (optical depth formulation)
- Power spectra normalized to PEAK instead of high-z value
- Peak redshift/xHI clearly marked in plots
- Units now in µK (physical units)
""")
