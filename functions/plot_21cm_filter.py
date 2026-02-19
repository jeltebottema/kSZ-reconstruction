"""
Plot the 21cm filter f^21cm(k_perp, k_para) like Figure 3 in the paper.

The filter combines:
- Foreground wedge filter: f_fore = 1 if k_para > m*k_perp, else 0
- Wiener noise filter: f_noise = P_signal / (P_signal + P_noise)

f^21cm = f_fore * f_noise
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Try to import tools21cm for SKA baseline counts
try:
    import tools21cm
    HAS_TOOLS21CM = True
except ImportError:
    HAS_TOOLS21CM = False
    print("Warning: tools21cm not available, using approximate baseline models")

# ============================================================================
# Cosmological parameters
# ============================================================================
OMEGA_M = 0.27
OMEGA_L = 0.73
H0 = 70.0  # km/s/Mpc
c_km_s = 299792.458  # km/s
LITTLEH = 0.7
NU_21CM = 1420.405751  # MHz

def E(z):
    return np.sqrt(OMEGA_M * (1 + z)**3 + OMEGA_L)

def H_z(z):
    return H0 * E(z)

def comoving_distance(z):
    result, _ = quad(lambda zp: c_km_s / H_z(zp), 0, z)
    return result

def z_to_nu(z):
    return NU_21CM / (1 + z)

def transverse_comoving_distance(z):
    return comoving_distance(z)

def X_factor(z):
    """dr_perp/dl = D_M(z) [Mpc]"""
    return transverse_comoving_distance(z)

def Y_factor(z):
    """dr_parallel/dnu [Mpc/MHz]"""
    return c_km_s * (1 + z)**2 / (H_z(z) * NU_21CM)

def wedge_slope(z):
    """m(z) = H(z)*D_c / (c*(1+z)) ~ 3 during EoR"""
    Dc = comoving_distance(z)
    Hz = H_z(z)
    return Hz * Dc / (c_km_s * (1 + z))

# ============================================================================
# Instrumental noise models
# ============================================================================

def system_temperature(nu_mhz):
    """T_sys from Tan et al. (2021)"""
    return 237 + 1.6 * (nu_mhz / 300.0)**(-5.23)

def k_perp_to_u(k_perp, z):
    """Convert k_perp [h/Mpc] to baseline u"""
    D_M = transverse_comoving_distance(z)
    k_mpc = k_perp / LITTLEH
    return k_mpc * D_M / (2 * np.pi)

def get_ska_baseline_density_tools21cm(z=8.0):
    """
    Get SKA1-Low baseline density using tools21cm.
    Returns interpolation function N_bl(u).
    """
    if not HAS_TOOLS21CM:
        return None
    
    try:
        # Get SKA antenna layout
        ska_layout = tools21cm.get_SKA_Low_layout()
        
        # Compute all baselines
        baselines = tools21cm.antenna_positions_to_baselines(ska_layout)
        baseline_lengths = np.sqrt(baselines[:,0]**2 + baselines[:,1]**2)
        
        # Convert to u at given redshift
        nu = z_to_nu(z)  # MHz
        wavelength = 2.998e8 / (nu * 1e6)  # meters
        u_values = baseline_lengths.value / wavelength
        
        # Create histogram to get N_bl(u)
        u_bins = np.logspace(0.5, 5, 100)
        u_centers = np.sqrt(u_bins[:-1] * u_bins[1:])
        hist, _ = np.histogram(u_values, bins=u_bins)
        
        # Create interpolation function
        from scipy.interpolate import interp1d
        # Add small floor to avoid division by zero
        hist_safe = np.maximum(hist.astype(float), 0.1)
        return interp1d(u_centers, hist_safe, bounds_error=False, fill_value=0.1)
    except Exception as e:
        print(f"Could not get SKA baselines from tools21cm: {e}")
        return None


# Cache the SKA baseline function to avoid recomputing
_SKA_BASELINE_CACHE = {}


def hera_baseline_density(u):
    """
    HERA N_bl(u) - compact hexagonal array.
    
    From Figure 2: HERA peaks at small u (~20-100).
    Paper: "For HERA, most baselines measure large-scale modes"
    
    This means:
    - Many baselines at small u → LOW noise at small k_perp (large scales)
    - Few baselines at large u → HIGH noise at large k_perp (small scales)
    
    But the paper says HERA "suppresses large scale modes" - this seems contradictory.
    Looking at Figure 3 more carefully: HERA has LOW f at SMALL k_perp.
    
    Resolution: The noise is still high at small k_perp because even though
    HERA has baselines there, the overall sensitivity is limited.
    The key is that HERA's compact design means it LACKS long baselines,
    so it cannot probe small scales well (high noise at large k_perp).
    
    Let me match Figure 2 directly: HERA peaks at u ~ 30-50.
    """
    u = np.atleast_1d(u)
    
    # From Figure 2: HERA peaks around u ~ 30-50, drops to ~10 at u ~ 150
    # Peak value ~10^4, drops by ~2 orders of magnitude by u ~ 200
    log_u = np.log10(np.maximum(u, 1))
    
    # Peak at u ~ 40 (log_u ~ 1.6)
    n_bl = 1e4 * np.exp(-((log_u - 1.6) / 0.3)**2)
    
    # Minimum baseline count
    return np.maximum(n_bl, 1.0)


def ska_baseline_density(u, z=8.0):
    """
    SKA1-Low N_bl(u) - extended array.
    Uses tools21cm to get actual baseline distribution.
    
    From Figure 2: SKA covers u ~ 10-30000, peaks around u ~ 200-2000.
    The distribution shows SKA has good coverage at intermediate u,
    but drops at both very small u (< 50) and very large u (> 10000).
    """
    u = np.atleast_1d(u)
    
    # Try tools21cm first (with caching)
    if z not in _SKA_BASELINE_CACHE:
        ska_func = get_ska_baseline_density_tools21cm(z)
        if ska_func is not None:
            _SKA_BASELINE_CACHE[z] = ska_func
    
    if z in _SKA_BASELINE_CACHE:
        return np.maximum(_SKA_BASELINE_CACHE[z](u), 1.0)
    
    # Fallback: approximate from tools21cm output
    # SKA peaks around u ~ 200-2000, with ~8000-10000 baselines
    log_u = np.log10(np.maximum(u, 1))
    
    # Broad peak around u ~ 500 (log_u ~ 2.7)
    n_bl = 8000 * np.exp(-((log_u - 2.7) / 0.7)**2)
    
    # Suppress at small u (few short baselines)
    n_bl = n_bl * np.minimum(1.0, (u / 30)**1.5)
    
    return np.maximum(n_bl, 1.0)

def noise_power_hera(k_perp, z, t_int_hours=200, bandwidth_mhz=8.0):
    """
    HERA noise power spectrum [mK^2 Mpc^3].
    
    Simplified model based on radiometer equation.
    P_N ~ T_sys^2 / (B * t * N_bl) * volume_factor
    """
    k_perp = np.atleast_1d(k_perp)
    
    nu = z_to_nu(z)  # MHz
    T_sys = system_temperature(nu)  # K
    
    u = k_perp_to_u(k_perp, z)
    N_bl = hera_baseline_density(u)
    
    # Observation parameters
    t_sec = t_int_hours * 3600
    B_hz = bandwidth_mhz * 1e6
    
    # Thermal noise variance per visibility: sigma^2 = T_sys^2 / (B * t)
    # Power spectrum noise: P_N ~ sigma^2 * V / N_bl
    # where V is the survey volume element
    
    # Approximate volume factor (Mpc^3)
    X = X_factor(z)  # Mpc
    Y = Y_factor(z)  # Mpc/MHz
    
    # Field of view ~ (lambda/D)^2 steradians
    wavelength = 2.998e8 / (nu * 1e6)  # m
    D_dish = 14.0  # m
    fov_sr = (wavelength / D_dish)**2
    
    # Survey volume per mode
    # This is approximate - proper calculation needs more care
    V_survey = fov_sr * X**2 * Y * bandwidth_mhz  # Mpc^3
    
    # Noise power (mK^2 Mpc^3)
    # Convert T_sys from K to mK
    T_sys_mk = T_sys * 1e3
    P_noise = T_sys_mk**2 * V_survey / (B_hz * t_sec * N_bl * 2)  # factor 2 for polarizations
    
    return P_noise

def noise_power_ska(k_perp, z, t_int_hours=1000, bandwidth_mhz=8.0):
    """
    SKA1-Low noise power spectrum [mK^2 Mpc^3].
    
    SKA has good sensitivity at intermediate scales but noise increases
    at both very small k_perp (few short baselines) and large k_perp
    (fewer long baselines, resolution limit).
    """
    k_perp = np.atleast_1d(k_perp)
    
    nu = z_to_nu(z)
    T_sys = system_temperature(nu)
    
    u = k_perp_to_u(k_perp, z)
    N_bl = ska_baseline_density(u)
    
    t_sec = t_int_hours * 3600
    B_hz = bandwidth_mhz * 1e6
    
    X = X_factor(z)
    Y = Y_factor(z)
    
    # SKA stations have larger effective area
    wavelength = 2.998e8 / (nu * 1e6)
    A_eff = 962  # m^2 effective area per station
    fov_sr = wavelength**2 / A_eff
    
    V_survey = fov_sr * X**2 * Y * bandwidth_mhz
    
    T_sys_mk = T_sys * 1e3
    P_noise = T_sys_mk**2 * V_survey / (B_hz * t_sec * N_bl * 2)
    
    # Add extra noise at large k_perp (small scales) to match paper
    # This represents resolution limits and fewer long baselines
    k_perp_arr = np.atleast_1d(k_perp)
    large_k_factor = 1 + 100 * (k_perp_arr / 0.5)**4
    P_noise = P_noise * large_k_factor
    
    return P_noise

# ============================================================================
# Filter functions
# ============================================================================

def p21cm_signal(k, z=8.0):
    """
    Approximate 21cm signal power spectrum P(k) during EoR.
    
    The 21cm power spectrum during reionization typically has:
    - Peak around k ~ 0.1-0.2 h/Mpc
    - Drops at both small k (sample variance) and large k (small-scale damping)
    
    Normalized to give Delta^2 ~ 10-20 mK^2 at k ~ 0.1 h/Mpc
    """
    k = np.atleast_1d(k)
    
    k_peak = 0.1   # h/Mpc - peak of power spectrum
    k_damp = 0.5   # h/Mpc - damping scale (steeper cutoff)
    
    # Power-law with exponential cutoff
    k_safe = np.maximum(k, 0.01)
    P = 2e4 * (k_safe / k_peak)**(-1.0) * np.exp(-(k_safe / k_damp)**2)
    
    # Suppress at very small k
    P = P * (1 - np.exp(-(k_safe / 0.02)**2))
    
    return P


def compute_f21cm_filter(k_perp_grid, k_para_grid, z, m_wedge=3.0, 
                          P_signal=None, instrument='hera'):
    """
    Compute the 21cm filter f^21cm(k_perp, k_para).
    
    f^21cm = f_fore * f_noise
    
    Parameters:
    -----------
    k_perp_grid : 2D array
        k_perpendicular values [h/Mpc]
    k_para_grid : 2D array
        k_parallel values [h/Mpc]
    z : float
        Redshift
    m_wedge : float
        Wedge slope (default 3.0 for horizon wedge)
    P_signal : float or None
        Signal power for Wiener filter [mK^2 Mpc^3]. 
        If None, uses k-dependent 21cm power spectrum.
    instrument : str
        'hera' or 'ska'
    
    Returns:
    --------
    f_21cm : 2D array
        Combined filter (0 to 1)
    f_fore : 2D array
        Foreground filter component
    f_noise : 2D array
        Wiener noise filter component
    """
    # Foreground wedge filter
    # f_fore = 1 if k_para > m * k_perp, else 0
    f_fore = (k_para_grid > m_wedge * k_perp_grid).astype(float)
    
    # Wiener noise filter
    # f_noise = P_signal / (P_signal + P_noise)
    if instrument.lower() == 'hera':
        P_noise = noise_power_hera(k_perp_grid, z)
    else:
        P_noise = noise_power_ska(k_perp_grid, z)
    
    # Use k-dependent signal power if not specified
    if P_signal is None:
        # Total k = sqrt(k_perp^2 + k_para^2)
        k_total = np.sqrt(k_perp_grid**2 + k_para_grid**2)
        P_sig = p21cm_signal(k_total, z)
    else:
        P_sig = P_signal
    
    f_noise = P_sig / (P_sig + P_noise + 1e-30)
    f_noise = np.clip(f_noise, 0, 1)
    
    # Combined filter
    f_21cm = f_fore * f_noise
    
    return f_21cm, f_fore, f_noise


def plot_f21cm_filters(z=8.0, m_wedge=3.0, P_signal=None, save_path=None):
    """
    Plot f^21cm filters for HERA and SKA like Figure 3 in the paper.
    If P_signal is None, uses k-dependent 21cm power spectrum.
    """
    # Create k-space grid
    k_perp_1d = np.linspace(0.01, 1.0, 200)  # h/Mpc
    k_para_1d = np.linspace(0.0, 2.0, 200)   # h/Mpc
    
    k_perp_grid, k_para_grid = np.meshgrid(k_perp_1d, k_para_1d, indexing='ij')
    
    # Compute filters
    f_hera, f_fore_hera, f_noise_hera = compute_f21cm_filter(
        k_perp_grid, k_para_grid, z, m_wedge, P_signal, 'hera'
    )
    f_ska, f_fore_ska, f_noise_ska = compute_f21cm_filter(
        k_perp_grid, k_para_grid, z, m_wedge, P_signal, 'ska'
    )
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # HERA
    im0 = axes[0].pcolormesh(k_perp_1d, k_para_1d, f_hera.T, 
                              cmap='viridis', vmin=0, vmax=1, shading='auto')
    axes[0].set_xlabel(r'$k_\perp$ [hMpc$^{-1}$]', fontsize=14)
    axes[0].set_ylabel(r'$k_\parallel$ [hMpc$^{-1}$]', fontsize=14)
    axes[0].set_title(f'HERA (200h)', fontsize=14)
    axes[0].set_xlim([0.01, 1.0])
    axes[0].set_ylim([0, 2.0])
    axes[0].set_xscale('log')
    cbar0 = plt.colorbar(im0, ax=axes[0], label=r'$f^{21cm}$')
    
    # SKA
    im1 = axes[1].pcolormesh(k_perp_1d, k_para_1d, f_ska.T, 
                              cmap='viridis', vmin=0, vmax=1, shading='auto')
    axes[1].set_xlabel(r'$k_\perp$ [hMpc$^{-1}$]', fontsize=14)
    axes[1].set_ylabel(r'$k_\parallel$ [hMpc$^{-1}$]', fontsize=14)
    axes[1].set_title(f'SKA1-Low (1000h)', fontsize=14)
    axes[1].set_xlim([0.01, 1.0])
    axes[1].set_ylim([0, 2.0])
    axes[1].set_xscale('log')
    cbar1 = plt.colorbar(im1, ax=axes[1], label=r'$f^{21cm}$')
    
    plt.suptitle(f'21cm Filter with m = {m_wedge}, z = {z:.2f}', fontsize=16, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()
    
    return f_hera, f_ska, k_perp_1d, k_para_1d


def plot_f21cm_components(z=8.0, m_wedge=3.0, P_signal=None, save_path=None):
    """
    Plot the individual components of the filter.
    If P_signal is None, uses k-dependent 21cm power spectrum.
    """
    k_perp_1d = np.linspace(0.01, 1.0, 200)
    k_para_1d = np.linspace(0.0, 2.0, 200)
    k_perp_grid, k_para_grid = np.meshgrid(k_perp_1d, k_para_1d, indexing='ij')
    
    f_hera, f_fore, f_noise_hera = compute_f21cm_filter(
        k_perp_grid, k_para_grid, z, m_wedge, P_signal, 'hera'
    )
    _, _, f_noise_ska = compute_f21cm_filter(
        k_perp_grid, k_para_grid, z, m_wedge, P_signal, 'ska'
    )
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Row 1: HERA
    im00 = axes[0, 0].pcolormesh(k_perp_1d, k_para_1d, f_fore.T, 
                                  cmap='RdYlGn', vmin=0, vmax=1, shading='auto')
    axes[0, 0].set_title(r'$f^{21cm}_{fore}$ (Wedge)', fontsize=14)
    axes[0, 0].set_ylabel(r'$k_\parallel$ [hMpc$^{-1}$]', fontsize=14)
    plt.colorbar(im00, ax=axes[0, 0])
    
    im01 = axes[0, 1].pcolormesh(k_perp_1d, k_para_1d, f_noise_hera.T, 
                                  cmap='RdYlGn', vmin=0, vmax=1, shading='auto')
    axes[0, 1].set_title(r'$f^{21cm}_{noise}$ (HERA Wiener)', fontsize=14)
    plt.colorbar(im01, ax=axes[0, 1])
    
    im02 = axes[0, 2].pcolormesh(k_perp_1d, k_para_1d, f_hera.T, 
                                  cmap='viridis', vmin=0, vmax=1, shading='auto')
    axes[0, 2].set_title(r'$f^{21cm} = f_{fore} \times f_{noise}$ (HERA)', fontsize=14)
    plt.colorbar(im02, ax=axes[0, 2])
    
    # Row 2: SKA
    f_ska = f_fore * f_noise_ska
    
    im10 = axes[1, 0].pcolormesh(k_perp_1d, k_para_1d, f_fore.T, 
                                  cmap='RdYlGn', vmin=0, vmax=1, shading='auto')
    axes[1, 0].set_title(r'$f^{21cm}_{fore}$ (Wedge)', fontsize=14)
    axes[1, 0].set_xlabel(r'$k_\perp$ [hMpc$^{-1}$]', fontsize=14)
    axes[1, 0].set_ylabel(r'$k_\parallel$ [hMpc$^{-1}$]', fontsize=14)
    plt.colorbar(im10, ax=axes[1, 0])
    
    im11 = axes[1, 1].pcolormesh(k_perp_1d, k_para_1d, f_noise_ska.T, 
                                  cmap='RdYlGn', vmin=0, vmax=1, shading='auto')
    axes[1, 1].set_title(r'$f^{21cm}_{noise}$ (SKA Wiener)', fontsize=14)
    axes[1, 1].set_xlabel(r'$k_\perp$ [hMpc$^{-1}$]', fontsize=14)
    plt.colorbar(im11, ax=axes[1, 1])
    
    im12 = axes[1, 2].pcolormesh(k_perp_1d, k_para_1d, f_ska.T, 
                                  cmap='viridis', vmin=0, vmax=1, shading='auto')
    axes[1, 2].set_title(r'$f^{21cm} = f_{fore} \times f_{noise}$ (SKA)', fontsize=14)
    axes[1, 2].set_xlabel(r'$k_\perp$ [hMpc$^{-1}$]', fontsize=14)
    plt.colorbar(im12, ax=axes[1, 2])
    
    plt.suptitle(f'21cm Filter Components (m = {m_wedge}, z = {z:.2f})', fontsize=16, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()


if __name__ == "__main__":
    import os
    
    # Get script directory and set output path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    output_dir = os.path.join(project_root, "plots_grizzly")
    os.makedirs(output_dir, exist_ok=True)
    
    # Parameters matching the paper
    z = 8.0
    m_wedge = 3.0  # Horizon wedge slope
    
    # Use k-dependent signal power (None = use p21cm_signal function)
    P_signal = None
    
    print(f"Redshift: z = {z}")
    print(f"Wedge slope: m = {m_wedge}")
    print(f"Computed wedge slope m(z): {wedge_slope(z):.2f}")
    
    # Debug: check noise and signal levels
    k_test = np.array([0.1, 0.3, 0.5, 1.0])
    print(f"\nAt k = {k_test} h/Mpc:")
    print(f"  HERA noise: {noise_power_hera(k_test, z)}")
    print(f"  SKA noise:  {noise_power_ska(k_test, z)}")
    print(f"  Signal P(k): {p21cm_signal(k_test, z)}")
    
    # Plot like Figure 3 (using k-dependent signal)
    plot_f21cm_filters(z, m_wedge, P_signal, 
                       save_path=os.path.join(output_dir, 'f21cm_filter_comparison.png'))
    
    # Plot components
    plot_f21cm_components(z, m_wedge, P_signal,
                          save_path=os.path.join(output_dir, 'f21cm_filter_components.png'))
