"""
ksz_pipeline.py — kSZ Reconstruction & Cross-Correlation Pipeline

Self-contained module for reconstructing the kinetic Sunyaev-Zel'dovich (kSZ)
signal from 21cm lightcone data and cross-correlating it with the true
(simulated) integrated kSZ signal.

Input:  21cmFAST lightcone arrays (density, xHI, velocity, redshifts)
Output: Reconstructed kSZ maps, cross-correlation r(k), power spectra

Usage:
    >>> from ksz_pipeline import KSZPipeline
    >>> pipe = KSZPipeline(box_mpc_over_h=300.0, littleh=0.7)
    >>> results = pipe.run(density, xHI, velocity_z, redshifts)
    >>> pipe.plot_results(results, save="my_analysis.png")
"""

import numpy as np
from scipy import fft
from scipy.integrate import quad
from dataclasses import dataclass, field
from typing import Optional
import warnings


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ChunkResult:
    """Results for a single redshift chunk."""
    chunk_idx: int
    mean_z: float
    z_min: float
    z_max: float
    mean_xhi: float
    r_real: float               # Real-space Pearson r(kSZ_real, kSZ_rec)
    k_values: np.ndarray        # Wavenumber bins [h/Mpc]
    r_k: np.ndarray             # Fourier-space correlation r(k)
    ell_values: np.ndarray      # Multipole ℓ values
    ksz_map_real: np.ndarray    # 2D integrated kSZ map (true velocity)
    ksz_map_rec: np.ndarray     # 2D integrated kSZ map (reconstructed velocity)


@dataclass
class PipelineResult:
    """Full pipeline output."""
    chunks: list                # List of ChunkResult
    # Full lightcone integration
    ksz_full_real: np.ndarray   # Summed kSZ map (true)
    ksz_full_rec: np.ndarray    # Summed kSZ map (reconstructed)
    r_full: float               # Real-space Pearson r for full integration
    k_full: np.ndarray          # k values for full integration
    r_k_full: np.ndarray        # r(k) for full integration
    ell_full: np.ndarray        # ell values for full integration


# ---------------------------------------------------------------------------
# Cosmology helpers
# ---------------------------------------------------------------------------

def _hubble_parameter(z, H0, omega_m, omega_l):
    """H(z) in km/s/Mpc."""
    return H0 * np.sqrt(omega_m * (1 + z)**3 + omega_l)


def _growth_rate(z, omega_m, omega_l):
    """Linear growth rate f ≈ Ω_m(z)^0.55."""
    a = 1.0 / (1.0 + z)
    Ez2 = omega_m / a**3 + omega_l
    Omega_m_z = (omega_m / a**3) / Ez2
    return Omega_m_z**0.55


def _comoving_distance(z, littleh, omega_m=0.27, omega_l=0.73):
    """Comoving distance χ(z) in Mpc/h."""
    def integrand(zp):
        return 1.0 / np.sqrt(omega_m * (1 + zp)**3 + omega_l)
    chi, _ = quad(integrand, 0, z)
    chi *= 3000.0 / littleh  # c/H0 in Mpc/h
    return chi


def k_to_ell(k, z, littleh=0.7):
    """Convert wavenumber k [h/Mpc] → multipole ℓ = k × χ(z)."""
    return k * _comoving_distance(z, littleh)


# ---------------------------------------------------------------------------
# Core statistics
# ---------------------------------------------------------------------------

def pearson_r(a, b):
    """Pearson correlation coefficient (NaN-safe)."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    m = np.isfinite(a) & np.isfinite(b)
    if not np.any(m):
        return np.nan
    a = a[m] - a[m].mean()
    b = b[m] - b[m].mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else np.nan


def fourier_correlation_2d(field1, field2, boxlength, bins=100):
    """
    Fourier-space correlation coefficient r(k) between two 2D fields.

    r(k) = P_XY(k) / sqrt(P_XX(k) * P_YY(k))

    Parameters
    ----------
    field1, field2 : 2D arrays
        Maps to cross-correlate.
    boxlength : float or [Ly, Lx]
        Physical box size(s) in Mpc/h.
    bins : int
        Number of k bins.

    Returns
    -------
    k_centers : 1D array  [h/Mpc]
    r_k       : 1D array  (correlation coefficient per bin)
    """
    if np.isscalar(boxlength):
        boxlength = [boxlength, boxlength]

    f1 = field1.astype(np.float64) - np.mean(field1)
    f2 = field2.astype(np.float64) - np.mean(field2)

    fft1 = np.fft.rfft2(f1)
    fft2 = np.fft.rfft2(f2)

    ny, nx = field1.shape
    dx = boxlength[1] / nx
    dy = boxlength[0] / ny

    kx = 2.0 * np.pi * np.fft.fftfreq(ny, d=dx)
    ky = 2.0 * np.pi * np.fft.rfftfreq(nx, d=dy)
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing="ij")
    k_mag = np.sqrt(kx_grid**2 + ky_grid**2)

    k_fund = 2.0 * np.pi / max(boxlength)
    k_max = k_mag.max()
    k_edges = np.linspace(k_fund, k_max, bins + 1)
    k_centers = 0.5 * (k_edges[:-1] + k_edges[1:])

    r_k = np.zeros(bins)
    for i in range(bins):
        mask = (k_mag >= k_edges[i]) & (k_mag < k_edges[i + 1])
        if not np.any(mask):
            continue
        f1b = fft1[mask]
        f2b = fft2[mask]
        cross = np.real(f1b * np.conj(f2b)).sum()
        auto1 = np.real(f1b * np.conj(f1b)).sum()
        auto2 = np.real(f2b * np.conj(f2b)).sum()
        denom = np.sqrt(auto1 * auto2)
        if denom > 0:
            r_k[i] = cross / denom

    return k_centers, r_k


def cross_power_spectrum_2d(field1, field2, boxlength, bins=100, log_bins=True):
    """
    Compute 2D auto/cross power spectra.

    Returns dimensionless power ℓ² P(ℓ).

    Parameters
    ----------
    field1, field2 : 2D arrays (pass field2=None for auto-power)
    boxlength : float  (Mpc/h, assumes square box)
    bins : int
    log_bins : bool

    Returns
    -------
    P_k : 1D array   (ℓ² C_ℓ or k² P(k))
    k   : 1D array   [h/Mpc]
    """
    try:
        from powerbox import get_power
    except ImportError:
        raise ImportError(
            "powerbox is required for power spectrum computation.  "
            "Install with:  pip install powerbox"
        )

    f1 = field1.astype(np.float32) - np.mean(field1)
    kw = dict(boxlength=boxlength, bins=bins, log_bins=log_bins, get_variance=True)

    if field2 is None:
        P, k, var = get_power(deltax=f1, **kw)
    else:
        f2 = field2.astype(np.float32) - np.mean(field2)
        P, k, var = get_power(deltax=f1, deltax2=f2, **kw)

    return P * k**2, k


# ---------------------------------------------------------------------------
# Velocity reconstruction (linear continuity equation)
# ---------------------------------------------------------------------------

def _kspace_rfft(n, rc, dtype=np.float32):
    """k-space coordinate arrays for an rfftn grid, with k=0 protection."""
    kx = (2.0 * np.pi * np.fft.fftfreq(n, d=rc)).astype(dtype)
    ky = (2.0 * np.pi * np.fft.fftfreq(n, d=rc)).astype(dtype)
    kz = (2.0 * np.pi * np.fft.rfftfreq(n, d=rc)).astype(dtype)
    tiny = np.finfo(dtype).tiny
    for arr in (kx, ky, kz):
        if arr.size:
            arr[0] = max(arr[0], tiny)
    return kx, ky, kz


def reconstruct_velocity_z(tracer_field, z, box_mpc_over_h, littleh,
                           omega_m=0.27, omega_l=0.73):
    """
    Reconstruct the line-of-sight velocity from a tracer field using the
    linearised continuity equation.

    v_z(k) = i × (aHf / k²) × k_z × δ(k)

    Parameters
    ----------
    tracer_field : 3D array  (nx, ny, nz)
        Mean-subtracted tracer field (e.g. (1+δ)×xHI or density contrast).
    z : float
        Representative redshift for cosmological factors.
    box_mpc_over_h : float
        Comoving box size [Mpc/h].
    littleh : float
        Dimensionless Hubble constant h.
    omega_m, omega_l : float
        Matter and dark-energy density parameters.

    Returns
    -------
    vz_rec : 3D array  (same shape, float32)
        Reconstructed LOS velocity in the same (arbitrary) units as the
        input field × cosmological factor.
    """
    nx, ny, nz = tracer_field.shape
    dtype = np.float32

    # FFT of the tracer field
    field_ms = tracer_field - tracer_field.mean()
    dlt_k = fft.rfftn(field_ms, workers=-1).astype(np.complex64, copy=False)

    # k-space grid
    rc = box_mpc_over_h / float(nx) / littleh  # cell size in Mpc
    kx, ky, kz_arr = _kspace_rfft(nx, rc, dtype)

    # Cosmological prefactor:  i × a × H(z) × f(z)
    a = dtype(1.0 / (1.0 + z))
    H0 = dtype(100.0 * littleh)
    Ha = dtype(_hubble_parameter(z, float(H0), omega_m, omega_l))
    f = dtype(_growth_rate(z, omega_m, omega_l))
    factor = np.complex64(Ha * a * f) * 1j

    kx2 = (kx * kx).astype(dtype)
    ky2 = (ky * ky).astype(dtype)
    kz2 = (kz_arr * kz_arr).astype(dtype)

    # v_z(k) = factor × k_z × δ(k) / k²
    tmp = dlt_k * factor
    tmp *= kz_arr[None, None, :]
    absk2 = kx2[:, None, None] + ky2[None, :, None] + kz2[None, None, :]
    np.divide(tmp, absk2, out=tmp, where=absk2 != 0)

    vz_rec = np.real(fft.irfftn(tmp, s=(nx, ny, nz), workers=-1)).astype(dtype)
    return vz_rec


# ---------------------------------------------------------------------------
# kSZ map computation
# ---------------------------------------------------------------------------

def compute_ksz_map(vz, xhi, density, axis=2):
    """
    Compute a projected kSZ map by integrating (1-xHI)×(1+δ)×v_z along LOS.

    Parameters
    ----------
    vz : 3D array
        Line-of-sight velocity field.
    xhi : 3D array
        Neutral hydrogen fraction (0 = fully ionised, 1 = fully neutral).
    density : 3D array
        Density field (unnormalised; mean will be computed internally).
    axis : int
        LOS axis for summation (default 2, i.e. the z-axis).

    Returns
    -------
    ksz_map : 2D array
    """
    mean_den = density.mean(dtype=np.float64).astype(np.float32)
    # Protect against xHI exactly = 1 (fully neutral → no free electrons)
    xe = 1.0 - np.clip(xhi, 0.0, 1.0 - 1e-8)
    one_plus_delta = density / mean_den if mean_den > 0 else density
    weighted = xe * one_plus_delta * vz
    return np.sum(weighted, axis=axis)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

class KSZPipeline:
    """
    End-to-end kSZ reconstruction and cross-correlation pipeline.

    Parameters
    ----------
    box_mpc_over_h : float
        Comoving simulation box size [Mpc/h].
    littleh : float
        Dimensionless Hubble parameter h.
    omega_m : float
        Matter density parameter Ω_m.
    omega_l : float
        Dark-energy density parameter Ω_Λ.
    chunk_size : int
        Number of LOS slices per redshift chunk.
    fourier_bins : int
        Number of k-bins for Fourier correlation.
    tracer : str
        Tracer field construction: "deltaXhi" → (1+δ)×xHI,
        "delta" → density contrast only.
    """

    def __init__(
        self,
        box_mpc_over_h: float = 300.0,
        littleh: float = 0.7,
        omega_m: float = 0.27,
        omega_l: float = 0.73,
        chunk_size: int = 200,
        fourier_bins: int = 100,
        tracer: str = "deltaXhi",
    ):
        self.box = box_mpc_over_h
        self.h = littleh
        self.omega_m = omega_m
        self.omega_l = omega_l
        self.chunk_size = chunk_size
        self.fourier_bins = fourier_bins
        self.tracer = tracer

    # ----- tracer field construction -----

    def _build_tracer(self, density, xhi):
        """Build the tracer field that gets FFT'd for velocity reconstruction."""
        mean_den = density.mean(dtype=np.float64).astype(np.float32)
        delta = density / mean_den if mean_den > 0 else density
        if self.tracer == "deltaXhi":
            return (1.0 + delta - 1.0) * xhi  # δ × xHI   (since delta = den/mean)
            # More precisely: (1+δ)*xHI, but delta = den/mean - 1
            # So (1 + delta)*xHI = (den/mean)*xHI
        elif self.tracer == "delta":
            return delta - 1.0  # just the overdensity
        else:
            raise ValueError(f"Unknown tracer: {self.tracer}")

    def _build_tracer_field(self, density, xhi):
        """Build the tracer field for velocity reconstruction."""
        mean_den = density.mean(dtype=np.float64).astype(np.float32)
        if self.tracer == "deltaXhi":
            # (1+δ) × xHI  where 1+δ = den / <den>
            return (density / mean_den) * xhi if mean_den > 0 else density * xhi
        elif self.tracer == "delta":
            return density / mean_den - 1.0 if mean_den > 0 else density
        else:
            raise ValueError(f"Unknown tracer: {self.tracer}")

    # ----- single chunk processing -----

    def _process_chunk(self, density, xhi, vz_real, redshifts, chunk_idx):
        """Reconstruct kSZ and cross-correlate for one redshift chunk."""
        nx, ny, nz_chunk = density.shape
        mean_z = float(np.mean(redshifts))
        mean_xhi = float(np.mean(xhi))

        # 1) Build tracer field and reconstruct LOS velocity
        tracer = self._build_tracer_field(density, xhi)
        vz_rec = reconstruct_velocity_z(
            tracer, mean_z, self.box, self.h, self.omega_m, self.omega_l
        )

        # 2) Compute kSZ maps (integrate along LOS)
        ksz_real = compute_ksz_map(vz_real, xhi, density)
        ksz_rec  = compute_ksz_map(-vz_rec, xhi, density)  # minus sign convention

        # 3) Real-space correlation
        r_real = pearson_r(ksz_real, ksz_rec)

        # 4) Fourier-space correlation
        dx = self.box / nx
        Lx = dx * nx
        Ly = dx * ny
        k_vals, r_k = fourier_correlation_2d(
            ksz_real, ksz_rec,
            boxlength=[Ly, Lx],
            bins=self.fourier_bins,
        )

        # 5) Convert k → ℓ
        ell_vals = k_to_ell(k_vals, mean_z, self.h)

        return ChunkResult(
            chunk_idx=chunk_idx,
            mean_z=mean_z,
            z_min=float(redshifts.min()),
            z_max=float(redshifts.max()),
            mean_xhi=mean_xhi,
            r_real=r_real,
            k_values=k_vals,
            r_k=r_k,
            ell_values=ell_vals,
            ksz_map_real=ksz_real,
            ksz_map_rec=ksz_rec,
        )

    # ----- main entry point -----

    def run(self, density, xhi, velocity_z, redshifts, n_chunks=None, verbose=True):
        """
        Run the full reconstruction + cross-correlation pipeline.

        Parameters
        ----------
        density : 3D array (nx, ny, n_los)
            Density field lightcone.  Can be raw ρ (unnormalised) or 1+δ.
        xhi : 3D array (nx, ny, n_los)
            Neutral fraction lightcone (0–1).
        velocity_z : 3D array (nx, ny, n_los)
            Line-of-sight velocity lightcone [any consistent units;
            the reconstruction cares about morphology, not amplitude].
        redshifts : 1D array (n_los,)
            Redshift for each LOS slice.
        n_chunks : int, optional
            Number of chunks.  Default: as many full chunks as fit.
        verbose : bool
            Print progress.

        Returns
        -------
        PipelineResult
        """
        nx, ny, n_los = density.shape
        cs = self.chunk_size

        if n_chunks is None:
            n_chunks = n_los // cs

        if verbose:
            print(f"Box: {self.box} Mpc/h,  h={self.h},  tracer={self.tracer}")
            print(f"Lightcone shape: {density.shape},  "
                  f"z=[{redshifts.min():.2f}, {redshifts.max():.2f}]")
            print(f"Processing {n_chunks} chunks of {cs} slices")

        chunks = []
        for i in range(n_chunks):
            s = i * cs
            e = s + cs
            if e > n_los:
                break

            if verbose:
                print(f"  Chunk {i+1}/{n_chunks}  (slices {s}:{e})", end="")

            den_c = np.array(density[:, :, s:e], dtype=np.float32)
            xhi_c = np.array(xhi[:, :, s:e], dtype=np.float32)
            vz_c  = np.array(velocity_z[:, :, s:e], dtype=np.float32)
            z_c   = redshifts[s:e]

            result = self._process_chunk(den_c, xhi_c, vz_c, z_c, i)
            chunks.append(result)

            if verbose:
                print(f"  z={result.mean_z:.2f}  <xHI>={result.mean_xhi:.3f}  "
                      f"r={result.r_real:.4f}")

        # Full lightcone integration
        ksz_full_real = np.sum([c.ksz_map_real for c in chunks], axis=0)
        ksz_full_rec  = np.sum([c.ksz_map_rec  for c in chunks], axis=0)
        r_full = pearson_r(ksz_full_real, ksz_full_rec)

        dx = self.box / nx
        Lx = dx * nx
        Ly = dx * ny
        k_full, r_k_full = fourier_correlation_2d(
            ksz_full_real, ksz_full_rec,
            boxlength=[Ly, Lx],
            bins=self.fourier_bins,
        )
        mean_z_all = np.mean([c.mean_z for c in chunks])
        ell_full = k_to_ell(k_full, mean_z_all, self.h)

        if verbose:
            print(f"\nFull lightcone:  r = {r_full:.4f}")

        return PipelineResult(
            chunks=chunks,
            ksz_full_real=ksz_full_real,
            ksz_full_rec=ksz_full_rec,
            r_full=r_full,
            k_full=k_full,
            r_k_full=r_k_full,
            ell_full=ell_full,
        )

    # ----- plotting -----

    @staticmethod
    def plot_results(result: PipelineResult, save: Optional[str] = None):
        """
        Generate a 4-panel summary figure.

        Top-left:     r(k) per chunk
        Top-right:    r(ℓ) per chunk
        Bottom-left:  Real-space r vs redshift & neutral fraction
        Bottom-right: Full lightcone r(k)
        """
        import matplotlib.pyplot as plt

        chunks = result.chunks
        n = len(chunks)
        colors = plt.cm.viridis(np.linspace(0, 1, n))

        fig, axes = plt.subplots(2, 2, figsize=(14, 11))

        # --- r(k) per chunk ---
        ax = axes[0, 0]
        for idx, c in enumerate(chunks):
            ax.plot(c.k_values, c.r_k, color=colors[idx], lw=1.8,
                    label=f"z={c.mean_z:.1f}, xHI={c.mean_xhi:.2f}")
        ax.axhline(0, color="k", lw=0.6, alpha=0.4)
        ax.axhline(1, color="grey", ls="--", lw=0.8, alpha=0.3)
        ax.set_xlabel("k [h/Mpc]")
        ax.set_ylabel("r(k)")
        ax.set_title("Fourier correlation per chunk")
        ax.set_ylim(-0.2, 1.15)
        ax.legend(fontsize=6, ncol=2, loc="lower right")
        ax.grid(True, alpha=0.25)

        # --- r(ℓ) per chunk ---
        ax = axes[0, 1]
        for idx, c in enumerate(chunks):
            ax.plot(c.ell_values, c.r_k, color=colors[idx], lw=1.8,
                    label=f"z={c.mean_z:.1f}")
        ax.axhline(0, color="k", lw=0.6, alpha=0.4)
        ax.set_xlabel("Multipole ℓ")
        ax.set_ylabel("r(ℓ)")
        ax.set_title("Fourier correlation (multipole space)")
        ax.set_ylim(-0.2, 1.15)
        ax.legend(fontsize=6, ncol=2, loc="lower right")
        ax.grid(True, alpha=0.25)

        # --- r vs z and xHI ---
        ax = axes[1, 0]
        zs   = [c.mean_z   for c in chunks]
        xhis = [c.mean_xhi for c in chunks]
        rs   = [c.r_real    for c in chunks]
        ax.plot(zs, rs, "o-", color="steelblue", lw=2, ms=6, label="r vs z")
        ax.set_xlabel("Mean redshift z")
        ax.set_ylabel("Real-space r")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.25)
        ax.set_title("Correlation vs redshift")

        ax2 = ax.twiny()
        ax2.plot(xhis, rs, "s--", color="seagreen", lw=1.5, ms=5, alpha=0.7,
                 label="r vs <xHI>")
        ax2.set_xlabel("<xHI>", color="seagreen")
        ax2.tick_params(axis="x", colors="seagreen")

        # --- Full lightcone r(k) ---
        ax = axes[1, 1]
        ax.plot(result.k_full, result.r_k_full, "r-", lw=2.5,
                label=f"Full LC  r={result.r_full:.3f}")
        ax.axhline(0, color="k", lw=0.6, alpha=0.4)
        ax.set_xlabel("k [h/Mpc]")
        ax.set_ylabel("r(k)")
        ax.set_title("Full lightcone integration")
        ax.set_ylim(-0.2, 1.15)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.25)

        fig.suptitle("kSZ Reconstruction Quality", fontsize=14, fontweight="bold")
        fig.tight_layout()

        if save:
            fig.savefig(save, dpi=200, bbox_inches="tight")
            print(f"Saved → {save}")
        plt.show()
        return fig


# ---------------------------------------------------------------------------
# Convenience: run directly on 21cmFAST lightcone .npy files
# ---------------------------------------------------------------------------

def run_from_files(
    density_file: str,
    xhi_file: str,
    velocity_z_file: str,
    redshifts_file: str,
    box_mpc_over_h: float = 300.0,
    littleh: float = 0.7,
    chunk_size: int = 200,
    n_chunks: Optional[int] = None,
    save_plot: Optional[str] = None,
) -> PipelineResult:
    """
    Load .npy lightcone files and run the full pipeline.

    Parameters
    ----------
    density_file : str
        Path to density lightcone  (nx, ny, n_los) .npy
    xhi_file : str
        Path to neutral fraction lightcone .npy
    velocity_z_file : str
        Path to LOS velocity lightcone .npy
    redshifts_file : str
        Path to 1D redshift array .npy
    box_mpc_over_h : float
        Comoving box side length [Mpc/h]
    littleh : float
        h parameter
    chunk_size : int
    n_chunks : int or None
    save_plot : str or None
        If given, save the summary figure to this path.

    Returns
    -------
    PipelineResult
    """
    print("Loading data...")
    density    = np.load(density_file, mmap_mode="r")
    xhi_data   = np.load(xhi_file, mmap_mode="r")
    vel_z      = np.load(velocity_z_file, mmap_mode="r")
    redshifts  = np.load(redshifts_file)
    print(f"  density  : {density.shape}")
    print(f"  xHI      : {xhi_data.shape}")
    print(f"  velocity : {vel_z.shape}")
    print(f"  redshifts: {redshifts.shape}  [{redshifts.min():.2f} – {redshifts.max():.2f}]")

    pipe = KSZPipeline(
        box_mpc_over_h=box_mpc_over_h,
        littleh=littleh,
        chunk_size=chunk_size,
    )
    result = pipe.run(density, xhi_data, vel_z, redshifts, n_chunks=n_chunks)

    if save_plot:
        KSZPipeline.plot_results(result, save=save_plot)

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="kSZ reconstruction pipeline")
    parser.add_argument("--density", required=True, help="density lightcone .npy")
    parser.add_argument("--xhi", required=True, help="xHI lightcone .npy")
    parser.add_argument("--velocity", required=True, help="LOS velocity lightcone .npy")
    parser.add_argument("--redshifts", required=True, help="redshifts .npy")
    parser.add_argument("--box", type=float, default=300.0, help="box size [Mpc/h]")
    parser.add_argument("--h", type=float, default=0.7, help="little h")
    parser.add_argument("--chunk-size", type=int, default=200)
    parser.add_argument("--n-chunks", type=int, default=None)
    parser.add_argument("--save", default=None, help="save figure path")
    args = parser.parse_args()

    run_from_files(
        density_file=args.density,
        xhi_file=args.xhi,
        velocity_z_file=args.velocity,
        redshifts_file=args.redshifts,
        box_mpc_over_h=args.box,
        littleh=args.h,
        chunk_size=args.chunk_size,
        n_chunks=args.n_chunks,
        save_plot=args.save,
    )
