"""
Reusable helpers for the 21cm noise-filter analysis.

Pipeline: UV coverage → P_noise(k_perp) → Wiener filter f_noise(k_perp, k_par)
          → combined with wedge mask f_total.

Used by the 01_noise_in_21cmfast and 03_uv_sweep notebooks.
"""
from __future__ import annotations

import numpy as np
from astropy import units as un
from scipy import fft

from py21cmsense import Observatory, Observation, GaussianBeam
from py21cmsense.antpos import hera as hera_antpos_fn
from tuesday.core import compute_uv_sampling, compute_thermal_rms_uvgrid, observe_coeval


# ---------------------------------------------------------------------------
# 1. Observation builder — one place to vary UV characteristics
# ---------------------------------------------------------------------------
def build_hera_observation(
    hex_num: int = 11,
    split_core: bool = True,
    outriggers: int = 2,
    dish_size_m: float = 14.0,
    latitude_deg: float = -30.0,
    track_hours: float = 6.0,
    time_per_day_hours: float = 6.0,
    n_days: int = 180,
):
    """Build a HERA-like Observation. Returns (observatory, observation, antpos)."""
    antpos = hera_antpos_fn(
        hex_num=hex_num, split_core=split_core, outriggers=outriggers,
    )
    observatory = Observatory(
        antpos=antpos,
        latitude=latitude_deg * un.deg,
        beam=GaussianBeam(dish_size=dish_size_m * un.m),
    )
    obs = Observation(
        observatory=observatory,
        track=track_hours * un.hour,
        time_per_day=time_per_day_hours * un.hour,
        n_days=n_days,
    )
    return observatory, obs, antpos


# ---------------------------------------------------------------------------
# 2. 2D power spectrum P(k_perp, k_par)
# ---------------------------------------------------------------------------
def compute_2d_ps(box, box_len, n_bins_perp=20, n_bins_par=20):
    """Cylindrically-averaged 2D power spectrum. Returns (kp_c, kpar_c, P_2d, count)."""
    n = box.shape[0]
    dk = 2 * np.pi / box_len
    V = box_len ** 3
    box_k = fft.rfftn(box - box.mean(), workers=-1)
    P_k = np.abs(box_k) ** 2 / V

    kx = dk * np.fft.fftfreq(n, d=1.0 / n)
    ky = dk * np.fft.fftfreq(n, d=1.0 / n)
    kz = dk * np.fft.rfftfreq(n, d=1.0 / n)
    kx3, ky3, kz3 = np.meshgrid(kx, ky, kz, indexing="ij")
    k_perp = np.sqrt(kx3 ** 2 + ky3 ** 2).ravel()
    k_par = np.abs(kz3).ravel()

    k_max = dk * n / 2
    kp_edges = np.logspace(np.log10(dk * 0.8), np.log10(k_max), n_bins_perp + 1)
    kpar_edges = np.linspace(0, k_max, n_bins_par + 1)
    ip = np.digitize(k_perp, kp_edges) - 1
    jp = np.digitize(k_par, kpar_edges) - 1
    valid = (ip >= 0) & (ip < n_bins_perp) & (jp >= 0) & (jp < n_bins_par)

    bidx = ip[valid] * n_bins_par + jp[valid]
    nf = n_bins_perp * n_bins_par
    P_sum = np.bincount(bidx, weights=P_k.ravel()[valid], minlength=nf)
    cnt = np.bincount(bidx, minlength=nf).astype(float)
    P_2d = np.full(nf, np.nan)
    P_2d[cnt > 0] = P_sum[cnt > 0] / cnt[cnt > 0]

    kp_c = np.sqrt(kp_edges[:-1] * kp_edges[1:])
    kpar_c = 0.5 * (kpar_edges[:-1] + kpar_edges[1:])
    return kp_c, kpar_c, P_2d.reshape(n_bins_perp, n_bins_par), cnt.reshape(n_bins_perp, n_bins_par)


# ---------------------------------------------------------------------------
# 3. Analytical noise power from tuesday
# ---------------------------------------------------------------------------
def analytical_noise_2d(observation, box_len, ncells, redshift,
                        kp_c, kpar_c):
    """
    Build 2D analytical P_noise(k_perp, k_par) by binning tuesday's sigma_uv.
    Thermal noise is white in frequency → flat in k_par.

    Returns (P_noise_2d, uv_sampling_2d) where uv_sampling_2d is the raw
    baseline-count grid (Nx, Nx//2+1).
    """
    freq_obs = 1420.405 * un.MHz / (1 + redshift)

    _, _, uv_sampling = compute_uv_sampling(
        observation=observation,
        freqs=un.Quantity([freq_obs]),
        box_length=box_len * un.Mpc,
        box_ncells=ncells,
        freq_dependent_uv_grid=False,
    )

    sigma_uv = compute_thermal_rms_uvgrid(
        observation=observation,
        uv_coverage=uv_sampling,
        freqs=un.Quantity([freq_obs]),
        box_length=box_len * un.Mpc,
    )
    # (Nx, Nx//2+1) in mK — noise RMS per UV cell
    sigma_2d = sigma_uv[:, :, 0].value
    P_noise_uv = sigma_2d ** 2

    # Bin to k_perp. tuesday's grid layout (see compute_uv_sampling docstring):
    #   axis 0: full u in ascending order from -k_nyq to +k_nyq (fftshift order, NOT fft order)
    #   axis 1: non-negative v in ascending order from 0 to k_nyq
    dk = 2 * np.pi / box_len
    kx = np.fft.fftshift(np.fft.fftfreq(ncells, d=1.0 / ncells)) * dk
    ky = np.arange(ncells // 2 + 1) * dk
    kx2, ky2 = np.meshgrid(kx, ky, indexing="ij")
    k_perp_uv = np.sqrt(kx2 ** 2 + ky2 ** 2).ravel()
    P_flat = P_noise_uv.ravel()

    n_bins_perp = len(kp_c)
    k_max = dk * ncells / 2
    kp_edges = np.logspace(np.log10(dk * 0.8), np.log10(k_max), n_bins_perp + 1)
    ip = np.digitize(k_perp_uv, kp_edges) - 1

    P_noise_1d = np.full(n_bins_perp, np.nan)
    for i in range(n_bins_perp):
        mask = (ip == i) & np.isfinite(P_flat) & (P_flat > 0)
        if mask.any():
            P_noise_1d[i] = np.mean(P_flat[mask])

    # Flat in k_par
    P_noise_2d = np.repeat(P_noise_1d[:, np.newaxis], len(kpar_c), axis=1)
    return P_noise_2d, P_noise_1d, uv_sampling[:, :, 0]


# ---------------------------------------------------------------------------
# 3b. Empirical noise from observe_coeval realizations
# ---------------------------------------------------------------------------
def empirical_noise_2d(observation, box_shape, box_len, redshift, *,
                       n_realizations=5, n_bins_perp=20, n_bins_par=20,
                       seed=99999):
    """
    P_noise(k_perp, k_par) by averaging compute_2d_ps over N zero-signal
    observe_coeval realizations. Bin layout matches compute_2d_ps with the
    same n_bins_perp/n_bins_par, so it lines up with the signal grid.

    Unmeasured (k_perp, k_par) bins (no UV coverage in any contributing cell)
    are returned as +inf so a downstream Wiener filter cleanly drops them to 0.

    Returns (kp_c, kpar_c, P_noise_2d, P_noise_1d).
    """
    ncells = box_shape[0]
    zero_box = np.zeros(box_shape) * un.mK
    realizations = observe_coeval(
        box=zero_box, box_length=box_len * un.Mpc,
        observation=observation, redshift=redshift,
        nrealizations=n_realizations,
        remove_wedge=False, remove_mean=False, multiply_by_beam=False,
        seed=seed,
    )

    P_sum = np.zeros((n_bins_perp, n_bins_par))
    cnt_sum = np.zeros((n_bins_perp, n_bins_par))
    for i in range(n_realizations):
        kp_c, kpar_c, P_2d, _ = compute_2d_ps(
            realizations[i].value, box_len, n_bins_perp, n_bins_par,
        )
        finite = np.isfinite(P_2d)
        P_sum[finite] += P_2d[finite]
        cnt_sum[finite] += 1
    P_noise_2d = np.where(cnt_sum > 0, P_sum / np.maximum(cnt_sum, 1), np.nan)

    # Mask k_perp bins outside the observed UV coverage. Use the actual UV
    # sampling grid to find which k_perp bins have any baselines.
    freq_obs = 1420.405 * un.MHz / (1 + redshift)
    _, _, uv_sampling = compute_uv_sampling(
        observation=observation, freqs=un.Quantity([freq_obs]),
        box_length=box_len * un.Mpc, box_ncells=ncells,
        freq_dependent_uv_grid=False,
    )
    has_bl = uv_sampling[:, :, 0] > 0
    dk = 2 * np.pi / box_len
    kx = np.fft.fftshift(np.fft.fftfreq(ncells, d=1.0 / ncells)) * dk
    ky = np.arange(ncells // 2 + 1) * dk
    kx2, ky2 = np.meshgrid(kx, ky, indexing="ij")
    k_perp_uv = np.sqrt(kx2 ** 2 + ky2 ** 2)
    k_max = dk * ncells / 2
    kp_edges = np.logspace(np.log10(dk * 0.8), np.log10(k_max), n_bins_perp + 1)
    ip = np.digitize(k_perp_uv.ravel(), kp_edges) - 1
    sampled_perp = np.zeros(n_bins_perp, dtype=bool)
    flat_has_bl = has_bl.ravel()
    for i in range(n_bins_perp):
        if (flat_has_bl & (ip == i)).any():
            sampled_perp[i] = True
    P_noise_2d[~sampled_perp, :] = np.inf

    P_noise_1d = np.nanmean(np.where(np.isinf(P_noise_2d), np.nan, P_noise_2d), axis=1)
    return kp_c, kpar_c, P_noise_2d, P_noise_1d


# ---------------------------------------------------------------------------
# 4. Wiener filter and wedge
# ---------------------------------------------------------------------------
def wiener_filter(P_signal, P_noise):
    """f_noise = P_s / (P_s + P_n). Returns 0 where denominator is nonpositive."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where((P_signal + P_noise) > 0,
                         P_signal / (P_signal + P_noise), 0.0)


def wedge_mask(kp_c, kpar_c, slope=1.0):
    """Return a (Nkp, Nkpar) mask: 1.0 above the wedge line k_par > slope * k_perp."""
    KP, KPAR = np.meshgrid(kp_c, kpar_c, indexing="ij")
    return np.where(KPAR > slope * KP, 1.0, 0.0)


# ---------------------------------------------------------------------------
# 5. Baseline / UV summary for panel (a)
# ---------------------------------------------------------------------------
def baseline_lengths(antpos):
    """Return all baseline lengths in metres from an antenna position array."""
    pos = antpos.to(un.m).value  # (N, 3)
    diffs = pos[:, None, :] - pos[None, :, :]
    dist = np.sqrt((diffs ** 2).sum(axis=-1))
    iu = np.triu_indices(len(pos), k=1)
    return dist[iu]
