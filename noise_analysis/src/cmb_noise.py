"""
cmb_noise.py
============
CMB noise models for realistic survey specifications.

Targets:
    - Simons Observatory (SO)  — LAT, baseline + goal noise
    - CMB-S4                   — wide + ultra-deep
    - Planck                   — for reference / large-scale anchor

What this module should provide (to be filled in):
    1. White-noise power spectrum N_l from
           Delta_T [muK-arcmin], beam_fwhm [arcmin]
       N_l = (Delta_T * pi/180/60)**2 * exp( l*(l+1) * sigma_b**2 ),
       with sigma_b = beam_fwhm/sqrt(8 ln 2) * pi/180/60.
    2. 1/f atmospheric component
           N_l_atm = N_l_white * (1 + (l_knee/l)**alpha_knee).
    3. ILC residual after multi-frequency component separation
       (foreground-deprojected, e.g. tSZ-free), supplied as a tabulated
       N_l_ILC(l) array (e.g. via pyilc / orphics).
    4. Helpers to:
           - generate Gaussian map realisations on a flat-sky patch,
           - convert to / from healpix maps,
           - return a unified NoiseModel object that noisy_reconstruction
             can consume alongside the 21cm noise side.

Conventions
-----------
Power spectra are returned as C_l in muK^2 sr (flat-sky convention),
matching the 21cm side and the orphics/ILC pipelines.

This is a placeholder — not implemented yet.
"""

import numpy as np


# -----------------------------------------------------------------------------
# Survey specs (placeholder values — populate from the SO / S4 papers).
# -----------------------------------------------------------------------------
SURVEYS = {
    "SO_baseline": {
        "freqs_GHz":   [27, 39, 93, 145, 225, 280],
        "noise_uK_arcmin": None,   # TODO
        "beam_arcmin":     None,   # TODO
        "l_knee":          None,   # TODO
        "alpha_knee":      None,   # TODO
        "f_sky":           0.4,
    },
    "CMB_S4_wide": {
        "freqs_GHz":   [30, 40, 90, 150, 220, 270],
        "noise_uK_arcmin": None,   # TODO
        "beam_arcmin":     None,   # TODO
        "l_knee":          None,   # TODO
        "alpha_knee":      None,   # TODO
        "f_sky":           0.5,
    },
}


def white_noise_cl(delta_T_uK_arcmin: float, beam_fwhm_arcmin: float, ell: np.ndarray) -> np.ndarray:
    """White detector noise + Gaussian-beam deconvolution.

    Returns C_l^N in muK^2 sr.
    """
    arcmin_to_rad = np.pi / (180.0 * 60.0)
    delta_T = delta_T_uK_arcmin * arcmin_to_rad           # muK rad
    sigma_b = beam_fwhm_arcmin * arcmin_to_rad / np.sqrt(8.0 * np.log(2.0))
    return delta_T**2 * np.exp(ell * (ell + 1) * sigma_b**2)


def atmospheric_factor(ell: np.ndarray, l_knee: float, alpha_knee: float) -> np.ndarray:
    """1/f-style multiplier (1 + (l_knee/l)**alpha_knee)."""
    ell_safe = np.where(ell == 0, 1, ell)
    return 1.0 + (l_knee / ell_safe) ** alpha_knee


def ilc_residual_cl(ell: np.ndarray, survey: str) -> np.ndarray:
    """ILC residual N_l after multi-frequency cleaning.

    TODO: load tabulated N_l from pyilc / orphics output for the given survey.
    """
    raise NotImplementedError("Hook this up to pyilc / orphics output.")


class NoiseModel:
    """Common interface that noisy_reconstruction/ can consume.

    Both this CMB model and the 21cm noise model should expose:
        .cl(ell)            -> C_l^N  (flat-sky convention, muK^2 sr)
        .realise_map(shape, dx_rad, rng) -> 2D noise map in muK
    """

    def cl(self, ell):
        raise NotImplementedError

    def realise_map(self, shape, dx_rad, rng=None):
        raise NotImplementedError
