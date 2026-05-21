"""
Plot helpers for the kSZ + non-Gaussianity pipeline.

Self-contained: takes arrays/dicts, returns Matplotlib Figures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np


def plot_evolution(
    evo: Dict[str, np.ndarray],
    out: Optional[str] = None,
    title: str = "kSZ non-Gaussianity through the EoR",
):
    """variance / skewness / kurtosis vs mean x_HI (top) and z (bottom).

    ``evo`` must be a dict with numeric array-like values at keys:
    ``mean_xHI``, ``redshift``, ``variance``, ``skewness``,
    ``kurtosis``.
    """
    metrics = [("variance", r"$\sigma^2$"),
               ("skewness", r"$\gamma_1$"),
               ("kurtosis", r"excess $\gamma_2$")]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, (key, label) in zip(axes[0], metrics):
        ax.plot(evo["mean_xHI"], evo[key], "o-", lw=2, color="C0")
        ax.set_xlabel(r"mean $x_{\rm HI}$")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()
    for ax, (key, label) in zip(axes[1], metrics):
        ax.plot(evo["redshift"], evo[key], "s-", lw=2, color="C3")
        ax.set_xlabel("redshift z")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()
    fig.suptitle(title)
    fig.tight_layout()
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
    return fig


def plot_pdf_evolution(
    fields: Sequence[np.ndarray],
    xHI: Sequence[float],
    redshifts: Sequence[float],
    bins: int = 80,
    out: Optional[str] = None,
):
    """Overlay the standardised 1-point PDFs, coloured by mean x_HI."""
    if len(fields) != len(xHI) or len(fields) != len(redshifts):
        raise ValueError("fields, xHI, redshifts must be aligned")

    order = np.argsort(-np.asarray(xHI))  # high x_HI first
    cmap = plt.get_cmap("viridis")

    fig, ax = plt.subplots(figsize=(7, 5))
    centers_ref = None
    for j, i in enumerate(order):
        arr = np.asarray(fields[i]).ravel()
        std = arr.std() or 1.0
        normed = arr / std
        hist, edges = np.histogram(normed, bins=bins, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        col = cmap(j / max(len(order) - 1, 1))
        ax.plot(centers, hist, color=col, lw=1.8,
                label=f"z={redshifts[i]:.2f}  $x_{{HI}}$={xHI[i]:.2f}")
        centers_ref = centers
    if centers_ref is not None:
        xg = np.linspace(centers_ref.min(), centers_ref.max(), 200)
        ax.plot(xg, np.exp(-0.5 * xg ** 2) / np.sqrt(2 * np.pi),
                "k--", lw=1.3, label="Gaussian")
    ax.set_yscale("log")
    ax.set_xlabel(r"field / $\sigma$")
    ax.set_ylabel("PDF")
    ax.set_title("kSZ PDF through the EoR")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
    return fig


def plot_power_spectrum(
    k: np.ndarray,
    Pk: np.ndarray,
    label: str = "",
    out: Optional[str] = None,
    ax=None,
    xlabel: str = r"$k$  [Mpc$^{-1}$]",
    ylabel: str = r"$P(k)$  [Mpc$^n$]",
):
    """Log-log 1D power spectrum."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure
    mask = np.isfinite(Pk) & (Pk > 0) & (k > 0)
    ax.loglog(k[mask], Pk[mask], lw=2, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.3)
    if label:
        ax.legend(fontsize=9)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
    return fig


def plot_lightcone_pdf(
    fields: Dict[str, np.ndarray],
    bins: int = 80,
    standardise: bool = True,
    out: Optional[str] = None,
    title: str = "lightcone kSZ 1-point PDF",
):
    """1-point PDF of one or more lightcone kSZ maps.

    Parameters
    ----------
    fields : dict {label: 2D array}
        One entry per map to overlay, e.g.
        ``{"rotation=True": ksz_rot.kSZ_box, "rotation=False": ksz_nor.kSZ_box}``.
    bins : int
        Number of histogram bins.
    standardise : bool
        If True, plot PDF of ``(field - mean) / std``; if False, plot
        the raw map PDF in microkelvin.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = plt.get_cmap("tab10").colors
    xs_all = []
    for i, (label, arr) in enumerate(fields.items()):
        x = np.asarray(arr).ravel().astype(np.float64)
        if standardise:
            std = x.std() or 1.0
            x = (x - x.mean()) / std
        else:
            x = (x - x.mean()) * 1e6  # K -> uK
        hist, edges = np.histogram(x, bins=bins, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax.plot(centers, hist, lw=1.8, color=colors[i % len(colors)], label=label)
        xs_all.append(centers)

    if standardise and xs_all:
        xg = np.linspace(min(c.min() for c in xs_all),
                         max(c.max() for c in xs_all), 400)
        ax.plot(xg, np.exp(-0.5 * xg ** 2) / np.sqrt(2 * np.pi),
                "k--", lw=1.3, label="Gaussian")
        ax.set_xlabel(r"field / $\sigma$")
    else:
        ax.set_xlabel(r"$\Delta T_{\rm kSZ}$  [$\mu$K]")
    ax.set_yscale("log")
    ax.set_ylabel("PDF")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
    return fig


def plot_ksz_map(
    dT_map: np.ndarray,
    box_len: float,
    title: str = "patchy kSZ",
    out: Optional[str] = None,
):
    """2D visualisation of a kSZ map (K -> uK)."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(dT_map * 1e6,
                   extent=[0, box_len, 0, box_len],
                   origin="lower", cmap="RdBu_r")
    ax.set_xlabel("cMpc")
    ax.set_ylabel("cMpc")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$\Delta T_{\rm kSZ}$  [$\mu$K]")
    fig.tight_layout()
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
    return fig
