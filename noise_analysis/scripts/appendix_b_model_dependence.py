"""
Appendix B — Model Dependence of the Cross-Correlation
=======================================================
Generates publication-quality figures from precomputed data:
  1. 3-panel r(x) vs <x_HI> for Fiducial / Early / Late models
  2. Ionization history comparison

Run locally — no cluster access needed.

Output → plots/ folder in working directory.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

# ---------------------------------------------------------------------------
# Simulation parameters (for cluster re-run):
#   BOX_LEN = 500 Mpc, DIM = 800, HII_DIM = 400, random_seed = 1
#
# Precomputed data below from Plotting.ipynb (BOX_LEN=300, DIM=256, HII_DIM=128)
# TODO: update with new simulation results once re-run with larger box
# ---------------------------------------------------------------------------
REDSHIFTS = np.array([4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10],
                     dtype=float)

# Mean neutral fractions (from notebook printed output)
XHI = {
    "Fiducial": np.array([0.031, 0.196, 0.432, 0.631, 0.768, 0.853, 0.904,
                           0.937, 0.959, 0.974, 0.984, 0.991, 0.995]),
    "Early":    np.array([0.000, 0.006, 0.058, 0.196, 0.376, 0.534, 0.654,
                           0.743, 0.809, 0.859, 0.900, 0.930, 0.955]),
    "Late":     np.array([0.679, 0.786, 0.855, 0.901, 0.933, 0.955, 0.971,
                           0.981, 0.988, 0.992, 0.995, 0.997, 0.998]),
}

# Pearson r_real values (read from scatter plot outputs)
# Fiducial: strong anti-correlation at low xHI, crossing zero ~0.85, positive at high xHI
R_REAL = {
    "Fiducial": np.array([-0.58, -0.88, -0.93, -0.92, -0.88, -0.67, -0.14,
                            0.54, 0.85, 0.96, 0.98, 0.99, 1.00]),
    # Early: reionization is earlier, so fully ionized at z=4,4.5
    # anti-correlation persists to higher xHI, crosses zero ~0.82
    "Early":    np.array([-0.06, -0.35, -0.78, -0.88, -0.93, -0.93, -0.90,
                           -0.82, -0.48, 0.19, 0.70, 0.91, 0.97]),
    # Late: only probes high xHI range (0.68–1.0), monotonic rise
    "Late":     np.array([-0.80, -0.58, 0.14, 0.69, 0.88, 0.97, 0.98,
                            0.99, 0.99, 0.99, 1.00, 1.00, 1.00]),
}

# Mean ionized fractions (1 - xHI, but use ionized_fraction from data)
# Read from ionization history plot
XE = {
    "Fiducial": 1.0 - XHI["Fiducial"],
    "Early":    1.0 - XHI["Early"],
    "Late":     1.0 - XHI["Late"],
}

OUTPUT_DIR = Path(__file__).parent / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Figure 1: 3-panel model dependence
# ---------------------------------------------------------------------------
def make_model_dependence_figure():
    mpl.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "font.family": "serif",
    })

    fig, axes = plt.subplots(
        1, 3, figsize=(15, 4.5), sharey=True,
        gridspec_kw={"wspace": 0.05, "right": 0.88},
    )

    models = ["Fiducial", "Early", "Late"]
    titles = ["Fiducial", "Early reionisation", "Late reionisation"]
    vmin, vmax = 4, 10

    for ax, model, title in zip(axes, models, titles):
        xhi = XHI[model]
        r = R_REAL[model]
        z = REDSHIFTS

        # Sort by xHI for connecting line
        idx = np.argsort(xhi)
        xhi, r, z = xhi[idx], r[idx], z[idx]

        ax.plot(xhi, r, color="gray", linestyle="--", alpha=0.4, zorder=1)
        sc = ax.scatter(
            xhi, r, c=z, cmap="viridis", s=60,
            edgecolors="k", linewidths=0.5, zorder=3,
            vmin=vmin, vmax=vmax,
        )
        ax.axhline(0, color="grey", linestyle="-", linewidth=0.5)
        ax.set_xlabel(r"$\langle x_{\mathrm{HI}} \rangle$")
        ax.set_xlim(-0.02, 1.02)
        ax.set_title(title)
        ax.grid(alpha=0.15)

    axes[0].set_ylabel(r"Pearson $r$")

    # Shared colourbar
    cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(sc, cax=cbar_ax)
    cbar.set_label(r"Redshift $z$")

    for fmt in ("pdf", "png"):
        out = OUTPUT_DIR / f"model_dependence_comparison.{fmt}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: Ionization history
# ---------------------------------------------------------------------------
def make_ionization_history_figure():
    mpl.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "font.family": "serif",
    })

    colours = {"Fiducial": "#E69F00", "Early": "#CC3311", "Late": "#0077BB"}
    labels = {"Fiducial": "Fiducial", "Early": "Early reionisation",
              "Late": "Late reionisation"}

    fig, ax = plt.subplots(figsize=(6, 4))

    for model in ["Fiducial", "Early", "Late"]:
        ax.plot(
            REDSHIFTS, XE[model],
            marker="o", markersize=5, color=colours[model],
            label=labels[model], linewidth=1.5,
        )

    ax.set_xlabel(r"Redshift $z$")
    ax.set_ylabel(r"$\langle x_e \rangle$")
    ax.legend(frameon=True, fontsize=10)
    ax.grid(alpha=0.15)
    ax.invert_xaxis()

    for fmt in ("pdf", "png"):
        out = OUTPUT_DIR / f"ionization_history_models.{fmt}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    make_ionization_history_figure()
    make_model_dependence_figure()
    print("Done!")
