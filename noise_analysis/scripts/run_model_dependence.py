"""
Run 21cmFAST simulations for 3 EoR models + kSZ reconstruction + plots.

New parameters: BOX_LEN=500, DIM=800, HII_DIM=400
Run this on the cluster, not locally.

Usage:
    python run_model_dependence.py
"""

import numpy as np
import logging
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
import sys

import py21cmfast as p21c
sys.path.insert(0, "/Users/jelte/Documents/PhD/Bachelor student/work alex")
from ksz_pipeline import reconstruct_and_correlate

logger = logging.getLogger("py21cmfast")
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BOX_LEN = 500
DIM = 800
HII_DIM = 400
RANDOM_SEED = 1
COEVAL_REDSHIFTS = [4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10]
MPC_TO_KM = 3.085677581e19

CACHE_ROOT = Path("/net/virgo01/data/users/astoica/kSZCache_500Mpc")
OUTPUT_DIR = Path(__file__).parent / "plots"
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Astrophysical parameters
# ---------------------------------------------------------------------------
base_astro = dict(
    HII_EFF_FACTOR=30.0,
    R_BUBBLE_MAX=50.0,
    R_BUBBLE_MIN=0.620350491,
    ION_Tvir_MIN=4.69897,
    L_X=40.5,
    NU_X_THRESH=500.0,
    X_RAY_SPEC_INDEX=1.0,
    X_RAY_Tvir_MIN=4.69897,
    F_H2_SHIELD=0.0,
    A_LW=2.0,
    BETA_LW=0.6,
    A_VCB=1.0,
    BETA_VCB=1.8,
    UPPER_STELLAR_TURNOVER_MASS=11.447,
    UPPER_STELLAR_TURNOVER_INDEX=-0.6,
    SIGMA_STAR=0.25,
    SIGMA_LX=0.5,
    SIGMA_SFR_LIM=0.19,
    SIGMA_SFR_INDEX=-0.12,
    T_RE=2e4,
    FIXED_VAVG=25.86,
    POP2_ION=5000.0,
    POP3_ION=44021.0,
    PHOTONCONS_CALIBRATION_END=3.5,
    CLUMPING_FACTOR=2.0,
    ALPHA_UVB=5.0,
    R_MAX_TS=200.0,
    N_STEP_TS=10,
    MAX_DVDR=0.2,
    DELTA_R_HII_FACTOR=1.1,
    NU_X_BAND_MAX=2000.0,
    NU_X_MAX=10000.0,
)

MODELS = {
    "Fiducial": dict(F_STAR10=-1.42, ALPHA_STAR=0.614, F_ESC10=-1.78,
                     ALPHA_ESC=0.474, M_TURN=8.62, t_STAR=0.392),
    "Early":    dict(F_STAR10=-1.10, ALPHA_STAR=0.40, F_ESC10=-1.40,
                     ALPHA_ESC=0.20, M_TURN=8.20, t_STAR=0.60),
    "Late":     dict(F_STAR10=-1.60, ALPHA_STAR=0.70, F_ESC10=-2.10,
                     ALPHA_ESC=0.30, M_TURN=9.00, t_STAR=0.30),
}


# ---------------------------------------------------------------------------
# 1. Run simulations and cache HDF5
# ---------------------------------------------------------------------------
def run_simulation(model_name):
    print(f"\n{'='*60}")
    print(f"Running {model_name} (BOX_LEN={BOX_LEN}, DIM={DIM}, HII_DIM={HII_DIM})")
    print(f"{'='*60}")

    astro = p21c.AstroParams(**base_astro, **MODELS[model_name])
    inputs = p21c.InputParameters(
        astro_params=astro,
        simulation_options=p21c.SimulationOptions(
            BOX_LEN=BOX_LEN, DIM=DIM, HII_DIM=HII_DIM,
        ),
        random_seed=RANDOM_SEED,
    )

    cache_dir = CACHE_ROOT / model_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = p21c.OutputCache(str(cache_dir))

    coevals = p21c.run_coeval(
        inputs=inputs,
        out_redshifts=COEVAL_REDSHIFTS,
        cache=cache,
        progressbar=True,
    )

    for coeval in coevals:
        z = coeval.redshift
        path = cache_dir / f"z{z:.1f}.h5"
        with h5py.File(path, "w") as f:
            f.create_dataset("density", data=coeval.density)
            f.create_dataset("neutral_fraction", data=coeval.neutral_fraction)
            f.create_dataset("ionized_fraction", data=1.0 - coeval.neutral_fraction)
            f.create_dataset("velocity_z", data=coeval.velocity_z)
        print(f"  Saved {path}")

    return coevals


# ---------------------------------------------------------------------------
# 2. Reconstruction + correlation
# ---------------------------------------------------------------------------
def compute_correlation_curve(model_name):
    print(f"\nComputing correlation curve for {model_name}...")
    cache_dir = CACHE_ROOT / model_name

    xHI_means, r_reals, redshifts = [], [], []
    for z in COEVAL_REDSHIFTS:
        path = cache_dir / f"z{z:.1f}.h5"
        with h5py.File(path, "r") as f:
            density = f["density"][:]
            xhi = f["neutral_fraction"][:]
            vz_raw = f["velocity_z"][:]

        a = 1.0 / (1.0 + z)
        vz_true = a * vz_raw * MPC_TO_KM

        out = reconstruct_and_correlate(density, xhi, vz_true, z, box_len=float(BOX_LEN))
        xHI_means.append(np.mean(xhi))
        r_reals.append(out["r_real"])
        redshifts.append(z)
        print(f"  z={z:.1f}  <xHI>={xHI_means[-1]:.4f}  r={r_reals[-1]:.4f}")

    return np.array(xHI_means), np.array(r_reals), np.array(redshifts)


# ---------------------------------------------------------------------------
# 3. Plots
# ---------------------------------------------------------------------------
def make_plots(results):
    mpl.rcParams.update({
        "font.size": 11, "axes.labelsize": 13,
        "xtick.labelsize": 10, "ytick.labelsize": 10,
        "font.family": "serif",
    })

    # --- 3-panel model dependence ---
    fig, axes = plt.subplots(
        1, 3, figsize=(15, 4.5), sharey=True,
        gridspec_kw={"wspace": 0.05, "right": 0.88},
    )
    titles = {"Fiducial": "Fiducial", "Early": "Early reionisation",
              "Late": "Late reionisation"}
    vmin, vmax = 4, 10

    for ax, model_name in zip(axes, ["Fiducial", "Early", "Late"]):
        xhi, r, z = results[model_name]
        idx = np.argsort(xhi)
        xhi, r, z = xhi[idx], r[idx], z[idx]

        ax.plot(xhi, r, color="gray", linestyle="--", alpha=0.4, zorder=1)
        sc = ax.scatter(xhi, r, c=z, cmap="viridis", s=60,
                        edgecolors="k", linewidths=0.5, zorder=3,
                        vmin=vmin, vmax=vmax)
        ax.axhline(0, color="grey", linestyle="-", linewidth=0.5)
        ax.set_xlabel(r"$\langle x_{\mathrm{HI}} \rangle$")
        ax.set_xlim(-0.02, 1.02)
        ax.set_title(titles[model_name])
        ax.grid(alpha=0.15)

    axes[0].set_ylabel(r"Pearson $r$")
    cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
    fig.colorbar(sc, cax=cbar_ax).set_label(r"Redshift $z$")

    for fmt in ("pdf", "png"):
        fig.savefig(OUTPUT_DIR / f"model_dependence_comparison.{fmt}",
                    dpi=300, bbox_inches="tight")
    print(f"Saved model_dependence_comparison to {OUTPUT_DIR}")
    plt.close(fig)

    # --- Ionization history ---
    colours = {"Fiducial": "#E69F00", "Early": "#CC3311", "Late": "#0077BB"}
    fig, ax = plt.subplots(figsize=(6, 4))
    for model_name in ["Fiducial", "Early", "Late"]:
        xhi = results[model_name][0]
        # Sort by redshift (results are already in COEVAL_REDSHIFTS order)
        ax.plot(COEVAL_REDSHIFTS, 1.0 - xhi, marker="o", markersize=5,
                color=colours[model_name], label=titles[model_name], linewidth=1.5)
    ax.set_xlabel(r"Redshift $z$")
    ax.set_ylabel(r"$\langle x_e \rangle$")
    ax.legend(frameon=True, fontsize=10)
    ax.grid(alpha=0.15)
    ax.invert_xaxis()

    for fmt in ("pdf", "png"):
        fig.savefig(OUTPUT_DIR / f"ionization_history_models.{fmt}",
                    dpi=300, bbox_inches="tight")
    print(f"Saved ionization_history_models to {OUTPUT_DIR}")
    plt.close(fig)

    # --- Print arrays for hardcoding later ---
    print("\n# Copy-paste these into appendix_b_model_dependence.py:")
    for model_name in ["Fiducial", "Early", "Late"]:
        xhi, r, z = results[model_name]
        print(f'    "{model_name}": np.array({np.array2string(xhi, separator=", ", precision=4)}),')
    print()
    for model_name in ["Fiducial", "Early", "Late"]:
        xhi, r, z = results[model_name]
        print(f'    "{model_name}": np.array({np.array2string(r, separator=", ", precision=4)}),')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Step 1: Run simulations
    for model_name in MODELS:
        run_simulation(model_name)

    # Step 2: Reconstruction + correlation
    results = {}
    for model_name in MODELS:
        results[model_name] = compute_correlation_curve(model_name)

    # Step 3: Plots
    make_plots(results)
    print("\nDone!")
