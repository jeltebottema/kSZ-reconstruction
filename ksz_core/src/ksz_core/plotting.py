"""Shared matplotlib defaults — paper and exploration figures share style."""
from __future__ import annotations


def set_default_style() -> None:
    """Apply project-default matplotlib rcParams. Idempotent."""
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 200,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })
