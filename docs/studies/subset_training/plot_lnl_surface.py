"""Visualise the 4D COMPAS-scale toy LnL surface (see ``toy_compas_lnl.py``).

Plots every pairwise 2D slice (6 total for 4 params) through the true maximum,
color-mapped on Delta lnL from the peak (clipped at the informative threshold
so the plateau doesn't wash out the peak's shape). Optionally overlays a
training set (e.g. from ``run_bo.py``) so you can see where points landed
relative to the ridge.

Run:  python plot_lnl_surface.py
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from toy_compas_lnl import LO, HI, MU, KEEP_DELTA, PARAMETERS_NAMES, true_lnl

N_GRID = 150
# Show a couple of decades of dynamic range around the peak; anything below
# this is "the floor" and would otherwise dominate the color scale.
VMIN = -5.0 * KEEP_DELTA


def slice_grid(dim_i: int, dim_j: int, n: int = N_GRID) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Delta-lnL on a 2D grid over dims (i, j), other dims fixed at MU."""
    gi = np.linspace(LO[dim_i], HI[dim_i], n)
    gj = np.linspace(LO[dim_j], HI[dim_j], n)
    Z = np.empty((n, n))
    theta = MU.copy()
    for a, vi in enumerate(gi):
        theta[dim_i] = vi
        for b, vj in enumerate(gj):
            theta[dim_j] = vj
            Z[b, a] = true_lnl(theta)
    return gi, gj, Z - Z.max()


def plot_surface(
    training_points: np.ndarray | None = None,
    training_lnl: np.ndarray | None = None,
    out_path: str | Path = "lnl_surface.png",
):
    pairs = list(combinations(range(4), 2))
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    best_lnl = true_lnl(MU)

    for ax, (i, j) in zip(axes.ravel(), pairs):
        gi, gj, Z = slice_grid(i, j)
        im = ax.pcolormesh(gi, gj, Z, vmin=VMIN, vmax=0.0, cmap="viridis", shading="auto")
        ax.contour(gi, gj, Z, levels=[-KEEP_DELTA], colors="white", linewidths=1.0, linestyles="--")
        ax.plot(MU[i], MU[j], "r*", markersize=14, label="truth")

        if training_points is not None:
            delta = training_lnl - best_lnl
            informative = delta > -KEEP_DELTA
            ax.scatter(
                training_points[~informative, i], training_points[~informative, j],
                s=6, c="gray", alpha=0.35, label="floor" if ax is axes.ravel()[0] else None,
            )
            ax.scatter(
                training_points[informative, i], training_points[informative, j],
                s=12, c="orange", edgecolors="k", linewidths=0.3,
                label="informative" if ax is axes.ravel()[0] else None,
            )

        ax.set_xlabel(PARAMETERS_NAMES[i])
        ax.set_ylabel(PARAMETERS_NAMES[j])
        fig.colorbar(im, ax=ax, label="Delta lnL" if (i, j) == pairs[-1] else None)

    axes.ravel()[0].legend(loc="upper right", fontsize=8)
    fig.suptitle(
        f"Toy 4D COMPAS-scale lnL (Delta from peak, dashed line = informative "
        f"threshold at -{KEEP_DELTA:,.0f})"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    plot_surface()
