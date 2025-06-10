import matplotlib.pyplot as plt
import numpy as np
from colormath.color_conversions import convert_color
from colormath.color_objects import LabColor, sRGBColor
from matplotlib.colors import (
    ListedColormap,
    LogNorm,
    PowerNorm,
    TwoSlopeNorm,
)
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter


def lab_to_rgb(*args):
    """Convert Lab color to sRGB, with components clipped to (0, 1)."""
    Lab = LabColor(*args)
    sRGB = convert_color(Lab, sRGBColor)
    return np.clip(sRGB.get_value_tuple(), 0, 1)


def get_cylon():
    L_samples = np.linspace(100, 0, 5)

    a_samples = (
        33.34664938,
        98.09940562,
        84.48361516,
        76.62970841,
        21.43276891,
    )

    b_samples = (
        62.73345997,
        2.09003022,
        37.28252236,
        76.22507582,
        16.24862535,
    )

    L = np.linspace(100, 0, 255)
    a = interp1d(L_samples, a_samples[::-1], "cubic")(L)
    b = interp1d(L_samples, b_samples[::-1], "cubic")(L)

    colors = [lab_to_rgb(Li, ai, bi) for Li, ai, bi in zip(L, a, b)]
    cmap = np.vstack(colors)
    return ListedColormap(cmap, name="myColorMap", N=cmap.shape[0])


def get_colors_from_cmap(
        vals, cmap="Blues", min_val=None, mid_val=None, max_val=None
):
    if min_val is not None:
        norm = TwoSlopeNorm(vmin=min_val, vcenter=mid_val, vmax=max_val)
        vals = norm(vals)
    cmap = plt.get_cmap(cmap)
    color = cmap(vals)
    return color


def get_top_color_of_cmap(cmap):
    return cmap(np.linspace(0, 1, 256))[-1]


CMAP = get_cylon()
CTOP = get_top_color_of_cmap(CMAP)

Mc = "srcmchirp"
Z = "z"

MC_LATEX = r"$\mathcal{M}_{\rm src}\ [M_{\odot}]$"
Z_LATEX = r"$z$"


def _fmt_ax(ax, bounds=None):
    ax.set_xlabel(Z_LATEX)
    ax.set_ylabel(MC_LATEX)
    if bounds:
        ax.set_xlim(bounds[Z])
        ax.set_ylim(bounds[Mc])


def _get_norm(x):
    log_x = np.log(x)
    log_x = log_x[np.isfinite(log_x)]
    if len(log_x) == 0:
        return LogNorm(vmin=0.1, vmax=1)
    vmin, vmax = np.exp(log_x.min()), x.max()
    # return LogNorm(vmin=np.exp(log_x.min()), vmax=x.max())
    return PowerNorm(gamma=0.3, vmin=vmin / 10, vmax=vmax * 3)


def plot_weights(weights: np.ndarray, mc_bins, z_bins, ax=None, contour=True):
    if ax is None:
        fig, ax = plt.subplots()
    _fmt_ax(
        ax, {Mc: [min(mc_bins), max(mc_bins)], Z: [min(z_bins), max(z_bins)]}
    )
    cmp = ax.pcolor(
        z_bins, mc_bins, weights.T, cmap=CMAP, norm=_get_norm(weights)
    )

    if contour:
        Zb, MCb = np.meshgrid(z_bins, mc_bins)
        ax.contour(
            Zb,
            MCb,
            weights.T,
            levels=1,
            colors="tab:orange",
            linewidths=[0, 2],
            alpha=0.1,
        )
    # add colorbar above the axes
    fig = ax.get_figure()
    cbar = fig.colorbar(cmp, ax=ax, orientation="vertical")
    cbar.set_label(r"$w_{z,\mathcal{M}_{\rm src}}$")
    return ax


def add_cntr(ax, X, Y, Z, color=CTOP):
    ax.contour(
        X,
        Y,
        gaussian_filter(Z, 1.2).T,
        levels=1,
        colors=color,
        linewidths=[0, 2],
        alpha=0.1,
    )
