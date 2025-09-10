import os

import numpy as np
import matplotlib.pyplot as plt
from .ratesSampler import MakeChirpMassBins, REDSHIFT_STEP, MAX_DETECTION_REDSHIFT, MAX_CHIRPMASS
from .utils import read_output, _param_str
import click
from typing import List, Optional

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




def _lab_to_rgb(*args):
    """Convert Lab color to sRGB, with components clipped to (0, 1)."""
    Lab = LabColor(*args)
    sRGB = convert_color(Lab, sRGBColor)
    return np.clip(sRGB.get_value_tuple(), 0, 1)


def _get_cylon():
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

    colors = [_lab_to_rgb(Li, ai, bi) for Li, ai, bi in zip(L, a, b)]
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


CMAP = _get_cylon()
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




def _add_cntr(ax, X, Y, Z, color=CTOP):
    ax.contour(
        X,
        Y,
        gaussian_filter(Z, 1.2).T,
        levels=1,
        colors=color,
        linewidths=[0, 2],
        alpha=0.1,
    )





def plot_matrix(
        matrix:np.ndarray,
        fname: str= "",
        params: Optional[List[float]]=None,
        ax: Optional[plt.Axes]=None,
        label: str = "",
        n_events=None
) -> None:
    """
    Plot the output data as a heatmap.
    """

    z_bins = np.arange(0.0, MAX_DETECTION_REDSHIFT + REDSHIFT_STEP, REDSHIFT_STEP)[:15]
    mc_bins, mc_bin_width = MakeChirpMassBins()
    mc_bins = mc_bins + [MAX_CHIRPMASS+10]  # Add an extra "Collection" bin to the end


    # If no axes object is provided, create a new figure and axes
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
        save_plot = True
    else:
        fig = ax.figure  # Get the figure from the provided axes
        save_plot = False

    _fmt_ax(
        ax, {Mc: [min(mc_bins), max(mc_bins)], Z: [min(z_bins), max(z_bins)]}
    )

    # Plotting using the provided or created axes object
    mesh = ax.pcolor(
        z_bins, mc_bins, matrix,
        norm=_get_norm(matrix),
        cmap=CMAP,
    )
    fig.colorbar(mesh, ax=ax)  # Attach colorbar to the correct axes

    param_str = f"{_param_str(params).replace('_', ' ')}" if params is not None else ""
    title = f'{label}\n{param_str}' if label else f'{param_str}'

    ax.set_title(title)
    if n_events is None:
        n_events = np.nansum(matrix)
    ax.annotate(
        f"Grid: {matrix.T.shape}\nN det: {n_events:.2f}/yr",
        xy=(1, 0),
        xycoords="axes fraction",
        xytext=(-5, 5),
        textcoords="offset points",
        ha="right",
        va="bottom",
        color="black",
    )
    ax.set_yscale('log')  # Set y-axis to logarithmic scale


    if save_plot:
        plt.tight_layout()
        plt.savefig(fname)
        plt.close(fig)  # Close the specific figure
        print(f"Plot saved to {fname}")




@click.command()
@click.argument('csv_fname', type=click.Path(exists=True))
@click.option('-i', default=0, type=int, help='row index to read from the CSV file')
@click.option('-o', '--outdir', default='out_rate_plots', type=str, help='output dir for the plot')
def main(csv_fname: str, i: int, outdir: str):
    matrix, params, _ = read_output(csv_fname, idx=i)
    print(f"Parameters: {params}, Shape: {matrix.shape}, Row {i} ")

    os.makedirs(outdir, exist_ok=True)
    fname = os.path.join(outdir, f"rate_plot_{_param_str(params)}_row_{i}.png")
    plot_matrix(matrix, fname, params)


if __name__ == "__main__":
    main()
