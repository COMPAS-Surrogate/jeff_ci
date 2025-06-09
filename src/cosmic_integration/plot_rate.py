import os

import numpy as np
import matplotlib.pyplot as plt
from .ratesSampler import MakeChirpMassBins, NUM_REDSHIFT_BINS, MAX_DETECTION_REDSHIFT
from .utils import read_output
import click
from typing import List, Optional


def plot_matrix(matrix:np.ndarray, fname: str= "", params: Optional[List[float]]=None):
    """
    Plot the output data as a heatmap.
    """

    ZbinEdges = np.linspace(0, MAX_DETECTION_REDSHIFT, NUM_REDSHIFT_BINS + 1)
    mc_bin_edge, mc_bin_width = MakeChirpMassBins()
    mc_bin_edge = np.array(mc_bin_edge)  # Convert to numpy array for consistency
    # ditch

    shape = matrix.shape

    plt.figure(figsize=(6, 5))
    output_data_trimmed = matrix[:111, :]  # Shape (111, 15)
    plt.pcolormesh(
        ZbinEdges, mc_bin_edge, output_data_trimmed, shading='flat',
        cmap='inferno'
    )
    plt.colorbar(label='Output Value')
    plt.title(f'Output Data Heatmap\nParameters: {params}\nShape: {shape}')
    plt.xlabel('Redshift Bins')
    plt.ylabel('Chirp Masses')
    n_events_per_year = np.nansum(output_data_trimmed)
    plt.annotate(
        f"Grid: {output_data_trimmed.T.shape}\nN det: {n_events_per_year:.2f}/yr",
        xy=(1, 0),
        xycoords="axes fraction",
        xytext=(-5, 5),
        textcoords="offset points",
        ha="right",
        va="bottom",
        color="white",
    )
    if fname:
        plt.tight_layout()
        plt.savefig(fname)
        plt.close()
        print(f"Plot saved to {fname}")
    else:
        plt.show()




@click.command()
@click.argument('csv_fname', type=click.Path(exists=True))
@click.option('-i', default=0, type=int, help='row index to read from the CSV file')
@click.option('-o', '--outdir', default='out_rate_plots', type=str, help='output dir for the plot')
def main(csv_fname: str, i: int, outdir: str):
    matrix, params, _ = read_output(csv_fname)
    print(f"Parameters: {params}")
    print(f"Shape: {matrix.shape}")

    os.makedirs(outdir, exist_ok=True)
    param_str = "_".join([f"{p:.3f}" for p in params])
    fname = os.path.join(outdir, f"rate_plot_{param_str}_row_{i}.png")
    plot_matrix(matrix, fname, params)


if __name__ == "__main__":
    main()
