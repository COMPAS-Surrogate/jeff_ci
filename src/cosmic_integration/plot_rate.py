import os

import numpy as np
import matplotlib.pyplot as plt
from .ratesSampler import MakeChirpMassBins, NUM_REDSHIFT_BINS, MAX_DETECTION_REDSHIFT
from .utils import read_output, _param_str
import click
from typing import List, Optional


def plot_matrix(matrix:np.ndarray, fname: str= "", params: Optional[List[float]]=None, ax: Optional[plt.Axes]=None, label: str = "") -> None:
    """
    Plot the output data as a heatmap.
    """

    ZbinEdges = np.linspace(0, MAX_DETECTION_REDSHIFT, NUM_REDSHIFT_BINS + 1)
    mc_bin_edge, mc_bin_width = MakeChirpMassBins()
    mc_bin_edge = np.array(mc_bin_edge)

    shape = matrix.shape

    # If no axes object is provided, create a new figure and axes
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.figure  # Get the figure from the provided axes

    output_data_trimmed = matrix[:111, :]  # Shape (111, 15)

    # Plotting using the provided or created axes object
    mesh = ax.pcolormesh(
        ZbinEdges, mc_bin_edge, output_data_trimmed, shading='flat',
        cmap='inferno'
    )
    fig.colorbar(mesh, ax=ax, label='Rate')  # Attach colorbar to the correct axes

    param_str = f"{_param_str(params).replace('_', ' ')}" if params is not None else ""
    shape_str = f'Shape: {shape}'
    title = f'{label}\n{param_str}\n{shape_str}' if label else f'{param_str}\n{shape_str}'

    ax.set_title(title)
    ax.set_xlabel('Redshift Bins')
    ax.set_ylabel('Chirp Masses')
    n_events_per_year = np.nansum(output_data_trimmed)
    ax.annotate(
        f"Grid: {output_data_trimmed.T.shape}\nN det: {n_events_per_year:.2f}/yr",
        xy=(1, 0),
        xycoords="axes fraction",
        xytext=(-5, 5),
        textcoords="offset points",
        ha="right",
        va="bottom",
        color="white",
    )

    # Only save or show if a new figure was created by the function
    if ax is None:  # This condition checks if 'ax' was initially None, implying a new figure was made
        if fname:
            plt.tight_layout()
            plt.savefig(fname)
            plt.close(fig)  # Close the specific figure
            print(f"Plot saved to {fname}")
        else:
            plt.show()




@click.command()
@click.argument('csv_fname', type=click.Path(exists=True))
@click.option('-i', default=0, type=int, help='row index to read from the CSV file')
@click.option('-o', '--outdir', default='out_rate_plots', type=str, help='output dir for the plot')
def main(csv_fname: str, i: int, outdir: str):
    matrix, params, _ = read_output(csv_fname, idx=i)
    print(f"Parameters: {params}")
    print(f"Shape: {matrix.shape}")
    print(f"Row {i} data: {matrix[i]}")

    os.makedirs(outdir, exist_ok=True)
    fname = os.path.join(outdir, f"rate_plot_{_param_str(params)}_row_{i}.png")
    plot_matrix(matrix, fname, params)


if __name__ == "__main__":
    main()
