import numpy as np
import matplotlib.pyplot as plt
from .ratesSampler import MakeChirpMassBins, NUM_REDSHIFT_BINS, MAX_DETECTION_REDSHIFT
import sys
from typing import List


def plot_matrix(matrix:np.ndarray, fname: str= "", params:List[float]=None):
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
        cmap='inferno',
        norm="linear",
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
    else:
        plt.show()


def main():
    fname = sys.argv[1] if len(sys.argv) > 1 else 'output.csv'
    params, shape, output_data = read_output(fname)
    print(f"Parameters: {params}")
    print(f"Shape: {shape}")
    print(f"Output Data:\n{output_data}")

    plot_matrix(params, shape, output_data, fname.replace('.csv', '.png'))


if __name__ == "__main__":
    main()
