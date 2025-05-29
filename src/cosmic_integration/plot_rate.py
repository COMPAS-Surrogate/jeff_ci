import numpy as np
import matplotlib.pyplot as plt
from cosmic_integration.ratesSampler import MakeChirpMassBins, NUM_REDSHIFT_BINS, MAX_DETECTION_REDSHIFT
import sys


def read_output(fname: str):
    """
    Read the output file and return the parameters, shape, and data.
    # first 4 floats are the parameters, next two floats are the shape (nrows, ncolumns), rest is the data
    """
    data = np.genfromtxt(fname, delimiter=',')

    if data.ndim == 1:
        # If the data is 1D, reshape it to a 2D array with one row
        data = data.reshape(1, -1)

    params = data[0, :4]
    shape = tuple(int(x) for x in data[0, 4:6])
    output_data = data[0, 6:].reshape(shape)

    return params, shape, output_data


def plot_output(params, shape, output_data, fname: str):
    """
    Plot the output data as a heatmap.
    """

    ZbinEdges = np.linspace(0, MAX_DETECTION_REDSHIFT, NUM_REDSHIFT_BINS + 1)
    mc_bin_edge, mc_bin_width = MakeChirpMassBins()
    mc_bin_edge = np.array(mc_bin_edge)  # Convert to numpy array for consistency
    # ditch

    plt.figure(figsize=(6, 5))
    output_data_trimmed = output_data[:111, :]  # Shape (111, 15)
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
    plt.savefig(fname)


def main():
    fname = sys.argv[1] if len(sys.argv) > 1 else 'output.csv'
    params, shape, output_data = read_output(fname)
    print(f"Parameters: {params}")
    print(f"Shape: {shape}")
    print(f"Output Data:\n{output_data}")

    plot_output(params, shape, output_data, fname.replace('.csv', '.png'))


if __name__ == "__main__":
    main()
