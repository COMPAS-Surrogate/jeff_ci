from cosmic_integration.plot_rate import plot_matrix
from cosmic_integration.utils import read_output
import os

def test_plot_matrix(observation_file, outdir):
    """
    Test the plot_matrix function to ensure it generates a plot without errors.
    """
    from cosmic_integration import ratesSampler

    # Load the observation
    matrix, params, _ = read_output(observation_file, 0)

    # Generate the plot
    plot_matrix(
        matrix,
        params=params,
        fname=f"{outdir}/test_plot_matrix.png",
    )

    # Check if the figure and axes are created
    os.path.exists(f"{outdir}/test_plot_matrix.png")