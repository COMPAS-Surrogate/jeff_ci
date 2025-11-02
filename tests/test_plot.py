import os

import numpy as np

from cosmic_integration.plot_rate import plot_matrix
from cosmic_integration.utils import read_output


def _write_rate_row(matrix: np.ndarray, params: np.ndarray, csv_path: str) -> None:
    row = np.concatenate([
        params.astype(float),
        np.array(matrix.shape, dtype=float),
        matrix.astype(float).ravel(),
    ])
    np.savetxt(csv_path, row.reshape(1, -1), delimiter=",")


def test_plot_matrix(outdir):
    """
    Test the plot_matrix function to ensure it generates a plot without errors.
    """
    params = np.array([-0.5, 0.1, 0.005, 4.2])
    matrix = np.linspace(1, 10, num=113 * 15, dtype=float).reshape(113, 15)

    csv_path = os.path.join(outdir, "test_plot_matrix.csv")
    _write_rate_row(matrix, params, csv_path)

    matrix_loaded, params_loaded, _ = read_output(csv_path, 0)

    # Generate the plot
    plot_matrix(
        matrix_loaded,
        params=params_loaded,
        fname=f"{outdir}/test_plot_matrix.png",
    )

    # Check if the figure and axes are created
    assert os.path.exists(f"{outdir}/test_plot_matrix.png")
