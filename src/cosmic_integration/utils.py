from typing import Optional, Tuple

import numpy as np


def read_output(fname: str, idx: int = 0) -> Tuple[np.ndarray, np.ndarray, Optional[float]]:
    """
    Read the output file and return the parameters, shape, and data.
    # first 4 floats are the parameters, next two floats are the shape (nrows, ncolumns), rest is the data
    :return: tuple of (matrix, params, lnl)
    """
    data = np.genfromtxt(fname, delimiter=',')

    if data.ndim == 1:
        # If the data is 1D, reshape it to a 2D array with one row
        data = data.reshape(1, -1)

    return row_to_matrix_params_lnl(data[idx])


def row_to_matrix_params_lnl(row: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Optional[float]]:
    """
    Process a row of data and return the parameters, shape, and matrix.
    """
    params = row[:4]
    shape = tuple(int(x) for x in row[4:6])
    n = shape[0] * shape[1]
    matrix = row[6:6 + n].reshape(shape)
    lnl = None

    # if there is 1 more datapoint, it is the lnL
    if len(row) > 6 + n:
        lnl = row[6 + n]

    return matrix, params, lnl
