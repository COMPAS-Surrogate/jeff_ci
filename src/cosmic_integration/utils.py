import numpy as np


def read_output(fname: str, idx: int = 0) -> tuple:
    """
    Read the output file and return the parameters, shape, and data.
    # first 4 floats are the parameters, next two floats are the shape (nrows, ncolumns), rest is the data
    """
    data = np.genfromtxt(fname, delimiter=',')

    if data.ndim == 1:
        # If the data is 1D, reshape it to a 2D array with one row
        data = data.reshape(1, -1)

    params = data[idx, :4]
    shape = tuple(int(x) for x in data[idx, 4:6])
    n = shape[0] * shape[1]
    matrix = data[idx, 6:6 + n].reshape(shape)
    lnl = None

    # if there is 1 more datapoint, it is the lnL
    if data.shape[1] > 6 + n:
        lnl = data[idx, 6 + n]

    return matrix, params, lnl
