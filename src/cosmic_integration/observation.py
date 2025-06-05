from dataclasses import dataclass

import numpy as np


@dataclass
class Observation:
    duration: float
    rate_matrix: np.ndarray[float]
    weights: np.ndarray[float] = None
    params: np.ndarray[float] = None

    @classmethod
    def from_jeff(self, fname: str, idx: int = 0) -> 'Observation':
        data = np.genfromtxt(fname, delimiter=',')

        if data.ndim == 1:
            # If the data is 1D, reshape it to a 2D array with one row
            data = data.reshape(1, -1)

        params = data[idx, :4]
        shape = tuple(int(x) for x in data[idx, 4:6])
        ndata = shape[0] * shape[1]

        matrix = data[idx, 6:6 + ndata].reshape(shape)
        return Observation(
            duration=1,
            rate_matrix=matrix,
            weights=None,  # Weights can be set later if needed
            params=params
        )

    @property
    def param_dict(self):
        return {
            'alpha': self.params[0],
            'sigma': self.params[1],
            'sfr_a': self.params[2],
            'sfr_d': self.params[3]
        }