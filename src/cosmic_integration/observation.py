from dataclasses import dataclass

import numpy as np

from .utils import read_output


@dataclass
class Observation:
    duration: float
    rate_matrix: np.ndarray[float]
    weights: np.ndarray[float] = None
    params: np.ndarray[float] = None

    @classmethod
    def from_jeff(self, fname: str, idx: int = 0) -> 'Observation':
        matrix, params, _ = read_output(fname, idx)
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