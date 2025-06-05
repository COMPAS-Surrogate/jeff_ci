from dataclasses import dataclass

import numpy as np

from typing import Optional, Dict
from .utils import read_output


@dataclass
class Observation:
    duration: float
    rate_matrix: np.ndarray
    weights: Optional[np.ndarray] = None
    params: Optional[np.ndarray] = None

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
    def param_dict(self) -> Dict[str, float]:
        """
        Returns a dictionary of parameters with keys 'alpha', 'sigma', 'sfr_a', and 'sfr_d'.
        Raises ValueError if parameters are not set or insufficient length.
        """
        if self.params is None or len(self.params) < 4:
            raise ValueError("Parameters are not set or insufficient length. Expected at least 4 parameters.")
        return {
            'alpha': self.params[0],
            'sigma': self.params[1],
            'sfr_a': self.params[2],
            'sfr_d': self.params[3]
        }