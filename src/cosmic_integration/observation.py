import os
import re
from dataclasses import dataclass
from typing import Optional, Dict

import h5py
import numpy as np

from .utils import read_output


@dataclass
class Observation:
    duration: float
    rate_matrix: Optional[np.ndarray] = None
    weights: Optional[np.ndarray] = None
    params: Optional[np.ndarray] = None

    @classmethod
    def from_jeff(self, fname: str, idx: int = 0) -> 'Observation':

        print(f"Reading observation from {fname} with idx {idx}")

        if "binned_rates_" in fname:
            matrix, params = read_jeff_binned_rate_file(fname)
            duration_days = 273.5
            duration_years = duration_days / 365.25
        else:
            matrix, params, _ = read_output(fname, idx)
            duration_years = 1.0  # Default duration in years for non-Jeff files

        return Observation(
            duration=duration_years,
            rate_matrix=matrix,
            weights=None,  # Weights can be set later if needed
            params=params
        )

    @classmethod
    def from_ilya(cls, fname: str) -> 'Observation':
        with h5py.File(fname, 'r') as f:
            weights = f['weights'][:]
        print(f"Loaded weights from {fname} with shape {weights.shape}")
        return cls(
            duration=273.5/365.25,  # Duration in years
            rate_matrix=None,
            weights=weights,  # Weights can be set later if needed
            params=[-0.325, 0.213, 0.012, 4.253]
        )

    def __post_init__(self):
        if self.rate_matrix is not None:
            if self.rate_matrix.ndim != 2:
                raise ValueError("Rate matrix must be a 2D array.")

            numChirpMassBins, numZBins = self.rate_matrix.shape
            if numChirpMassBins < numZBins:
                raise ValueError(
                    f"Rate matrix must be oriented with (chirp_mass bins, z bins), got shape {self.rate_matrix.shape}.")

        if self.params is None or len(self.params) != 4:
            raise ValueError(f"Parameters must be a list 4 elements, got {self.params}, len = {len(self.params)}.")

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


def _parse_float_list(line):
    return np.array([
        float(x) if x.lower() != "inf" else np.inf
        for x in line.split(",")
    ])


def read_jeff_binned_rate_file(fname: str):
    """
    
    Reads a binned rate file in the format used by Jeff's code.
    
    fname: str binned_rates_alpha-0.325_sigma0.213_asf0.012_dsf4.253.csv
    
    """
    with open(fname, "r") as f:
        lines = f.read().splitlines()

    # param (from the filename)
    fname = os.path.basename(fname)
    param_str = fname.replace("binned_rates_", "").replace(".csv", "")
    params = param_str.split("_")
    # "alpha-0.12", "sigma0.12", "asf0.112", "dsf4.253"
    params = dict(map(lambda s: re.match(r"([a-zA-Z]+)(-?\d*\.?\d+)", s).groups(), params))
    params = {k: float(v) for k, v in params.items()}

    # Parse header
    num_Mc_bins, num_z_bins = map(int, lines[0].split(","))

    # Mc bin right edges
    Mc_right_edges = _parse_float_list(lines[1])
    Mc_widths = _parse_float_list(lines[2])
    Mc_left_edges = Mc_right_edges - Mc_widths
    Mc_left_edges[-1] = Mc_right_edges[-1]
    Mc_edges = np.concatenate((Mc_left_edges, Mc_right_edges[-1:]))

    # z bin right edges
    z_right_edges = _parse_float_list(lines[3])
    z_widths = _parse_float_list(lines[4])
    z_left_edges = z_right_edges - z_widths[0]
    z_edges = np.concatenate((z_left_edges, z_right_edges[-1:]))

    # LOAD DATA BLOCK
    data_lines = lines[5:5 + num_z_bins]
    data = np.array([
        _parse_float_list(line)
        for line in data_lines
    ])

    # # drop the inf bin (last Mc bin)
    # data = data[:, :-2]
    # Mc_edges = Mc_edges[:-2]
    return (data.T, list(params.values()))
