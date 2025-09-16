import abc
from dataclasses import dataclass
import numpy as np
import h5py
from typing import Dict
from .plotting import plot_weights, plot_event_summaries


@dataclass
class ObservationBase(abc.ABC):
    population_weights: np.ndarray  # shape (n_events, n_mc_bins, n_z_bins)
    mc_bin_edges: np.ndarray  # right edges of MC bins
    mc_bin_widths: np.ndarray  # widths of MC bins
    z_bin_edges: np.ndarray  # left edges of Z bins (+ implicit right edge)
    mc_prior: np.ndarray  # prior density in each MC bin
    z_prior: np.ndarray  # prior density in each Z bin
    posterior_quantiles: np.ndarray  # shape (n_events, 2, 3)
    duration: float
    params: np.ndarray = None
    rate_matrix: np.ndarray = None

    def save_h5(self, filepath: str):
        with h5py.File(filepath, 'w') as f:
            f.create_dataset('population_weights', data=self.population_weights)
            f.create_dataset('mc_bin_edges', data=self.mc_bin_edges)
            f.create_dataset('mc_bin_widths', data=self.mc_bin_widths)
            f.create_dataset('z_bin_edges', data=self.z_bin_edges)
            f.create_dataset('mc_prior', data=self.mc_prior)
            f.create_dataset('z_prior', data=self.z_prior)
            f.create_dataset('posterior_quantiles', data=self.posterior_quantiles)

            # if rate_matrix is not None, save it
            if self.rate_matrix is not None:
                f.create_dataset('rate_matrix', data=self.rate_matrix)

            # Add metadata
            f.attrs['n_events'] = self.population_weights.shape[0]
            f.attrs['n_z_bins'] = len(self.z_bin_edges)
            f.attrs['n_mc_bins'] = len(self.mc_bin_edges)
            f.attrs['duration'] = self.duration
            if self.params is not None:
                f.attrs['params'] = self.params

        print(f"Saved Observation to {filepath}")
        print(f"Shape: {self.population_weights.shape} (n_events, n_mc_bins, n_z_bins)")

    @classmethod
    def load_h5(cls, filepath: str):
        with h5py.File(filepath, 'r') as f:
            duration = f.attrs['duration']
            params = f.attrs.get('params', None)
            if 'rate_matrix' in f:
                rate_matrix = f['rate_matrix'][:]
            else:
                rate_matrix = None

            return cls(
                population_weights=f['population_weights'][:],
                mc_bin_edges=f['mc_bin_edges'][:],
                mc_bin_widths=f['mc_bin_widths'][:],
                z_bin_edges=f['z_bin_edges'][:],
                mc_prior=f['mc_prior'][:],
                z_prior=f['z_prior'][:],
                posterior_quantiles=f['posterior_quantiles'][:],
                duration=duration,
                params=params,
                rate_matrix=rate_matrix
            )

    def plot(self, *args, **kwargs):
        # Swap axes for plotting: prior_2d shape (n_mc_bins, n_z_bins)
        return plot_weights(
            prior_2d=np.outer(self.mc_prior, self.z_prior),
            population_weights=self.population_weights,
            *args, **kwargs
        )

    def plot_event_summaries(self, *args, **kwargs):
        return plot_event_summaries(
            self.posterior_quantiles,
            *args, **kwargs
        )

    @property
    def n_events(self):
        return self.population_weights.shape[0]

    @property
    def normalized_weights(self):
        weights_norm = self.population_weights.copy()
        for i in range(len(weights_norm)):
            if np.sum(weights_norm[i]) > 0:
                weights_norm[i] = weights_norm[i] / np.sum(weights_norm[i])
        return weights_norm

    def __post_init__(self):
        if self.rate_matrix is not None:
            if self.rate_matrix.ndim != 2:
                raise ValueError("Rate matrix must be a 2D array.")
            n_mc_bins, n_z_bins = self.rate_matrix.shape
            # mc bins should be more than z bins
            if n_z_bins > n_mc_bins:
                raise ValueError(f"Rate matrix shape invalid: {self.rate_matrix.shape}. Expected (n_mc_bins, n_z_bins) with n_mc_bins >= n_z_bins.")

        if self.params is not None:
            if len(self.params) != 4:
                raise ValueError(f"Parameters must be a list 4 elements, got {self.params}, len = {len(self.params)}.")

        weights = np.zeros_like(self.population_weights)

        # ensure every event has normalized weights
        idx_to_drop = []
        for i in range(len(self.population_weights)):
            if np.sum(self.population_weights[i]) > 0:
                weights[i] = self.population_weights[i] / np.sum(self.population_weights[i])

            # if any event has all zero weights, drop the event and warn
            if np.sum(self.population_weights[i]) == 0:
                idx_to_drop.append(i)

        if len(idx_to_drop) > 0:
            weights = np.delete(weights, idx_to_drop, axis=0)
            self.posterior_quantiles = np.delete(self.posterior_quantiles, idx_to_drop, axis=0)
            print(f"Dropped {len(idx_to_drop)} events with all zero weights. New number of events: {weights.shape[0]}")
        self.population_weights = weights



    @property
    def param_dict(self) -> Dict:
        if self.params is None or len(self.params) < 4:
            raise ValueError("Parameters are not set or insufficient length. Expected at least 4 parameters.")
        return {
            'alpha': self.params[0],
            'sigma': self.params[1],
            'sfr_a': self.params[2],
            'sfr_d': self.params[3]
        }


    def summary(self) -> str:
        """Return a summary string of the MockObservation."""
        summary_str = (
            f"MockObservation Summary:\n"
            f"  Number of events: {self.n_events}\n"
            f"  Number of MC bins: {len(self.mc_bin_edges)}\n"
            f"  Number of Z bins: {len(self.z_bin_edges)}\n"
            f"  Population weights shape: {self.population_weights.shape}\n"
            f"  MC prior sum: {np.sum(self.mc_prior)}\n"
            f"  Z prior sum: {np.sum(self.z_prior)}\n"
        )
        if self.params is not None:
            summary_str += (
                f"  Model Parameters:\n"
                f"    alpha: {self.params[0]}\n"
                f"    sigma: {self.params[1]}\n"
                f"    sfr_a: {self.params[2]}\n"
                f"    sfr_d: {self.params[3]}\n"
            )

        # table of z, mc quantiles
        if self.posterior_quantiles is not None:
            summary_str += "  Posterior Quantiles :\n"
            summary_str += "    Event |   z +/- 95%   |  Mc +/- 95% \n"
            summary_str += "    -----------------------------------------------\n"
            for i in range(self.n_events):
                z_q = self.posterior_quantiles[i, 0]
                mc_q = self.posterior_quantiles[i, 1]
                z_err = (z_q[2] - z_q[0]) / 2
                mc_err = (mc_q[2] - mc_q[0]) / 2
                summary_str += f"    {i+1:5d} | {z_q[1]:.3f} +/- {z_err:.3f} | {mc_q[1]:.2f} +/- {mc_err:.2f}\n"
        return summary_str
