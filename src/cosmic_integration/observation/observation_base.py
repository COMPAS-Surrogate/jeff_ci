import abc
from dataclasses import dataclass
from typing import Dict

import h5py
import numpy as np

from ..ratesSampler.binned_cosmic_integrator import get_default_mc_z_bins
from .plotting import plot_event_summaries, plot_weights


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
            duration = f.attrs.get('duration', None)
            if duration is None:
                # Backwards compatibility with legacy files that did not store duration explicitly
                duration = 1.0
            if isinstance(duration, (np.ndarray, list)):
                duration = float(np.asarray(duration).reshape(-1)[0])
            else:
                duration = float(duration)
            params = f.attrs.get('params', None)

            population_weights = None
            mc_bin_edges = None
            mc_bin_widths = None
            z_bin_edges = None
            mc_prior = None
            z_prior = None
            posterior_quantiles = None
            rate_matrix = f['rate_matrix'][:] if 'rate_matrix' in f else None

            if 'population_weights' in f:
                population_weights = f['population_weights'][:]
                mc_bin_edges = f['mc_bin_edges'][:] if 'mc_bin_edges' in f else None
                mc_bin_widths = f['mc_bin_widths'][:] if 'mc_bin_widths' in f else None
                z_bin_edges = f['z_bin_edges'][:] if 'z_bin_edges' in f else None
                mc_prior = f['mc_prior'][:] if 'mc_prior' in f else None
                z_prior = f['z_prior'][:] if 'z_prior' in f else None
                posterior_quantiles = f['posterior_quantiles'][:] if 'posterior_quantiles' in f else None
            elif 'weights' in f:
                # Legacy file layout
                population_weights = f['weights'][:]
                n_events, n_mc_bins, n_z_bins = population_weights.shape
                default_mc_edges, default_mc_widths, default_z_edges = get_default_mc_z_bins()

                # Adjust chirp-mass edges to match the stored weights shape if required
                if len(default_mc_edges) == n_mc_bins:
                    mc_bin_edges = default_mc_edges
                    mc_bin_widths = default_mc_widths
                elif len(default_mc_edges) + 1 == n_mc_bins:
                    # Extend with one more edge using the last known spacing
                    last_width = default_mc_widths[-1] if len(default_mc_widths) > 0 else 1.0
                    extension = default_mc_edges[-1] + last_width
                    mc_bin_edges = np.concatenate([default_mc_edges, [extension]])
                    mc_bin_widths = np.concatenate([default_mc_widths, [last_width]])
                else:
                    # Fall back to evenly spaced edges covering the index range
                    mc_bin_edges = np.linspace(0.5, 0.5 + 0.5 * (n_mc_bins - 1), n_mc_bins)
                    mc_bin_widths = np.full(n_mc_bins, mc_bin_edges[1] - mc_bin_edges[0])

                if len(default_z_edges) == n_z_bins:
                    z_bin_edges = default_z_edges
                else:
                    z_bin_edges = np.linspace(0.0, 0.1 * (n_z_bins - 1), n_z_bins)

                mc_prior = np.full(n_mc_bins, 1.0 / n_mc_bins)
                z_prior = np.full(n_z_bins, 1.0 / n_z_bins)
                posterior_quantiles = np.zeros((n_events, 2, 3), dtype=float)

                if params is None:
                    from ..ratesSampler.ratesSampler import DEFAULT_PARAMS
                    params = np.asarray(DEFAULT_PARAMS, dtype=float)

            if population_weights is not None:
                n_events, n_mc_bins, n_z_bins = population_weights.shape
                default_mc_edges, default_mc_widths, default_z_edges = get_default_mc_z_bins()

                if mc_bin_edges is None:
                    if len(default_mc_edges) == n_mc_bins:
                        mc_bin_edges = default_mc_edges
                    elif len(default_mc_edges) + 1 == n_mc_bins:
                        last_width = default_mc_widths[-1] if len(default_mc_widths) > 0 else 1.0
                        extension = default_mc_edges[-1] + last_width
                        mc_bin_edges = np.concatenate([default_mc_edges, [extension]])
                    else:
                        mc_bin_edges = np.linspace(0.5, 0.5 + 0.5 * (n_mc_bins - 1), n_mc_bins)

                if mc_bin_widths is None:
                    if len(mc_bin_edges) > 1:
                        widths = np.diff(np.concatenate([mc_bin_edges, [mc_bin_edges[-1]]]))
                        if np.allclose(widths, 0):
                            widths = np.full(n_mc_bins, 1.0)
                        mc_bin_widths = widths
                    else:
                        mc_bin_widths = np.full(n_mc_bins, 1.0)

                if z_bin_edges is None:
                    if len(default_z_edges) == n_z_bins:
                        z_bin_edges = default_z_edges
                    else:
                        z_bin_edges = np.linspace(0.0, 0.1 * (n_z_bins - 1), n_z_bins)

                if mc_prior is None:
                    mc_prior = np.full(n_mc_bins, 1.0 / n_mc_bins)

                if z_prior is None:
                    z_prior = np.full(n_z_bins, 1.0 / n_z_bins)

                if posterior_quantiles is None:
                    posterior_quantiles = np.zeros((n_events, 2, 3), dtype=float)

            if population_weights is None:
                raise ValueError(
                    f"Unsupported observation file format for {filepath}. Missing required datasets."
                )

            if params is not None:
                params = np.asarray(params, dtype=float)

            return cls(
                population_weights=population_weights,
                mc_bin_edges=np.asarray(mc_bin_edges, dtype=float) if mc_bin_edges is not None else None,
                mc_bin_widths=np.asarray(mc_bin_widths, dtype=float) if mc_bin_widths is not None else None,
                z_bin_edges=np.asarray(z_bin_edges, dtype=float) if z_bin_edges is not None else None,
                mc_prior=np.asarray(mc_prior, dtype=float) if mc_prior is not None else None,
                z_prior=np.asarray(z_prior, dtype=float) if z_prior is not None else None,
                posterior_quantiles=np.asarray(posterior_quantiles, dtype=float) if posterior_quantiles is not None else None,
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
        def _fmt(value):
            if isinstance(value, str):
                return value
            return f"{value:.3f}"

        mc_bins = self.population_weights.shape[1]
        z_bins = self.population_weights.shape[2]

        mc_edges = np.asarray(self.mc_bin_edges) if self.mc_bin_edges is not None else None
        z_edges = np.asarray(self.z_bin_edges) if self.z_bin_edges is not None else None

        mc_edge_count = len(mc_edges) if mc_edges is not None else "NA"
        z_edge_count = len(z_edges) if z_edges is not None else "NA"

        def _mc_integral():
            if self.mc_prior is None:
                return "NA"
            mc_prior = np.asarray(self.mc_prior, dtype=float)
            mc_widths = None
            if self.mc_bin_widths is not None:
                mc_widths = np.asarray(self.mc_bin_widths, dtype=float)
            elif mc_edges is not None and mc_edges.size > 0:
                left = np.concatenate(([0.0], mc_edges[:-1]))
                mc_widths = mc_edges - left
            if mc_widths is None:
                return float(np.sum(mc_prior))
            if mc_widths.shape[0] < mc_prior.shape[0]:
                pad = mc_prior.shape[0] - mc_widths.shape[0]
                mc_widths = np.concatenate([mc_widths, np.full(pad, mc_widths[-1])])
            elif mc_widths.shape[0] > mc_prior.shape[0]:
                mc_widths = mc_widths[:mc_prior.shape[0]]
            return float(np.sum(mc_prior * mc_widths))

        def _z_integral():
            if self.z_prior is None:
                return "NA"
            z_prior = np.asarray(self.z_prior, dtype=float)
            if z_edges is None or z_edges.size == 0:
                return float(np.sum(z_prior))
            if z_edges.size == 1:
                widths = np.array([0.1], dtype=float)
            else:
                last_step = z_edges[-1] - z_edges[-2]
                widths = np.diff(np.concatenate([z_edges, [z_edges[-1] + last_step]]))
            if widths.shape[0] < z_prior.shape[0]:
                pad = z_prior.shape[0] - widths.shape[0]
                widths = np.concatenate([widths, np.full(pad, widths[-1])])
            elif widths.shape[0] > z_prior.shape[0]:
                widths = widths[:z_prior.shape[0]]
            return float(np.sum(z_prior * widths))

        lines = [
            "MockObservation Summary:",
            f"  Number of events: {self.n_events}",
            f"  Population weights shape: {self.population_weights.shape}",
            f"  MC bins (weights): {mc_bins}; stored edge count: {mc_edge_count}",
            f"  Z bins (weights): {z_bins}; stored edge count: {z_edge_count}",
            f"  MC prior integral: {_fmt(_mc_integral())}",
            f"  Z prior integral: {_fmt(_z_integral())}",
        ]

        if self.params is not None:
            lines.extend(
                [
                    "  Model Parameters:",
                    f"    alpha: {self.params[0]}",
                    f"    sigma: {self.params[1]}",
                    f"    sfr_a: {self.params[2]}",
                    f"    sfr_d: {self.params[3]}",
                ]
            )

        summary_str = "\n".join(lines)

        if self.posterior_quantiles is not None and len(self.posterior_quantiles) > 0:
            max_rows = min(5, self.posterior_quantiles.shape[0])
            summary_str += "\n  Posterior Quantiles (first %d events):\n" % max_rows
            summary_str += "    Event |   z +/- 95%   |  Mc +/- 95% \n"
            summary_str += "    -----------------------------------------------\n"
            for i in range(max_rows):
                z_q = self.posterior_quantiles[i, 0]
                mc_q = self.posterior_quantiles[i, 1]
                z_err = (z_q[2] - z_q[0]) / 2
                mc_err = (mc_q[2] - mc_q[0]) / 2
                summary_str += f"    {i+1:5d} | {z_q[1]:.3f} +/- {z_err:.3f} | {mc_q[1]:.2f} +/- {mc_err:.2f}\n"
            if self.posterior_quantiles.shape[0] > max_rows:
                summary_str += f"    ... ({self.posterior_quantiles.shape[0]} events total)\n"

        return summary_str
