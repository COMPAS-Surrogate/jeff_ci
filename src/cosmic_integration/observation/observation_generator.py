import numpy as np
import h5py
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional
import warnings
import os
from scipy.interpolate import interp1d
from tqdm import tqdm
import tempfile

import numpy as np
import h5py
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, Tuple, List
import warnings
import os
from scipy.interpolate import interp1d
from tqdm import tqdm
from matplotlib.ticker import ScalarFormatter

from ..ratesSampler import get_default_mc_z_bins


MC_BIN_R_EDGE, MC_BIN_WDT, Z_BIN_L_EDGE = get_default_mc_z_bins()


@dataclass
class MockObservation:
    population_weights: np.ndarray  # shape (n_events, n_z_bins, n_mc_bins)
    mc_bin_edges: np.ndarray  # right edges of MC bins
    mc_bin_widths: np.ndarray  # widths of MC bins
    z_bin_edges: np.ndarray  # left edges of Z bins (+ implicit right edge)
    mc_prior: np.ndarray  # prior density in each MC bin
    z_prior: np.ndarray  # prior density in each Z bin
    posterior_quantiles: np.ndarray # shape (n_events, 2, 3)


    def __post_init__(self):
        print(self.summary())

    @classmethod
    def generate_from_rates(cls,
                            rates: np.ndarray,
                            chirp_masses: np.ndarray,
                            mc_bin_edges: np.ndarray = MC_BIN_R_EDGE,  # right edges
                            mc_bin_widths: np.ndarray = MC_BIN_WDT,  # bin widths
                            z_bin_edges: np.ndarray = Z_BIN_L_EDGE,  # left edges
                            n_samples: int = int(1e6),
                            n_posterior_samples: int = int(1e4),
                            output_file: Optional[str] = None):
        """
        Generate MockObservation from detection rates.

        Parameters:
        -----------
        rates : np.ndarray
            Detection rates matrix
        chirp_masses : np.ndarray
            Chirp mass array
        mc_bin_edges : np.ndarray
            Right edges of chirp mass bins
        mc_bin_widths : np.ndarray
            Widths of chirp mass bins
        z_bin_edges : np.ndarray
            Left edges of redshift bins
        n_samples : int
            Number of samples for prior generation
        n_posterior_samples : int
            Maximum number of posterior samples per event
        output_file : str, optional
            If provided, save to this HDF5 file
        """

        print(f"Total rate sum: {np.sum(rates)}")
        print(f"Maximum rate: {np.max(rates)}")
        if np.max(rates) >= 1:
            warnings.warn(f"Rates should be probabilities (max < 1), but max={np.max(rates)}")

        n_mc, n_z = rates.shape
        print(f"Chirp mass array length: {len(chirp_masses)}, Rate matrix Mc dimension: {n_mc}")
        print(f"Number of MC bins: {len(mc_bin_edges)}, Number of Z bins: {len(z_bin_edges)}")

        # Sample detected events
        mc_found, z_found = cls._sample_detected_events(rates, chirp_masses, len(z_bin_edges))
        print(f"Number of detected events: {len(mc_found)}")

        # Generate priors
        mc_prior, z_prior = cls._generate_priors(
            n_samples, mc_bin_edges, mc_bin_widths, z_bin_edges
        )

        # Generate population weights
        population_weights, posterior_quantiles = cls._generate_population_weights(
            mc_found, z_found, mc_bin_edges, mc_bin_widths, z_bin_edges,
            mc_prior, z_prior, n_posterior_samples
        )

        # Create observation object
        observation = cls(
            population_weights=population_weights,
            mc_bin_edges=mc_bin_edges,
            mc_bin_widths=mc_bin_widths,
            z_bin_edges=z_bin_edges,
            mc_prior=mc_prior,
            z_prior=z_prior,
            posterior_quantiles=posterior_quantiles
        )

        # Save if requested
        if output_file:
            observation.save_h5(output_file)

        return observation

    @classmethod
    def load_h5(cls, filepath: str):
        """Load MockObservation from HDF5 file."""
        with h5py.File(filepath, 'r') as f:
            posterior_quantiles = f['posterior_quantiles'][:] if 'posterior_quantiles' in f else None
            return cls(
                population_weights=f['population_weights'][:],
                mc_bin_edges=f['mc_bin_edges'][:],
                mc_bin_widths=f['mc_bin_widths'][:],
                z_bin_edges=f['z_bin_edges'][:],
                mc_prior=f['mc_prior'][:],
                z_prior=f['z_prior'][:],
                posterior_quantiles=posterior_quantiles
            )

    def save_h5(self, filepath: str):
        """Save MockObservation to HDF5 file."""
        with h5py.File(filepath, 'w') as f:
            f.create_dataset('population_weights', data=self.population_weights)
            f.create_dataset('mc_bin_edges', data=self.mc_bin_edges)
            f.create_dataset('mc_bin_widths', data=self.mc_bin_widths)
            f.create_dataset('z_bin_edges', data=self.z_bin_edges)
            f.create_dataset('mc_prior', data=self.mc_prior)
            f.create_dataset('z_prior', data=self.z_prior)
            if self.posterior_quantiles is not None:
                f.create_dataset('posterior_quantiles', data=self.posterior_quantiles)

            # Add metadata
            f.attrs['n_events'] = self.population_weights.shape[0]
            f.attrs['n_z_bins'] = len(self.z_bin_edges)
            f.attrs['n_mc_bins'] = len(self.mc_bin_edges)

        print(f"Saved MockObservation to {filepath}")
        print(f"Shape: {self.population_weights.shape} (n_events, n_z_bins, n_mc_bins)")

    def plot(self, figsize=(8, 4.5), cmap='Blues', fname=None,
             show_bin_edges=False, bin_alpha=0.3, bin_lw=0.5, bin_color='white'):
        """
        Plot prior and sum of normalized weights using imshow.

        Parameters:
        -----------
        figsize : tuple
            Figure size
        cmap : str
            Colormap for plots
        fname : str, optional
            Filename to save plot
        show_bin_edges : bool
            Whether to show bin edges as lines
        bin_alpha : float
            Transparency of bin edge lines (0-1)
        bin_lw : float
            Line width of bin edges
        bin_color : str
            Color of bin edge lines
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Create bin centers
        mc_centers = self._get_mc_bin_centers()
        z_centers = self._get_z_bin_centers()

        # Create 2D prior by outer product
        prior_2d = np.outer(self.z_prior, self.mc_prior)

        # Normalize each event's weights and sum
        weights_normalized = self.population_weights.copy()
        for i in range(len(weights_normalized)):
            if np.sum(weights_normalized[i]) > 0:
                weights_normalized[i] = weights_normalized[i] / np.sum(weights_normalized[i])

        weights_sum = np.sum(weights_normalized, axis=0)
        z_left_edges = self.z_bin_edges
        mc_left_edges = self.mc_bin_edges - self.mc_bin_widths
        mc_centers = self._get_mc_bin_centers()

        prior_2d_log = np.log10(np.clip(prior_2d.T, 1e-10, None))
        weights_sum_log = np.log10(np.clip(weights_sum.T, 1e-10, None))
        # Plot prior
        axes[0].pcolormesh(z_left_edges, mc_left_edges, prior_2d_log, cmap=cmap, shading='auto')
        axes[1].pcolormesh(z_left_edges, mc_left_edges, weights_sum_log, cmap=cmap, shading='auto')
        axes[0].set_title("Prior Density")
        axes[1].set_title("Sum Weights(events)")
        for ax in axes:
            ax.set_xlabel("Redshift")
            ax.set_ylabel("Chirp Mass [M☉]")
            _fmt_yaxes(ax)




        plt.tight_layout()

        if fname:
            fig.savefig(fname, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {fname}")

        return fig, axes

    def plot_event_summaries(self, color="C0", figsize=(10, 6), fname=None, **kwgs):
        """
        Plot per-event redshift and chirp mass medians with 16/84% uncertainties.

        Only events with non-zero total weight are plotted. Returns (fig, axes, plotted_idx)
        where `plotted_idx` maps plotted rows back to original event indices.
        """
        posterior_quantiles = self.posterior_quantiles
        n_events = self.n_events
        y = np.arange(n_events)


        # get errors
        zqtl = posterior_quantiles[:, 0]  # shape (n_events, 3)
        mcqtl = posterior_quantiles[:, 1]  # shape (n_events, 3)
        zerr = np.abs(zqtl - zqtl[:, 1].reshape(-1, 1))[:, [0, 2]].T
        mcerr = np.abs(mcqtl - mcqtl[:, 1].reshape(-1, 1))[:, [0, 2]].T


        fig, axes = plt.subplots(1, 2, figsize=(10, n_events * 0.3), sharey=True)

        axes[0].errorbar(
            zqtl[:, 1], y, xerr=zerr, fmt='o', color='black', **kwgs
        )
        axes[1].errorbar(
            mcqtl[:,1], y, xerr=mcerr, fmt='o', color='black', **kwgs
        )
        # labels
        axes[0].set_xlabel("Redshift")
        axes[1].set_xlabel("Chirp Mass [M☉]")

        # styling
        axes[0].set_ylim(-0.5, n_events - 0.5)
        axes[0].set_xlim(0, 1)
        axes[0].set_xticks([0, 0.5, 0.8])
        axes[1].set_xticks([5, 30, 70])

        if fname:
            plt.savefig(fname, dpi=300, bbox_inches='tight')
            print(f"Saved event summary plot to {fname}")

        return axes


    def _get_mc_bin_centers(self):
        """Get chirp mass bin centers."""
        mc_left_edges = self.mc_bin_edges - self.mc_bin_widths
        return (mc_left_edges + self.mc_bin_edges) / 2

    def _get_z_bin_centers(self):
        """Get redshift bin centers."""
        z_widths = np.diff(
            np.concatenate([self.z_bin_edges, [self.z_bin_edges[-1] + (self.z_bin_edges[-1] - self.z_bin_edges[-2])]]))
        return self.z_bin_edges + z_widths / 2

    @property
    def n_events(self):
        """Number of detected events."""
        return self.population_weights.shape[0]

    @property
    def normalized_weights(self):
        """Return population weights normalized per event."""
        weights_norm = self.population_weights.copy()
        for i in range(len(weights_norm)):
            if np.sum(weights_norm[i]) > 0:
                weights_norm[i] = weights_norm[i] / np.sum(weights_norm[i])
        return weights_norm

    @staticmethod
    def _sample_detected_events(rates: np.ndarray, mc: np.ndarray, n_z: int):
        """Sample detected events based on detection rates."""
        mc_found = []
        z_found = []

        for i in range(n_z):
            r = np.random.rand(len(mc))
            detected_indices = np.where(r < rates[:, i])[0]
            mc_found.extend(mc[detected_indices])
            z = (i) * 0.1 + 0.05  # redshift bin center
            z_found.extend([z] * len(detected_indices))

        return np.array(mc_found), np.array(z_found)

    @staticmethod
    def _generate_priors(n_samples: int, mc_bin_edges: np.ndarray,
                         mc_bin_widths: np.ndarray, z_bin_edges: np.ndarray):
        """Generate prior density arrays for chirp mass and redshift bins."""

        # Generate chirp mass prior
        m1 = 1 + 999 * np.random.rand(n_samples)
        m2 = 1 + 999 * np.random.rand(n_samples)
        q = m2 / m1
        mc_prior_samples = (m1 ** 0.6 * m2 ** 0.6) / (m1 + m2) ** 0.2

        # Apply cuts for physical binary systems
        mc_cut = (q > 0.05) & (q < 1) & (mc_prior_samples > 1) & (mc_prior_samples < 200)
        mc_prior_samples = mc_prior_samples[mc_cut]

        # Count samples in each MC bin
        mc_prior_counts = np.zeros(len(mc_bin_edges))
        for mc in mc_prior_samples:
            bin_idx = MockObservation._chirp_mass_bin(mc, mc_bin_edges)
            if 0 <= bin_idx < len(mc_bin_edges):
                mc_prior_counts[bin_idx] += 1

        # Convert to density (counts / (total_counts * bin_width))
        mc_prior_density = mc_prior_counts / (np.sum(mc_prior_counts) * mc_bin_widths)

        # Generate redshift prior
        z_prior_samples = 1.5 * np.random.rand(int(1e6)) ** (1 / 3)

        # Count samples in each Z bin
        z_bin_widths = np.diff(np.concatenate([z_bin_edges, [z_bin_edges[-1] + (z_bin_edges[-1] - z_bin_edges[-2])]]))
        z_prior_counts = np.zeros(len(z_bin_edges))

        for z in z_prior_samples:
            bin_idx = MockObservation._redshift_bin(z, z_bin_edges)
            if 0 <= bin_idx < len(z_bin_edges):
                z_prior_counts[bin_idx] += 1

        # Convert to density
        z_prior_density = z_prior_counts / (np.sum(z_prior_counts) * z_bin_widths)

        return mc_prior_density, z_prior_density

    @staticmethod
    def _chirp_mass_bin(chirp_mass: float, chirp_mass_bins: np.ndarray) -> int:
        """Find chirp mass bin using the same logic as the original function."""
        bin_idx = 0
        while chirp_mass >= chirp_mass_bins[bin_idx]:
            bin_idx += 1
            if bin_idx >= len(chirp_mass_bins):
                break
        return bin_idx

    @staticmethod
    def _redshift_bin(redshift: float, z_bin_edges: np.ndarray) -> int:
        """Find redshift bin (left edges)."""
        if redshift < z_bin_edges[0]:
            return -1

        for i in range(len(z_bin_edges) - 1):
            if z_bin_edges[i] <= redshift < z_bin_edges[i + 1]:
                return i

        # Handle last bin (include right edge)
        if len(z_bin_edges) > 1 and redshift >= z_bin_edges[-1]:
            return len(z_bin_edges) - 1

        return -1

    @staticmethod
    def _generate_population_weights(mc_found: np.ndarray, z_found: np.ndarray,
                                     mc_bin_edges: np.ndarray, mc_bin_widths: np.ndarray,
                                     z_bin_edges: np.ndarray, mc_prior: np.ndarray,
                                     z_prior: np.ndarray, n_posterior_samples: int,
                                     debug=True):
        """Generate weights for the entire population of events."""

        n_events = len(mc_found)
        population_weights = np.zeros((n_events, len(z_bin_edges), len(mc_bin_edges)))

        # posterior summaries per event

        posterior_quantiles = []

        for i in tqdm(range(n_events), desc="Generating event weights"):
            # SNR sampling (CDF ~ SNR^{-3})
            rho = 12 * np.random.rand() ** (-1 / 3)

            # Generate posterior samples with measurement uncertainty
            posterior_samples = MockObservation._generate_event_posterior(
                mc_found[i], z_found[i], rho, n_posterior_samples
            )





            # collect z and mc quantiles
            posterior_quantiles.append([
                np.percentile(posterior_samples[:,1],  [16, 50, 84]),
                np.percentile(posterior_samples[:,0],  [16, 50, 84])
            ])

            if len(posterior_samples) > 0:
                # Calculate weights for this event
                event_weights, best_mc_z = MockObservation._get_weights_for_one_posterior(
                    posterior_samples, mc_bin_edges, mc_bin_widths, z_bin_edges,
                    mc_prior, z_prior
                )

                if debug:

                    # 2d histogram of posterior samples
                    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
                    axes[0].hist2d(
                        posterior_samples[:, 1], posterior_samples[:, 0],
                        cmap='Blues'
                    )
                    axes[1].hist2d(
                        posterior_samples[:, 1], posterior_samples[:, 0],
                        bins=[z_bin_edges, np.concatenate([mc_bin_edges - mc_bin_widths, [mc_bin_edges[-1]]])],
                        cmap='Blues'
                    )
                    z_left_edges = z_bin_edges
                    mc_left_edges = mc_bin_edges - mc_bin_widths
                    axes[2].pcolormesh(
                        z_left_edges, mc_left_edges, event_weights.T,
                        cmap='Blues', shading='auto'
                    )

                    axes[0].set_title('p(z, Mc)')
                    axes[1].set_title('p(z, Mc) binned')
                    axes[2].set_title('Weights')

                    for ax in [axes[1], axes[2]]:
                        _fmt_yaxes(ax)


                    for ax in axes:
                        ax.axvline(z_found[i], color='red', linestyle='--', alpha=0.2)
                        ax.axhline(mc_found[i], color='red', linestyle='--', alpha=0.2)
                        ax.set_xlabel('Redshift')
                        ax.set_ylabel('Chirp Mass [M☉]')


                    # save in tmpdir
                    tmpdir = tempfile.gettempdir()
                    plt.tight_layout()
                    plt.suptitle(f'Event {i + 1}')
                    plt.savefig(os.path.join(tmpdir, f'event_{i + 1}_posterior.png'), dpi=300, bbox_inches='tight')
                    plt.close()


                if best_mc_z is not None:
                    qtls = np.array(posterior_quantiles[-1])[:, 1][::-1]
                    # now check if qtls fall within best bin edges
                    for j in range(2):
                        if not (best_mc_z[j][0] <= qtls[j] <= best_mc_z[j][1]):
                            warnings.warn(
                                f"Event {i}: Posterior median {['z','mc'][j]}={qtls[j]:.2f} "
                                f"outside best bin edges {best_mc_z[j]}"
                            )

                population_weights[i, :, :] = event_weights





            else:
                population_weights[i, :, :] = np.nan

        if debug:
            print(f"Posterior plots at {tmpdir}")

        return population_weights, np.array(posterior_quantiles)

    @staticmethod
    def _generate_event_posterior(mc_true: float, z_true: float, rho: float, n_samples: int):
        """Generate posterior samples for a single event with measurement uncertainty."""

        # Chirp mass measurement uncertainty (Powell+ 2019, eq. 4)
        r0_mc = np.random.randn()
        r_mc = np.random.randn(n_samples)
        mc_out = mc_true * (1 + 0.03 * 12 / rho * (r0_mc + r_mc))
        mc_out = mc_out[(mc_out > 0.1) & (mc_out < 199.9)]

        # Redshift measurement uncertainty
        r0_z = np.random.randn()
        r_z = np.random.randn(n_samples)
        z_out = z_true * (1 + 0.3 * 12 / rho * (r0_z + r_z))
        z_out = z_out[(z_out > 0) & (z_out < 1.49)]

        # Combine samples
        out_length = min(len(mc_out), len(z_out))
        if out_length > 0:
            return np.column_stack([mc_out[:out_length], z_out[:out_length]])
        else:
            return np.array([]).reshape(0, 2)

    @staticmethod
    def _get_weights_for_one_posterior(posterior_samples: np.ndarray,
                                       mc_bin_edges: np.ndarray, mc_bin_widths: np.ndarray,
                                       z_bin_edges: np.ndarray, mc_prior: np.ndarray,
                                       z_prior: np.ndarray) -> np.ndarray:
        """Calculate weights for one event's posterior samples."""
        n_z_bins, n_mc_bins = len(z_bin_edges), len(mc_bin_edges)
        weights = np.zeros((n_z_bins, n_mc_bins))


        for mc, z in posterior_samples:
            # Find bins using proper binning functions
            mc_bin = MockObservation._chirp_mass_bin(mc, mc_bin_edges)
            z_bin = MockObservation._redshift_bin(z, z_bin_edges)

            # Check if bins are valid
            if 0 <= mc_bin < n_mc_bins and 0 <= z_bin < n_z_bins:
                # Get prior probability density for this bin
                prior_prob = mc_prior[mc_bin] * z_prior[z_bin]
                if prior_prob > 0:
                    weights[z_bin, mc_bin] += 1 / prior_prob

        # get max weight sample's mc and z bin edges
        best_bin_edges = None
        if np.sum(weights) > 0:
            max_idx = np.unravel_index(np.argmax(weights, axis=None), weights.shape)
            mc_right = mc_bin_edges[max_idx[1]]
            z_left = z_bin_edges[max_idx[0]]
            # get both left and right edges of mc bin
            best_mc_bin_edges = [mc_right - mc_bin_widths[max_idx[1]], mc_right]
            best_z_bin_edges = [z_left, z_left + (z_bin_edges[1] - z_bin_edges[0])]
            best_bin_edges = np.array([best_mc_bin_edges, best_z_bin_edges]).round(2)

        if len(posterior_samples) > 0:
            weights /= len(posterior_samples)
        return weights, best_bin_edges

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


def _fmt_yaxes(ax: plt.Axes):
    ax.set_yscale('log')
    ax.set_ylim(MC_BIN_R_EDGE[0], MC_BIN_R_EDGE[-1])
    # add many more yticks -- dont use log-formatter but scalar formatter
    ax.yaxis.set_major_formatter(ScalarFormatter())
    # 8 ticks from MC_BIN_R_EDGE
    tick_locs = np.unique(np.logspace(np.log10(MC_BIN_R_EDGE[0]), np.log10(MC_BIN_R_EDGE[-1]), 8).round(1))
    ax.set_yticks(tick_locs)
    # add tick labels
    ax.set_yticklabels([f"{t:.1f}" for t in tick_locs])