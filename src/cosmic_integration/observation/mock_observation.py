import tempfile

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional
import warnings
import os
from tqdm import tqdm

from ..ratesSampler.binned_cosmic_integrator import get_default_mc_z_bins, ChirpMassBin, MakeChirpMassBins

# Use the same bin edges as FindBinnedDetectionRate
MC_BIN_R_EDGE, MC_BIN_WDT = MakeChirpMassBins()  # 114 edges, 114 widths
MC_BIN_R_EDGE = np.array(MC_BIN_R_EDGE)
MC_BIN_WDT = np.array(MC_BIN_WDT)
Z_BIN_L_EDGE = get_default_mc_z_bins()[2]
# Use full edges and widths to match BinnedCosmicIntegrator (115 bins)
MC_BIN_R_EDGE_BINS = MC_BIN_R_EDGE  # 114 edges for 115 bins
MC_BIN_WDT_BINS = MC_BIN_WDT  # 114 widths

from .observation_base import ObservationBase
from ..plot_rate import CMAP, _get_norm, MC_LATEX, Z_LATEX, plot_matrix

@dataclass
class MockObservation(ObservationBase):

    @classmethod
    def generate_from_rates(cls,
                            rates: np.ndarray,
                            chirp_masses: np.ndarray,
                            params: np.ndarray,
                            duration: float,
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

        # Use bin edges and widths for consistency with model
        mc_bin_edges: np.ndarray = MC_BIN_R_EDGE_BINS  # 114 edges
        mc_bin_widths: np.ndarray = MC_BIN_WDT_BINS    # 114 widths
        z_bin_edges: np.ndarray = Z_BIN_L_EDGE  # left edges

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
            posterior_quantiles=posterior_quantiles,
            duration=duration,
            params=params,
            rate_matrix=rates
        )

        # Save if requested
        if output_file:
            observation.save_h5(output_file)

        print(observation.summary())
        return observation

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
    def _chirp_mass_bin(chirp_mass: float, chirp_mass_bins: np.ndarray) -> int:
        # Use the same binning logic as ratesSampler.ChirpMassBin
        return ChirpMassBin(chirp_mass, chirp_mass_bins)

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
        n_mc_bins = len(mc_bin_edges) + 1
        mc_prior_counts = np.zeros(n_mc_bins)
        for mc in mc_prior_samples:
            bin_idx = MockObservation._chirp_mass_bin(mc, mc_bin_edges)
            if 0 <= bin_idx < n_mc_bins:
                mc_prior_counts[bin_idx] += 1

        # Convert to density (counts / (total_counts * bin_width))
        mc_bin_widths_extended = np.append(mc_bin_widths, mc_bin_widths[-1])
        mc_prior_density = mc_prior_counts / (np.sum(mc_prior_counts) * mc_bin_widths_extended)

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
        n_mc_bins = len(mc_bin_edges) + 1  # Should be 115
        n_z_bins = len(z_bin_edges)
        # Change shape to (n_events, n_mc_bins, n_z_bins)
        population_weights = np.zeros((n_events, n_mc_bins, n_z_bins))

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
                np.percentile(posterior_samples[:, 1], [16, 50, 84]),
                np.percentile(posterior_samples[:, 0], [16, 50, 84])
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
                        cmap=CMAP
                    )
                    axes[1].hist2d(
                        posterior_samples[:, 1], posterior_samples[:, 0],
                        bins=[z_bin_edges, np.concatenate([mc_bin_edges - mc_bin_widths, [mc_bin_edges[-1]]])],
                        cmap=CMAP
                    )
                    plot_matrix(
                        event_weights, ax=axes[2], label='Weights'
                    )

                    axes[0].set_title('p(z, Mc)')
                    axes[1].set_title('p(z, Mc) binned')
                    axes[2].set_title('Weights')

                    for ax in axes:
                        ax.axvline(z_found[i], color='red', linestyle='--', alpha=0.2)
                        ax.axhline(mc_found[i], color='red', linestyle='--', alpha=0.2)
                        ax.set_xlabel(Z_LATEX)
                        ax.set_ylabel(MC_LATEX)

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
                                f"Event {i}: Posterior median {['z', 'mc'][j]}={qtls[j]:.2f} "
                                f"outside best bin edges {best_mc_z[j]}"
                            )

                # Assign event_weights to population_weights with new shape
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
        n_mc_bins = len(mc_bin_edges) + 1  # Should be 115
        n_z_bins = len(z_bin_edges)
        weights = np.zeros((n_mc_bins, n_z_bins))

        for mc, z in posterior_samples:
            mc_bin = MockObservation._chirp_mass_bin(mc, mc_bin_edges)
            z_bin = MockObservation._redshift_bin(z, z_bin_edges)

            if 0 <= mc_bin < n_mc_bins and 0 <= z_bin < n_z_bins:
                prior_prob = mc_prior[mc_bin] * z_prior[z_bin]
                if prior_prob > 0:
                    weights[mc_bin, z_bin] += 1 / prior_prob

        # get max weight sample's mc and z bin edges
        best_bin_edges = None
        if np.sum(weights) > 0:
            max_idx = np.unravel_index(np.argmax(weights, axis=None), weights.shape)
            mc_bin_widths_extended = np.append(mc_bin_widths, mc_bin_widths[-1])
            if max_idx[0] < len(mc_bin_edges):
                mc_right = mc_bin_edges[max_idx[0]]
            else:
                mc_right = mc_bin_edges[-1] + mc_bin_widths[-1]
            z_left = z_bin_edges[max_idx[1]]
            # get both left and right edges of mc bin
            best_mc_bin_edges = [mc_right - mc_bin_widths_extended[max_idx[0]], mc_right]
            best_z_bin_edges = [z_left, z_left + (z_bin_edges[1] - z_bin_edges[0])]
            best_bin_edges = np.array([best_mc_bin_edges, best_z_bin_edges]).round(2)

        if len(posterior_samples) > 0:
            weights /= len(posterior_samples)
        return weights, best_bin_edges
