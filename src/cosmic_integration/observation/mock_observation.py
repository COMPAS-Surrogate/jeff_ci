import tempfile

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional
import warnings
import os
from tqdm import tqdm

from ..ratesSampler.binned_cosmic_integrator import get_default_mc_z_bins, ChirpMassBin
from .observation_base import ObservationBase
from ..plot_rate import CMAP, MC_LATEX, Z_LATEX, plot_matrix

MC_BIN_R_EDGE_BINS, MC_BIN_WDT_BINS, Z_BIN_L_EDGE = get_default_mc_z_bins()


@dataclass
class MockObservation(ObservationBase):

    @classmethod
    def generate_from_rates(cls, rates: np.ndarray, chirp_masses: np.ndarray,
                            params: np.ndarray, duration: float,
                            n_samples: int = int(1e6), n_posterior_samples: int = int(1e4),
                            output_file: Optional[str] = None,
                            measurement_uncertainty: bool = True,
                            plot_posteriors: bool = False
                            ):
        """Generate MockObservation from detection rates.

        Args:
            rates: Detection rates matrix.
            chirp_masses: Chirp mass array.
            params: Model parameters.
            duration: Observation duration.
            n_samples: Number of prior samples.
            n_posterior_samples: Number of posterior samples per event.
            output_file: Optional output file path.
            measurement_uncertainty: If True, generate posteriors with measurement uncertainty. If False, posteriors are delta functions at the true value.
        """

        outdir = os.path.dirname(output_file) if output_file else None

        # Validation
        _validate_inputs(rates, chirp_masses)

        # Sample detected events
        mc_found, z_found = sample_detected_events(rates*duration, chirp_masses)
        print(f"Number of sampled events: {len(mc_found)} (original sum(rates*duration)={np.sum(rates)*duration:.1f})")

        # Generate priors
        mc_prior, z_prior = generate_priors(n_samples)

        # Generate population weights
        population_weights, posterior_quantiles = generate_population_weights(
            mc_found, z_found, n_posterior_samples, mc_prior, z_prior, plot_posteriors=plot_posteriors, measurement_uncertainty=measurement_uncertainty,
            outdir=outdir
        )

        # Create observation
        observation = cls(
            population_weights=population_weights,
            posterior_quantiles=posterior_quantiles,
            duration=duration,
            params=params,
            rate_matrix=rates,
            mc_bin_edges=MC_BIN_R_EDGE_BINS,
            mc_bin_widths=MC_BIN_WDT_BINS,
            z_bin_edges=Z_BIN_L_EDGE,
            mc_prior=mc_prior,
            z_prior=z_prior
        )

        # Optional save and return
        if output_file:
            observation.save_h5(output_file)

        print(observation.summary())
        return observation



def _validate_inputs(rates, chirp_masses):
    """Validate input parameters and print diagnostics."""
    print(f"Total rate sum: {np.sum(rates)}")
    print(f"Maximum rate: {np.max(rates)}")

    if np.max(rates) >= 1:
        warnings.warn(f"Rates should be probabilities (max < 1), but max={np.max(rates)}")

    n_mc, n_z = rates.shape
    print(f"Chirp mass array length: {len(chirp_masses)}, Rate matrix Mc dimension: {n_mc}")
    print(f"Number of MC bins: {len(MC_BIN_R_EDGE_BINS)}, Number of Z bins: {len(Z_BIN_L_EDGE)}")


def sample_detected_events(rates: np.ndarray, mc: np.ndarray):
    """Sample detected events based on detection rates using Poisson sampling."""
    mc_found, z_found = [], []
    n_z = len(Z_BIN_L_EDGE)

    for i in range(n_z):
        # For each redshift bin, sample number of events from each chirp mass bin
        for j, rate in enumerate(rates[:, i]):
            # Expected number of events = rate * duration
            expected_events = np.sum(rate)

            # Sample actual number of events from Poisson distribution
            n_events = np.random.poisson(expected_events)

            # Add the detected events
            if n_events > 0:
                mc_found.extend([mc[j]] * n_events)
                z_center = i * 0.1 + 0.05  # redshift bin center
                z_found.extend([z_center] * n_events)

    mc_found, z_found = np.array(mc_found), np.array(z_found)
    # check that we dont have duplicates
    if len(mc_found) != len(set(zip(mc_found, z_found))):
        num_duplicates = len(mc_found) - len(set(zip(mc_found, z_found)))
        print(f"Found {num_duplicates} duplicate (mc, z) pairs in sampled events")

    # dropping duplicates
    # mc_found, z_found = list(set(zip(mc_found, z_found)))
    return np.array(mc_found), np.array(z_found)

def get_chirp_mass_bin_index(chirp_mass: float) -> int:
    """Get chirp mass bin index using same logic as ratesSampler.ChirpMassBin."""
    return ChirpMassBin(chirp_mass, MC_BIN_R_EDGE_BINS)


def get_redshift_bin_index(redshift: float) -> int:
    """Find redshift bin index (left edges)."""
    if redshift < Z_BIN_L_EDGE[0]:
        return -1

    for i in range(len(Z_BIN_L_EDGE) - 1):
        if Z_BIN_L_EDGE[i] <= redshift < Z_BIN_L_EDGE[i + 1]:
            return i

    # Handle last bin (include right edge)
    if len(Z_BIN_L_EDGE) > 1 and redshift >= Z_BIN_L_EDGE[-1]:
        return len(Z_BIN_L_EDGE) - 1

    return -1


def _samples_to_density(samples: np.ndarray):
    """Convert samples to density histogram."""
    n_bins = len(MC_BIN_R_EDGE_BINS) + 1
    counts = np.zeros(n_bins)

    for sample in samples:
        bin_idx = get_chirp_mass_bin_index(sample)
        if 0 <= bin_idx < n_bins:
            counts[bin_idx] += 1

    # Extend bin widths to match number of bins
    bin_widths_extended = np.append(MC_BIN_WDT_BINS, MC_BIN_WDT_BINS[-1])
    density = counts / (np.sum(counts) * bin_widths_extended)

    return density


def _generate_chirp_mass_prior(n_samples: int):
    """Generate chirp mass prior density."""
    # Sample from mass distribution
    m1 = 1 + 999 * np.random.rand(n_samples)
    m2 = 1 + 999 * np.random.rand(n_samples)
    q = m2 / m1
    mc_samples = (m1 ** 0.6 * m2 ** 0.6) / (m1 + m2) ** 0.2

    # Apply physical cuts
    valid_mask = (q > 0.05) & (q < 1) & (mc_samples > 1) & (mc_samples < 200)
    mc_samples = mc_samples[valid_mask]

    # Convert to density
    return _samples_to_density(mc_samples)


def _generate_redshift_prior():
    """Generate redshift prior density."""
    z_samples = 1.5 * np.random.rand(int(1e6)) ** (1 / 3)

    # Calculate bin widths
    z_bin_widths = np.diff(np.concatenate([Z_BIN_L_EDGE, [Z_BIN_L_EDGE[-1] + (Z_BIN_L_EDGE[-1] - Z_BIN_L_EDGE[-2])]]))

    # Count samples in bins
    counts = np.zeros(len(Z_BIN_L_EDGE))
    for z in z_samples:
        bin_idx = get_redshift_bin_index(z)
        if 0 <= bin_idx < len(Z_BIN_L_EDGE):
            counts[bin_idx] += 1

    # Convert to density
    density = counts / (np.sum(counts) * z_bin_widths)
    return density


def generate_priors(n_samples: int):
    """Generate prior density arrays for chirp mass and redshift bins."""
    mc_prior = _generate_chirp_mass_prior(n_samples)
    z_prior = _generate_redshift_prior()
    return mc_prior, z_prior


def sample_event_posterior(mc_true: float, z_true: float, rho: float, n_samples: int):
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


def _process_single_event(mc_true: float, z_true: float, n_posterior_samples: int):
    """Process a single event to get posterior samples and quantiles."""
    # Sample SNR from CDF ~ SNR^{-3}
    rho = 12 * np.random.rand() ** (-1 / 3)

    # Generate posterior samples
    posterior_samples = sample_event_posterior(mc_true, z_true, rho, n_posterior_samples)

    # Calculate quantiles
    if len(posterior_samples) > 0:
        quantiles = [
            np.percentile(posterior_samples[:, 1], [16, 50, 84]),  # z quantiles
            np.percentile(posterior_samples[:, 0], [16, 50, 84])  # mc quantiles
        ]
    else:
        quantiles = [np.array([np.nan, np.nan, np.nan]), np.array([np.nan, np.nan, np.nan])]

    return posterior_samples, quantiles


def _calculate_event_weights(posterior_samples: np.ndarray, mc_prior: np.ndarray, z_prior: np.ndarray):
    """Calculate weights for one event's posterior samples."""
    n_mc_bins = len(MC_BIN_R_EDGE_BINS) + 1
    n_z_bins = len(Z_BIN_L_EDGE)
    weights = np.zeros((n_mc_bins, n_z_bins))

    for mc, z in posterior_samples:
        mc_bin = get_chirp_mass_bin_index(mc)
        z_bin = get_redshift_bin_index(z)

        if 0 <= mc_bin < n_mc_bins and 0 <= z_bin < n_z_bins:
            prior_prob = mc_prior[mc_bin] * z_prior[z_bin]
            if prior_prob > 0:
                weights[mc_bin, z_bin] += 1 / prior_prob

    # Normalize by number of samples
    if len(posterior_samples) > 0:
        weights /= len(posterior_samples)

    return weights


def _debug_plot_event(posterior_samples, event_weights, z_found, mc_found, event_idx, outdir):
    """Create debug plots for posterior analysis."""
    outdir = outdir if outdir else tempfile.gettempdir()
    outdir = f"{outdir}/mock_posterior_plots"
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Raw posterior samples
    axes[0].hist2d(posterior_samples[:, 1], posterior_samples[:, 0], cmap=CMAP)
    axes[0].set_title('p(z, Mc)')

    # Binned posterior samples
    mc_bin_edges_plot = np.concatenate([MC_BIN_R_EDGE_BINS - MC_BIN_WDT_BINS, [MC_BIN_R_EDGE_BINS[-1]]])
    axes[1].hist2d(posterior_samples[:, 1], posterior_samples[:, 0],
                   bins=[Z_BIN_L_EDGE, mc_bin_edges_plot], cmap=CMAP)
    axes[1].set_title('p(z, Mc) binned')

    # Weights
    plot_matrix(event_weights, ax=axes[2], label='Weights')
    axes[2].set_title('Weights')

    # Add true values
    for ax in axes:
        ax.axvline(z_found, color='red', linestyle='--', alpha=0.2)
        ax.axhline(mc_found, color='red', linestyle='--', alpha=0.2)
        ax.set_xlabel(Z_LATEX)
        ax.set_ylabel(MC_LATEX)

    # Save plot
    tmpdir = tempfile.gettempdir()
    plt.tight_layout()
    plt.suptitle(f'Event {event_idx + 1}')
    plt.savefig(os.path.join(outdir, f'event_{event_idx + 1}_posterior.png'), dpi=300, bbox_inches='tight')
    plt.close()


def generate_population_weights(mc_found: np.ndarray, z_found: np.ndarray,
                                n_posterior_samples: int, mc_prior: np.ndarray,
                                z_prior: np.ndarray, plot_posteriors=False, measurement_uncertainty=True, outdir: Optional[str] = None):
    """Generate weights for the entire population of events.

    Args:
        mc_found: Array of detected chirp masses.
        z_found: Array of detected redshifts.
        n_posterior_samples: Number of posterior samples per event.
        mc_prior: Chirp mass prior density.
        z_prior: Redshift prior density.
        debug: If True, generate debug plots.
        measurement_uncertainty: If True, generate posteriors with measurement uncertainty. If False, posteriors are delta functions at the true value.
    """
    n_events = len(mc_found)
    n_mc_bins = len(MC_BIN_R_EDGE_BINS) + 1
    n_z_bins = len(Z_BIN_L_EDGE)

    population_weights = np.zeros((n_events, n_mc_bins, n_z_bins))
    posterior_quantiles = []

    for i in tqdm(range(n_events), desc="Generating event weights"):
        if not measurement_uncertainty:
            # Find bin indices for true values
            mc_val = float(mc_found[i])
            z_val = float(z_found[i])
            mc_bin = get_chirp_mass_bin_index(mc_val)
            z_bin = get_redshift_bin_index(z_val)
            event_weights = np.zeros((n_mc_bins, n_z_bins))
            if 0 <= mc_bin < n_mc_bins and 0 <= z_bin < n_z_bins:
                event_weights[mc_bin, z_bin] = 1.0
            population_weights[i, :, :] = event_weights
            # Quantiles are just the true value
            quantiles = [np.array([z_val]*3), np.array([mc_val]*3)]
            posterior_quantiles.append(quantiles)
        else:
            # Process single event
            posterior_samples, quantiles = _process_single_event(
                mc_found[i], z_found[i], n_posterior_samples
            )
            posterior_quantiles.append(quantiles)

            # Calculate weights if we have valid samples
            if len(posterior_samples) > 0:
                event_weights = _calculate_event_weights(
                    posterior_samples, mc_prior, z_prior
                )
                population_weights[i, :, :] = event_weights

                # Optional debug plotting
                if plot_posteriors:
                    _debug_plot_event(posterior_samples, event_weights,
                                      z_found[i], mc_found[i], i, outdir)
            else:
                population_weights[i, :, :] = np.nan

    if plot_posteriors:
        print(f"Posterior plots saved to {tempfile.gettempdir()}")

    return population_weights, np.array(posterior_quantiles)
