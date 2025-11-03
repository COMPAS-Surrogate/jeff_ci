
import logging
import os
import tempfile
from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from ..plot_rate import CMAP, MC_LATEX, Z_LATEX, plot_matrix
from ..ratesSampler.binned_cosmic_integrator import get_default_mc_z_bins
from .observation_base import ObservationBase


logger = logging.getLogger(__name__)

MC_BIN_R_EDGE_BINS, MC_BIN_WDT_BINS, Z_BIN_L_EDGE = get_default_mc_z_bins()


def _mc_bin_edges(n_bins: int) -> np.ndarray:
    right_edges = np.asarray(MC_BIN_R_EDGE_BINS, dtype=float)
    widths = np.asarray(MC_BIN_WDT_BINS, dtype=float)
    left_edge0 = max(0.0, right_edges[0] - widths[0])
    edges = np.concatenate(([left_edge0], right_edges))
    while edges.size <= n_bins:
        edges = np.append(edges, edges[-1] + widths[-1])
    return edges[: n_bins + 1]


def _z_bin_edges(n_bins: int) -> np.ndarray:
    left_edges = np.asarray(Z_BIN_L_EDGE, dtype=float)
    if left_edges.size < 2:
        step = 0.1
    else:
        step = np.diff(left_edges).mean()
    edges = left_edges.copy()
    if edges.size == n_bins:
        edges = np.append(edges, edges[-1] + step)
    while edges.size <= n_bins:
        edges = np.append(edges, edges[-1] + step)
    return edges[: n_bins + 1]


def _bin_indices(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(edges, values, side="right") - 1
    return np.clip(idx, 0, edges.size - 2)


@dataclass
class MockObservation(ObservationBase):

    @classmethod
    def generate_from_rates(
        cls,
        rates: np.ndarray,
        params: np.ndarray,
        duration: float,
        n_samples: int = int(1e6),
        n_posterior_samples: int = int(1e4),
        output_file: Optional[str] = None,
        measurement_uncertainty: bool = True,
        plot_posteriors: bool = False,
        rng: Optional[np.random.Generator] = None,
    ) -> "MockObservation":
        """Generate a mock observation sampled from a binned detection-rate matrix."""

        rng = rng or np.random.default_rng()
        outdir = os.path.dirname(output_file) if output_file else None

        _validate_inputs(rates)

        n_mc_bins, n_z_bins = rates.shape
        mc_edges = _mc_bin_edges(n_mc_bins)
        z_edges = _z_bin_edges(n_z_bins)

        mc_found, z_found = sample_detected_events(
            rates,
            duration,
            mc_edges,
            z_edges,
            rng=rng,
        )
        logger.info(
            "Generated %d detected events (expected %.1f)",
            len(mc_found),
            float(np.sum(rates) * duration),
        )

        mc_prior, z_prior = generate_priors(
            n_samples,
            mc_edges,
            z_edges,
            rng=rng,
        )

        population_weights, posterior_quantiles = generate_population_weights(
            mc_found,
            z_found,
            n_posterior_samples,
            mc_prior,
            z_prior,
            mc_edges,
            z_edges,
            plot_posteriors=plot_posteriors,
            measurement_uncertainty=measurement_uncertainty,
            outdir=outdir,
            rng=rng,
        )

        observation = cls(
            population_weights=population_weights,
            posterior_quantiles=posterior_quantiles,
            duration=duration,
            params=params,
            rate_matrix=rates,
            mc_bin_edges=mc_edges[1:],
            mc_bin_widths=np.diff(mc_edges),
            z_bin_edges=z_edges[:-1],
            mc_prior=mc_prior,
            z_prior=z_prior,
        )

        if output_file:
            observation.save_h5(output_file)

        logger.info("\n%s", observation.summary())
        return observation


def _validate_inputs(rates: np.ndarray) -> None:
    total_rate = float(np.sum(rates))
    max_rate = float(np.max(rates))
    n_mc, n_z = rates.shape

    logger.info("Rate diagnostics: total=%0.3f, max=%0.3f", total_rate, max_rate)
    if max_rate >= 1:
        logger.info(
            "Detection-rate entries represent expected counts per bin-year; values above 1"
            " indicate bins with multiple expected detections (max=%0.3f)",
            max_rate,
        )
    logger.info("Binned grid: mc bins=%d, z bins=%d", n_mc, n_z)


def sample_detected_events(
    rates: np.ndarray,
    duration: float,
    mc_edges: np.ndarray,
    z_edges: np.ndarray,
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, np.ndarray]:
    rng = rng or np.random.default_rng()
    n_mc_bins, n_z_bins = rates.shape
    mc_left = mc_edges[:-1]
    mc_right = mc_edges[1:]
    z_left = z_edges[:-1]
    z_right = z_edges[1:]

    mc_events: list[np.ndarray] = []
    z_events: list[np.ndarray] = []
    expected_total = 0.0

    for i in range(n_mc_bins):
        for j in range(n_z_bins):
            lam = float(rates[i, j] * duration)
            expected_total += lam
            if lam <= 0:
                continue
            n_events = int(rng.poisson(lam))
            if n_events <= 0:
                continue
            mc_samples = rng.uniform(mc_left[i], mc_right[i], size=n_events)
            z_samples = rng.uniform(z_left[j], z_right[j], size=n_events)
            mc_events.append(mc_samples)
            z_events.append(z_samples)

    if mc_events:
        mc_found = np.concatenate(mc_events)
        z_found = np.concatenate(z_events)
    else:
        mc_found = np.array([], dtype=float)
        z_found = np.array([], dtype=float)

    logger.info(
        "Total expected detections=%.3f; sampled events=%d",
        expected_total,
        mc_found.size,
    )

    return mc_found, z_found


def generate_priors(
    n_samples: int,
    mc_edges: np.ndarray,
    z_edges: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    m1 = 1 + 999 * rng.random(n_samples)
    m2 = 1 + 999 * rng.random(n_samples)
    q = m2 / m1
    mc_samples = (m1 ** 0.6 * m2 ** 0.6) / (m1 + m2) ** 0.2
    valid_mask = (q > 0.05) & (q < 1) & (mc_samples > mc_edges[0]) & (mc_samples < mc_edges[-1])
    mc_samples = mc_samples[valid_mask]
    counts_mc, _ = np.histogram(mc_samples, bins=mc_edges)
    widths_mc = np.diff(mc_edges)
    mc_prior = np.zeros_like(widths_mc)
    if counts_mc.sum() > 0:
        mc_prior = counts_mc / (counts_mc.sum() * widths_mc)

    z_samples = 1.5 * rng.random(int(1e6)) ** (1 / 3)
    counts_z, _ = np.histogram(z_samples, bins=z_edges)
    widths_z = np.diff(z_edges)
    z_prior = np.zeros_like(widths_z)
    if counts_z.sum() > 0:
        z_prior = counts_z / (counts_z.sum() * widths_z)

    return mc_prior, z_prior


def sample_event_posterior(
    mc_true: float,
    z_true: float,
    rho: float,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    r0_mc = rng.normal()
    r_mc = rng.normal(size=n_samples)
    mc_out = mc_true * (1 + 0.03 * 12 / rho * (r0_mc + r_mc))
    mc_out = mc_out[(mc_out > 0.1) & (mc_out < 199.9)]

    r0_z = rng.normal()
    r_z = rng.normal(size=n_samples)
    z_out = z_true * (1 + 0.3 * 12 / rho * (r0_z + r_z))
    z_out = z_out[(z_out > 0) & (z_out < 1.49)]

    out_length = min(len(mc_out), len(z_out))
    if out_length > 0:
        return np.column_stack([mc_out[:out_length], z_out[:out_length]])
    return np.empty((0, 2))


def _process_single_event(
    mc_true: float,
    z_true: float,
    n_posterior_samples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[np.ndarray]]:
    rho = 12 * rng.random() ** (-1 / 3)
    posterior_samples = sample_event_posterior(mc_true, z_true, rho, n_posterior_samples, rng)

    if len(posterior_samples) > 0:
        quantiles = [
            np.percentile(posterior_samples[:, 1], [16, 50, 84]),
            np.percentile(posterior_samples[:, 0], [16, 50, 84]),
        ]
    else:
        quantiles = [
            np.array([np.nan, np.nan, np.nan]),
            np.array([np.nan, np.nan, np.nan]),
        ]
    return posterior_samples, quantiles


def _calculate_event_weights(
    posterior_samples: np.ndarray,
    mc_edges: np.ndarray,
    z_edges: np.ndarray,
) -> np.ndarray:
    n_mc_bins = mc_edges.size - 1
    n_z_bins = z_edges.size - 1
    weights = np.zeros((n_mc_bins, n_z_bins))
    if posterior_samples.size == 0:
        return weights
    mc_idx = _bin_indices(posterior_samples[:, 0], mc_edges)
    z_idx = _bin_indices(posterior_samples[:, 1], z_edges)
    np.add.at(weights, (mc_idx, z_idx), 1.0)
    total = weights.sum()
    if total > 0:
        weights /= total
    return weights


def _debug_plot_event(
    posterior_samples: np.ndarray,
    event_weights: np.ndarray,
    z_found: float,
    mc_found: float,
    event_idx: int,
    mc_edges: np.ndarray,
    z_edges: np.ndarray,
    outdir: Optional[str],
) -> None:
    outdir = outdir if outdir else tempfile.gettempdir()
    outdir = f"{outdir}/mock_posterior_plots"
    os.makedirs(outdir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    if posterior_samples.size > 0:
        axes[0].hist2d(posterior_samples[:, 1], posterior_samples[:, 0], bins=50, cmap=CMAP)
        axes[1].hist2d(
            posterior_samples[:, 1],
            posterior_samples[:, 0],
            bins=[z_edges, mc_edges],
            cmap=CMAP,
        )
    axes[0].set_title('p(z, Mc)')
    axes[1].set_title('p(z, Mc) binned')

    plot_matrix(event_weights, ax=axes[2], label='Weights')
    axes[2].set_title('Weights')

    for ax in axes:
        ax.axvline(z_found, color='red', linestyle='--', alpha=0.2)
        ax.axhline(mc_found, color='red', linestyle='--', alpha=0.2)
        ax.set_xlabel(Z_LATEX)
        ax.set_ylabel(MC_LATEX)

    plt.tight_layout()
    plt.suptitle(f'Event {event_idx + 1}')
    plt.savefig(os.path.join(outdir, f'event_{event_idx + 1}_posterior.png'), dpi=300, bbox_inches='tight')
    plt.close()


def generate_population_weights(
    mc_found: np.ndarray,
    z_found: np.ndarray,
    n_posterior_samples: int,
    mc_prior: np.ndarray,
    z_prior: np.ndarray,
    mc_edges: np.ndarray,
    z_edges: np.ndarray,
    plot_posteriors: bool = False,
    measurement_uncertainty: bool = True,
    outdir: Optional[str] = None,
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, np.ndarray]:
    n_events = len(mc_found)
    n_mc_bins = mc_edges.size - 1
    n_z_bins = z_edges.size - 1
    population_weights = np.zeros((n_events, n_mc_bins, n_z_bins))
    posterior_quantiles = []
    rng = rng or np.random.default_rng()

    for i in tqdm(range(n_events), desc="Generating event weights"):
        mc_val = float(mc_found[i])
        z_val = float(z_found[i])

        if not measurement_uncertainty:
            mc_bin = _bin_indices(np.array([mc_val]), mc_edges)[0]
            z_bin = _bin_indices(np.array([z_val]), z_edges)[0]
            event_weights = np.zeros((n_mc_bins, n_z_bins))
            event_weights[mc_bin, z_bin] = 1.0
            population_weights[i] = event_weights
            posterior_quantiles.append([np.array([z_val]*3), np.array([mc_val]*3)])
            continue

        posterior_samples, quantiles = _process_single_event(
            mc_val,
            z_val,
            n_posterior_samples,
            rng,
        )

        if len(posterior_samples) > 0:
            event_weights = _calculate_event_weights(posterior_samples, mc_edges, z_edges)
            population_weights[i] = event_weights
            posterior_quantiles.append(quantiles)
            if plot_posteriors:
                _debug_plot_event(
                    posterior_samples,
                    event_weights,
                    z_val,
                    mc_val,
                    i,
                    mc_edges,
                    z_edges,
                    outdir,
                )
        else:
            event_weights = np.zeros((n_mc_bins, n_z_bins))
            mc_bin = _bin_indices(np.array([mc_val]), mc_edges)[0]
            z_bin = _bin_indices(np.array([z_val]), z_edges)[0]
            event_weights[mc_bin, z_bin] = 1.0
            population_weights[i] = event_weights
            posterior_quantiles.append([
                np.array([z_val, z_val, z_val]),
                np.array([mc_val, mc_val, mc_val]),
            ])

    if plot_posteriors:
        logger.info("Posterior plots saved to %s", tempfile.gettempdir())

    return population_weights, np.asarray(posterior_quantiles)
