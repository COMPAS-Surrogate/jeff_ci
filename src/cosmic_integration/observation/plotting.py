import matplotlib.pyplot as plt
import numpy as np

from ..ratesSampler.binned_cosmic_integrator import get_default_mc_z_bins
from ..plot_rate import plot_matrix, MC_LATEX, Z_LATEX

MC_BIN_R_EDGE, MC_BIN_WDT, Z_BIN_L_EDGE = get_default_mc_z_bins()


def plot_weights(prior_2d,
                 population_weights, figsize=(8, 4.5), fname=None):
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    # Normalize each event's weights and sum
    weights_normalized = population_weights.copy()
    for i in range(len(weights_normalized)):
        if np.sum(weights_normalized[i]) > 0:
            weights_normalized[i] = weights_normalized[i] / np.sum(weights_normalized[i])

    weights_sum = np.sum(weights_normalized, axis=0)

    # Plot prior
    plot_matrix(prior_2d, ax=axes[0], label="Prior Density")
    # Plot weights sum
    plot_matrix(weights_sum, ax=axes[1], label="Sum Weights(events)")

    plt.tight_layout()

    if fname:
        fig.savefig(fname, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {fname}")

    return fig, axes


def plot_event_summaries(posterior_quantiles, color="C0", figsize=(10, 6), fname=None, **kwgs):
    n_events = len(posterior_quantiles)
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
        mcqtl[:, 1], y, xerr=mcerr, fmt='o', color='black', **kwgs
    )
    # labels
    axes[0].set_xlabel(Z_LATEX)
    axes[1].set_xlabel(MC_LATEX)

    # styling
    axes[0].set_ylim(-0.5, n_events - 0.5)
    axes[0].set_xlim(0, 1)
    axes[0].set_xticks([0, 0.5, 0.8])
    axes[1].set_xticks([5, 30, 70])

    if fname:
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        print(f"Saved event summary plot to {fname}")

    return axes

