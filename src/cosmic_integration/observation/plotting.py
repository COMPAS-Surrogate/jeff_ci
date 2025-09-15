import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter

from ..ratesSampler import get_default_mc_z_bins

MC_BIN_R_EDGE, MC_BIN_WDT, Z_BIN_L_EDGE = get_default_mc_z_bins()


def plot_weights(prior_2d,
                 population_weights, figsize=(8, 4.5), cmap='Blues', fname=None):
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    # Normalize each event's weights and sum
    weights_normalized = population_weights.copy()
    for i in range(len(weights_normalized)):
        if np.sum(weights_normalized[i]) > 0:
            weights_normalized[i] = weights_normalized[i] / np.sum(weights_normalized[i])

    weights_sum = np.sum(weights_normalized, axis=0)
    z_left_edges = Z_BIN_L_EDGE
    mc_left_edges = MC_BIN_R_EDGE - MC_BIN_WDT

    prior_2d_log = np.log10(np.clip(prior_2d.T, 1e-10, None))
    weights_sum_log = np.log10(np.clip(weights_sum.T, 1e-10, None))
    # Plot prior
    z_edges = np.append(z_left_edges, z_left_edges[-1] + (z_left_edges[-1] - z_left_edges[-2]))
    mc_edges = np.append(mc_left_edges, MC_BIN_R_EDGE[-1])
    axes[0].pcolormesh(z_edges, mc_edges, prior_2d_log, cmap=cmap, shading='auto')
    axes[1].pcolormesh(z_edges, mc_edges, weights_sum_log, cmap=cmap, shading='auto')
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
