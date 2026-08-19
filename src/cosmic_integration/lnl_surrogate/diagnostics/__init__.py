"""Diagnostics tooling for the GPJax surrogate."""
from .plots import plot_diagnostics, plot_trace, scatter_matrix
from .gp_truth import gp_accuracy_vs_distance
from .posterior_kl import consecutive_posterior_kl, posterior_kl_vs_reference
from .simulation_study import (
    local_peak_diagnostics_vs_round,
    plot_gp_vs_true_1d,
    posterior_convergence_vs_round,
    summarise_surrogate_optimum,
)

__all__ = [
    "plot_diagnostics",
    "plot_trace",
    "scatter_matrix",
    "gp_accuracy_vs_distance",
    "consecutive_posterior_kl",
    "posterior_kl_vs_reference",
    "local_peak_diagnostics_vs_round",
    "plot_gp_vs_true_1d",
    "posterior_convergence_vs_round",
    "summarise_surrogate_optimum",
]
