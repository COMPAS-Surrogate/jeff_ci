"""
Backward-compatible shim.

The implementation moved to `cosmic_integration.lnl_surrogate.diagnostics.plots`.
"""

from .diagnostics.plots import plot_diagnostics, plot_trace, scatter_matrix  # noqa: F401

__all__ = [
    "plot_diagnostics",
    "plot_trace",
    "scatter_matrix",
]

