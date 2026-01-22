"""
Backward-compatible shim.

The implementation moved to `cosmic_integration.lnl_surrogate.diagnostics.offline`.
"""

from .diagnostics.offline import (  # noqa: F401
    OfflineDiagnosticsResult,
    OfflineRoundResult,
    fit_surrogate_from_samples,
    offline_surrogate_diagnostics,
)

__all__ = [
    "OfflineDiagnosticsResult",
    "OfflineRoundResult",
    "fit_surrogate_from_samples",
    "offline_surrogate_diagnostics",
]

