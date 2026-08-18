import logging

from .adaptive_robust_scalar import AdaptiveRobustScaler
from .lnl_surrogate import LnLSurrogate

# Heavy dependencies (tensorflow/gpflow/trieste) live behind these imports.
# Keep the package importable for lightweight tasks (e.g. reading constants,
# type checking) while still providing the same public API when deps exist.
try:  # pragma: no cover
    from .active_learner import ActiveLearner
    from .offline_diagnostics import (
        OfflineDiagnosticsResult,
        OfflineRoundResult,
        offline_surrogate_diagnostics,
    )
except ModuleNotFoundError:  # pragma: no cover
    ActiveLearner = None  # type: ignore[assignment]
    OfflineDiagnosticsResult = None  # type: ignore[assignment]
    OfflineRoundResult = None  # type: ignore[assignment]
    offline_surrogate_diagnostics = None  # type: ignore[assignment]
except ImportError:  # pragma: no cover
    ActiveLearner = None  # type: ignore[assignment]
    OfflineDiagnosticsResult = None  # type: ignore[assignment]
    OfflineRoundResult = None  # type: ignore[assignment]
    offline_surrogate_diagnostics = None  # type: ignore[assignment]

# Set up logging for the lnl_surrogate module
logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    'LnLSurrogate',
    'ActiveLearner',
    'AdaptiveRobustScaler',
    'OfflineRoundResult',
    'offline_surrogate_diagnostics',
    'OfflineDiagnosticsResult',
]
