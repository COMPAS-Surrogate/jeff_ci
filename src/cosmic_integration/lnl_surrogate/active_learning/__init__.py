"""
Internal implementation details for active learning / Trieste-based surrogate fitting.

Public surface area remains in:
- `cosmic_integration.lnl_surrogate.active_learner.ActiveLearner`
- `cosmic_integration.lnl_surrogate.lnl_surrogate.LnLSurrogate`
"""

from .gp_model import build_and_optimize_gpr
from .persistence import load_round_model, save_round_model
from .trieste_loop import run_active_learning

__all__ = [
    "build_and_optimize_gpr",
    "load_round_model",
    "run_active_learning",
    "save_round_model",
]

