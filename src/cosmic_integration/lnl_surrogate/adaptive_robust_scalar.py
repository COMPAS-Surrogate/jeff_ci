from __future__ import annotations

import json
import logging
from typing import Callable, Dict, Optional

import numpy as np


class AdaptiveRobustScaler:
    """
    Simplified robust scaler tailored for log-likelihood values.

    * Anchors the scale at the best (maximum) observed log-likelihood.
    * Uses the median and interquartile range of the clipped distribution
      for robustness to heavy tails.
    * Optionally applies a smooth tanh-based clipping in transformed space
      to keep extreme values bounded.
    * Optionally clips raw log-likelihoods below a chosen percentile before
      transformation so that very poor regions do not dominate the scale.
    """

    def __init__(
        self,
        *,
        lower_clip_percentile: Optional[float] = 1.0,
        lower_clip_value: Optional[float] = None,
        soft_clipping: bool = True,
        clip_factor: float = 3.0,
    ):
        self.lower_clip_percentile = lower_clip_percentile
        self.lower_clip_value = lower_clip_value
        self.soft_clipping = soft_clipping
        self.clip_factor = float(clip_factor)
        if self.soft_clipping and self.clip_factor <= 0:
            raise ValueError("clip_factor must be positive when soft_clipping is enabled.")

        self.reference_value: float = 0.0
        self.median: float = 0.0
        self.scale: float = 1.0
        self.lower_clip_value: Optional[float] = lower_clip_value
        self.initialized: bool = False

    # ------------------------------------------------------------------ #
    # Initialisation and core helpers
    # ------------------------------------------------------------------ #
    def initialize_with_data(self, initial_lnls: np.ndarray) -> None:
        if initial_lnls is None or len(initial_lnls) == 0:
            raise ValueError("initial_lnls must contain at least one value.")

        values = np.asarray(initial_lnls, dtype=float)
        self.reference_value = float(np.max(values))

        if self.lower_clip_value is not None:
            self.lower_clip_value = float(self.lower_clip_value)
        elif self.lower_clip_percentile is not None:
            self.lower_clip_value = float(
                np.percentile(values, self.lower_clip_percentile)
            )
        else:
            self.lower_clip_value = None

        clipped = self._clip(values)
        shifted = clipped - self.reference_value

        span = None
        if self.lower_clip_value is not None and np.isfinite(self.lower_clip_value):
            span = float(self.reference_value - self.lower_clip_value)
            span = max(span, 1e-6)

        if span is not None:
            target_width = self.clip_factor if self.soft_clipping else 3.0
            target_width = max(float(target_width), 1e-6)
            self.median = 0.0
            self.scale = span / target_width
        else:
            self.median = float(np.median(shifted))
            iqr = np.subtract(*np.percentile(shifted, [75, 25]))
            self.scale = float(max(iqr, 1e-6))
        self.initialized = True

    def _ensure_initialized(self) -> None:
        if not self.initialized:
            raise RuntimeError("AdaptiveRobustScaler.initialize_with_data must be called first.")

    def _clip(self, arr: np.ndarray) -> np.ndarray:
        if self.lower_clip_value is None:
            return arr
        return np.maximum(arr, self.lower_clip_value)

    def clip_value(self, value: float) -> float:
        self._ensure_initialized()
        if self.lower_clip_value is None:
            return float(value)
        return float(max(value, self.lower_clip_value))

    # ------------------------------------------------------------------ #
    # Transform / inverse-transform
    # ------------------------------------------------------------------ #
    def transform(self, raw_value: np.ndarray | float) -> np.ndarray | float:
        """
        Transform raw log-likelihood(s) into a scaled space suitable for GP training.
        """
        self._ensure_initialized()
        raw = np.asarray(raw_value, dtype=float)
        clipped = self._clip(raw)
        standardized = (clipped - self.reference_value - self.median) / self.scale

        if self.soft_clipping:
            standardized = np.tanh(standardized / self.clip_factor) * self.clip_factor

        if np.isscalar(raw_value):
            return float(standardized)
        return standardized

    def inverse_transform(self, transformed_value: np.ndarray | float) -> np.ndarray | float:
        """
        Map scaled values back to the original log-likelihood space.
        """
        self._ensure_initialized()
        transformed = np.asarray(transformed_value, dtype=float)

        if self.soft_clipping:
            scaled = transformed / self.clip_factor
            scaled = np.clip(scaled, -0.999, 0.999)
            standardized = np.arctanh(scaled) * self.clip_factor
        else:
            standardized = transformed

        raw = standardized * self.scale + self.median + self.reference_value

        if np.isscalar(transformed_value):
            return float(raw)
        return raw

    # ------------------------------------------------------------------ #
    # Diagnostics / persistence
    # ------------------------------------------------------------------ #
    def get_diagnostics(self) -> Dict[str, float]:
        self._ensure_initialized()
        return {
            "reference_value": self.reference_value,
            "median": self.median,
            "scale": self.scale,
            "lower_clip_value": self.lower_clip_value,
            "soft_clipping": self.soft_clipping,
            "clip_factor": self.clip_factor if self.soft_clipping else None,
        }

    def save(self, outdir: str) -> None:
        self._ensure_initialized()
        state = {
            "lower_clip_percentile": self.lower_clip_percentile,
            "lower_clip_value": self.lower_clip_value,
            "soft_clipping": self.soft_clipping,
            "clip_factor": self.clip_factor,
            "reference_value": self.reference_value,
            "median": self.median,
            "scale": self.scale,
            "lower_clip_value": self.lower_clip_value,
            "initialized": self.initialized,
        }
        fname = f"{outdir}/scaler.json"
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

        logger = logging.getLogger(__name__)
        logger.info(
            "Scaler saved: reference=%.2f median=%.2f scale=%.2f clip=%s",
            self.reference_value,
            self.median,
            self.scale,
            self.clip_factor if self.soft_clipping else "none",
        )

    @classmethod
    def load(cls, outdir: str) -> "AdaptiveRobustScaler":
        fname = f"{outdir}/scaler.json"
        with open(fname, "r", encoding="utf-8") as f:
            state = json.load(f)

        if "lower_clip_percentile" in state or "lower_clip_value" in state:
            scaler = cls(
                lower_clip_percentile=state.get("lower_clip_percentile"),
                lower_clip_value=state.get("lower_clip_value"),
                soft_clipping=state.get("soft_clipping", True),
                clip_factor=state.get("clip_factor", 3.0),
            )
            scaler.reference_value = state["reference_value"]
            scaler.median = state["median"]
            scaler.scale = max(state["scale"], 1e-6)
            scaler.lower_clip_value = state.get("lower_clip_value")
            scaler.initialized = state.get("initialized", True)
            return scaler

        # Backwards compatibility with the legacy adaptive scaler dump
        scaler = cls(
            lower_clip_percentile=None,
            lower_clip_value=state.get("rejection_threshold"),
            soft_clipping=state.get("soft_clipping", True),
            clip_factor=state.get("clip_factor", 3.0),
        )
        scaler.reference_value = state.get("reference_value", 0.0)
        scaler.median = state.get("median", 0.0)
        scaler.scale = max(state.get("scale", 1.0), 1e-6)
        scaler.initialized = True
        return scaler


# -------------------------------------------------------------------------- #
# Factory for negative log-likelihood computer
# -------------------------------------------------------------------------- #
def robust_neg_lnl_computer_factory(
    lnl_computer: Callable[..., float],
    initial_lnls: np.ndarray,
    *,
    lower_clip_percentile: Optional[float] = 1.0,
    lower_clip_value: Optional[float] = None,
    soft_clipping: bool = True,
    clip_factor: float = 3.0,
) -> Callable[..., float]:
    """
    Wrap a raw LnL evaluator with robust scaling and optional clipping so the GP
    sees a well-behaved target.
    """
    scaler = AdaptiveRobustScaler(
        lower_clip_percentile=lower_clip_percentile,
        lower_clip_value=lower_clip_value,
        soft_clipping=soft_clipping,
        clip_factor=clip_factor,
    )
    scaler.initialize_with_data(initial_lnls)

    def robust_neg_lnl_computer(*params: float) -> float:
        try:
            raw_lnl = lnl_computer(*params)
        except Exception as exc:  # pragma: no cover - defensive
            logging.getLogger(__name__).warning("LnL evaluation failed: %s", exc)
            raw_lnl = scaler.reference_value - 1_000.0

        transformed = scaler.transform(raw_lnl)
        return -float(transformed)

    robust_neg_lnl_computer.scaler = scaler  # type: ignore[attr-defined]
    return robust_neg_lnl_computer
