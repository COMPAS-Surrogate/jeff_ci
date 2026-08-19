from __future__ import annotations

import json
import logging
from typing import Callable, Dict, Optional

import numpy as np


def suggest_lower_clip_value(
    lnls: np.ndarray,
    *,
    best_fraction: float = 0.05,
    delta_factor: float = 20.0,
    min_delta: float = 200.0,
    max_delta: float = 200.0,
    min_points: int = 10,
) -> float:
    """
    Heuristic "auto" lower clip for log-likelihoods.

    The likelihood surface can have an enormous dynamic range (e.g. very poor regions
    at ~-1e5). For surrogate-based inference, we mostly care about *relative* structure
    close to the maximum. Using a low percentile as the clip floor can make the scale
    dominated by extreme tails and compress the region of interest.

    This helper picks a floor based on the spread of the *best* fraction of points.
    """

    values = np.asarray(lnls, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("lnls must contain at least one finite value.")

    reference = float(np.max(values))
    best_fraction = float(best_fraction)
    if not (0.0 < best_fraction <= 1.0):
        raise ValueError("best_fraction must be in (0, 1].")

    threshold = float(np.quantile(values, max(0.0, 1.0 - best_fraction)))
    top = values[values >= threshold]
    if top.size < min_points:
        top = values

    deltas = reference - top  # >= 0 (up to numerical tolerance)
    if deltas.size == 0:
        return reference - float(min_delta)

    q25, q75 = np.percentile(deltas, [25.0, 75.0])
    iqr = float(q75 - q25)
    std = float(np.std(deltas))
    scale_est = max(iqr, std, 1.0)

    delta = max(float(min_delta), float(delta_factor) * scale_est)
    delta = min(delta, float(max_delta))
    return reference - delta


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
        focus_fraction: float = 0.05,
        max_scale: float = 10.0,
        compression: str = "none",
        compression_scale: float = 10.0,
    ):
        self.lower_clip_percentile = lower_clip_percentile
        self.lower_clip_value = lower_clip_value
        self.soft_clipping = soft_clipping
        self.clip_factor = float(clip_factor)
        if self.soft_clipping and self.clip_factor <= 0:
            raise ValueError("clip_factor must be positive when soft_clipping is enabled.")
        self.focus_fraction = float(focus_fraction)
        if not (0.0 < self.focus_fraction <= 1.0):
            raise ValueError("focus_fraction must be in (0, 1].")
        self.max_scale = float(max_scale)
        if self.max_scale <= 0:
            raise ValueError("max_scale must be positive.")
        # How to compress the drop below the best lnL before scaling.
        #   none - linear. Keeps tail gradient (good for BO acquisition) but
        #          leaves the posterior a ~1e-4 sliver of the range when the
        #          raw lnL spans 1e5, so the GP cannot resolve its shape.
        #   log  - log1p(delta). Monotone, never flat (so BO can still walk
        #          downhill from anywhere) yet most sensitive near the peak,
        #          which is where the posterior lives. The usable compromise.
        #   sqrt - intermediate.
        # Hard clipping alone is NOT a substitute: it flattens the tail and
        # blinds the acquisition function.
        self.compression = str(compression).lower()
        if self.compression not in ("none", "log", "sqrt", "softlog"):
            raise ValueError(f"Unknown compression: {compression!r}")
        # Only used by "softlog": the Delta lnL below which the transform is
        # effectively linear. Should cover the posterior bulk (a few lnL units).
        self.compression_scale = float(compression_scale)
        if self.compression_scale <= 0:
            raise ValueError("compression_scale must be positive.")

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
        values = values[np.isfinite(values)]
        if values.size == 0:
            raise ValueError("initial_lnls must contain at least one finite value.")
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
        deltas = self.reference_value - clipped  # >= 0

        # Focus scaling on the "good" region to avoid compressing structure near the peak.
        # We take the best `focus_fraction` of points (smallest deltas) when available.
        deltas_for_scale = deltas
        if deltas.size >= 20:
            k = int(max(10, round(self.focus_fraction * float(deltas.size))))
            k = min(k, int(deltas.size))
            if k >= 10:
                kth = float(np.partition(deltas, k - 1)[k - 1])
                focus = deltas[deltas <= kth]
                if focus.size >= 10:
                    deltas_for_scale = focus

        deltas_for_scale = self._compress(deltas_for_scale)

        q25, q75 = np.percentile(deltas_for_scale, [25.0, 75.0])
        iqr = float(q75 - q25)
        std = float(np.std(deltas_for_scale))

        # Keep reference_value mapped close to 0 in transformed space for interpretability.
        self.median = 0.0
        self.scale = float(max(min(max(iqr, std), self.max_scale), 1e-6))
        self.initialized = True

    def _compress(self, delta: np.ndarray) -> np.ndarray:
        """Compress a non-negative drop below the reference value."""
        delta = np.maximum(np.asarray(delta, dtype=float), 0.0)
        if self.compression == "log":
            return np.log1p(delta)
        if self.compression == "sqrt":
            return np.sqrt(delta)
        if self.compression == "softlog":
            d0 = self.compression_scale
            return d0 * np.log1p(delta / d0)
        return delta

    def _decompress(self, value: np.ndarray) -> np.ndarray:
        value = np.maximum(np.asarray(value, dtype=float), 0.0)
        if self.compression == "log":
            return np.expm1(value)
        if self.compression == "sqrt":
            return value ** 2
        if self.compression == "softlog":
            d0 = self.compression_scale
            return d0 * np.expm1(np.minimum(value / d0, 700.0))
        return value

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
        standardized = -self._compress(self.reference_value + self.median - clipped) / self.scale

        if self.soft_clipping:
            # Asymmetric soft-clipping:
            # - Clip only the *low* tail (very poor LnL regions) to avoid extreme dynamic range.
            # - Keep the *high* tail linear so that if we later discover a much better region
            #   than the initial "reference_value", the GP still sees meaningful differences
            #   instead of saturating at +clip_factor.
            standardized = np.where(
                standardized < 0.0,
                np.tanh(standardized / self.clip_factor) * self.clip_factor,
                standardized,
            )

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
            # Invert the asymmetric transform:
            # - For transformed <= 0, invert tanh.
            # - For transformed > 0, it was linear (no clipping), so pass through.
            scaled = transformed / self.clip_factor
            scaled = np.clip(scaled, -0.999, 0.999)
            neg_branch = np.arctanh(scaled) * self.clip_factor
            standardized = np.where(transformed <= 0.0, neg_branch, transformed)
        else:
            standardized = transformed

        raw = self.reference_value + self.median - self._decompress(-standardized * self.scale)

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
            "compression": self.compression,
            "compression_scale": self.compression_scale,
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
            "focus_fraction": self.focus_fraction,
            "max_scale": self.max_scale,
            "reference_value": self.reference_value,
            "median": self.median,
            "scale": self.scale,
            "compression": self.compression,
            "compression_scale": self.compression_scale,
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
                focus_fraction=state.get("focus_fraction", 0.05),
                max_scale=state.get("max_scale", 10.0),
                compression=state.get("compression", "none"),
                compression_scale=state.get("compression_scale", 10.0),
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
    focus_fraction: float = 0.05,
    max_scale: float = 10.0,
    compression: str = "none",
    compression_scale: float = 10.0,
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
        focus_fraction=focus_fraction,
        max_scale=max_scale,
        compression=compression,
        compression_scale=compression_scale,
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
