"""A 4D toy log-likelihood at COMPAS-realistic scale.

Same correlated-Gaussian-peak idea as ``test_subset.py``, but tuned so the
*numbers* look like a real run rather than a small synthetic one:

  - parameter box: the real ``(alpha, sigma, sfr_a, sfr_d)`` bounds
    (`cosmic_integration.lnl_surrogate.lnl_surrogate.BOUNDS`).
  - dynamic range: production's "auto" floor uses
    ``min_delta = max(2e4, 200 * n_events)`` and
    ``max_delta = max(2e5, 900 * n_events)`` (see
    ``LnLSurrogate.train`` / ``adaptive_robust_scalar.suggest_lower_clip_value``).
    For a ~90-event catalogue that is a floor 1.8e4-8.1e4 lnL below the peak,
    so this toy scales its quadratic form to reach a comparable magnitude at
    the box edges.
  - informative fraction: ~1% within `KEEP_DELTA` of the best point, matching
    the observed 3-yr-run plateau (see ``test_subset.py``).

Importing this module should not require GPJax/JAX -- it is pure numpy so it
can be reused by lightweight diagnostics too.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cosmic_integration.lnl_surrogate.lnl_surrogate import BOUNDS, PARAMETERS

PARAMETERS_NAMES = PARAMETERS
LO, HI = np.asarray(BOUNDS[0], float), np.asarray(BOUNDS[1], float)
MU = LO + 0.5 * (HI - LO)

# Peak width relative to the box, and a correlation between the two star
# formation parameters (sfr_a, sfr_d) -- these two are physically degenerate
# in COMPAS rate calculations, so a real posterior banana between them is
# expected.
WIDTH = 0.05 * (HI - LO)
_C = np.eye(4)
_C[2, 3] = _C[3, 2] = 0.85
_BASE_COV = np.diag(WIDTH) @ _C @ np.diag(WIDTH)

# Scale the quadratic form so a box-corner point lands near the production
# floor magnitude (tens of thousands of lnL below the peak), not the O(10-50)
# scale used in the small `test_subset.py` toy.
_CORNER_DELTA = 0.5 * np.sum(((HI - LO) / (2 * WIDTH)) ** 2)  # unscaled chi2 at a corner
N_EVENTS = 90
TARGET_FLOOR_DELTA = max(2.0e4, 200.0 * N_EVENTS)  # matches "auto" min_delta
SCALE = TARGET_FLOOR_DELTA / _CORNER_DELTA
COV = _BASE_COV / SCALE
PREC = np.linalg.inv(COV)

# "informative" = within a small Delta lnL of the best point (Delta lnL ~ O(1)
# sets a 1-sigma credible interval; O(10) covers the posterior bulk with
# margin). This threshold does NOT scale with the floor magnitude above --
# it is the physical width of the peak, calibrated (in test_subset.py) to
# reproduce the observed ~1% informative fraction. The unscaled peak's
# quadratic form is scaled up by SCALE along with everything else, so the
# threshold must be scaled the same way to stay a fixed multiple of the peak
# width.
_BASE_KEEP_DELTA = 10.0
KEEP_DELTA = _BASE_KEEP_DELTA * SCALE


def true_lnl(theta: np.ndarray) -> float:
    d = np.asarray(theta, float) - MU
    return float(-0.5 * d @ PREC @ d)


@dataclass(frozen=True)
class ToyTrainingSet:
    X: np.ndarray
    lnl: np.ndarray
    informative: np.ndarray


def build_training_set(
    n_total: int, n_near: int, *, seed: int = 0, keep_delta: float = KEEP_DELTA
) -> ToyTrainingSet:
    """Training set shaped like a real active-learning run: `n_near` points
    drawn near the peak, the rest uniform over the box (the "floor")."""
    rng = np.random.default_rng(seed)
    x_far = rng.uniform(LO, HI, size=(n_total - n_near, 4))
    x_near = np.clip(rng.multivariate_normal(MU, COV * 4.0, size=n_near), LO, HI)
    X = np.vstack([x_near, x_far])
    lnl = np.array([true_lnl(row) for row in X])
    informative = (lnl - lnl.max()) > -keep_delta
    return ToyTrainingSet(X=X, lnl=lnl, informative=informative)


if __name__ == "__main__":
    ts = build_training_set(2000, 20, seed=0)
    print(f"box: lo={LO}, hi={HI}")
    print(f"best possible lnL (at MU): {true_lnl(MU):.2f}")
    print(f"lnL at a box corner: {true_lnl(HI):.2f} (target floor ~ -{TARGET_FLOOR_DELTA:,.0f})")
    print(
        f"training set: {len(ts.X)} pts, {ts.informative.sum()} informative "
        f"({100*ts.informative.mean():.2f}%)"
    )
