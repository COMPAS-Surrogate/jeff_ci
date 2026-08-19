"""Why does BO fail to find the peak, and does the GP target choice fix it?

``run_bo.py`` showed BO acquiring *fewer* informative points than random
sampling. The suspected cause is the hard clip we introduced to stop the target
transform wasting its dynamic range on the far tail:

  - for SAMPLING, flattening the tail is right: it is irrelevant to the
    posterior and it costs GP resolution where the posterior lives.
  - for ACQUISITION, flattening the tail is fatal: that gradient is the only
    thing telling BO which way the peak is.

The two uses want opposite things from one transform. This script tests
monotone compressions that tame the dynamic range *without* flattening:

  clip    - hard clip at ref - KEEP_DELTA (current default)
  sqrt    - t = -sqrt(delta)     (delta = best - lnl >= 0)
  log     - t = -log1p(delta)
  linear  - no compression at all, for reference

Metric: how many informative points BO acquires for a fixed budget, versus a
matched random baseline.

Run:  python test_acquisition_target.py
"""
from __future__ import annotations

import numpy as np

from cosmic_integration.lnl_surrogate.jax_active_learner import JaxActiveLearner, JaxGPConfig

from toy_compas_lnl import LO, HI, KEEP_DELTA, true_lnl

INITIAL_POINTS = 60
TOTAL_STEPS = 150
SEED = 0


def make_targets(ref: float):
    """Monotone maps from raw lnL to the GP target (lower = better, as the
    learner minimises). All are decreasing in lnL, so the minimum is the peak."""
    scale_lin = max(abs(ref), 1.0)

    def clip(lnl):
        d = ref - np.asarray(lnl, float)
        return np.minimum(d, KEEP_DELTA) / KEEP_DELTA

    def sqrt(lnl):
        d = np.maximum(ref - np.asarray(lnl, float), 0.0)
        return np.sqrt(d) / np.sqrt(KEEP_DELTA)

    def log(lnl):
        d = np.maximum(ref - np.asarray(lnl, float), 0.0)
        return np.log1p(d) / np.log1p(KEEP_DELTA)

    def linear(lnl):
        return (ref - np.asarray(lnl, float)) / scale_lin

    return {"clip": clip, "sqrt": sqrt, "log": log, "linear": linear}


def main() -> None:
    rng = np.random.default_rng(SEED)
    x0 = rng.uniform(LO, HI, size=(INITIAL_POINTS, 4))
    lnl0 = np.array([true_lnl(r) for r in x0])
    ref = float(lnl0.max())
    n_inf0 = int(np.sum(lnl0 - ref > -KEEP_DELTA))
    print(f"initial batch: {INITIAL_POINTS} pts, best lnL={ref:.1f}, "
          f"informative={n_inf0}/{INITIAL_POINTS}")

    budget = INITIAL_POINTS + TOTAL_STEPS
    x_rand = rng.uniform(LO, HI, size=(budget, 4))
    lnl_rand = np.array([true_lnl(r) for r in x_rand])
    n_rand = int(np.sum(lnl_rand - lnl_rand.max() > -KEEP_DELTA))
    print(f"random baseline ({budget} pts): {n_rand} informative "
          f"({100*n_rand/budget:.1f}%), best lnL={lnl_rand.max():.1f}\n")

    print(f"{'target':>8} {'informative':>12} {'frac':>7} {'best lnL':>12} {'vs random':>10}")
    for name, fn in make_targets(ref).items():
        y0 = np.asarray(fn(lnl0), float).reshape(-1, 1)
        learner = JaxActiveLearner(
            trainable_function=lambda *t: float(fn(true_lnl(np.asarray(t)))),
            bounds=np.array([LO, HI]),
            initial_data_x=x0, initial_data_y=y0,
            random_seed=SEED, config=JaxGPConfig(), outdir=None,
        )
        learner.run(total_steps=TOTAL_STEPS, steps_per_round=None)

        X = np.asarray(learner.data.query_points, float)
        lnl = np.array([true_lnl(r) for r in X])
        n_inf = int(np.sum(lnl - lnl.max() > -KEEP_DELTA))
        verdict = "better" if n_inf > n_rand else ("same" if n_inf == n_rand else "WORSE")
        print(f"{name:>8} {n_inf:>8}/{len(X):<3} {100*n_inf/len(X):>6.1f}% "
              f"{lnl.max():>12.1f} {verdict:>10}")


if __name__ == "__main__":
    main()
