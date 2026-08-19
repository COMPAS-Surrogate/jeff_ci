"""Does cycling explore/exploit beat one big block of each?

`predictive_variance` maximises posterior variance, so it samples *away* from
existing data -- including away from the peak. The old schedule spent a single
2/3 block on it up front, then a single 1/3 block on expected improvement.

This compares that against interleaved cycles (1/3 explore, 2/3 exploit,
repeating), and against the pure strategies, on the COMPAS-scale toy with a
known truth.

Metric: informative points acquired and best lnL found, for a fixed budget.

Run:  python test_acquisition_schedule.py
"""
from __future__ import annotations

import time

import numpy as np

from cosmic_integration.lnl_surrogate.adaptive_robust_scalar import AdaptiveRobustScaler
from cosmic_integration.lnl_surrogate.jax_active_learner import JaxActiveLearner, JaxGPConfig

from toy_compas_lnl import LO, HI, KEEP_DELTA, true_lnl

INITIAL_POINTS = 60
TOTAL_STEPS = 150
SEED = 0

# (exploration_fraction, cycle_length, label)
SCHEDULES = [
    (2 / 3, None, "old: 2/3 explore block"),
    (1 / 3, None, "1/3 explore block"),
    (1 / 3, 30, "NEW: 1/3 explore, cycle 30"),
    (1 / 3, 15, "1/3 explore, cycle 15"),
    (0.0, None, "pure exploit (EI)"),
    (1.0, None, "pure explore (var)"),
]


def main() -> None:
    rng = np.random.default_rng(SEED)
    x0 = rng.uniform(LO, HI, size=(INITIAL_POINTS, 4))
    lnl0 = np.array([true_lnl(r) for r in x0])

    budget = INITIAL_POINTS + TOTAL_STEPS
    x_rand = rng.uniform(LO, HI, size=(budget, 4))
    lnl_rand = np.array([true_lnl(r) for r in x_rand])
    n_rand = int(np.sum(lnl_rand - lnl_rand.max() > -KEEP_DELTA))
    print(f"random baseline ({budget} pts): {n_rand} informative "
          f"({100*n_rand/budget:.1f}%), best lnL={lnl_rand.max():.1f}\n")

    hdr = f"{'schedule':>28} {'informative':>13} {'best lnL':>11} {'time':>8}"
    print(hdr)
    print("-" * len(hdr))

    for frac, cycle, label in SCHEDULES:
        scaler = AdaptiveRobustScaler(
            soft_clipping=False, lower_clip_value=float(lnl0.min()),
            focus_fraction=0.05, max_scale=1e9, compression="sqrt",
        )
        scaler.initialize_with_data(lnl0)
        y0 = -np.array([scaler.transform(v) for v in lnl0]).reshape(-1, 1)

        t0 = time.time()
        learner = JaxActiveLearner(
            trainable_function=lambda *t: -scaler.transform(true_lnl(np.asarray(t))),
            bounds=np.array([LO, HI]), initial_data_x=x0, initial_data_y=y0,
            random_seed=SEED, config=JaxGPConfig(), outdir=None, refit_every=15,
        )
        learner.run(total_steps=TOTAL_STEPS, exploration_fraction=frac,
                    cycle_length=cycle)
        elapsed = time.time() - t0

        X = np.asarray(learner.data.query_points, float)
        lnl = np.array([true_lnl(r) for r in X])
        n_inf = int(np.sum(lnl - lnl.max() > -KEEP_DELTA))
        print(f"{label:>28} {n_inf:>7}/{len(X):<5} {lnl.max():>11.1f} {elapsed:>7.0f}s")


if __name__ == "__main__":
    main()
