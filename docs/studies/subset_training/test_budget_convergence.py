"""Does the sqrt-target surrogate converge to the truth as the budget grows?

`test_compression.py` left the best option (sqrt compression, no clip) with a
posterior ~2x too wide -- conservative, but not correct. The question that
decides whether the method is sound is whether that excess shrinks with more
evaluations, or whether it is a floor.

Runs the full loop (BO -> GP -> NUTS) at several budgets and tracks the width
ratio, the recovered sfr_a/sfr_d correlation, and the bias, all against the
known truth.

Run:  python test_budget_convergence.py
"""
from __future__ import annotations

import shutil
from pathlib import Path

import os

import numpy as np

from cosmic_integration.lnl_surrogate.adaptive_robust_scalar import AdaptiveRobustScaler
from cosmic_integration.lnl_surrogate.jax_active_learner import JaxActiveLearner, JaxGPConfig
from cosmic_integration.lnl_surrogate.nuts_sampler import run_nuts

from toy_compas_lnl import LO, HI, MU, WIDTH, KEEP_DELTA, true_lnl

OUT = Path(__file__).parent / "budget_out"
if OUT.exists():
    shutil.rmtree(OUT)

INITIAL_POINTS = 60
BUDGETS = [int(b) for b in os.environ.get("BUDGETS", "150,350,700").split(",")]
COMPRESSION = os.environ.get("COMPRESSION", "sqrt")
SEED = 0


def main() -> None:
    rng = np.random.default_rng(SEED)
    x0 = rng.uniform(LO, HI, size=(INITIAL_POINTS, 4))
    lnl0 = np.array([true_lnl(r) for r in x0])

    hdr = (f"{'BO steps':>9} {'n_total':>8} {'informative':>12} {'best lnL':>10} "
           f"{'conv':>6} {'width ratio':>12} {'corr':>8} {'|bias|/sig':>11}")
    print(f"compression = {COMPRESSION}")
    print(hdr)
    print("-" * len(hdr))

    for steps in BUDGETS:
        md = OUT / f"steps_{steps}" / "gp_model"
        md.mkdir(parents=True)

        scaler = AdaptiveRobustScaler(
            soft_clipping=False, lower_clip_value=float(lnl0.min()),
            focus_fraction=0.05, max_scale=1e9, compression=COMPRESSION,
        )
        scaler.initialize_with_data(lnl0)
        y0 = -np.array([scaler.transform(v) for v in lnl0]).reshape(-1, 1)

        learner = JaxActiveLearner(
            trainable_function=lambda *t: -scaler.transform(true_lnl(np.asarray(t))),
            bounds=np.array([LO, HI]), initial_data_x=x0, initial_data_y=y0,
            random_seed=SEED, config=JaxGPConfig(), outdir=str(md),
        )
        learner.run(total_steps=steps, steps_per_round=None)

        X = np.asarray(learner.data.query_points, float)
        lnl = np.array([true_lnl(r) for r in X])
        n_inf = int(np.sum(lnl - lnl.max() > -KEEP_DELTA))

        learner.save_model(round_idx=0)
        scaler.save(str(md))
        s = run_nuts(lnl_model_path=str(md / "models"), outdir=str(md.parent / "run"),
                     target="mean", round_idx=0, num_warmup=500, num_samples=1500,
                     num_chains=2, seed=SEED)
        smp = np.load(md.parent / "run" / "posterior_samples.npy")
        med, sd = np.median(smp, axis=0), smp.std(axis=0)
        wr = float(np.mean(sd / WIDTH))
        corr = float(np.corrcoef(smp[:, 2], smp[:, 3])[0, 1])
        bias = float(np.max(np.abs(med - MU) / WIDTH))

        print(f"{steps:>9} {len(X):>8} {n_inf:>7}/{len(X):<4} {lnl.max():>10.1f} "
              f"{str(s['converged']):>6} {wr:>12.2f} {corr:>8.3f} {bias:>11.2f}")

    print("\ntruth: width ratio 1.00, corr 0.850, bias 0.00")


if __name__ == "__main__":
    main()
