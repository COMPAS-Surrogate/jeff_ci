"""Which GP target compression gives BOTH good acquisition and a good posterior?

Acquisition and sampling pull in opposite directions:

  - BO needs gradient in the far tail to walk toward the peak. Hard clipping
    flattens it and BO does worse than random (``test_acquisition_target.py``).
  - Sampling needs resolution near the peak. With no compression the posterior
    is a ~1e-4 sliver of the GP's range and comes out ~10x too narrow
    (``test_two_stage.py``).

``log`` compression should serve both: monotone and never flat, but most
sensitive near the peak. This runs the full loop (BO -> GP -> NUTS) for each
option and scores the posterior against the known truth.

Run:  python test_compression.py
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from cosmic_integration.lnl_surrogate.adaptive_robust_scalar import AdaptiveRobustScaler
from cosmic_integration.lnl_surrogate.jax_active_learner import JaxActiveLearner, JaxGPConfig
from cosmic_integration.lnl_surrogate.nuts_sampler import run_nuts

from toy_compas_lnl import LO, HI, MU, WIDTH, KEEP_DELTA, true_lnl

OUT = Path(__file__).parent / "compression_out"
if OUT.exists():
    shutil.rmtree(OUT)

INITIAL_POINTS = 60
TOTAL_STEPS = 150
SEED = 0

# No hard clip: compression alone tames the dynamic range, so the tail keeps
# the gradient BO needs. max_scale is large so `scale` is data-driven.
VARIANTS = {
    "none": dict(compression="none", max_scale=1e9),
    "sqrt": dict(compression="sqrt", max_scale=1e9),
    "log": dict(compression="log", max_scale=1e9),
    "clip(none)": dict(compression="none", max_scale=5.0, clip=True),
}


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

    hdr = (f"{'target':>11} {'informative':>12} {'best lnL':>11} {'conv':>6} "
           f"{'width ratio':>12} {'corr':>8} {'|bias|/sig':>11}")
    print(hdr)
    print("-" * len(hdr))

    for name, cfg in VARIANTS.items():
        md = OUT / name.replace("(", "_").replace(")", "") / "gp_model"
        md.mkdir(parents=True)

        clip = cfg.pop("clip", False)
        scaler = AdaptiveRobustScaler(
            soft_clipping=False,
            lower_clip_value=float(lnl0.max() - KEEP_DELTA) if clip else float(lnl0.min()),
            focus_fraction=0.05,
            **cfg,
        )
        scaler.initialize_with_data(lnl0)

        y0 = -np.array([scaler.transform(v) for v in lnl0]).reshape(-1, 1)
        learner = JaxActiveLearner(
            trainable_function=lambda *t: -scaler.transform(true_lnl(np.asarray(t))),
            bounds=np.array([LO, HI]),
            initial_data_x=x0, initial_data_y=y0,
            random_seed=SEED, config=JaxGPConfig(), outdir=str(md),
        )
        learner.run(total_steps=TOTAL_STEPS, steps_per_round=None)

        X = np.asarray(learner.data.query_points, float)
        lnl = np.array([true_lnl(r) for r in X])
        n_inf = int(np.sum(lnl - lnl.max() > -KEEP_DELTA))

        learner.save_model(round_idx=0)
        scaler.save(str(md))
        s = run_nuts(lnl_model_path=str(md / "models"),
                     outdir=str(md.parent / "run"), target="mean", round_idx=0,
                     num_warmup=500, num_samples=1500, num_chains=2, seed=SEED)
        smp = np.load(md.parent / "run" / "posterior_samples.npy")
        med, sd = np.median(smp, axis=0), smp.std(axis=0)
        wr = float(np.mean(sd / WIDTH))
        corr = float(np.corrcoef(smp[:, 2], smp[:, 3])[0, 1])
        bias = float(np.max(np.abs(med - MU) / WIDTH))

        print(f"{name:>11} {n_inf:>7}/{len(X):<4} {lnl.max():>11.1f} "
              f"{str(s['converged']):>6} {wr:>12.2f} {corr:>8.3f} {bias:>11.2f}")

    print("\ntruth: width ratio 1.00, corr 0.850, bias 0.00")


if __name__ == "__main__":
    main()
