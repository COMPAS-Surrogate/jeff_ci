"""Acquisition and sampling want opposite transforms. Do we need two?

``test_acquisition_target.py`` showed BO only finds the peak when the GP target
keeps its gradient in the far tail (linear/sqrt), and fails completely when the
tail is clipped flat. But the clip was introduced because an uncompressed target
spends nearly all its dynamic range far from the posterior.

So: run BO with a gradient-preserving target, then compare two ways of getting a
posterior out of the acquired points:

  one-stage  - sample the same (linear) GP that BO used
  two-stage  - refit a GP on a clipped target, using the same acquired points,
               then sample that

Scored against the known truth. If one-stage is good enough, we do not need the
extra machinery.

Run:  python test_two_stage.py
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from cosmic_integration.lnl_surrogate.adaptive_robust_scalar import AdaptiveRobustScaler
from cosmic_integration.lnl_surrogate.jax_active_learner import JaxActiveLearner, JaxGPConfig
from cosmic_integration.lnl_surrogate.nuts_sampler import run_nuts

from toy_compas_lnl import LO, HI, MU, WIDTH, KEEP_DELTA, true_lnl

OUT = Path(__file__).parent / "two_stage_out"
if OUT.exists():
    shutil.rmtree(OUT)

INITIAL_POINTS = 60
TOTAL_STEPS = 150
SEED = 0


def score(samples: np.ndarray) -> tuple[float, float, float]:
    med, sd = np.median(samples, axis=0), samples.std(axis=0)
    return (
        float(np.mean(sd / WIDTH)),
        float(np.corrcoef(samples[:, 2], samples[:, 3])[0, 1]),
        float(np.max(np.abs(med - MU) / WIDTH)),
    )


def sample_and_score(name: str, X: np.ndarray, lnl: np.ndarray, scaler) -> None:
    md = OUT / name / "gp_model"
    md.mkdir(parents=True)
    y = -np.array([scaler.transform(v) for v in lnl]).reshape(-1, 1)
    learner = JaxActiveLearner(
        trainable_function=lambda *t: -scaler.transform(true_lnl(np.asarray(t))),
        bounds=np.array([LO, HI]), outdir=str(md),
        initial_data_x=X, initial_data_y=y, random_seed=SEED, config=JaxGPConfig(),
    )
    learner.save_model(round_idx=0)
    scaler.save(str(md))
    s = run_nuts(lnl_model_path=str(md / "models"), outdir=str(OUT / name / "run"),
                 target="mean", round_idx=0, num_warmup=500, num_samples=1500,
                 num_chains=2, seed=SEED)
    smp = np.load(OUT / name / "run" / "posterior_samples.npy")
    wr, corr, bias = score(smp)
    print(f"{name:>11} {str(s['converged']):>6} {wr:>12.2f} {corr:>9.3f} {bias:>16.2f}")


def main() -> None:
    rng = np.random.default_rng(SEED)
    x0 = rng.uniform(LO, HI, size=(INITIAL_POINTS, 4))
    lnl0 = np.array([true_lnl(r) for r in x0])
    ref = float(lnl0.max())
    spread = float(np.ptp(lnl0)) or 1.0

    # --- BO with a gradient-preserving (linear) target ---------------------
    def linear_target(*theta) -> float:
        return (ref - true_lnl(np.asarray(theta))) / spread

    y0 = np.array([(ref - v) / spread for v in lnl0]).reshape(-1, 1)
    learner = JaxActiveLearner(
        trainable_function=linear_target, bounds=np.array([LO, HI]),
        initial_data_x=x0, initial_data_y=y0,
        random_seed=SEED, config=JaxGPConfig(), outdir=None,
    )
    learner.run(total_steps=TOTAL_STEPS, steps_per_round=None)

    X = np.asarray(learner.data.query_points, float)
    lnl = np.array([true_lnl(r) for r in X])
    n_inf = int(np.sum(lnl - lnl.max() > -KEEP_DELTA))
    print(f"BO (linear target): {len(X)} pts, {n_inf} informative "
          f"({100*n_inf/len(X):.1f}%), best lnL={lnl.max():.1f}\n")

    print(f"{'variant':>11} {'conv':>6} {'width ratio':>12} {'corr':>9} {'max|bias|/sigma':>16}")

    # one-stage: no clipping, the same target BO used
    sc_lin = AdaptiveRobustScaler(
        soft_clipping=False, lower_clip_value=float(lnl.min()),
        focus_fraction=0.05, max_scale=float(np.ptp(lnl)),
    )
    sc_lin.initialize_with_data(lnl)
    sample_and_score("one-stage", X, lnl, sc_lin)

    # two-stage: refit on a clipped target, same points
    sc_clip = AdaptiveRobustScaler(
        soft_clipping=False, lower_clip_value=float(lnl.max() - KEEP_DELTA),
        focus_fraction=0.05, max_scale=5.0,
    )
    sc_clip.initialize_with_data(lnl)
    sample_and_score("two-stage", X, lnl, sc_clip)

    print("\ntruth: width ratio 1.00, corr 0.850, bias 0.00")


if __name__ == "__main__":
    main()
