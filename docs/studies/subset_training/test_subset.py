"""Does dropping the far-tail training points make the surrogate better?

Mimics the real training set: a 4D correlated-Gaussian peak, with only a small
fraction of points anywhere near the posterior and the rest at terrible lnL
(the observed 86-90% plateau in the moderate scenario; ~99% in the harsh one,
matching the 3-yr run). Compares strategies for what to feed the GP, scored
against a known truth.

  all     - keep everything, hard-clipped at the floor (current default)
  topN    - keep only points within `keep_delta` lnL of the best
  thinned - keep all informative points + a random 10% of the floor points

Two scenarios are run:
  moderate - ~12-24% informative points (as in the original exploratory test)
  harsh    - ~1% informative points (matching the observed 3-yr run)

Run:  python test_subset.py
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from cosmic_integration.lnl_surrogate.adaptive_robust_scalar import AdaptiveRobustScaler
from cosmic_integration.lnl_surrogate.jax_active_learner import JaxActiveLearner, JaxGPConfig
from cosmic_integration.lnl_surrogate.lnl_surrogate import BOUNDS
from cosmic_integration.lnl_surrogate.nuts_sampler import run_nuts

ROOT = Path(__file__).parent / "subset_test_out"
if ROOT.exists():
    shutil.rmtree(ROOT)

lo, hi = np.asarray(BOUNDS[0], float), np.asarray(BOUNDS[1], float)
MU = lo + 0.5 * (hi - lo)
WIDTH = 0.05 * (hi - lo)
C = np.eye(4)
C[2, 3] = C[3, 2] = 0.85
COV = np.diag(WIDTH) @ C @ np.diag(WIDTH)
PREC = np.linalg.inv(COV)


def true_lnl(*theta):
    d = np.asarray(theta, float) - MU
    return float(-0.5 * d @ PREC @ d)


def build_training_set(n_total: int, n_near: int, seed: int = 0):
    """Training set shaped like the real one: `n_near` points drawn near the
    peak, the rest uniform over the box (the "floor")."""
    rng = np.random.default_rng(seed)
    x_far = rng.uniform(lo, hi, size=(n_total - n_near, 4))
    x_near = np.clip(rng.multivariate_normal(MU, COV * 4.0, size=n_near), lo, hi)
    X = np.vstack([x_near, x_far])
    LNL = np.array([true_lnl(*r) for r in X])
    return X, LNL, rng


SCENARIOS = {
    # (n_total, n_near, keep_delta) -> ~24% informative, as originally explored
    "moderate": (250, 30, 50.0),
    # matches the observed 3-yr run: ~1% informative, ~99% at the floor
    "harsh": (2000, 20, 10.0),
}

for scenario, (n_total, n_near, keep_delta) in SCENARIOS.items():
    X, LNL, rng = build_training_set(n_total, n_near)
    delta = LNL - LNL.max()
    informative = delta > -keep_delta
    print(f"\n=== scenario: {scenario} (keep_delta={keep_delta}) ===")
    print(f"training set: {len(X)} pts, {informative.sum()} informative "
          f"({100*informative.mean():.1f}%), {100*(~informative).mean():.1f}% at the floor")

    STRATEGIES = {
        "all": np.ones(len(X), bool),
        "topN": informative,
        "thinned": informative | (rng.random(len(X)) < 0.10),
    }

    print(f"\n{'strategy':>9} {'n_train':>8} {'conv':>5} "
          f"{'width ratio (mean)':>19} {'corr(a,d)':>10} {'max |bias|/sigma':>17} {'lengthscales':>34}")
    for name, mask in STRATEGIES.items():
        if mask.sum() < 8:
            print(f"{name:>9} {int(mask.sum()):>8}  -- too few points to fit a GP, skipped --")
            continue

        out = ROOT / scenario / name
        md = out / "gp_model"
        md.mkdir(parents=True)

        xs, ls = X[mask], LNL[mask]
        sc = AdaptiveRobustScaler(soft_clipping=False, lower_clip_value=float(LNL.max() - keep_delta),
                                  focus_fraction=0.05, max_scale=5.0)
        sc.initialize_with_data(ls)
        ys = -np.array([sc.transform(v) for v in ls]).reshape(-1, 1)

        learner = JaxActiveLearner(
            trainable_function=lambda *t: -sc.transform(true_lnl(*t)),
            bounds=np.asarray(BOUNDS, float), outdir=str(md),
            initial_data_x=xs, initial_data_y=ys, random_seed=0, config=JaxGPConfig(),
        )
        learner.save_model(round_idx=0)
        sc.save(str(md))

        try:
            ls_fit = np.asarray(
                learner.model.posterior.prior.kernel.lengthscale, dtype=float
            )
        except Exception:
            ls_fit = np.full(4, np.nan)

        s = run_nuts(lnl_model_path=str(md / "models"), outdir=str(out / "run"),
                      target="mean", round_idx=0, num_warmup=500, num_samples=1500,
                      num_chains=2, seed=0)
        smp = np.load(out / "run" / "posterior_samples.npy")
        med, sd = np.median(smp, axis=0), smp.std(axis=0)
        wr = float(np.mean(sd / WIDTH))
        corr = float(np.corrcoef(smp[:, 2], smp[:, 3])[0, 1])
        bias = float(np.max(np.abs(med - MU) / WIDTH))
        lstr = " ".join(f"{v:.3g}" for v in ls_fit / (hi - lo))
        print(f"{name:>9} {int(mask.sum()):>8} {str(s['converged']):>5} "
              f"{wr:>19.2f} {corr:>10.3f} {bias:>17.2f} {lstr:>34}")

    print(f"\ntruth: width ratio 1.00, corr 0.850, bias 0.00")
