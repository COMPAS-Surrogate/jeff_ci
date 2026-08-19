"""How many informative points, and how peaked a problem, can the GP handle?

Answers three questions with corrected scoring:

  1. How many training points near the peak are needed for the surrogate to
     resolve the posterior?
  2. Is the toy simply too hard (too peaked)? Does a broader peak help?
  3. matern32 vs matern52 -- is the earlier difference real, or seed noise?

SCORING FIXES (both bugs affected every earlier table):

  * The true posterior sigma is ``sqrt(diag(COV))``. Earlier scripts divided by
    ``WIDTH``, which is ``sqrt(SCALE)`` = 10x larger, so a *correct* posterior
    scored 0.10 and every "width ratio" was 10x optimistic.
  * "Peak region" was ``Delta lnL < KEEP_DELTA`` (=1000), which spans ~45 sigma
    -- far outside the posterior. Accuracy is now measured where the posterior
    actually lives, ``Delta lnL < POSTERIOR_DELTA``.

The meaningful accuracy target follows from the likelihood itself: the 1-sigma
contour sits at Delta lnL = 0.5, so a surrogate whose RMSE there approaches ~0.5
cannot resolve the posterior. Target: RMSE << 0.5.

Run:  python test_informative_fraction.py
"""
from __future__ import annotations

import itertools

import numpy as np

from cosmic_integration.lnl_surrogate.adaptive_robust_scalar import AdaptiveRobustScaler
from cosmic_integration.lnl_surrogate.jax_active_learner import (
    JaxGPConfig,
    JaxTrainingData,
    fit_exact_gp,
)
from cosmic_integration.lnl_surrogate.lnl_surrogate import BOUNDS

LO, HI = np.asarray(BOUNDS[0], float), np.asarray(BOUNDS[1], float)
MU = LO + 0.5 * (HI - LO)

SEED = 0
N_SEEDS = 5
N_TRAIN = 250
N_TEST = 600
POSTERIOR_DELTA = 5.0        # ~3 sigma in 4D; where the posterior lives
FLOOR_DELTA = 1.8e4          # dynamic range, as in a real run

# For a Gaussian, peak width and floor depth are the SAME knob: normalising the
# corner to a fixed Delta lnL cancels the width exactly. So sweep the dynamic
# range instead -- a shallower floor IS a broader peak.
FLOORS = [1.0e2, 1.0e3, 1.8e4]
FRAC_NEAR = [0.05, 0.15, 0.30, 0.60]
KERNELS = ["matern32", "matern52"]


def make_toy(floor_delta: float):
    """A 4D correlated-Gaussian peak whose box corner sits `floor_delta` below
    the peak. Smaller floor_delta == broader peak. Returns (true_lnl, sigma, cov)."""
    sigma = 0.05 * (HI - LO)
    c = np.eye(4)
    c[2, 3] = c[3, 2] = 0.85
    cov = np.diag(sigma) @ c @ np.diag(sigma)
    # Scale so a box corner sits at ~FLOOR_DELTA below the peak.
    corner = float(0.5 * (HI - MU) @ np.linalg.inv(cov) @ (HI - MU))
    cov = cov * (corner / floor_delta)
    prec = np.linalg.inv(cov)

    def true_lnl(theta):
        d = np.asarray(theta, float) - MU
        return float(-0.5 * d @ prec @ d)

    return true_lnl, np.sqrt(np.diag(cov)), cov


def make_set(n, frac_near, cov, true_lnl, rng):
    n_near = int(round(n * frac_near))
    parts = []
    if n_near:
        parts.append(np.clip(rng.multivariate_normal(MU, cov * 4.0, size=n_near), LO, HI))
    if n - n_near:
        parts.append(rng.uniform(LO, HI, size=(n - n_near, 4)))
    x = np.vstack(parts)
    return x, np.array([true_lnl(r) for r in x])


def evaluate(kernel, X, LNL, Xt, LNLt, best):
    scaler = AdaptiveRobustScaler(
        soft_clipping=False, lower_clip_value=float(LNL.min()),
        focus_fraction=0.05, max_scale=1e9, compression="sqrt",
    )
    scaler.initialize_with_data(LNL)
    y = -np.array([scaler.transform(v) for v in LNL]).reshape(-1, 1)
    model = fit_exact_gp(JaxTrainingData(X, y), np.array([LO, HI]),
                         config=JaxGPConfig(kernel=kernel), seed=SEED)
    mean, var = model.predict_f(Xt)
    pred = np.array([scaler.inverse_transform(-m) for m in mean[:, 0]])

    near = (LNLt - best) > -POSTERIOR_DELTA
    if near.sum() < 5:
        return np.nan, int(near.sum())
    return float(np.sqrt(np.mean((LNLt[near] - pred[near]) ** 2))), int(near.sum())


def main() -> None:
    print(f"accuracy measured where the posterior lives (Delta lnL < {POSTERIOR_DELTA})")
    print(f"1-sigma contour is at Delta lnL = 0.5, so RMSE must be << 0.5\n")

    for floor in FLOORS:
        true_lnl, sig, cov = make_toy(floor)
        frac_box = float(np.mean(sig / (HI - LO)))
        vol = float(np.prod(sig / (HI - LO)))
        print(f"=== dynamic range {floor:.3g} lnL  ->  peak sigma = {frac_box:.1%} of box "
              f"(volume ~{vol:.1e} of prior) ===")
        print(f"{'frac_near':>10} {'kernel':>9} {'n_post_test':>12} "
              f"{'RMSE @ posterior (median [min,max])':>40}")
        for frac, kernel in itertools.product(FRAC_NEAR, KERNELS):
            vals, npost = [], []
            for k in range(N_SEEDS):
                rng = np.random.default_rng(SEED + k)
                X, LNL = make_set(N_TRAIN, frac, cov, true_lnl, rng)
                rng_t = np.random.default_rng(SEED + 1000 + k)
                Xt, LNLt = make_set(N_TEST, 0.5, cov, true_lnl, rng_t)
                best = float(max(LNL.max(), LNLt.max(), true_lnl(MU)))
                v, npt = evaluate(kernel, X, LNL, Xt, LNLt, best)
                if np.isfinite(v):
                    vals.append(v); npost.append(npt)
            if not vals:
                print(f"{frac:>10.0%} {kernel:>9} {'--':>12}   no test points near the peak")
                continue
            v = np.array(vals)
            flag = "  <-- usable" if np.median(v) < 0.5 else ""
            print(f"{frac:>10.0%} {kernel:>9} {int(np.mean(npost)):>12} "
                  f"{np.median(v):>14.3g} [{v.min():.2g}, {v.max():.2g}]{flag}")
        print()


if __name__ == "__main__":
    main()
