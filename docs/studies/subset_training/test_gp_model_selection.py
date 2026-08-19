"""GP model selection on a FIXED training set -- no BO.

Surrogate quality is a supervised regression question, so it should be measured
as one. Fixing the training set:

  * removes BO's stochasticity, which was confounding every earlier comparison;
  * keeps the array shapes constant, so JAX compiles once instead of once per
    step (BO was doing ~127 XLA compilations per step);
  * lets us score held-out calibration, which needs no ground truth and so
    transfers to the real COMPAS problem.

Two training regimes:
  random  - uniform over the box. The pessimistic case: few points near the
            peak, exactly the regime the real runs were stuck in.
  mixed   - uniform plus a peak-concentrated component, i.e. what a working
            acquisition scheme would hand us.

Metrics (per config):
  rmse_peak / nlpd_peak / cov68_peak - held-out points within KEEP_DELTA of the
      best. This is the region the posterior occupies, so it is the one that
      matters; `cov68` is the fraction inside the GP's own 1-sigma interval and
      should be ~0.68 if the uncertainties are honest.
  rmse_all - held-out points anywhere, to catch configs that fit the peak by
      abandoning the rest of the space (which would break acquisition).

Run:  python test_gp_model_selection.py
"""
from __future__ import annotations

import itertools
import time

import numpy as np

from cosmic_integration.lnl_surrogate.adaptive_robust_scalar import AdaptiveRobustScaler
from cosmic_integration.lnl_surrogate.jax_active_learner import (
    JaxGPConfig,
    JaxTrainingData,
    fit_exact_gp,
)

from toy_compas_lnl import LO, HI, COV, MU, KEEP_DELTA, true_lnl

SEED = 0
N_SEEDS = 5   # seed-to-seed scatter is large; one draw is not a result
N_TRAIN = 250
N_TEST = 400

KERNELS = ["matern32", "matern52"]
COMPRESSIONS = ["none", "sqrt", "log", "softlog"]


def make_set(n: int, frac_near: float, rng) -> tuple[np.ndarray, np.ndarray]:
    n_near = int(round(n * frac_near))
    x = np.vstack([
        np.clip(rng.multivariate_normal(MU, COV * 4.0, size=n_near), LO, HI),
        rng.uniform(LO, HI, size=(n - n_near, 4)),
    ]) if n_near else rng.uniform(LO, HI, size=(n, 4))
    return x, np.array([true_lnl(r) for r in x])


def evaluate(kernel: str, compression: str, X, LNL, Xt, LNLt, best) -> dict:
    scaler = AdaptiveRobustScaler(
        soft_clipping=False, lower_clip_value=float(LNL.min()),
        focus_fraction=0.05, max_scale=1e9, compression=compression,
    )
    scaler.initialize_with_data(LNL)
    y = -np.array([scaler.transform(v) for v in LNL]).reshape(-1, 1)

    model = fit_exact_gp(
        JaxTrainingData(X, y), np.array([LO, HI]),
        config=JaxGPConfig(kernel=kernel), seed=SEED,
    )
    mean, var = model.predict_f(Xt)
    # Back to lnL space: that is where accuracy actually matters.
    pred = np.array([scaler.inverse_transform(-m) for m in mean[:, 0]])
    # Local slope of the inverse transform, to carry sigma into lnL units.
    eps = 1e-4
    slope = np.abs(np.array([
        (scaler.inverse_transform(-(m - eps)) - scaler.inverse_transform(-(m + eps))) / (2 * eps)
        for m in mean[:, 0]
    ]))
    sigma = np.sqrt(np.maximum(var[:, 0], 1e-300)) * slope

    resid = LNLt - pred
    near = (LNLt - best) > -KEEP_DELTA
    out = {"kernel": kernel, "compression": compression}
    out["rmse_all"] = float(np.sqrt(np.mean(resid**2)))
    if near.sum() >= 5:
        r, s = resid[near], np.maximum(sigma[near], 1e-12)
        out["rmse_peak"] = float(np.sqrt(np.mean(r**2)))
        out["nlpd_peak"] = float(np.mean(0.5 * np.log(2 * np.pi * s**2) + r**2 / (2 * s**2)))
        out["cov68_peak"] = float(np.mean(np.abs(r) <= s))
        out["n_peak"] = int(near.sum())
    else:
        out.update(rmse_peak=np.nan, nlpd_peak=np.nan, cov68_peak=np.nan, n_peak=int(near.sum()))
    return out


def main() -> None:
    for regime, frac in (("random", 0.0), ("mixed", 0.25)):
        print(f"\n=== training regime: {regime}  ({N_SEEDS} seeds, mean +- sd) ===")
        print(f"{'kernel':>9} {'compress':>9} {'rmse_peak':>18} {'cov68_peak':>14} {'rmse_all':>18}")
        summary = []
        for kernel, comp in itertools.product(KERNELS, COMPRESSIONS):
            rp, cv, ra = [], [], []
            for k in range(N_SEEDS):
                X, LNL = make_set(N_TRAIN, frac, np.random.default_rng(SEED + k))
                Xt, LNLt = make_set(N_TEST, 0.35, np.random.default_rng(SEED + 1000 + k))
                best = float(max(LNL.max(), LNLt.max()))
                r = evaluate(kernel, comp, X, LNL, Xt, LNLt, best)
                if np.isfinite(r["rmse_peak"]):
                    rp.append(r["rmse_peak"]); cv.append(r["cov68_peak"]); ra.append(r["rmse_all"])
            if not rp:
                continue
            rp, cv, ra = np.array(rp), np.array(cv), np.array(ra)
            summary.append((float(np.median(rp)), kernel, comp, rp, cv, ra))
            print(f"{kernel:>9} {comp:>9} "
                  f"{np.median(rp):>9.3g} [{rp.min():.2g},{rp.max():.2g}] "
                  f"{np.mean(cv):>8.2f}+-{np.std(cv):<4.2f} "
                  f"{np.median(ra):>9.3g} [{ra.min():.2g},{ra.max():.2g}]")
        if summary:
            summary.sort()
            med, kernel, comp, rp, cv, ra = summary[0]
            print(f"  -> best median rmse_peak: {kernel} + {comp} = {med:.3g} "
                  f"(spread {rp.min():.2g}-{rp.max():.2g}, cov68 {np.mean(cv):.2f})")


if __name__ == "__main__":
    main()
