"""Diagnostics for the GP surrogate at COMPAS-realistic scale (1% informative).

Uses ``toy_compas_lnl.py`` (4D correlated Gaussian peak, box + dynamic range
matched to a real run) to test four ideas raised while investigating whether
to subset/downweight far-tail training points (see ``test_subset.py``):

  A. input whitening  - can rescaling/rotating theta before the GP kernel
     recover the correlation a diagonal ARD kernel cannot represent?
  B. leave-one-out CV  - without ground truth, does the GP's predictive
     interval cover held-out informative points?
  C. bootstrap stability - does the GP's implied correlation structure
     (curvature at the best point) change wildly under resampling? If so,
     don't trust a single fit's correlation estimate.
  D. posterior-predictive re-evaluation - re-run the (expensive) true
     function at posterior draws and check the surrogate wasn't overconfident.

Run:  python test_diagnostics.py
"""
from __future__ import annotations

import shutil
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import cho_solve, cholesky, solve_triangular

from cosmic_integration.lnl_surrogate.adaptive_robust_scalar import AdaptiveRobustScaler
from cosmic_integration.lnl_surrogate.jax_active_learner import JaxActiveLearner, JaxGPConfig
import cosmic_integration.lnl_surrogate.nuts_sampler as nuts_sampler
from cosmic_integration.lnl_surrogate.nuts_sampler import run_nuts

from toy_compas_lnl import LO, HI, MU, WIDTH, KEEP_DELTA, true_lnl, build_training_set

ROOT = Path(__file__).parent / "diagnostics_out"
if ROOT.exists():
    shutil.rmtree(ROOT)

CONFIG = JaxGPConfig()


def make_scaler(lnl: np.ndarray) -> AdaptiveRobustScaler:
    sc = AdaptiveRobustScaler(
        soft_clipping=False,
        lower_clip_value=float(lnl.max() - KEEP_DELTA),
        focus_fraction=0.05,
        max_scale=5.0,
    )
    sc.initialize_with_data(lnl)
    return sc


def fit_gp(X: np.ndarray, lnl: np.ndarray, scaler: AdaptiveRobustScaler, seed: int = 0):
    y = -np.array([scaler.transform(v) for v in lnl]).reshape(-1, 1)
    learner = JaxActiveLearner(
        trainable_function=lambda *t: -scaler.transform(true_lnl(np.asarray(t))),
        bounds=np.asarray([LO, HI]), outdir=None,
        initial_data_x=X, initial_data_y=y, random_seed=seed, config=CONFIG,
    )
    return learner


# --------------------------------------------------------------------------
# Build one harsh (~1% informative) training set shared by all diagnostics.
# --------------------------------------------------------------------------
ts = build_training_set(2000, 20, seed=0)
X, LNL, informative = ts.X, ts.lnl, ts.informative
print(f"training set: {len(X)} pts, {informative.sum()} informative "
      f"({100*informative.mean():.2f}%)")
SCALER = make_scaler(LNL)


# ==========================================================================
# A. Whitening: does rescaling inputs by the informative-point covariance
#    let a diagonal ARD kernel recover the true correlation?
# ==========================================================================
def whiten(X_full: np.ndarray, informative_mask: np.ndarray):
    """Whitening transform from a shrinkage covariance of informative points.

    Shrinks toward a broad diagonal prior when there are too few informative
    points to estimate a full covariance reliably.
    """
    dim = X_full.shape[1]
    center = X_full[informative_mask].mean(axis=0) if informative_mask.sum() > 0 else X_full.mean(axis=0)
    prior_cov = np.diag(((HI - LO) / 6.0) ** 2)
    n_inf = int(informative_mask.sum())
    if n_inf >= dim + 1:
        cov_inf = np.cov(X_full[informative_mask], rowvar=False)
    else:
        cov_inf = prior_cov
    alpha = min(1.0, n_inf / (5.0 * dim))
    cov_hat = alpha * cov_inf + (1.0 - alpha) * prior_cov
    L = cholesky(cov_hat, lower=True)
    X_white = solve_triangular(L, (X_full - center).T, lower=True).T
    return X_white, center, L


def score_against_truth(samples: np.ndarray) -> dict:
    med, sd = np.median(samples, axis=0), samples.std(axis=0)
    return dict(
        width_ratio=float(np.mean(sd / WIDTH)),
        corr=float(np.corrcoef(samples[:, 2], samples[:, 3])[0, 1]),
        bias=float(np.max(np.abs(med - MU) / WIDTH)),
    )


def run_variant(name: str, X_fit: np.ndarray, bounds: np.ndarray, to_original) -> dict:
    out = ROOT / "whitening" / name
    md = out / "gp_model"
    md.mkdir(parents=True)
    ys = -np.array([SCALER.transform(v) for v in LNL]).reshape(-1, 1)
    learner = JaxActiveLearner(
        trainable_function=lambda *t: -SCALER.transform(true_lnl(np.asarray(t))),
        bounds=bounds, outdir=str(md),
        initial_data_x=X_fit, initial_data_y=ys, random_seed=0, config=CONFIG,
    )
    learner.save_model(round_idx=0)
    SCALER.save(str(md))

    saved_bounds = nuts_sampler.BOUNDS
    nuts_sampler.BOUNDS = bounds
    try:
        s = run_nuts(lnl_model_path=str(md / "models"), outdir=str(out / "run"),
                      target="mean", round_idx=0, num_warmup=500, num_samples=1500,
                      num_chains=2, seed=0)
    finally:
        nuts_sampler.BOUNDS = saved_bounds
    smp = np.load(out / "run" / "posterior_samples.npy")
    smp_orig = to_original(smp)
    metrics = score_against_truth(smp_orig)
    metrics["converged"] = s["converged"]
    return metrics


print("\n=== A. whitening vs raw (NUTS, scored against truth) ===")
X_white, center, L = whiten(X, informative)
white_corners = np.array([
    [lo_i if b & (1 << d) == 0 else hi_i for d, (lo_i, hi_i) in enumerate(zip(LO, HI))]
    for b in range(16)
])
white_corners_t = solve_triangular(L, (white_corners - center).T, lower=True).T
margin = 0.05 * (white_corners_t.max(axis=0) - white_corners_t.min(axis=0))
white_bounds = np.array([
    np.minimum(white_corners_t.min(axis=0), X_white.min(axis=0)) - margin,
    np.maximum(white_corners_t.max(axis=0), X_white.max(axis=0)) + margin,
])

results = {
    "raw": run_variant("raw", X, np.array([LO, HI]), lambda s: s),
    "whitened": run_variant("whitened", X_white, white_bounds, lambda s: center + s @ L.T),
}
print(f"{'variant':>9} {'conv':>5} {'width ratio':>12} {'corr':>8} {'max|bias|/sigma':>16}")
for name, m in results.items():
    print(f"{name:>9} {str(m['converged']):>5} {m['width_ratio']:>12.2f} {m['corr']:>8.3f} {m['bias']:>16.2f}")
print("truth: width ratio 1.00, corr 0.850, bias 0.00")


# ==========================================================================
# B. Leave-one-out calibration: refit without each held-out informative
#    point, check whether its true lnL falls inside the GP's predictive
#    interval.
# ==========================================================================
print("\n=== B. leave-one-out calibration (informative points) ===")
inf_idx = np.flatnonzero(informative)
rng = np.random.default_rng(1)
loo_idx = rng.choice(inf_idx, size=min(6, len(inf_idx)), replace=False)
z_scores = []
for i in loo_idx:
    mask = np.ones(len(X), bool)
    mask[i] = False
    learner = fit_gp(X[mask], LNL[mask], SCALER, seed=int(i))
    mean, var = learner.model.predict_f(X[i : i + 1])
    true_y = -SCALER.transform(LNL[i])
    z = float((true_y - mean[0, 0]) / np.sqrt(max(var[0, 0], 1e-12)))
    z_scores.append(z)
z_scores = np.array(z_scores)
print(f"held-out z-scores: {np.round(z_scores, 2)}")
print(f"|z| <= 2 (should be ~most of them): {int(np.sum(np.abs(z_scores) <= 2))}/{len(z_scores)}")


# ==========================================================================
# C. Bootstrap stability of the implied correlation at the best point.
# ==========================================================================
print("\n=== C. bootstrap stability of implied correlation (sfr_a, sfr_d) ===")
best_theta = X[np.argmax(LNL)]


def implied_correlation(learner: JaxActiveLearner, theta: np.ndarray) -> float:
    """Correlation implied by the local curvature of the GP mean at theta."""
    def scalar_mean(t):
        m, _ = learner.model.predict_f_jax(t.reshape(1, -1))
        return m[0]

    hess = np.asarray(jax.hessian(scalar_mean)(jnp.asarray(theta, dtype=jnp.float64)))
    hess = -0.5 * (hess + hess.T)  # symmetrize; mean is being minimised (neg-lnL-like)
    try:
        cov = np.linalg.inv(hess)
    except np.linalg.LinAlgError:
        return float("nan")
    denom = np.sqrt(cov[2, 2] * cov[3, 3])
    if not np.isfinite(denom) or denom <= 0:
        return float("nan")
    return float(cov[2, 3] / denom)


n_boot = 8
boot_corrs = []
rng = np.random.default_rng(2)
for b in range(n_boot):
    idx = rng.choice(len(X), size=len(X), replace=True)
    learner = fit_gp(X[idx], LNL[idx], SCALER, seed=b)
    boot_corrs.append(implied_correlation(learner, best_theta))
boot_corrs = np.array(boot_corrs)
print(f"bootstrap implied corr(sfr_a, sfr_d): {np.round(boot_corrs, 3)}")
print(f"mean={np.nanmean(boot_corrs):.3f} std={np.nanstd(boot_corrs):.3f} (truth: 0.850)")
if np.nanstd(boot_corrs) > 0.3 or np.nanmean(np.abs(boot_corrs)) < 0.3:
    print("-> unstable / weak: do not trust this fit's correlation, consider a better kernel.")
else:
    print("-> reasonably stable across resamples.")


# ==========================================================================
# D. Posterior-predictive re-evaluation: draw posterior samples, re-evaluate
#    the (stand-in for expensive) true function there, compare to the
#    surrogate's own prediction and its claimed uncertainty.
# ==========================================================================
print("\n=== D. posterior-predictive re-evaluation ===")
best_variant = "whitened" if results["whitened"]["width_ratio"] < results["raw"]["width_ratio"] else "raw"
run_dir = ROOT / "whitening" / best_variant / "run"
smp = np.load(run_dir / "posterior_samples.npy")
if best_variant == "whitened":
    smp_orig = center + smp @ L.T
    gp_x = smp
    learner_for_eval = fit_gp(X_white, LNL, SCALER, seed=0)
else:
    smp_orig = smp
    gp_x = smp
    learner_for_eval = fit_gp(X, LNL, SCALER, seed=0)

k = min(10, len(smp_orig))
draw_idx = np.random.default_rng(3).choice(len(smp_orig), size=k, replace=False)
resid = []
for i in draw_idx:
    theta_orig = np.clip(smp_orig[i], LO, HI)
    true_y = true_lnl(theta_orig)
    mean, var = learner_for_eval.model.predict_f(gp_x[i : i + 1])
    pred_y = SCALER.inverse_transform(-mean[0, 0])
    sigma_y = float(np.sqrt(max(var[0, 0], 1e-12))) * SCALER.scale
    z = (true_y - pred_y) / max(sigma_y, 1e-9)
    resid.append(z)
resid = np.array(resid)
print(f"({best_variant}) re-evaluated z-scores at posterior draws: {np.round(resid, 2)}")
print(f"mean z={np.mean(resid):.2f}, std z={np.std(resid):.2f} (well-calibrated: mean~0, std~1)")
