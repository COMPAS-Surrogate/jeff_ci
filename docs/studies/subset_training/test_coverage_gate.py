"""Overconfidence gate: simulation-based calibration for the GP surrogate.

Being too WIDE is conservative and safe. Being too NARROW means claiming a
precision we do not have, and can exclude the true value -- the one failure mode
that invalidates a result rather than merely weakening it. Width ratio alone
cannot tell us which we have, because it is scored against a truth we will not
know for real data.

Simulation-based calibration does tell us, and needs no ground-truth posterior:
draw a true parameter vector, build the surrogate, sample its posterior, and
record the quantile at which the truth sits in each 1D marginal. If the
surrogate is calibrated, those quantiles are Uniform(0,1) across simulations.

  quantiles pile up in the MIDDLE  -> posterior too WIDE   (conservative)
  quantiles pile up at the EDGES   -> posterior too NARROW (OVERCONFIDENT)

Empirical coverage of the nominal 68% and 90% intervals is reported directly:
coverage below nominal is the dangerous direction, and this is the gate that
should run before any result is trusted -- especially after narrowing the prior,
which pushes widths down and can turn a safe surrogate into an overconfident one.

Run:  python test_coverage_gate.py
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
from scipy import stats

from cosmic_integration.lnl_surrogate.adaptive_robust_scalar import AdaptiveRobustScaler
from cosmic_integration.lnl_surrogate.jax_active_learner import JaxActiveLearner, JaxGPConfig
from cosmic_integration.lnl_surrogate.lnl_surrogate import BOUNDS, PARAMETERS
from cosmic_integration.lnl_surrogate.nuts_sampler import run_nuts

LO, HI = np.asarray(BOUNDS[0], float), np.asarray(BOUNDS[1], float)

OUT = Path(__file__).parent / "coverage_out"
if OUT.exists():
    shutil.rmtree(OUT)

N_SIM = 24
N_TRAIN = 250
FRAC_NEAR = 0.30          # the sweet spot from test_informative_fraction.py
FLOOR_DELTA = 1.0e3       # regime where the GP demonstrably works
KERNEL = "matern52"
SEED = 0


def make_toy(centre: np.ndarray):
    sigma = 0.05 * (HI - LO)
    c = np.eye(4)
    c[2, 3] = c[3, 2] = 0.85
    cov = np.diag(sigma) @ c @ np.diag(sigma)
    corner = float(0.5 * (HI - centre) @ np.linalg.inv(cov) @ (HI - centre))
    cov = cov * (corner / FLOOR_DELTA)
    prec = np.linalg.inv(cov)

    def true_lnl(theta):
        d = np.asarray(theta, float) - centre
        return float(-0.5 * d @ prec @ d)

    return true_lnl, cov


def one_simulation(idx: int) -> np.ndarray | None:
    """Return the quantile of the truth in each 1D marginal, or None on failure."""
    rng = np.random.default_rng(SEED + idx)
    # Keep the peak away from the box edge so the truth is interior.
    centre = LO + (0.25 + 0.5 * rng.random(4)) * (HI - LO)
    true_lnl, cov = make_toy(centre)

    n_near = int(round(N_TRAIN * FRAC_NEAR))
    X = np.vstack([
        np.clip(rng.multivariate_normal(centre, cov * 4.0, size=n_near), LO, HI),
        rng.uniform(LO, HI, size=(N_TRAIN - n_near, 4)),
    ])
    LNL = np.array([true_lnl(r) for r in X])

    scaler = AdaptiveRobustScaler(
        soft_clipping=False, lower_clip_value=float(LNL.min()),
        focus_fraction=0.05, max_scale=1e9, compression="sqrt",
    )
    scaler.initialize_with_data(LNL)
    y = -np.array([scaler.transform(v) for v in LNL]).reshape(-1, 1)

    md = OUT / f"sim_{idx}" / "gp_model"
    md.mkdir(parents=True)
    learner = JaxActiveLearner(
        trainable_function=lambda *t: -scaler.transform(true_lnl(np.asarray(t))),
        bounds=np.array([LO, HI]), outdir=str(md),
        initial_data_x=X, initial_data_y=y,
        random_seed=SEED, config=JaxGPConfig(kernel=KERNEL),
    )
    learner.save_model(round_idx=0)
    scaler.save(str(md))

    try:
        run_nuts(lnl_model_path=str(md / "models"), outdir=str(md.parent / "run"),
                 target="mean", round_idx=0, num_warmup=300, num_samples=1000,
                 num_chains=2, seed=SEED)
        smp = np.load(md.parent / "run" / "posterior_samples.npy")
    except Exception:
        return None
    if smp.shape[0] < 100:
        return None
    return np.array([float(np.mean(smp[:, i] < centre[i])) for i in range(4)])


def main() -> None:
    print(f"simulation-based calibration: {N_SIM} runs, {FRAC_NEAR:.0%} informative, "
          f"dynamic range {FLOOR_DELTA:.0g}\n")
    rows = []
    for i in range(N_SIM):
        q = one_simulation(i)
        if q is not None:
            rows.append(q)
        print(f"  sim {i + 1}/{N_SIM}" + ("" if q is not None else "  (failed)"), end="\r")
    print(" " * 40, end="\r")

    if len(rows) < 5:
        print(f"only {len(rows)} usable simulations -- cannot assess calibration")
        return

    Q = np.array(rows)
    print(f"usable simulations: {len(Q)}\n")
    print(f"{'param':>8} {'cov68':>8} {'cov90':>8} {'KS p':>8}   verdict")
    for i, name in enumerate(PARAMETERS):
        q = Q[:, i]
        c68 = float(np.mean(np.abs(q - 0.5) <= 0.34))
        c90 = float(np.mean(np.abs(q - 0.5) <= 0.45))
        ks = float(stats.kstest(q, "uniform").pvalue)
        if c68 < 0.68 - 0.15:
            verdict = "OVERCONFIDENT (dangerous)"
        elif c68 > 0.68 + 0.15:
            verdict = "too wide (conservative)"
        else:
            verdict = "calibrated"
        print(f"{name:>8} {c68:>8.2f} {c90:>8.2f} {ks:>8.3f}   {verdict}")

    q = Q.ravel()
    c68 = float(np.mean(np.abs(q - 0.5) <= 0.34))
    print(f"\noverall cov68 = {c68:.2f} (nominal 0.68), "
          f"KS p = {stats.kstest(q, 'uniform').pvalue:.3f}")
    print("\ncoverage BELOW nominal is the dangerous direction: the surrogate would")
    print("be claiming precision it does not have.")


if __name__ == "__main__":
    main()
