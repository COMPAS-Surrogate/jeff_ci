"""Does a LOCAL GP fix the too-wide posterior?

The lnL surface is non-stationary: a near-flat plateau over most of the box plus
a narrow peak. A stationary kernel (Matern52 with one lengthscale per dimension)
cannot represent both, so it compromises -- long lengthscales that smooth the
peak, which widens the posterior. That is the likely reason the width plateaus
at ~2.5x and does not improve with budget.

If that diagnosis is right, fitting a SECOND GP restricted to the peak region --
where the surface *is* roughly stationary, and where no output compression is
needed because the local dynamic range is small -- should recover the width.

  global  - one GP on all points, sqrt-compressed target (current best)
  local   - GP on points within `radius` lnL of the best, linear target,
            lengthscales refit on that subset only

Acquisition is identical in both cases (the same BO run supplies the points), so
this isolates the surrogate-fitting choice.

Run:  python test_local_gp.py
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from cosmic_integration.lnl_surrogate.adaptive_robust_scalar import AdaptiveRobustScaler
from cosmic_integration.lnl_surrogate.jax_active_learner import JaxActiveLearner, JaxGPConfig
from cosmic_integration.lnl_surrogate.nuts_sampler import run_nuts

from toy_compas_lnl import LO, HI, MU, WIDTH, KEEP_DELTA, true_lnl

OUT = Path(__file__).parent / "local_gp_out"
if OUT.exists():
    shutil.rmtree(OUT)

INITIAL_POINTS = 60
TOTAL_STEPS = 150
SEED = 0
RADII = [KEEP_DELTA, 5 * KEEP_DELTA]


def score(smp: np.ndarray) -> tuple[float, float, float]:
    med, sd = np.median(smp, axis=0), smp.std(axis=0)
    return (float(np.mean(sd / WIDTH)),
            float(np.corrcoef(smp[:, 2], smp[:, 3])[0, 1]),
            float(np.max(np.abs(med - MU) / WIDTH)))


def fit_and_sample(name: str, X: np.ndarray, lnl: np.ndarray, scaler) -> None:
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
    print(f"{name:>22} {len(X):>8} {str(s['converged']):>6} "
          f"{wr:>12.2f} {corr:>9.3f} {bias:>12.2f}")


def main() -> None:
    rng = np.random.default_rng(SEED)
    x0 = rng.uniform(LO, HI, size=(INITIAL_POINTS, 4))
    lnl0 = np.array([true_lnl(r) for r in x0])

    # --- one BO run, shared by every variant -------------------------------
    scaler_bo = AdaptiveRobustScaler(
        soft_clipping=False, lower_clip_value=float(lnl0.min()),
        focus_fraction=0.05, max_scale=1e9, compression="sqrt",
    )
    scaler_bo.initialize_with_data(lnl0)
    y0 = -np.array([scaler_bo.transform(v) for v in lnl0]).reshape(-1, 1)
    learner = JaxActiveLearner(
        trainable_function=lambda *t: -scaler_bo.transform(true_lnl(np.asarray(t))),
        bounds=np.array([LO, HI]), initial_data_x=x0, initial_data_y=y0,
        random_seed=SEED, config=JaxGPConfig(), outdir=None, refit_every=15,
    )
    learner.run(total_steps=TOTAL_STEPS, exploration_fraction=1 / 3, cycle_length=30)
    X = np.asarray(learner.data.query_points, float)
    LNL = np.array([true_lnl(r) for r in X])
    print(f"BO acquired {len(X)} pts, best lnL={LNL.max():.1f}, "
          f"{int(np.sum(LNL - LNL.max() > -KEEP_DELTA))} informative\n")

    print(f"{'variant':>22} {'n_train':>8} {'conv':>6} "
          f"{'width ratio':>12} {'corr':>9} {'|bias|/sig':>12}")

    fit_and_sample("global (sqrt, all)", X, LNL, scaler_bo)

    for radius in RADII:
        mask = (LNL - LNL.max()) > -radius
        if mask.sum() < 12:
            print(f"{'local r=' + str(int(radius)):>22} {int(mask.sum()):>8}  -- too few points --")
            continue
        # Local range is small, so no compression is needed.
        sc = AdaptiveRobustScaler(
            soft_clipping=False, lower_clip_value=float(LNL[mask].min()),
            focus_fraction=0.5, max_scale=1e9, compression="none",
        )
        sc.initialize_with_data(LNL[mask])
        fit_and_sample(f"local r={int(radius)}", X[mask], LNL[mask], sc)

    print("\ntruth: width ratio 1.00, corr 0.850, bias 0.00")


if __name__ == "__main__":
    main()
