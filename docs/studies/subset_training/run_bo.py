"""Run real BO (active learning) on the toy COMPAS-scale lnL and see whether
targeted acquisition reaches a higher informative fraction than random
sampling would, for the same budget.

Also fits a final GP + NUTS on the BO-acquired data and overlays the acquired
points on the lnL surface plot (see ``plot_lnl_surface.py``).

Run:  python run_bo.py
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path

import numpy as np

from cosmic_integration.lnl_surrogate.adaptive_robust_scalar import AdaptiveRobustScaler
from cosmic_integration.lnl_surrogate.jax_active_learner import JaxActiveLearner, JaxGPConfig
from cosmic_integration.lnl_surrogate.nuts_sampler import run_nuts

from plot_lnl_surface import plot_surface
from toy_compas_lnl import LO, HI, MU, WIDTH, KEEP_DELTA, true_lnl

OUT = Path(__file__).parent / "bo_run_out"
if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True)

INITIAL_POINTS = 60
TOTAL_STEPS = 150  # BO-selected points added on top of the initial batch

rng = np.random.default_rng(0)


def true_lnl_row(*theta) -> float:
    return true_lnl(np.asarray(theta))


# --- initial random batch, used to calibrate the scaler (mirrors production) ---
x0 = rng.uniform(LO, HI, size=(INITIAL_POINTS, 4))
lnl0 = np.array([true_lnl_row(*row) for row in x0])
print(f"initial batch: min={lnl0.min():.1f} max={lnl0.max():.1f} "
      f"informative={(lnl0 - lnl0.max() > -KEEP_DELTA).sum()}/{INITIAL_POINTS}")

scaler = AdaptiveRobustScaler(
    soft_clipping=False,
    lower_clip_value=float(lnl0.max() - KEEP_DELTA),
    focus_fraction=0.05,
    max_scale=5.0,
)
scaler.initialize_with_data(lnl0)


def trainable(*theta) -> float:
    return -scaler.transform(true_lnl_row(*theta))


y0 = -np.array([scaler.transform(v) for v in lnl0]).reshape(-1, 1)

t0 = time.time()
learner = JaxActiveLearner(
    trainable_function=trainable,
    bounds=np.array([LO, HI]),
    initial_data_x=x0, initial_data_y=y0,
    random_seed=0, config=JaxGPConfig(), outdir=str(OUT),
)
learner.run(total_steps=TOTAL_STEPS, steps_per_round=None)
print(f"BO done: {len(learner.data.query_points)} total points, {time.time()-t0:.0f}s")

X_final = learner.data.query_points
LNL_final = np.array([true_lnl(row) for row in X_final])
delta = LNL_final - LNL_final.max()
informative = delta > -KEEP_DELTA
print(f"BO-acquired set: {len(X_final)} pts, {informative.sum()} informative "
      f"({100*informative.mean():.1f}%)")

# --- matched-size random baseline, for comparison ---
x_rand = rng.uniform(LO, HI, size=(len(X_final), 4))
lnl_rand = np.array([true_lnl(row) for row in x_rand])
inf_rand = (lnl_rand - lnl_rand.max() > -KEEP_DELTA)
print(f"random baseline (same budget): {inf_rand.sum()} informative "
      f"({100*inf_rand.mean():.1f}%)")

# --- fit + NUTS on the final BO data, scored against the known truth ---
learner.save_model(round_idx=0)
scaler.save(str(OUT))
s = run_nuts(lnl_model_path=str(OUT / "models"), outdir=str(OUT / "run"),
             target="mean", round_idx=0, num_warmup=500, num_samples=1500,
             num_chains=2, seed=0)
smp = np.load(OUT / "run" / "posterior_samples.npy")
med, sd = np.median(smp, axis=0), smp.std(axis=0)
wr = float(np.mean(sd / WIDTH))
corr = float(np.corrcoef(smp[:, 2], smp[:, 3])[0, 1])
bias = float(np.max(np.abs(med - MU) / WIDTH))
print(f"\nBO-trained surrogate posterior: converged={s['converged']} "
      f"width_ratio={wr:.2f} corr={corr:.3f} (truth 0.850) bias/sigma={bias:.2f}")

plot_surface(training_points=X_final, training_lnl=LNL_final, out_path=str(OUT / "lnl_surface_bo.png"))
