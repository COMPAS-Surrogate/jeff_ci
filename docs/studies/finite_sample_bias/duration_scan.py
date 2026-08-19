"""Does the sfr_a--sfr_d ridge soften when the mock observation is shorter?

Hypothesis under test: the spurious ridge in the Jan 2026 posteriors is driven
(at least partly) by finite-COMPAS-sample bias in the likelihood. That bias is
controlled by

    t = N_eff(COMPAS run) / N_obs(data)

Holding the COMPAS run fixed and *shortening* the mock observation raises t
without touching the forward model, so:

  * if the ridge softens as duration drops -> finite-sample bias is contributing
  * if the ridge is unchanged             -> it is the GP / target scaling

We use the 32M run throughout (N_eff ~ 1266) and scan duration.

    duration   N_obs     t      Poisson-term bias
       3.0 yr   ~2250   0.56          -0.51
       1.0 yr    ~750   1.69          -0.23
       0.1 yr     ~75  16.9           -0.03

Measurement uncertainty is off, matching the "no uncertainty" slides where the
ridge is clearest.

Run:  python duration_scan.py [duration ...]
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

from cosmic_integration.lnl_surrogate.workflow import (
    SurrogateWorkflowConfig,
    run_surrogate_workflow,
)
from cosmic_integration.observation.mock_observation import MockObservation
from cosmic_integration.ratesSampler import BinnedCosmicIntegrator

TRUE = dict(p_Alpha=-0.325, p_Sigma=0.213, p_SFRa=0.012, p_SFRd=4.253)
TRUE_VEC = np.array([-0.325, 0.213, 0.012, 4.253])

REPO = Path(__file__).resolve().parents[3]
COMPAS_H5 = REPO / "tests/large_test_data/h5out_32M_reduced.h5"
OUTROOT = REPO / "docs/studies/finite_sample_bias/duration_scan_out"

SEED = 0


def make_observation(rates: np.ndarray, duration: float, path: Path) -> Path:
    if path.exists():
        logging.info("reusing %s", path)
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    MockObservation.generate_from_rates(
        rates=rates,
        params=TRUE_VEC,
        duration=float(duration),
        output_file=str(path),
        measurement_uncertainty=False,
        rng=np.random.default_rng(SEED),
    )
    return path


def main(durations: list[float]) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    ci = BinnedCosmicIntegrator.from_compas_fpath(str(COMPAS_H5))
    rates = ci.FindBinnedDetectionRate(**TRUE)
    logging.info("32M rate matrix: shape=%s total=%.1f/yr", rates.shape, rates.sum())

    for duration in durations:
        tag = f"{duration:g}yr".replace(".", "p")
        outdir = OUTROOT / tag
        obs_file = outdir / f"mock_obs_{tag}.h5"
        make_observation(rates, duration, obs_file)

        cfg = SurrogateWorkflowConfig(
            compas_h5=str(COMPAS_H5),
            observation_file=str(obs_file),
            outdir=str(outdir),
            initial_points=50,
            total_steps=200,
            steps_per_round=50,
            seed=SEED,
            postprocess_every=1,
            # 2000 is too short: bilby's burn-in estimate can exceed the chain
            # length and raise SamplerError, killing the whole BO run.
            mcmc_kwargs={"nwalkers": 32, "iterations": 10000},
            # A sampler hiccup in one round must not throw away the BO work.
            callback_fail_fast=False,
            # Keep the GP-vs-truth diagnostic cheap: each probe is a real LnL call.
            gp_truth_n_per_fraction=40,
            gp_truth_fractions=(0.05, 0.2),
        )
        logging.info("=== running duration=%g yr -> %s ===", duration, outdir)
        summary = run_surrogate_workflow(cfg)
        logging.info("done %s: %s", tag, summary.get("outdir"))


if __name__ == "__main__":
    args = [float(a) for a in sys.argv[1:]] or [3.0, 1.0, 0.1]
    main(args)
