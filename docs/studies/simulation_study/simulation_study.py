"""
Simulation study for cosmic integration with active learning and 1D likelihood sanity checks.

This script performs the following steps:

1. Build the fiducial detection-rate matrix using the COMPAS catalogue
   distributed alongside this study (:mod:`h5out_5M.h5`) and the parameter
   vector :math:`\lambda = (\alpha, \sigma, a_{\rm SF}, d_{\rm SF}) =
   (-0.325, 0.213, 0.012, 4.253)`.
2. Generate a mock LVK-like catalogue corresponding to a chosen observing
   duration (default 0.1 year). Optionally include stochastic measurement
   uncertainties.
3. Perform a brute-force 1D log-likelihood scan along each of the four
   parameters to check the likelihood behavior.
4. Train the Gaussian-process active learner (:class:`LnLSurrogate`)
   on the generated catalogue and store all intermediary artefacts
   (cached likelihood evaluations, GP checkpoints, diagnostic plots)
   in a dedicated output directory.

Usage
-----

Run the full study from the repository root::

    PYTHONPATH=src python docs/studies/simulation_study/simulation_study.py [--noise] [--force] [--quick]

By default the script produces three outputs inside
``docs/studies/simulation_study/outputs``. Runs that include measurement noise
and noise-free runs are isolated in ``outputs/noise`` and
``outputs/no_noise`` respectively:

``full_mock_observation.h5``
    One-year catalogue with 578 events.

``mock_observation.h5``
    Shorter catalogue (default 0.1 year) used for active learning and likelihood scans.

``analysis``
    Directory with GP training logs, cached likelihood evaluations, and trained surrogate weights.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from cosmic_integration.lnl_surrogate.lnl_surrogate import LnLSurrogate, PARAMETERS
from cosmic_integration.lnl_surrogate.run_sampler import sample_lnl_surrogate
from cosmic_integration.observation import load_observation
from cosmic_integration.observation.mock_observation import MockObservation
from cosmic_integration.ratesSampler import BinnedCosmicIntegrator, CosmicIntegration
from cosmic_integration.plot_rate import plot_matrix
from cosmic_integration.lnl_computer import LnLComputer


# ---------------------------------------------------------------------------
# Configuration

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR
OUTPUT_DIR = BASE_DIR / "outputs"

# Fiducial parameter vector λ = (α, σ, a_SF, d_SF)
FIDUCIAL_PARAMS = np.array([-0.325, 0.213, 0.012, 4.253], dtype=float)

# Observation settings
FULL_DURATION_YEARS = 1.0
SUBSET_DURATION_YEARS = 3.0
FULL_EVENT_TARGET = 578
POSTERIOR_SAMPLES_PER_EVENT = int(5e3)

# Active learning settings
ACTIVE_LEARNING_INITIAL_POINTS = 40
ACTIVE_LEARNING_TOTAL_STEPS = 200
ACTIVE_LEARNING_STEPS_PER_ROUND = 20

# Quick-mode overrides
QUICK_POSTERIOR_SAMPLES_PER_EVENT = int(1e3)
QUICK_ACTIVE_LEARNING_INITIAL_POINTS = 20
QUICK_ACTIVE_LEARNING_TOTAL_STEPS = 80
QUICK_ACTIVE_LEARNING_STEPS_PER_ROUND = 10

MCMC_DEFAULT_SETTINGS = {"nwalkers": 48, "iterations": 3000}
MCMC_QUICK_SETTINGS = {"nwalkers": 24, "iterations": 3000}

# Randomness control
SEED = int(os.environ.get("SIM_STUDY_SEED", 20240623))




# ---------------------------------------------------------------------------
# Utilities

def _rng(seed: Optional[int] = None) -> np.random.Generator:
    return np.random.default_rng(SEED if seed is None else seed)


def _ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _compas_catalogue() -> Path:
    compas_path = DATA_DIR / "h5out_5M.h5"
    if not compas_path.exists():
        raise FileNotFoundError(
            "The COMPAS catalogue 'h5out_5M.h5' was not found. "
            "Please download it to docs/studies/simulation_study before running the study."
        )
    return compas_path


def _create_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("simulation_study")
    logger.setLevel(logging.INFO)
    if not any(isinstance(h, logging.FileHandler) and h.baseFilename == str(log_path) for h in logger.handlers):
        handler = logging.FileHandler(log_path, mode="a")
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(handler)
    return logger


@dataclass(frozen=True)
class StudyArtifacts:
    full_observation: Path
    subset_observation: Path
    analysis_dir: Path


def _resolve_settings(quick: bool) -> dict:
    return {
        "posterior_samples_per_event": QUICK_POSTERIOR_SAMPLES_PER_EVENT if quick else POSTERIOR_SAMPLES_PER_EVENT,
        "active_learning_initial_points": QUICK_ACTIVE_LEARNING_INITIAL_POINTS if quick else ACTIVE_LEARNING_INITIAL_POINTS,
        "active_learning_total_steps": QUICK_ACTIVE_LEARNING_TOTAL_STEPS if quick else ACTIVE_LEARNING_TOTAL_STEPS,
        "active_learning_steps_per_round": QUICK_ACTIVE_LEARNING_STEPS_PER_ROUND if quick else ACTIVE_LEARNING_STEPS_PER_ROUND,
        "mcmc_settings": MCMC_QUICK_SETTINGS if quick else MCMC_DEFAULT_SETTINGS,
        "seed": SEED,
    }


# ---------------------------------------------------------------------------
# Stage 1 – generate the mock catalogue

def _generate_catalogue(
    artifacts: StudyArtifacts,
    *,
    duration: float,
    posterior_samples: int,
    with_noise: bool,
    force: bool,
    rng: np.random.Generator,
) -> MockObservation:
    """Generate a mock catalogue for the given duration."""
    destination = artifacts.full_observation if duration == FULL_DURATION_YEARS else artifacts.subset_observation
    compas_h5 = _compas_catalogue()
    _ensure_directory(destination.parent)
    _ensure_directory(artifacts.analysis_dir)
    logger = _create_logger(artifacts.analysis_dir / "simulation.log")

    bci = BinnedCosmicIntegrator.from_compas_h5(
        inputPath=os.path.dirname(compas_h5),
        inputName=os.path.basename(compas_h5),
    )
    binned_rates = bci.FindBinnedDetectionRate(
        p_Alpha=float(FIDUCIAL_PARAMS[0]),
        p_Sigma=float(FIDUCIAL_PARAMS[1]),
        p_SFRa=float(FIDUCIAL_PARAMS[2]),
        p_SFRd=float(FIDUCIAL_PARAMS[3]),
    )
    plot_matrix(
        binned_rates,
        params=FIDUCIAL_PARAMS,
        label="Detection rate (expected counts per yr)",
        fname=str(artifacts.analysis_dir / "rates_matrix.png"),
    )

    if destination.exists() and not force:
        logger.info("Reusing cached catalogue: %s", destination)
        observation = load_observation(str(destination))
    else:
        logger.info("Generating new catalogue (noise=%s): %s", with_noise, destination)
        observation = MockObservation.generate_from_rates(
            rates=binned_rates,
            params=FIDUCIAL_PARAMS,
            duration=duration,
            n_posterior_samples=posterior_samples,
            output_file=str(destination),
            measurement_uncertainty=with_noise,
            rng=rng,
        )

    plot_matrix(
        np.nansum(observation.population_weights, axis=0),
        params=observation.params,
        label=f"DATA ({duration:.1f} yr)",
        fname=str(artifacts.analysis_dir / f"data_matrix_{duration:.1f}yr.png"),
    )
    return observation, str(destination)


# ---------------------------------------------------------------------------
# Stage 2 – brute-force 1D log-likelihood scan

def _evaluate_lnl(observation, test_params: np.ndarray) -> float:
    """Compute log-likelihood for given parameters."""
    ci = CosmicIntegration.from_compas_h5(
        os.path.dirname(_compas_catalogue()),
        os.path.basename(_compas_catalogue()),
    )
    rates, _ = ci.FindDetectionRate(
        p_Alpha=float(test_params[0]),
        p_Sigma=float(test_params[1]),
        p_SFRa=float(test_params[2]),
        p_SFRd=float(test_params[3]),
    )
    # Placeholder: replace with your actual likelihood evaluation logic
    # Here we use a simple log-sum over population weights and rates
    model = np.clip(rates, 1e-12, None)
    data = np.clip(np.nansum(observation.population_weights, axis=0), 1e-12, None)
    return np.sum(data * np.log(model) - model)


def _lnl_1d_scan(observation, output_dir: Path) -> None:
    """Compute and plot LnL as a function of each parameter around truth."""

    _lnl_comptuer = LnLComputer.load(
        observation_file=str(observation),
        compas_h5=str(_compas_catalogue()),
    )

    param_names = PARAMETERS
    fid = FIDUCIAL_PARAMS
    n_points = 10
    offsets = np.linspace(-0.5, 0.5, n_points)

    fig, axes = plt.subplots(4, 1, figsize=(5, 8))
    results_payload = {"fiducial": FIDUCIAL_PARAMS.tolist(), "curves": []}
    for i, (ax, pname) in enumerate(zip(axes, param_names)):
        print("Computing 1D LnL scan for parameter:", pname)
        xvals = fid[i] + offsets * abs(fid[i] if fid[i] != 0 else 1)
        yvals = []
        for val in xvals:
            test_params = fid.copy()
            test_params[i] = val
            _lnl = _lnl_comptuer(
                alpha=float(test_params[0]),
                sigma=float(test_params[1]),
                sfr_a=float(test_params[2]),
                sfr_d=float(test_params[3]),
            )
            yvals.append(_lnl)
        yvals = np.array(yvals)
        ax.plot(xvals, yvals, label="LnL", color="C0")
        ax.axvline(fid[i], color="r", ls="--")
        ax.axhline(np.max(yvals), color="r", ls="--")
        ax.set_xlabel(pname)
        ax.set_ylabel("LnL")

        # Store data for later GP validation overlay
        results_payload["curves"].append({
            "param": pname,
            "x": [float(x) for x in xvals],
            "lnl": [float(y) for y in yvals],
        })

    fig.suptitle("True: {}".format(", ".join(f"{p}={v:.3f}" for p, v in zip(param_names, fid))))

    fig.tight_layout()
    fig.savefig(output_dir / "lnl_1d_scan.png", dpi=200)
    plt.close(fig)

    # Persist numerical data for later reuse during GP diagnostics
    (output_dir / "lnl_1d").mkdir(parents=True, exist_ok=True)
    with open(output_dir / "lnl_1d" / "scan_data.json", "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)


# ---------------------------------------------------------------------------
# Stage 3 – active learning analysis

def _train_surrogate(
    artifacts: StudyArtifacts,
    *,
    force: bool,
    initial_points: int,
    total_steps: int,
    steps_per_round: int,
    seed: int,
) -> Path:
    analysis_dir = artifacts.analysis_dir
    model_dir = analysis_dir / "gp_model"

    if model_dir.exists() and not force:
        return model_dir

    _ensure_directory(analysis_dir)
    logger = _create_logger(analysis_dir / "simulation.log")
    logger.info(
        "Starting active learning training (steps=%d, steps/round=%d, initial=%d)",
        total_steps,
        steps_per_round,
        initial_points,
    )

    compas_h5 = _compas_catalogue()
    observation_path = str(artifacts.subset_observation)
    np.random.seed(seed)

    LnLSurrogate.train(
        observation_file=observation_path,
        compas_h5=str(compas_h5),
        outdir=str(analysis_dir),
        initial_points=initial_points,
        total_steps=total_steps,
        steps_per_round=steps_per_round,
        truth=FIDUCIAL_PARAMS,

    )

    logger.info("Active learning completed. Artefacts stored in %s", analysis_dir)
    return model_dir


def _plot_gp_vs_true_1d(analysis_dir: Path) -> None:
    """Overlay GP-predicted LnL against the precomputed true 1D scans.

    Expects scan data from _lnl_1d_scan stored under analysis_dir/lnl_1d/scan_data.json
    and a trained GP model under analysis_dir/gp_model/models.
    """
    data_path = analysis_dir / "lnl_1d" / "scan_data.json"
    model_dir = analysis_dir / "gp_model" / "models"
    if not data_path.exists() or not model_dir.exists():
        return

    try:
        with open(data_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return

    try:
        surrogate = LnLSurrogate.load(str(model_dir))
    except Exception:
        return

    fid = np.array(payload.get("fiducial", FIDUCIAL_PARAMS), dtype=float)
    curves = payload.get("curves", [])

    fig, axes = plt.subplots(4, 1, figsize=(6, 9))
    for ax, curve in zip(axes, curves):
        pname = curve["param"]
        xvals = np.array(curve["x"], dtype=float)
        true_y = np.array(curve["lnl"], dtype=float)

        # Build points along the 1D slice
        pts = np.tile(fid, (xvals.size, 1))
        idx = PARAMETERS.index(pname)
        pts[:, idx] = xvals

        # Predict with GP (in transformed space) and invert to LnL
        try:
            preds_tf, _ = surrogate.gp_model.predict_f(pts)
            preds = preds_tf.numpy().reshape(-1)
            transformed = -preds
            gp_y = np.array([surrogate.scaler.inverse_transform(v) for v in transformed])
        except Exception:
            gp_y = np.full_like(xvals, np.nan, dtype=float)

        ax.plot(xvals, true_y, label="True LnL", color="C0")
        ax.plot(xvals, gp_y, label="GP pred", color="C1", linestyle="--")
        ax.axvline(fid[idx], color="r", ls="--", alpha=0.7)
        ax.set_xlabel(pname)
        ax.set_ylabel("LnL")
        ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(analysis_dir / "lnl_1d_gp_vs_true.png", dpi=200)
    plt.close(fig)


def _run_mcmc_and_plots(
    artifacts: StudyArtifacts,
    *,
    mcmc_settings: dict,
    truths: dict,
) -> None:
    analysis_dir = artifacts.analysis_dir
    model_dir = analysis_dir / "gp_model" / "models"
    if not model_dir.exists():
        logging.getLogger("simulation_study").warning(
            "Skipped MCMC sampling; model directory %s not found.", model_dir
        )
        return

    mcmc_dir = analysis_dir / "MCMC"
    result = sample_lnl_surrogate(
        lnl_model_path=str(model_dir),
        outdir=str(mcmc_dir),
        verbose=False,
        truths=truths,
        mcmc_kwargs=dict(mcmc_settings),
    )

    if result is None or result.posterior is None or "log_likelihood" not in result.posterior:
        return

    posterior = result.posterior
    best_idx = posterior["log_likelihood"].idxmax()
    best_params_series = posterior.loc[best_idx, PARAMETERS]
    best_params = np.array([float(best_params_series[p]) for p in PARAMETERS])

    compas_h5 = _compas_catalogue()
    bci = BinnedCosmicIntegrator.from_compas_h5(
        inputPath=str(compas_h5.parent),
        inputName=compas_h5.name,
    )
    model_matrix = bci.FindBinnedDetectionRate(
        p_Alpha=best_params[0],
        p_Sigma=best_params[1],
        p_SFRa=best_params[2],
        p_SFRd=best_params[3],
    )

    plot_matrix(
        model_matrix,
        params=best_params,
        label="MODEL (posterior mode)",
        fname=str(analysis_dir / "posterior_mode_matrix.png"),
    )

    summary_path = mcmc_dir / "posterior_mode.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_payload = {
        "best_parameters": {param: float(val) for param, val in zip(PARAMETERS, best_params)},
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Entry point

def run(force: bool = False, quick: bool = False, with_noise: bool = False) -> StudyArtifacts:
    _ensure_directory(OUTPUT_DIR)
    scenario_dir = OUTPUT_DIR / ("noise" if with_noise else "no_noise")
    _ensure_directory(scenario_dir)
    artifacts = StudyArtifacts(
        full_observation=scenario_dir / "full_mock_observation.h5",
        subset_observation=scenario_dir / "mock_observation.h5",
        analysis_dir=scenario_dir / "analysis",
    )

    settings = _resolve_settings(quick)
    rng = _rng(settings["seed"])

    observation, obs_fn = _generate_catalogue(
        artifacts,
        duration=SUBSET_DURATION_YEARS,
        posterior_samples=settings["posterior_samples_per_event"],
        with_noise=with_noise,
        force=force,
        rng=rng,
    )

    _lnl_1d_scan(obs_fn, artifacts.analysis_dir)

    _train_surrogate(
        artifacts,
        force=force,
        initial_points=settings["active_learning_initial_points"],
        total_steps=settings["active_learning_total_steps"],
        steps_per_round=settings["active_learning_steps_per_round"],
        seed=settings["seed"],
    )

    # Compare GP predictions against the precomputed 1D LnL scans
    _plot_gp_vs_true_1d(artifacts.analysis_dir)

    truths_dict = {p: float(v) for p, v in zip(PARAMETERS, FIDUCIAL_PARAMS)}
    _run_mcmc_and_plots(
        artifacts,
        mcmc_settings=dict(settings["mcmc_settings"]),
        truths=truths_dict,
    )

    return artifacts


def main(args: Optional[Sequence[str]] = None) -> None:
    force = quick = with_noise = False
    if args is None:
        args = []
    for opt in args:
        if opt in {"--force", "-f"}:
            force = True
        elif opt in {"--quick", "-q"}:
            quick = True
        elif opt in {"--noise", "-n"}:
            with_noise = True

    artifacts = run(force=force, quick=quick, with_noise=with_noise)
    print("Simulation study completed.")
    print(f"  Full catalogue:     {artifacts.full_observation}")
    print(f"  Mock observation:   {artifacts.subset_observation}")
    print(f"  Analysis artefacts: {artifacts.analysis_dir}")


if __name__ == "__main__":  # pragma: no cover
    import sys
    main(sys.argv[1:])
