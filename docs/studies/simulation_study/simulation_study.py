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

Optional brute-force check (uniform random sampling over the bounds)::

    PYTHONPATH=src python docs/studies/simulation_study/simulation_study.py --bruteforce
    PYTHONPATH=src python docs/studies/simulation_study/simulation_study.py --bruteforce-samples 5000 --bruteforce-temperature 100
    PYTHONPATH=src python docs/studies/simulation_study/simulation_study.py --duration-years 3 --bruteforce --bruteforce-only

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
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from cosmic_integration.lnl_surrogate.lnl_surrogate import LnLSurrogate, PARAMETERS, BOUNDS
from cosmic_integration.lnl_surrogate.offline_diagnostics import (
    offline_surrogate_diagnostics,
    fit_surrogate_from_samples,
)
from cosmic_integration.lnl_surrogate.diagnostics.simulation_study import (
    local_peak_diagnostics_vs_round as _local_peak_diagnostics_vs_round_impl,
    plot_gp_vs_true_1d as _plot_gp_vs_true_1d_impl,
    posterior_convergence_vs_round as _posterior_convergence_vs_round_impl,
    summarise_surrogate_optimum as _summarise_surrogate_optimum_impl,
)
from cosmic_integration.lnl_surrogate.run_sampler import sample_lnl_surrogate
from cosmic_integration.cli_tools.diagnose_surrogate_chain import run_chain_diagnostics
from cosmic_integration.observation import load_observation
from cosmic_integration.observation.mock_observation import MockObservation
from cosmic_integration.ratesSampler import BinnedCosmicIntegrator, CosmicIntegration
from cosmic_integration.plot_rate import plot_matrix
from cosmic_integration.lnl_computer import LnLComputer
from cosmic_integration.utils import row_to_matrix_params_lnl


# ---------------------------------------------------------------------------
# Configuration

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR
OUTPUT_DIR = BASE_DIR / "outputs"

# Fiducial parameter vector λ = (α, σ, a_SF, d_SF)
FIDUCIAL_PARAMS = np.array([-0.325, 0.213, 0.012, 4.253], dtype=float)

# Observation settings
FULL_DURATION_YEARS = 1.0
SUBSET_DURATION_YEARS = 0.1  # Matches Riley paper: 58 events for small dataset
FULL_EVENT_TARGET = 578
POSTERIOR_SAMPLES_PER_EVENT = int(5e3)

# Active learning settings
ACTIVE_LEARNING_INITIAL_POINTS = 40
ACTIVE_LEARNING_TOTAL_STEPS = 250
ACTIVE_LEARNING_STEPS_PER_ROUND = 20

# Quick-mode overrides
QUICK_POSTERIOR_SAMPLES_PER_EVENT = int(1e3)
QUICK_ACTIVE_LEARNING_INITIAL_POINTS = 20
QUICK_ACTIVE_LEARNING_TOTAL_STEPS = 80
QUICK_ACTIVE_LEARNING_STEPS_PER_ROUND = 10

MCMC_DEFAULT_SETTINGS = {"nwalkers": 48, "iterations": 6000, "nburn": 2000}
MCMC_QUICK_SETTINGS = {"nwalkers": 24, "iterations": 4000, "nburn": 1000}

# Brute-force sampling diagnostics (optional)
BRUTEFORCE_SAMPLES = 5000
QUICK_BRUTEFORCE_SAMPLES = 2000
BRUTEFORCE_CORNER_TEMPERATURE = 200.0
BRUTEFORCE_CORNER_MAX_POINTS = 5000

# Surrogate scaler defaults
SCALER_SOFT_CLIPPING = True
SCALER_CLIP_FACTOR = 5.0
SCALER_LOWER_CLIP_PERCENTILE = "auto"

# Brute-force GP baseline (random sampling) defaults
BRUTEFORCE_GP_DEFAULT_TRAIN_N = ACTIVE_LEARNING_INITIAL_POINTS + ACTIVE_LEARNING_TOTAL_STEPS
BRUTEFORCE_GP_DEFAULT_STRATEGY = "mix"  # random | top | mix
BRUTEFORCE_GP_DEFAULT_TOP_FRACTION = 0.5
BRUTEFORCE_GP_DEFAULT_DELTA = 200.0  # train on points with LnL >= max(LnL) - delta

SCALER_TRAIN_KWARGS = {
    "scaler_soft_clipping": SCALER_SOFT_CLIPPING,
    "scaler_clip_factor": SCALER_CLIP_FACTOR,
    "scaler_lower_clip_percentile": SCALER_LOWER_CLIP_PERCENTILE,
}

OFFLINE_BASELINE_ROUND_TOTALS = [200, 400, 600]
OFFLINE_BASELINE_TRAIN = OFFLINE_BASELINE_ROUND_TOTALS[-1]
OFFLINE_BASELINE_TEST = 400
OFFLINE_BASELINE_MCMC_KWARGS = {"nwalkers": 16, "iterations": 1000}

# Randomness control
SEED = int(os.environ.get("SIM_STUDY_SEED", 20240623))

CONVERGENCE_DEFAULT_EVERY = 2
CONVERGENCE_DEFAULT_UNCERTAINTY_BETA = 0.0

LOCAL_PEAK_DEFAULT_EVERY = 2
LOCAL_PEAK_DEFAULT_N = 200
LOCAL_PEAK_DEFAULT_FRACTION = 0.02
LOCAL_PEAK_DEFAULT_UNCERTAINTY_BETA = 0.0




# ---------------------------------------------------------------------------
# Utilities

def _rng(seed: Optional[int] = None) -> np.random.Generator:
    return np.random.default_rng(SEED if seed is None else seed)


def _ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _duration_tag(duration: float) -> str:
    """Format durations consistently for directory names."""
    return f"dur_{float(duration):g}yr"


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
        "bruteforce_samples": QUICK_BRUTEFORCE_SAMPLES if quick else BRUTEFORCE_SAMPLES,
        "bruteforce_temperature": BRUTEFORCE_CORNER_TEMPERATURE,
        "bruteforce_max_points": BRUTEFORCE_CORNER_MAX_POINTS,
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
    expected_events = float(np.nansum(binned_rates) * duration)
    logger.info(
        "Expected detections from fiducial rate matrix: %.1f (sum(rates)=%.1f /yr, duration=%.3g yr)",
        expected_events,
        float(np.nansum(binned_rates)),
        float(duration),
    )
    plot_matrix(
        binned_rates,
        params=FIDUCIAL_PARAMS,
        label="Detection rate (expected counts per yr)",
        fname=str(artifacts.analysis_dir / "rates_matrix.png"),
    )
    plot_matrix(
        binned_rates * duration,
        params=FIDUCIAL_PARAMS,
        label=f"Expected counts ({duration:.1f} yr)",
        fname=str(artifacts.analysis_dir / f"expected_matrix_{duration:.1f}yr.png"),
        n_events=expected_events,
    )

    if destination.exists() and not force:
        logger.info("Reusing cached catalogue: %s", destination)
        observation = load_observation(str(destination))
        if abs(float(observation.duration) - float(duration)) > 1e-9:
            logger.warning(
                "Cached observation duration mismatch (have %.3g yr, expected %.3g yr). "
                "Regenerating catalogue: %s",
                float(observation.duration),
                float(duration),
                destination,
            )
            observation = MockObservation.generate_from_rates(
                rates=binned_rates,
                params=FIDUCIAL_PARAMS,
                duration=duration,
                n_posterior_samples=posterior_samples,
                output_file=str(destination),
                measurement_uncertainty=with_noise,
                rng=rng,
            )
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
    logger.info(
        "Observed detections in mock catalogue: %d (duration=%.3g yr)",
        int(observation.population_weights.shape[0]),
        float(observation.duration),
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


def _lnl_1d_scan(lnl_computer: LnLComputer, output_dir: Path) -> None:
    """Compute and plot LnL as a function of each parameter around truth."""

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
            _lnl = lnl_computer(
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
    observation_file: str,
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
    observation_path = str(observation_file)
    np.random.seed(seed)

    LnLSurrogate.train(
        observation_file=observation_path,
        compas_h5=str(compas_h5),
        outdir=str(analysis_dir),
        initial_points=initial_points,
        total_steps=total_steps,
        steps_per_round=steps_per_round,
        truth=FIDUCIAL_PARAMS,
        **SCALER_TRAIN_KWARGS,
    )

    logger.info("Active learning completed. Artefacts stored in %s", analysis_dir)
    return model_dir


def _plot_gp_vs_true_1d(analysis_dir: Path) -> None:
    _plot_gp_vs_true_1d_impl(analysis_dir, fiducial_params=FIDUCIAL_PARAMS, parameters=PARAMETERS)


def _summarise_surrogate_optimum(
    analysis_dir: Path,
    lnl_computer: LnLComputer,
    *,
    rng: np.random.Generator,
    truths: Dict[str, float],
) -> None:
    _create_logger(analysis_dir / "simulation.log")
    _summarise_surrogate_optimum_impl(analysis_dir, lnl_computer, rng=rng, truths=truths)


def _offline_baseline_diagnostics(
    analysis_dir: Path,
    lnl_computer: LnLComputer,
    *,
    force: bool,
    seed: int,
    truths: Dict[str, float],
) -> None:
    diag_dir = analysis_dir / "offline_baseline"
    metrics_path = diag_dir / "metrics.json"
    if metrics_path.exists() and not force:
        return

    if force and diag_dir.exists():
        shutil.rmtree(diag_dir)
    diag_dir.mkdir(parents=True, exist_ok=True)

    result = offline_surrogate_diagnostics(
        evaluator=lambda params: lnl_computer(*params),
        bounds=BOUNDS,
        n_train=OFFLINE_BASELINE_TRAIN,
        n_test=OFFLINE_BASELINE_TEST,
        round_totals=OFFLINE_BASELINE_ROUND_TOTALS,
        seed=seed,
        scaler_kwargs={
            "soft_clipping": SCALER_SOFT_CLIPPING,
            "clip_factor": SCALER_CLIP_FACTOR,
            "lower_clip_percentile": SCALER_LOWER_CLIP_PERCENTILE,
        },
        outdir=diag_dir,
        truths=truths,
        mcmc_kwargs=OFFLINE_BASELINE_MCMC_KWARGS,
        param_labels=PARAMETERS,
        corner_temperature=500.0,
    )

    metrics_payload = {
        "rounds": [
            {
                "round_index": round_result.round_index,
                "n_train": round_result.n_train,
                **round_result.metrics,
                "scatter_path": str(round_result.scatter_path) if round_result.scatter_path else None,
                "residual_path": str(round_result.residual_path) if round_result.residual_path else None,
                "model_dir": str(round_result.model_dir) if round_result.model_dir else None,
                "sampler_dir": str(round_result.sampler_dir) if round_result.sampler_dir else None,
            }
            for round_result in result.rounds
        ],
        "round_totals": OFFLINE_BASELINE_ROUND_TOTALS,
        "n_test": OFFLINE_BASELINE_TEST,
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    np.savez(
        diag_dir / "offline_samples.npz",
        train_points=result.final_train_points,
        train_lnls=result.final_train_lnls,
        test_points=result.test_points,
        test_lnls=result.test_lnls,
        predicted_lnls=result.final_predictions,
    )


def _bruteforce_random_scan(
    analysis_dir: Path,
    lnl_computer: LnLComputer,
    *,
    n_samples: int,
    seed: int,
    temperature: float,
    max_points: int,
    truths: Dict[str, float],
    force: bool,
) -> None:
    """
    Uniform random scan over `BOUNDS`, saving a weighted corner plot of true LnL values.

    This is intended as a sanity check for multi-modality / broad degeneracies and to
    confirm whether the evaluated optimum is plausible, not as an efficient optimizer.
    """

    outdir = analysis_dir / "bruteforce"
    outdir.mkdir(parents=True, exist_ok=True)
    out_npz = outdir / "samples.npz"
    out_png = outdir / "weighted_corner.png"
    out_json = outdir / "summary.json"

    logger = _create_logger(analysis_dir / "simulation.log")
    n_samples = int(max(n_samples, 1))

    existing_points = np.empty((0, len(PARAMETERS)), dtype=float)
    existing_lnls = np.empty((0,), dtype=float)
    if out_npz.exists() and not force:
        try:
            prior = np.load(out_npz)
            existing_points = np.asarray(prior["points"], dtype=float)
            existing_lnls = np.asarray(prior["lnls"], dtype=float)
            if existing_points.ndim != 2 or existing_points.shape[1] != len(PARAMETERS):
                raise ValueError("samples.npz points array has wrong shape")
            if existing_lnls.ndim != 1 or existing_lnls.shape[0] != existing_points.shape[0]:
                raise ValueError("samples.npz lnls array has wrong shape")
        except Exception as exc:
            logger.warning("Failed to load existing brute-force samples (%s); starting fresh.", exc)
            existing_points = np.empty((0, len(PARAMETERS)), dtype=float)
            existing_lnls = np.empty((0,), dtype=float)

    n_existing = int(existing_points.shape[0])
    n_needed = max(int(n_samples) - n_existing, 0)
    if n_needed > 0:
        logger.info(
            "Starting brute-force scan (target=%d, have=%d, adding=%d, seed=%d)",
            int(n_samples),
            n_existing,
            n_needed,
            int(seed),
        )
        # Derive a new seed when appending to avoid repeating the initial sequence.
        rng = np.random.default_rng(int(seed) + n_existing)
        new_points = rng.uniform(BOUNDS[0], BOUNDS[1], size=(n_needed, len(PARAMETERS)))
        new_lnls = np.empty(n_needed, dtype=float)
        for i, row in enumerate(tqdm(new_points, desc="Brute-force LnL scan", unit="pt")):
            new_lnls[i] = float(lnl_computer(*row))
        points = np.vstack([existing_points, new_points])
        lnls = np.concatenate([existing_lnls, new_lnls])
    else:
        points = existing_points
        lnls = existing_lnls
        logger.info(
            "Brute-force scan already has %d samples (requested %d); regenerating plots/summary.",
            n_existing,
            int(n_samples),
        )

    best_idx = int(np.nanargmax(lnls))
    best_lnl = float(lnls[best_idx])
    best_params = points[best_idx]

    # Weighted corner plot: w ∝ exp((LnL - max) / T)
    try:
        import corner

        plot_points = points
        plot_lnls = lnls
        if max_points is not None and int(max_points) > 0 and len(plot_lnls) > int(max_points):
            idx = rng.choice(len(plot_lnls), size=int(max_points), replace=False)
            plot_points = plot_points[idx]
            plot_lnls = plot_lnls[idx]

        delta = plot_lnls - float(np.max(plot_lnls))
        T = max(float(temperature), 1e-6)
        weights = np.exp(delta / T)
        truths_vec = [float(truths[p]) for p in PARAMETERS] if truths else None
        fig = corner.corner(
            plot_points,
            labels=list(PARAMETERS),
            weights=weights,
            show_titles=True,
            truths=truths_vec,
        )
        fig.savefig(out_png, dpi=180)
        plt.close(fig)
    except Exception as exc:
        logger.warning("Failed to generate brute-force corner plot: %s", exc)

    truth_lnl = None
    if truths:
        try:
            truth_vec = np.array([truths[p] for p in PARAMETERS], dtype=float)
            truth_lnl = float(lnl_computer(*truth_vec))
        except Exception:
            truth_lnl = None

    payload = {
        "n_samples": int(points.shape[0]),
        "n_samples_requested": int(n_samples),
        "n_samples_added": int(n_needed),
        "bounds": [[float(v) for v in BOUNDS[0]], [float(v) for v in BOUNDS[1]]],
        "temperature": float(temperature),
        "max_points": int(max_points),
        "best_lnl": best_lnl,
        "best_params": {p: float(v) for p, v in zip(PARAMETERS, best_params)},
        "truths": dict(truths),
        "truth_lnl": truth_lnl,
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    np.savez(out_npz, points=points, lnls=lnls)
    logger.info("Brute-force scan completed. Best LnL=%.2f; outputs in %s", best_lnl, outdir)


def _bruteforce_gp_baseline(
    analysis_dir: Path,
    *,
    train_n: int,
    seed: int,
    truths: Dict[str, float],
    mcmc_settings: dict,
    force: bool,
    strategy: str = BRUTEFORCE_GP_DEFAULT_STRATEGY,
    top_fraction: float = BRUTEFORCE_GP_DEFAULT_TOP_FRACTION,
    delta: Optional[float] = BRUTEFORCE_GP_DEFAULT_DELTA,
) -> None:
    """
    Train a GP surrogate on the brute-force random-scan samples (random subset),
    then run the same MCMC sampling as for the BO-trained GP.
    """

    logger = _create_logger(analysis_dir / "simulation.log")
    brute_npz = analysis_dir / "bruteforce" / "samples.npz"
    if not brute_npz.exists():
        logger.warning("Bruteforce samples not found at %s; skipping bruteforce GP baseline.", brute_npz)
        return

    outdir = analysis_dir / "bruteforce_gp"
    if force and outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    samples = np.load(brute_npz)
    points = np.asarray(samples["points"], dtype=float)
    lnls = np.asarray(samples["lnls"], dtype=float).reshape(-1)
    if points.ndim != 2 or points.shape[1] != len(PARAMETERS) or lnls.shape[0] != points.shape[0]:
        raise ValueError(f"Invalid bruteforce samples arrays in {brute_npz}")

    train_n = int(max(train_n, 10))
    n_available = int(points.shape[0])
    if train_n > n_available:
        train_n = n_available

    strategy = str(strategy).strip().lower()
    if strategy not in {"random", "top", "mix"}:
        raise ValueError(f"Unknown bruteforce GP strategy: {strategy}")

    top_fraction = float(top_fraction)
    if not (0.0 <= top_fraction <= 1.0):
        raise ValueError("top_fraction must be in [0, 1].")

    rng = np.random.default_rng(int(seed))
    lnl_max = float(np.nanmax(lnls))
    eligible = np.ones_like(lnls, dtype=bool)
    if delta is not None:
        eligible = lnls >= (lnl_max - float(delta))
        if int(np.sum(eligible)) < max(10, min(train_n, 50)):
            logger.warning(
                "Bruteforce GP: only %d/%d points within delta=%.2f of max; falling back to all points.",
                int(np.sum(eligible)),
                n_available,
                float(delta),
            )
            eligible = np.ones_like(lnls, dtype=bool)

    eligible_idx = np.nonzero(eligible)[0]
    eligible_points = points[eligible_idx]
    eligible_lnls = lnls[eligible_idx]

    if strategy == "random":
        chosen = rng.choice(eligible_points.shape[0], size=train_n, replace=False)
        train_points = eligible_points[chosen]
        train_lnls = eligible_lnls[chosen]
    elif strategy == "top":
        order = np.argsort(eligible_lnls)[::-1]
        chosen = order[:train_n]
        train_points = eligible_points[chosen]
        train_lnls = eligible_lnls[chosen]
    else:  # mix
        n_top = int(round(top_fraction * train_n))
        n_top = max(0, min(train_n, n_top))
        n_rand = train_n - n_top

        order = np.argsort(eligible_lnls)[::-1]
        top_idx = order[:n_top] if n_top > 0 else np.array([], dtype=int)
        remaining_idx = order[n_top:]
        if n_rand > 0:
            if remaining_idx.shape[0] < n_rand:
                rand_idx = remaining_idx
            else:
                rand_idx = rng.choice(remaining_idx, size=n_rand, replace=False)
            chosen = np.concatenate([top_idx, rand_idx]) if top_idx.size else np.asarray(rand_idx, dtype=int)
        else:
            chosen = np.asarray(top_idx, dtype=int)
        train_points = eligible_points[chosen]
        train_lnls = eligible_lnls[chosen]

    models_dir = fit_surrogate_from_samples(
        train_points=train_points,
        train_lnls=train_lnls,
        bounds=BOUNDS,
        outdir=outdir,
        scaler_kwargs={
            "soft_clipping": SCALER_SOFT_CLIPPING,
            "clip_factor": SCALER_CLIP_FACTOR,
            "lower_clip_percentile": SCALER_LOWER_CLIP_PERCENTILE,
        },
        overwrite=True,
    )

    (outdir / "training_summary.json").write_text(
        json.dumps(
            {
                "train_n": int(train_n),
                "seed": int(seed),
                "source": str(brute_npz),
                "strategy": strategy,
                "top_fraction": float(top_fraction),
                "delta": float(delta) if delta is not None else None,
                "n_available": int(n_available),
                "n_eligible": int(eligible_points.shape[0]),
                "lnl_max": float(lnl_max),
                "lnl_min_train": float(np.min(train_lnls)),
                "lnl_max_train": float(np.max(train_lnls)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    mcmc_dir = analysis_dir / "MCMC_bruteforce_gp"
    sample_lnl_surrogate(
        lnl_model_path=str(models_dir),
        outdir=str(mcmc_dir),
        verbose=False,
        truths=truths,
        mcmc_kwargs=dict(mcmc_settings),
    )


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
    try:
        result = sample_lnl_surrogate(
            lnl_model_path=str(model_dir),
            outdir=str(mcmc_dir),
            verbose=False,
            truths=truths,
            mcmc_kwargs=dict(mcmc_settings),
        )
    except Exception as exc:  # pragma: no cover - defensive
        logging.getLogger("simulation_study").warning(
            "MCMC sampling failed (%s). Skipping posterior diagnostics.", exc,
        )
        return

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


def _posterior_convergence_vs_round(
    artifacts: StudyArtifacts,
    *,
    mcmc_settings: dict,
    truths: dict,
    initial_points: int,
    steps_per_round: int,
    every: int = CONVERGENCE_DEFAULT_EVERY,
    uncertainty_beta: float = CONVERGENCE_DEFAULT_UNCERTAINTY_BETA,
    bins: int = 60,
) -> None:
    _posterior_convergence_vs_round_impl(
        artifacts.analysis_dir,
        models_dir=artifacts.analysis_dir / "gp_model" / "models",
        mcmc_settings=mcmc_settings,
        truths=truths,
        initial_points=initial_points,
        steps_per_round=steps_per_round,
        every=every,
        uncertainty_beta=uncertainty_beta,
        bins=bins,
        parameters=PARAMETERS,
    )


def _local_peak_diagnostics_vs_round(
    artifacts: StudyArtifacts,
    lnl_computer: LnLComputer,
    *,
    truths: dict,
    initial_points: int,
    steps_per_round: int,
    every: int = LOCAL_PEAK_DEFAULT_EVERY,
    n_points: int = LOCAL_PEAK_DEFAULT_N,
    fraction: float = LOCAL_PEAK_DEFAULT_FRACTION,
    seed: int = SEED,
    uncertainty_beta: float = LOCAL_PEAK_DEFAULT_UNCERTAINTY_BETA,
) -> None:
    _local_peak_diagnostics_vs_round_impl(
        artifacts.analysis_dir,
        artifacts.analysis_dir / "gp_model" / "models",
        lnl_computer,
        truths=truths,
        initial_points=initial_points,
        steps_per_round=steps_per_round,
        every=every,
        n_points=n_points,
        fraction=fraction,
        seed=seed,
        uncertainty_beta=uncertainty_beta,
        parameters=PARAMETERS,
        bounds=BOUNDS,
    )


# ---------------------------------------------------------------------------
# Entry point

def run(
    force: bool = False,
    quick: bool = False,
    with_noise: bool = False,
    *,
    bruteforce: bool = False,
    bruteforce_samples: Optional[int] = None,
    bruteforce_temperature: Optional[float] = None,
    bruteforce_only: bool = False,
    bruteforce_gp: bool = False,
    bruteforce_gp_train_n: Optional[int] = None,
    bruteforce_gp_strategy: str = BRUTEFORCE_GP_DEFAULT_STRATEGY,
    bruteforce_gp_top_fraction: float = BRUTEFORCE_GP_DEFAULT_TOP_FRACTION,
    bruteforce_gp_delta: Optional[float] = BRUTEFORCE_GP_DEFAULT_DELTA,
    duration_years: Optional[float] = None,
    posterior_convergence: bool = False,
    posterior_convergence_every: int = CONVERGENCE_DEFAULT_EVERY,
    posterior_convergence_uncertainty_beta: float = CONVERGENCE_DEFAULT_UNCERTAINTY_BETA,
    local_peak_diagnostics: bool = False,
    local_peak_every: int = LOCAL_PEAK_DEFAULT_EVERY,
    local_peak_n: int = LOCAL_PEAK_DEFAULT_N,
    local_peak_fraction: float = LOCAL_PEAK_DEFAULT_FRACTION,
    local_peak_uncertainty_beta: float = LOCAL_PEAK_DEFAULT_UNCERTAINTY_BETA,
    surrogate_chain_diagnostics: bool = True,
    surrogate_chain_max_points: int = 2000,
    surrogate_chain_burnin: int = 0,
    surrogate_chain_burnin_fraction: Optional[float] = None,
    surrogate_chain_tail_fraction: float = 1.0,
    surrogate_chain_uncertainty_betas: str = "0,1",
    surrogate_chain_top_fractions: str = "0.05,0.01,0.001",
) -> StudyArtifacts:
    _ensure_directory(OUTPUT_DIR)
    subset_duration = float(duration_years) if duration_years is not None else float(SUBSET_DURATION_YEARS)
    scenario_root = OUTPUT_DIR / ("noise" if with_noise else "no_noise")
    _ensure_directory(scenario_root)
    subset_dir = scenario_root / _duration_tag(subset_duration)
    full_dir = scenario_root / _duration_tag(FULL_DURATION_YEARS)
    _ensure_directory(subset_dir)
    _ensure_directory(full_dir)
    artifacts = StudyArtifacts(
        full_observation=full_dir / "full_mock_observation.h5",
        subset_observation=subset_dir / "mock_observation.h5",
        analysis_dir=subset_dir / "analysis",
    )

    settings = _resolve_settings(quick)
    if bruteforce_samples is not None:
        settings["bruteforce_samples"] = int(bruteforce_samples)
    if bruteforce_temperature is not None:
        settings["bruteforce_temperature"] = float(bruteforce_temperature)
    rng = _rng(settings["seed"])

    _, obs_fn = _generate_catalogue(
        artifacts,
        duration=subset_duration,
        posterior_samples=settings["posterior_samples_per_event"],
        with_noise=with_noise,
        force=force,
        rng=rng,
    )

    lnl_computer = LnLComputer.load(
        observation_file=str(obs_fn),
        compas_h5=str(_compas_catalogue()),
    )

    truths_dict = {p: float(v) for p, v in zip(PARAMETERS, FIDUCIAL_PARAMS)}

    if bruteforce:
        _bruteforce_random_scan(
            artifacts.analysis_dir,
            lnl_computer,
            n_samples=int(settings["bruteforce_samples"]),
            seed=int(settings["seed"]) + 11,
            temperature=float(settings["bruteforce_temperature"]),
            max_points=int(settings["bruteforce_max_points"]),
            truths=truths_dict,
            force=force,
        )
        if bruteforce_only:
            return artifacts

        if bruteforce_gp:
            _bruteforce_gp_baseline(
                artifacts.analysis_dir,
                train_n=int(
                    bruteforce_gp_train_n
                    if bruteforce_gp_train_n is not None
                    else BRUTEFORCE_GP_DEFAULT_TRAIN_N
                ),
                seed=int(settings["seed"]) + 17,
                truths=truths_dict,
                mcmc_settings=dict(settings["mcmc_settings"]),
                force=force,
                strategy=str(bruteforce_gp_strategy),
                top_fraction=float(bruteforce_gp_top_fraction),
                delta=bruteforce_gp_delta,
            )

    # _lnl_1d_scan(lnl_computer, artifacts.analysis_dir)

    _offline_baseline_diagnostics(
        artifacts.analysis_dir,
        lnl_computer,
        force=force,
        seed=settings["seed"],
        truths=truths_dict,
    )

    _train_surrogate(
        artifacts,
        observation_file=obs_fn,
        force=force,
        initial_points=settings["active_learning_initial_points"],
        total_steps=settings["active_learning_total_steps"],
        steps_per_round=settings["active_learning_steps_per_round"],
        seed=settings["seed"],
    )

    _summarise_surrogate_optimum(
        artifacts.analysis_dir,
        lnl_computer,
        rng=_rng(settings["seed"] + 1),
        truths=truths_dict,
    )

    # Compare GP predictions against the precomputed 1D LnL scans
    _plot_gp_vs_true_1d(artifacts.analysis_dir)

    _run_mcmc_and_plots(
        artifacts,
        mcmc_settings=dict(settings["mcmc_settings"]),
        truths=truths_dict,
    )

    if posterior_convergence:
        _posterior_convergence_vs_round(
            artifacts,
            mcmc_settings=dict(settings["mcmc_settings"]),
            truths=truths_dict,
            initial_points=int(settings["active_learning_initial_points"]),
            steps_per_round=int(settings["active_learning_steps_per_round"]),
            every=int(posterior_convergence_every),
            uncertainty_beta=float(posterior_convergence_uncertainty_beta),
        )

    if local_peak_diagnostics:
        _local_peak_diagnostics_vs_round(
            artifacts,
            lnl_computer,
            truths=truths_dict,
            initial_points=int(settings["active_learning_initial_points"]),
            steps_per_round=int(settings["active_learning_steps_per_round"]),
            every=int(local_peak_every),
            n_points=int(local_peak_n),
            fraction=float(local_peak_fraction),
            seed=int(settings["seed"]),
            uncertainty_beta=float(local_peak_uncertainty_beta),
        )

    if surrogate_chain_diagnostics:
        chain_path = artifacts.analysis_dir / "MCMC" / "emcee_lnl_surrogate" / "chain.dat"
        if not chain_path.exists():
            logging.getLogger("simulation_study").warning(
                "Skipped surrogate-chain diagnostics; chain file %s not found.", chain_path
            )
        else:
            try:
                outdir = artifacts.analysis_dir / "surrogate_chain_diagnostics"
                default_burnin = int(mcmc_settings.get("nburn", 0))
                effective_burnin = int(surrogate_chain_burnin) if surrogate_chain_burnin else default_burnin
                run_chain_diagnostics(
                    observation_file=str(obs_fn),
                    compas_h5=str(_compas_catalogue()),
                    model_dir=str(artifacts.analysis_dir / "gp_model" / "models"),
                    chain_path=str(chain_path),
                    outdir=str(outdir),
                    max_points=int(surrogate_chain_max_points),
                    seed=int(settings["seed"]) + 123,
                    burnin=effective_burnin,
                    burnin_fraction=surrogate_chain_burnin_fraction,
                    tail_fraction=float(surrogate_chain_tail_fraction),
                    uncertainty_betas=[float(x) for x in surrogate_chain_uncertainty_betas.replace(",", " ").split()],
                    top_fractions=[float(x) for x in surrogate_chain_top_fractions.replace(",", " ").split()],
                )
            except Exception as exc:  # pragma: no cover - defensive
                logging.getLogger("simulation_study").warning(
                    "Surrogate-chain diagnostics failed (%s).", exc
                )

    return artifacts


def main(args: Optional[Sequence[str]] = None) -> None:
    force = quick = with_noise = False
    bruteforce = False
    bruteforce_samples: Optional[int] = None
    bruteforce_temperature: Optional[float] = None
    bruteforce_only = False
    bruteforce_gp = False
    bruteforce_gp_train_n: Optional[int] = None
    bruteforce_gp_strategy: str = BRUTEFORCE_GP_DEFAULT_STRATEGY
    bruteforce_gp_top_fraction: float = BRUTEFORCE_GP_DEFAULT_TOP_FRACTION
    bruteforce_gp_delta: Optional[float] = BRUTEFORCE_GP_DEFAULT_DELTA
    duration_years: Optional[float] = None
    posterior_convergence = False
    posterior_convergence_every: int = CONVERGENCE_DEFAULT_EVERY
    posterior_convergence_uncertainty_beta: float = CONVERGENCE_DEFAULT_UNCERTAINTY_BETA
    local_peak_diagnostics = False
    local_peak_every: int = LOCAL_PEAK_DEFAULT_EVERY
    local_peak_n: int = LOCAL_PEAK_DEFAULT_N
    local_peak_fraction: float = LOCAL_PEAK_DEFAULT_FRACTION
    local_peak_uncertainty_beta: float = LOCAL_PEAK_DEFAULT_UNCERTAINTY_BETA
    surrogate_chain_diagnostics = True
    surrogate_chain_max_points: int = 2000
    surrogate_chain_burnin: int = 0
    surrogate_chain_burnin_fraction: Optional[float] = None
    surrogate_chain_tail_fraction: float = 1.0
    surrogate_chain_uncertainty_betas: str = "0,1"
    surrogate_chain_top_fractions: str = "0.05,0.01,0.001"
    if args is None:
        args = []
    i = 0
    while i < len(args):
        opt = args[i]
        if opt in {"--force", "-f"}:
            force = True
        elif opt in {"--quick", "-q"}:
            quick = True
        elif opt in {"--noise", "-n"}:
            with_noise = True
        elif opt == "--bruteforce":
            bruteforce = True
        elif opt.startswith("--bruteforce-samples"):
            bruteforce = True
            if "=" in opt:
                bruteforce_samples = int(opt.split("=", 1)[1])
            else:
                i += 1
                bruteforce_samples = int(args[i])
        elif opt.startswith("--bruteforce-temperature"):
            bruteforce = True
            if "=" in opt:
                bruteforce_temperature = float(opt.split("=", 1)[1])
            else:
                i += 1
                bruteforce_temperature = float(args[i])
        elif opt == "--bruteforce-only":
            bruteforce = True
            bruteforce_only = True
        elif opt == "--bruteforce-gp":
            bruteforce = True
            bruteforce_gp = True
        elif opt.startswith("--bruteforce-gp-train-n"):
            bruteforce = True
            bruteforce_gp = True
            if "=" in opt:
                bruteforce_gp_train_n = int(opt.split("=", 1)[1])
            else:
                i += 1
                bruteforce_gp_train_n = int(args[i])
        elif opt.startswith("--bruteforce-gp-strategy"):
            bruteforce = True
            bruteforce_gp = True
            if "=" in opt:
                bruteforce_gp_strategy = str(opt.split("=", 1)[1])
            else:
                i += 1
                bruteforce_gp_strategy = str(args[i])
        elif opt.startswith("--bruteforce-gp-top-fraction"):
            bruteforce = True
            bruteforce_gp = True
            if "=" in opt:
                bruteforce_gp_top_fraction = float(opt.split("=", 1)[1])
            else:
                i += 1
                bruteforce_gp_top_fraction = float(args[i])
        elif opt.startswith("--bruteforce-gp-delta"):
            bruteforce = True
            bruteforce_gp = True
            if "=" in opt:
                value = opt.split("=", 1)[1]
            else:
                i += 1
                value = str(args[i])
            if value.strip().lower() in {"none", "off"}:
                bruteforce_gp_delta = None
            else:
                bruteforce_gp_delta = float(value)
        elif opt.startswith("--duration-years"):
            if "=" in opt:
                duration_years = float(opt.split("=", 1)[1])
            else:
                i += 1
                duration_years = float(args[i])
        elif opt == "--posterior-convergence":
            posterior_convergence = True
        elif opt.startswith("--posterior-convergence-every"):
            posterior_convergence = True
            if "=" in opt:
                posterior_convergence_every = int(opt.split("=", 1)[1])
            else:
                i += 1
                posterior_convergence_every = int(args[i])
        elif opt.startswith("--posterior-convergence-uncertainty-beta"):
            posterior_convergence = True
            if "=" in opt:
                posterior_convergence_uncertainty_beta = float(opt.split("=", 1)[1])
            else:
                i += 1
                posterior_convergence_uncertainty_beta = float(args[i])
        elif opt == "--local-peak-diagnostics":
            local_peak_diagnostics = True
        elif opt.startswith("--local-peak-every"):
            local_peak_diagnostics = True
            if "=" in opt:
                local_peak_every = int(opt.split("=", 1)[1])
            else:
                i += 1
                local_peak_every = int(args[i])
        elif opt.startswith("--local-peak-n"):
            local_peak_diagnostics = True
            if "=" in opt:
                local_peak_n = int(opt.split("=", 1)[1])
            else:
                i += 1
                local_peak_n = int(args[i])
        elif opt.startswith("--local-peak-fraction"):
            local_peak_diagnostics = True
            if "=" in opt:
                local_peak_fraction = float(opt.split("=", 1)[1])
            else:
                i += 1
                local_peak_fraction = float(args[i])
        elif opt.startswith("--local-peak-uncertainty-beta"):
            local_peak_diagnostics = True
            if "=" in opt:
                local_peak_uncertainty_beta = float(opt.split("=", 1)[1])
            else:
                i += 1
                local_peak_uncertainty_beta = float(args[i])
        elif opt == "--surrogate-chain-diagnostics":
            surrogate_chain_diagnostics = True
        elif opt.startswith("--surrogate-chain-max-points"):
            surrogate_chain_diagnostics = True
            if "=" in opt:
                surrogate_chain_max_points = int(opt.split("=", 1)[1])
            else:
                i += 1
                surrogate_chain_max_points = int(args[i])
        elif opt.startswith("--surrogate-chain-burnin"):
            surrogate_chain_diagnostics = True
            if "=" in opt:
                surrogate_chain_burnin = int(opt.split("=", 1)[1])
            else:
                i += 1
                surrogate_chain_burnin = int(args[i])
        elif opt.startswith("--surrogate-chain-burnin-fraction"):
            surrogate_chain_diagnostics = True
            if "=" in opt:
                surrogate_chain_burnin_fraction = float(opt.split("=", 1)[1])
            else:
                i += 1
                surrogate_chain_burnin_fraction = float(args[i])
        elif opt.startswith("--surrogate-chain-tail-fraction"):
            surrogate_chain_diagnostics = True
            if "=" in opt:
                surrogate_chain_tail_fraction = float(opt.split("=", 1)[1])
            else:
                i += 1
                surrogate_chain_tail_fraction = float(args[i])
        elif opt.startswith("--surrogate-chain-uncertainty-betas"):
            surrogate_chain_diagnostics = True
            if "=" in opt:
                surrogate_chain_uncertainty_betas = str(opt.split("=", 1)[1])
            else:
                i += 1
                surrogate_chain_uncertainty_betas = str(args[i])
        elif opt.startswith("--surrogate-chain-top-fractions"):
            surrogate_chain_diagnostics = True
            if "=" in opt:
                surrogate_chain_top_fractions = str(opt.split("=", 1)[1])
            else:
                i += 1
                surrogate_chain_top_fractions = str(args[i])
        elif opt == "--no-surrogate-chain-diagnostics":
            surrogate_chain_diagnostics = False
        i += 1

    artifacts = run(
        force=force,
        quick=quick,
        with_noise=with_noise,
        bruteforce=bruteforce,
        bruteforce_samples=bruteforce_samples,
        bruteforce_temperature=bruteforce_temperature,
        bruteforce_only=bruteforce_only,
        bruteforce_gp=bruteforce_gp,
        bruteforce_gp_train_n=bruteforce_gp_train_n,
        bruteforce_gp_strategy=bruteforce_gp_strategy,
        bruteforce_gp_top_fraction=bruteforce_gp_top_fraction,
        bruteforce_gp_delta=bruteforce_gp_delta,
        duration_years=duration_years,
        posterior_convergence=posterior_convergence,
        posterior_convergence_every=posterior_convergence_every,
        posterior_convergence_uncertainty_beta=posterior_convergence_uncertainty_beta,
        local_peak_diagnostics=local_peak_diagnostics,
        local_peak_every=local_peak_every,
        local_peak_n=local_peak_n,
        local_peak_fraction=local_peak_fraction,
        local_peak_uncertainty_beta=local_peak_uncertainty_beta,
        surrogate_chain_diagnostics=surrogate_chain_diagnostics,
        surrogate_chain_max_points=surrogate_chain_max_points,
        surrogate_chain_burnin=surrogate_chain_burnin,
        surrogate_chain_burnin_fraction=surrogate_chain_burnin_fraction,
        surrogate_chain_tail_fraction=surrogate_chain_tail_fraction,
        surrogate_chain_uncertainty_betas=surrogate_chain_uncertainty_betas,
        surrogate_chain_top_fractions=surrogate_chain_top_fractions,
    )
    print("Simulation study completed.")
    print(f"  Full catalogue:     {artifacts.full_observation}")
    print(f"  Mock observation:   {artifacts.subset_observation}")
    print(f"  Analysis artefacts: {artifacts.analysis_dir}")


if __name__ == "__main__":  # pragma: no cover
    import sys
    main(sys.argv[1:])
