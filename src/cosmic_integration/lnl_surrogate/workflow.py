from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
from tqdm.auto import tqdm

from ..lnl_computer import LnLComputer
from ..observation import load_observation
from .active_learner import ActiveLearner
from .adaptive_robust_scalar import robust_neg_lnl_computer_factory, suggest_lower_clip_value
from .diagnostics.gp_truth import gp_accuracy_vs_distance
from .diagnostics.posterior_kl import consecutive_posterior_kl
from .lnl_surrogate import BOUNDS, PARAMETERS, sample_points
from .run_sampler import sample_lnl_surrogate


@dataclass(frozen=True)
class SurrogateWorkflowConfig:
    compas_h5: str
    observation_file: str
    outdir: str

    initial_points: int = 50
    total_steps: int = 300
    steps_per_round: int = 30

    seed: int = 0
    force_dataset: bool = False

    postprocess_every: int = 2
    mcmc_kwargs: dict[str, Any] = field(default_factory=lambda: {"nwalkers": 32, "iterations": 2000})
    mcmc_uncertainty_beta: float | Sequence[float] = 0.0

    gp_truth_n_per_fraction: int = 200
    gp_truth_fractions: Sequence[float] = (0.01, 0.02, 0.05, 0.1, 0.2, 0.4)

    scaler_soft_clipping: bool = True
    scaler_clip_factor: float = 3.0
    scaler_lower_clip_percentile: float | None | str = "auto"

    callback_fail_fast: bool = True

    def to_json(self) -> str:
        return json.dumps(
            {
                "compas_h5": self.compas_h5,
                "observation_file": self.observation_file,
                "outdir": self.outdir,
                "initial_points": int(self.initial_points),
                "total_steps": int(self.total_steps),
                "steps_per_round": int(self.steps_per_round),
                "seed": int(self.seed),
                "force_dataset": bool(self.force_dataset),
                "postprocess_every": int(self.postprocess_every),
                "mcmc_kwargs": dict(self.mcmc_kwargs),
                "mcmc_uncertainty_beta": self.mcmc_uncertainty_beta,
                "gp_truth_n_per_fraction": int(self.gp_truth_n_per_fraction),
                "gp_truth_fractions": list(float(x) for x in self.gp_truth_fractions),
                "scaler_soft_clipping": bool(self.scaler_soft_clipping),
                "scaler_clip_factor": float(self.scaler_clip_factor),
                "scaler_lower_clip_percentile": self.scaler_lower_clip_percentile,
                "callback_fail_fast": bool(self.callback_fail_fast),
            },
            indent=2,
        )


def _truths_from_observation(observation_file: str) -> tuple[np.ndarray, Dict[str, float]]:
    obs = load_observation(observation_file)
    truth_vec = np.asarray(obs.params, dtype=float).reshape(-1)
    truths = {param: float(truth_vec[i]) for i, param in enumerate(PARAMETERS) if i < truth_vec.size}
    return truth_vec, truths


def _ensure_initial_dataset(
    *,
    dataset_path: Path,
    lnl_computer: LnLComputer,
    n: int,
    seed: int,
    force: bool,
) -> tuple[np.ndarray, np.ndarray]:
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    if dataset_path.exists() and not force:
        payload = np.load(dataset_path)
        x = np.asarray(payload["x"], dtype=float)
        lnls = np.asarray(payload["lnl"], dtype=float)
        if x.ndim != 2 or lnls.ndim != 1 or x.shape[0] != lnls.shape[0]:
            raise ValueError(f"Invalid dataset file: {dataset_path}")
        return x, lnls

    rng = np.random.default_rng(int(seed))
    np.random.seed(int(seed))

    x = sample_points(int(n), parameters=list(PARAMETERS), seed=int(seed))
    lnls = np.array([lnl_computer(*row) for row in tqdm(x, desc="Initial LnL", unit="pt")], dtype=float)
    np.savez_compressed(dataset_path, x=x, lnl=lnls)
    return x, lnls


def _resolve_lower_clip(
    *,
    initial_lnls: np.ndarray,
    lnl_computer: LnLComputer,
    lower_clip_percentile: float | None | str,
) -> tuple[Optional[float], Optional[float]]:
    lower_clip_value = None
    lower_clip_percentile_value = None

    if isinstance(lower_clip_percentile, str):
        if lower_clip_percentile.lower() != "auto":
            raise ValueError(f"Unknown scaler_lower_clip_percentile string: {lower_clip_percentile}")
        try:
            n_events = int(lnl_computer.observation.population_weights.shape[0])
        except Exception:
            n_events = 0

        min_delta = max(2.0e4, 200.0 * float(max(n_events, 1)))
        max_delta = max(2.0e5, 900.0 * float(max(n_events, 1)))
        lower_clip_value = suggest_lower_clip_value(
            initial_lnls,
            best_fraction=0.05,
            delta_factor=50.0,
            min_delta=min_delta,
            max_delta=max_delta,
        )
    else:
        lower_clip_percentile_value = lower_clip_percentile

    return lower_clip_value, lower_clip_percentile_value


def run_surrogate_workflow(config: SurrogateWorkflowConfig) -> dict:
    """
    End-to-end workflow:
      1) Ensure initial dataset exists (raw LnL samples)
      2) Run BO/active learning with per-round GP checkpoints
      3) At configurable intervals, run per-round post-processing:
           - MCMC on current surrogate (posterior samples + corner)
           - GP-vs-truth accuracy vs distance diagnostics
      4) Final analysis: consecutive KL divergence between posteriors
    """

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    outdir = Path(config.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "workflow_config.json").write_text(config.to_json(), encoding="utf-8")

    truth_vec, truths = _truths_from_observation(config.observation_file)
    lnl_computer = LnLComputer.load(
        observation_file=config.observation_file,
        compas_h5=config.compas_h5,
        cache_fn=str(outdir / "lnl_cache.csv"),
    )

    dataset_path = outdir / "dataset" / "initial_dataset.npz"
    initial_x, initial_lnls = _ensure_initial_dataset(
        dataset_path=dataset_path,
        lnl_computer=lnl_computer,
        n=int(config.initial_points),
        seed=int(config.seed),
        force=bool(config.force_dataset),
    )

    lower_clip_value, lower_clip_percentile = _resolve_lower_clip(
        initial_lnls=initial_lnls,
        lnl_computer=lnl_computer,
        lower_clip_percentile=config.scaler_lower_clip_percentile,
    )

    neg_lnl_computer = robust_neg_lnl_computer_factory(
        lnl_computer,
        initial_lnls,
        soft_clipping=bool(config.scaler_soft_clipping),
        clip_factor=float(config.scaler_clip_factor),
        lower_clip_percentile=lower_clip_percentile,
        lower_clip_value=lower_clip_value,
        focus_fraction=0.05,
        max_scale=10.0,
    )

    scaler = neg_lnl_computer.scaler
    initial_y = -np.asarray([scaler.transform(v) for v in initial_lnls], dtype=float).reshape(-1, 1)

    model_dir = outdir / "gp_model"
    learner = ActiveLearner(
        trainable_function=neg_lnl_computer,
        bounds=np.asarray(BOUNDS, dtype=float),
        outdir=str(model_dir),
        initial_data_x=initial_x,
        initial_data_y=initial_y,
        true_minima=np.asarray(truth_vec, dtype=float),
        random_seed=int(config.seed),
    )

    rounds_root = outdir / "rounds"
    rounds_root.mkdir(parents=True, exist_ok=True)
    processed_rounds: list[int] = []

    def _postprocess_round(round_idx: int, active_learner: ActiveLearner) -> None:
        if int(config.postprocess_every) <= 0:
            return
        if int(round_idx) % int(config.postprocess_every) != 0:
            return

        round_dir = rounds_root / f"round_{int(round_idx)}"
        round_dir.mkdir(parents=True, exist_ok=True)

        points = active_learner.current_dataset.query_points.numpy()
        obs = active_learner.current_dataset.observations.numpy().reshape(-1)
        best_idx = int(np.argmin(obs)) if obs.size else 0
        best_x = points[best_idx].tolist() if points.size else None
        best_lnl = None
        if obs.size:
            best_lnl = float(active_learner.trainable_function.scaler.inverse_transform(-float(obs[best_idx])))

        (round_dir / "bo_summary.json").write_text(
            json.dumps(
                {
                    "round": int(round_idx),
                    "n_train": int(points.shape[0]),
                    "best_x": best_x,
                    "best_lnl": best_lnl,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        mcmc_dir = round_dir / "MCMC"
        mcmc_dir.mkdir(parents=True, exist_ok=True)
        sample_lnl_surrogate(
            lnl_model_path=str(model_dir / "models"),
            outdir=str(mcmc_dir),
            verbose=False,
            truths=truths,
            mcmc_kwargs=dict(config.mcmc_kwargs),
            uncertainty_beta=config.mcmc_uncertainty_beta,
            model_round_idx=int(round_idx),
        )

        gp_accuracy_vs_distance(
            outdir=round_dir / "gp_truth",
            lnl_computer=lnl_computer,
            models_dir=model_dir / "models",
            round_idx=int(round_idx),
            truths=truths,
            seed=int(config.seed),
            n_per_fraction=int(config.gp_truth_n_per_fraction),
            fractions=tuple(float(x) for x in config.gp_truth_fractions),
            uncertainty_beta=0.0,
        )

        processed_rounds.append(int(round_idx))

    learner.run(
        total_steps=int(config.total_steps),
        steps_per_round=int(config.steps_per_round),
        round_callback=_postprocess_round,
        callback_fail_fast=bool(config.callback_fail_fast),
    )

    model_dir.mkdir(parents=True, exist_ok=True)
    scaler.save(str(model_dir))

    final_dir = outdir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    kl_series = []
    if len(processed_rounds) >= 2:
        kl_series = consecutive_posterior_kl(
            postprocess_root=rounds_root,
            rounds=sorted(set(processed_rounds)),
            outdir=final_dir / "posterior_kl",
        )

    summary = {
        "outdir": str(outdir.resolve()),
        "models_dir": str((model_dir / "models").resolve()),
        "dataset": str(dataset_path.resolve()),
        "processed_rounds": sorted(set(int(r) for r in processed_rounds)),
        "kl_series_path": str((final_dir / "posterior_kl" / "kl_consecutive.json").resolve())
        if kl_series
        else None,
    }
    (outdir / "workflow_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
