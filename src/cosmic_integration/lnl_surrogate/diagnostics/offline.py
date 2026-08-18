from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Union

import corner
import gpflow
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy.stats import qmc
from tqdm.auto import tqdm

from trieste.acquisition.rule import RandomSampling
from trieste.models.gpflow import GaussianProcessRegression
from trieste.space import Box

from ..adaptive_robust_scalar import AdaptiveRobustScaler, suggest_lower_clip_value
from ..lnl_surrogate import BOUNDS
from ..run_sampler import sample_lnl_surrogate


def _module_with_variables(model_obj):
    """Return a tf.Module-like object carrying variables for SavedModel export."""
    return getattr(model_obj, "model", model_obj)


def _latin_hypercube(bounds: np.ndarray, n: int, seed: Optional[int] = None) -> np.ndarray:
    sampler = qmc.LatinHypercube(d=bounds.shape[1], seed=seed)
    samples = sampler.random(n=n)
    return qmc.scale(samples, bounds[0], bounds[1])


def _build_gpr_model(x: tf.Tensor, y: tf.Tensor, bounds: np.ndarray) -> gpflow.models.GPR:
    widths = (bounds[1] - bounds[0]).astype(np.float64)
    init_ls = np.clip(0.3 * widths, 1e-6, np.inf)

    kernel = gpflow.kernels.Matern52(
        variance=np.float64(1.0),
        lengthscales=init_ls,
    )
    mean_init = np.float64(np.mean(y.numpy()))
    mean_function = gpflow.mean_functions.Constant(mean_init)

    model = gpflow.models.GPR(
        data=(x, y),
        kernel=kernel,
        # Must be strictly above GPflow's positive transform lower bound (often 1e-6),
        # otherwise the unconstrained parameter becomes -inf and TF will error.
        noise_variance=np.float64(1e-5),
        mean_function=mean_function,
    )
    gpflow.set_trainable(model.likelihood.variance, False)
    opt = gpflow.optimizers.Scipy()
    opt.minimize(
        model.training_loss,
        variables=model.trainable_variables,
        options={"maxiter": 1000},
    )
    return model


def _compute_metrics(true_vals: np.ndarray, pred_vals: np.ndarray) -> Dict[str, float]:
    residuals = pred_vals - true_vals
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    mae = float(np.mean(np.abs(residuals)))
    r2 = 1.0 - float(np.sum(residuals ** 2) / (np.sum((true_vals - np.mean(true_vals)) ** 2) + 1e-12))
    max_err = float(np.max(np.abs(residuals)))
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "max_abs_err": max_err,
    }


def _maybe_plot_scatter(true_vals: np.ndarray, pred_vals: np.ndarray, outdir: Optional[Path]) -> Optional[Path]:
    if outdir is None:
        return None

    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "offline_gp_true_vs_pred.png"

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(true_vals, pred_vals, s=14, alpha=0.7)
    lims = [
        min(np.min(true_vals), np.min(pred_vals)),
        max(np.max(true_vals), np.max(pred_vals)),
    ]
    ax.plot(lims, lims, color="k", linestyle="--", linewidth=1)
    ax.set_xlabel("True LnL")
    ax.set_ylabel("Predicted LnL")
    ax.set_title("Offline GP true vs predicted")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _maybe_plot_residuals(true_vals: np.ndarray, pred_vals: np.ndarray, outdir: Optional[Path]) -> Optional[Path]:
    if outdir is None:
        return None

    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "offline_gp_residuals.png"
    residuals = pred_vals - true_vals

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(residuals, bins=40, alpha=0.8)
    ax.axvline(0.0, color="k", linestyle="--")
    ax.set_xlabel("Residual (Pred - True)")
    ax.set_ylabel("Count")
    ax.set_title("Offline GP residuals")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _run_round_diagnostics(
        *,
        round_index: int,
        train_points: np.ndarray,
        train_lnls: np.ndarray,
        test_points: np.ndarray,
        test_lnls: np.ndarray,
        bounds: np.ndarray,
        soft_clipping: bool,
        clip_factor: float,
        lower_clip_percentile: Optional[float],
        lower_clip_value: Optional[float],
        round_dir: Optional[Path],
        truths: Optional[Dict[str, float]],
        mcmc_kwargs: Optional[Dict[str, Union[int, float]]],
        param_labels: Optional[Sequence[str]] = None,
        corner_temperature: float = 500.0,
) -> tuple["OfflineRoundResult", np.ndarray]:
    if lower_clip_percentile is None and lower_clip_value is None:
        effective_lower_clip_value = suggest_lower_clip_value(
            train_lnls,
            best_fraction=0.05,
            delta_factor=20.0,
            min_delta=200.0,
            max_delta=200.0,
        )
        effective_lower_clip_percentile = None
    else:
        effective_lower_clip_value = lower_clip_value
        effective_lower_clip_percentile = lower_clip_percentile

    scaler = AdaptiveRobustScaler(
        lower_clip_percentile=effective_lower_clip_percentile,
        lower_clip_value=effective_lower_clip_value,
        soft_clipping=soft_clipping,
        clip_factor=clip_factor,
    )
    scaler.initialize_with_data(train_lnls)

    y_train = -np.asarray(scaler.transform(train_lnls), dtype=float).reshape(-1, 1)
    x_tf = tf.convert_to_tensor(train_points, dtype=tf.float64)
    y_tf = tf.convert_to_tensor(y_train, dtype=tf.float64)
    model = _build_gpr_model(x_tf, y_tf, bounds)

    x_test_tf = tf.convert_to_tensor(test_points, dtype=tf.float64)
    pred_mean, _ = model.predict_f(x_test_tf)
    pred_transformed = -pred_mean.numpy().reshape(-1)
    predicted_lnls = np.asarray(scaler.inverse_transform(pred_transformed), dtype=float)

    metrics = _compute_metrics(test_lnls, predicted_lnls)
    diagnostics = scaler.get_diagnostics()
    metrics.update({
        "reference_value": diagnostics.get("reference_value"),
        "median": diagnostics.get("median"),
        "scale": diagnostics.get("scale"),
        "lower_clip_value": float(effective_lower_clip_value) if effective_lower_clip_value is not None else None,
        "lower_clip_percentile": (float(effective_lower_clip_percentile)
                                    if effective_lower_clip_percentile is not None
                                    else None),
    })

    model_dir_path = None
    sampler_dir_path = None
    if round_dir is not None:
        if round_dir.exists():
            shutil.rmtree(round_dir)
        plots_dir = round_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        scatter_path = _maybe_plot_scatter(test_lnls, predicted_lnls, plots_dir)
        residual_path = _maybe_plot_residuals(test_lnls, predicted_lnls, plots_dir)

        # Weighted corner plot of the random training samples using LnL as weights
        try:
            labels = list(param_labels) if param_labels is not None else [f"p{i}" for i in range(train_points.shape[1])]
            delta = train_lnls - np.max(train_lnls)
            temp = max(float(corner_temperature), 1e-6)
            weights = np.exp(delta / temp)
            # Prepare truths vector if names are available
            truths_vec = None
            if truths is not None:
                try:
                    truths_vec = [float(truths.get(name)) for name in labels]
                except Exception:
                    truths_vec = None
            fig = corner.corner(train_points, labels=labels, weights=weights, show_titles=True, truths=truths_vec)
            fig.savefig(plots_dir / "weighted_corner.png", dpi=180)
            plt.close(fig)
        except Exception:
            pass

        gp_root = round_dir / "gp_model"
        model_dir = gp_root / "models" / "round_0"
        model_dir.mkdir(parents=True, exist_ok=True)

        trieste_model = GaussianProcessRegression(model, num_kernel_samples=250)
        module = _module_with_variables(trieste_model)
        module.predict_f = tf.function(
            model.predict_f,
            input_signature=[tf.TensorSpec(shape=[None, train_points.shape[1]], dtype=tf.float64)],
        )
        tf.saved_model.save(module, str(model_dir))
        scaler.save(str(gp_root))
        model_dir_path = gp_root / "models"

        sampler_dir_path = round_dir / "MCMC"
        mcmc_settings = dict(mcmc_kwargs or {})
        sample_lnl_surrogate(
            lnl_model_path=str(model_dir_path),
            outdir=str(sampler_dir_path),
            verbose=False,
            truths=truths,
            mcmc_kwargs=mcmc_settings,
        )
    else:
        scatter_path = _maybe_plot_scatter(test_lnls, predicted_lnls, None)
        residual_path = _maybe_plot_residuals(test_lnls, predicted_lnls, None)

    round_result = OfflineRoundResult(
        round_index=round_index,
        n_train=int(train_points.shape[0]),
        metrics=metrics,
        scatter_path=scatter_path,
        residual_path=residual_path,
        model_dir=model_dir_path,
        sampler_dir=sampler_dir_path,
    )
    return round_result, predicted_lnls


@dataclass
class OfflineRoundResult:
    round_index: int
    n_train: int
    metrics: Dict[str, float]
    scatter_path: Optional[Path]
    residual_path: Optional[Path]
    model_dir: Optional[Path]
    sampler_dir: Optional[Path]


@dataclass
class OfflineDiagnosticsResult:
    rounds: Sequence[OfflineRoundResult]
    final_train_points: np.ndarray
    final_train_lnls: np.ndarray
    test_points: np.ndarray
    test_lnls: np.ndarray
    final_predictions: np.ndarray


def offline_surrogate_diagnostics(
        evaluator: Callable[[Sequence[float]], float],
        *,
        bounds: np.ndarray = BOUNDS,
        n_train: int = 400,
        n_test: int = 400,
        round_totals: Optional[Sequence[int]] = None,
        seed: Optional[int] = None,
        scaler_kwargs: Optional[Dict[str, object]] = None,
        outdir: Optional[Union[str, Path]] = None,
        truths: Optional[Dict[str, float]] = None,
        mcmc_kwargs: Optional[Dict[str, Union[int, float]]] = None,
        param_labels: Optional[Sequence[str]] = None,
        corner_temperature: float = 500.0,
) -> OfflineDiagnosticsResult:
    """
    Fit a GP surrogate from scratch using Trieste's RandomSampling rule as a baseline
    and produce diagnostic artefacts (scatter, residuals, optional MCMC corner plots).
    """
    logger = logging.getLogger(__name__)

    if n_test <= 0:
        raise ValueError("n_test must be positive.")

    if round_totals is None:
        round_totals = [n_train]
    else:
        round_totals = list(round_totals)

    if len(round_totals) == 0:
        raise ValueError("round_totals must contain at least one positive integer.")

    last_total = 0
    for total in round_totals:
        if total <= 0:
            raise ValueError("round_totals entries must be positive.")
        if total <= last_total:
            raise ValueError("round_totals must be strictly increasing.")
        last_total = total

    if seed is not None:
        tf.random.set_seed(int(seed))

    scaler_kw = dict(scaler_kwargs or {})
    soft_clipping = scaler_kw.pop("soft_clipping", True)
    clip_factor = scaler_kw.pop("clip_factor", 3.0)
    lower_clip_percentile = scaler_kw.pop("lower_clip_percentile", None)
    if isinstance(lower_clip_percentile, str):
        if lower_clip_percentile.lower() == "auto":
            lower_clip_percentile = None
        else:
            raise ValueError(f"Unknown lower_clip_percentile string: {lower_clip_percentile}")
    lower_clip_value = scaler_kw.pop("lower_clip_value", None)

    if scaler_kw:
        unused = ", ".join(sorted(scaler_kw.keys()))
        raise ValueError(f"Unknown scaler_kwargs: {unused}")

    dim = bounds.shape[1]
    search_space = Box(bounds[0], bounds[1])

    test_points = _latin_hypercube(bounds, n_test, seed=None if seed is None else seed + 1)
    logger.info("Evaluating log-likelihoods for test set (%d points)", n_test)
    test_lnls = np.array(
        [evaluator(pt) for pt in tqdm(test_points, desc="Test LnL", unit="pt")],
        dtype=float,
    )

    train_points = np.empty((0, dim), dtype=float)
    train_lnls = np.empty((0,), dtype=float)
    round_results: list[OfflineRoundResult] = []
    final_predictions = np.empty_like(test_lnls)

    base_dir = Path(outdir) if outdir is not None else None
    if base_dir is not None:
        base_dir.mkdir(parents=True, exist_ok=True)

    for round_idx, target_total in enumerate(round_totals):
        to_add = target_total - train_points.shape[0]
        logger.info(
            "Offline diagnostics round %d: sampling %d new points (total %d)",
            round_idx + 1,
            to_add,
            target_total,
        )
        new_points_batches = []
        remaining = to_add
        while remaining > 0:
            batch = min(64, remaining)
            rule = RandomSampling(num_query_points=batch)
            sampled = rule.acquire(search_space, models={})
            new_points_batches.append(sampled.numpy())
            remaining -= batch
        if new_points_batches:
            new_points_arr = np.vstack(new_points_batches)
            new_lnls = np.array(
                [evaluator(pt) for pt in tqdm(new_points_arr, desc=f"Round {round_idx+1} LnL", unit="pt", leave=False)],
                dtype=float,
            )
            train_points = np.vstack([train_points, new_points_arr])
            train_lnls = np.concatenate([train_lnls, new_lnls])

        round_dir = base_dir / f"round_{round_idx}" if base_dir is not None else None
        round_result, preds = _run_round_diagnostics(
            round_index=round_idx,
            train_points=train_points,
            train_lnls=train_lnls,
            test_points=test_points,
            test_lnls=test_lnls,
            bounds=bounds,
            soft_clipping=soft_clipping,
            clip_factor=clip_factor,
            lower_clip_percentile=lower_clip_percentile,
            lower_clip_value=lower_clip_value,
            round_dir=round_dir,
            truths=truths,
            mcmc_kwargs=mcmc_kwargs,
            param_labels=param_labels,
            corner_temperature=corner_temperature,
        )
        round_results.append(round_result)
        final_predictions = preds

    # Final aggregated weighted corner plot across all training samples
    if base_dir is not None and train_points.size > 0:
        try:
            labels = list(param_labels) if param_labels is not None else [f"p{i}" for i in range(train_points.shape[1])]
            delta = train_lnls - np.max(train_lnls)
            temp = max(float(corner_temperature), 1e-6)
            weights = np.exp(delta / temp)
            truths_vec = None
            if truths is not None:
                try:
                    truths_vec = [float(truths.get(name)) for name in labels]
                except Exception:
                    truths_vec = None
            fig = corner.corner(train_points, labels=labels, weights=weights, show_titles=True, truths=truths_vec)
            fig.savefig(base_dir / "weighted_corner.png", dpi=180)
            plt.close(fig)
        except Exception:
            pass

    return OfflineDiagnosticsResult(
        rounds=tuple(round_results),
        final_train_points=train_points,
        final_train_lnls=train_lnls,
        test_points=test_points,
        test_lnls=test_lnls,
        final_predictions=final_predictions,
    )


def fit_surrogate_from_samples(
    train_points: np.ndarray,
    train_lnls: np.ndarray,
    *,
    bounds: np.ndarray = BOUNDS,
    outdir: Union[str, Path],
    scaler_kwargs: Optional[Dict[str, object]] = None,
    overwrite: bool = True,
) -> Path:
    """
    Fit and persist a GP surrogate from a precomputed set of (points, LnL) samples.
    """

    out_path = Path(outdir)
    gp_root = out_path / "gp_model"
    if overwrite and gp_root.exists():
        shutil.rmtree(gp_root)
    gp_root.mkdir(parents=True, exist_ok=True)

    x = np.asarray(train_points, dtype=float)
    y = np.asarray(train_lnls, dtype=float).reshape(-1)
    if x.ndim != 2:
        raise ValueError("train_points must be a 2D array [N, D].")
    if y.ndim != 1 or y.shape[0] != x.shape[0]:
        raise ValueError("train_lnls must be a 1D array with length N.")

    scaler_kw = dict(scaler_kwargs or {})
    soft_clipping = scaler_kw.pop("soft_clipping", True)
    clip_factor = float(scaler_kw.pop("clip_factor", 5.0))
    focus_fraction = float(scaler_kw.pop("focus_fraction", 0.05))
    max_scale = float(scaler_kw.pop("max_scale", 10.0))
    lower_clip_value = scaler_kw.pop("lower_clip_value", None)
    lower_clip_percentile = scaler_kw.pop("lower_clip_percentile", None)
    if isinstance(lower_clip_percentile, str):
        if lower_clip_percentile.lower() == "auto":
            lower_clip_percentile = None
        else:
            raise ValueError(f"Unknown lower_clip_percentile string: {lower_clip_percentile}")
    if scaler_kw:
        unused = ", ".join(sorted(scaler_kw.keys()))
        raise ValueError(f"Unknown scaler_kwargs: {unused}")

    if lower_clip_percentile is None and lower_clip_value is None:
        lower_clip_value = suggest_lower_clip_value(
            y,
            best_fraction=0.05,
            delta_factor=20.0,
            min_delta=200.0,
            max_delta=200.0,
        )
        lower_clip_percentile = None

    scaler = AdaptiveRobustScaler(
        lower_clip_percentile=lower_clip_percentile,
        lower_clip_value=lower_clip_value,
        soft_clipping=soft_clipping,
        clip_factor=clip_factor,
        focus_fraction=focus_fraction,
        max_scale=max_scale,
    )
    scaler.initialize_with_data(y)

    y_train = -np.asarray(scaler.transform(y), dtype=float).reshape(-1, 1)
    x_tf = tf.convert_to_tensor(x, dtype=tf.float64)
    y_tf = tf.convert_to_tensor(y_train, dtype=tf.float64)
    model = _build_gpr_model(x_tf, y_tf, bounds)

    model_dir = gp_root / "models" / "round_0"
    model_dir.mkdir(parents=True, exist_ok=True)
    trieste_model = GaussianProcessRegression(model, num_kernel_samples=250)
    module = _module_with_variables(trieste_model)
    module.predict_f = tf.function(
        model.predict_f,
        input_signature=[tf.TensorSpec(shape=[None, x.shape[1]], dtype=tf.float64)],
    )
    tf.saved_model.save(module, str(model_dir))
    scaler.save(str(gp_root))

    return gp_root / "models"

