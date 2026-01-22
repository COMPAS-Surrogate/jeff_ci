from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from ..lnl_surrogate import BOUNDS, LnLSurrogate, PARAMETERS
from ...utils import row_to_matrix_params_lnl
from ..run_sampler import sample_lnl_surrogate


def plot_gp_vs_true_1d(
    analysis_dir: str | Path,
    *,
    fiducial_params: Optional[np.ndarray] = None,
    parameters: Sequence[str] = PARAMETERS,
) -> None:
    """Overlay GP-predicted LnL against precomputed 1D scans (if present)."""
    analysis_dir = Path(analysis_dir)
    data_path = analysis_dir / "lnl_1d" / "scan_data.json"
    model_dir = analysis_dir / "gp_model" / "models"
    if not data_path.exists() or not model_dir.exists():
        return

    try:
        payload = json.loads(data_path.read_text(encoding="utf-8"))
    except Exception:
        return

    try:
        surrogate = LnLSurrogate.load(str(model_dir))
    except Exception:
        return

    fid = payload.get("fiducial")
    if fid is None:
        fid = fiducial_params.tolist() if fiducial_params is not None else None
    fid = np.array(fid if fid is not None else np.zeros(len(parameters)), dtype=float)
    curves = payload.get("curves", [])
    if not curves:
        return

    n_params = len(curves)
    fig, axes = plt.subplots(n_params, 2, figsize=(10, 2.6 * n_params))
    if axes.ndim == 1:
        axes = axes.reshape(n_params, 1)

    scaler = surrogate.scaler

    for row_idx, curve in enumerate(curves):
        pname = curve["param"]
        xvals = np.array(curve["x"], dtype=float)
        true_y = np.array(curve["lnl"], dtype=float)

        pts = np.tile(fid, (xvals.size, 1))
        idx = list(parameters).index(pname)
        pts[:, idx] = xvals

        try:
            preds_tf, _ = surrogate.gp_model.predict_f(pts)
            preds = preds_tf.numpy().reshape(-1)
            transformed = -preds
            gp_y = np.array([surrogate.scaler.inverse_transform(v) for v in transformed])
            true_transformed = np.array([surrogate.scaler.transform(v) for v in true_y])
            gp_transformed = transformed
        except Exception:
            gp_y = np.full_like(xvals, np.nan, dtype=float)
            true_transformed = np.full_like(xvals, np.nan, dtype=float)
            gp_transformed = np.full_like(xvals, np.nan, dtype=float)

        ax_lnl = axes[row_idx, 0]
        ax_lnl.plot(xvals, true_y, label="True LnL", color="C0")
        ax_lnl.plot(xvals, gp_y, label="GP pred", color="C1", linestyle="--")
        ax_lnl.axvline(fid[idx], color="r", ls="--", alpha=0.7)
        ax_lnl.set_xlabel(pname)
        ax_lnl.set_ylabel("LnL")
        ax_lnl.legend(loc="best", fontsize=8)

        ax_trans = axes[row_idx, 1]
        ax_trans.plot(xvals, true_transformed, label="True (scaled)", color="C2")
        ax_trans.plot(xvals, gp_transformed, label="GP pred (scaled)", color="C3", linestyle="--")
        ax_trans.axvline(fid[idx], color="r", ls="--", alpha=0.7)
        ax_trans.set_xlabel(pname)
        ax_trans.set_ylabel("Scaled LnL")
        ax_trans.legend(loc="best", fontsize=8)

    lower_clip = getattr(surrogate.scaler, "lower_clip_value", None)
    lower_clip_txt = f"{lower_clip:.1f}" if lower_clip is not None else "none"
    clip_txt = (
        f"{getattr(scaler, 'clip_factor', np.nan):.1f}"
        if getattr(scaler, "soft_clipping", False)
        else "off"
    )
    scaler_summary = (
        f"Scaler stats → reference={scaler.reference_value:.2f}, "
        f"median={scaler.median:.2f}, scale={scaler.scale:.2f}, "
        f"soft_clip={clip_txt}, lower_clip={lower_clip_txt}"
    )
    fig.suptitle(scaler_summary, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(analysis_dir / "lnl_1d_gp_vs_true.png", dpi=200)
    plt.close(fig)


def summarise_surrogate_optimum(
    analysis_dir: str | Path,
    lnl_computer,
    *,
    rng: np.random.Generator,
    truths: Dict[str, float],
    parameters: Sequence[str] = PARAMETERS,
    bounds: np.ndarray = BOUNDS,
) -> None:
    """
    Summarise whether the BO run evaluated (and whether the surrogate predicts) the best region.

    Writes `surrogate_opt_check.json` under `analysis_dir`.
    """
    analysis_dir = Path(analysis_dir)
    logger = logging.getLogger("simulation_study")
    cache_fn = analysis_dir / "lnl_cache.csv"
    model_dir = analysis_dir / "gp_model" / "models"
    if not cache_fn.exists():
        return

    try:
        data = np.genfromtxt(cache_fn, delimiter=",")
        if data.ndim == 1:
            data = data.reshape(1, -1)
    except Exception:
        return

    params: list[np.ndarray] = []
    lnls: list[float] = []
    for row in data:
        try:
            _, p, lnl = row_to_matrix_params_lnl(row)
        except Exception:
            continue
        if lnl is None or not np.isfinite(lnl):
            continue
        params.append(np.asarray(p, dtype=float))
        lnls.append(float(lnl))

    if not lnls:
        return

    params_arr = np.vstack(params)
    lnls_arr = np.asarray(lnls, dtype=float)
    best_idx = int(np.nanargmax(lnls_arr))
    best_true_lnl = float(lnls_arr[best_idx])
    best_true_params = params_arr[best_idx]

    logger.info(
        "BO evaluated best LnL=%.2f at alpha=%.4f sigma=%.4f sfr_a=%.5f sfr_d=%.4f",
        best_true_lnl,
        best_true_params[0],
        best_true_params[1],
        best_true_params[2],
        best_true_params[3],
    )

    summary: dict = {
        "best_true_lnl": best_true_lnl,
        "best_true_params": {p: float(v) for p, v in zip(parameters, best_true_params)},
    }

    try:
        truth_vec = np.array([truths[p] for p in parameters], dtype=float)
        truth_lnl = float(lnl_computer(*truth_vec))
        summary["truth_lnl"] = truth_lnl
        summary["truth_params"] = {p: float(v) for p, v in zip(parameters, truth_vec)}
        summary["best_true_minus_truth_lnl"] = float(best_true_lnl - truth_lnl)
    except Exception:
        pass

    if not model_dir.exists():
        (analysis_dir / "surrogate_opt_check.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        return

    try:
        import tensorflow as tf

        surrogate = LnLSurrogate.load(str(model_dir))
        x = tf.convert_to_tensor(params_arr, dtype=tf.float64)
        y_pred, _ = surrogate.gp_model.predict_f(x)
        y_pred = np.asarray(y_pred.numpy().reshape(-1), dtype=float)
        transformed = -y_pred
        lnl_pred = np.asarray(surrogate.scaler.inverse_transform(transformed), dtype=float)

        err = lnl_pred - lnls_arr
        summary["surrogate_rmse_all"] = float(np.sqrt(np.mean(err**2)))
        summary["surrogate_mae_all"] = float(np.mean(np.abs(err)))

        q = float(np.quantile(lnls_arr, 0.95))
        mask = lnls_arr >= q
        if int(np.sum(mask)) >= 3:
            err_top = err[mask]
            summary["surrogate_top5_threshold"] = q
            summary["surrogate_rmse_top5"] = float(np.sqrt(np.mean(err_top**2)))
            summary["surrogate_mae_top5"] = float(np.mean(np.abs(err_top)))

        n_search = int(np.clip(int(2000), 100, 50000))
        random_points = rng.uniform(bounds[0], bounds[1], size=(n_search, len(parameters)))
        y_rs, _ = surrogate.gp_model.predict_f(tf.convert_to_tensor(random_points, dtype=tf.float64))
        y_rs = np.asarray(y_rs.numpy().reshape(-1), dtype=float)
        lnl_rs = np.asarray(surrogate.scaler.inverse_transform(-y_rs), dtype=float)
        j = int(np.argmax(lnl_rs))
        proposed = random_points[j]
        proposed_pred_lnl = float(lnl_rs[j])
        proposed_true_lnl = float(lnl_computer(*proposed))

        summary["surrogate_random_search"] = {
            "n_samples": n_search,
            "proposed_params": {p: float(v) for p, v in zip(parameters, proposed)},
            "predicted_lnl": proposed_pred_lnl,
            "true_lnl": proposed_true_lnl,
            "pred_minus_true_lnl": float(proposed_pred_lnl - proposed_true_lnl),
            "true_minus_best_true_lnl": float(proposed_true_lnl - best_true_lnl),
        }
    except Exception as exc:
        logger.warning("Surrogate optimum summary failed: %s", exc)

    (analysis_dir / "surrogate_opt_check.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def posterior_convergence_vs_round(
    analysis_dir: str | Path,
    *,
    models_dir: str | Path,
    mcmc_settings: dict,
    truths: dict,
    initial_points: int,
    steps_per_round: int,
    every: int,
    uncertainty_beta: float,
    bins: int = 60,
    parameters: Sequence[str] = PARAMETERS,
) -> None:
    analysis_dir = Path(analysis_dir)
    models_dir = Path(models_dir)
    if not models_dir.exists():
        logging.getLogger("simulation_study").warning(
            "Skipped posterior convergence; models directory %s not found.", models_dir
        )
        return

    round_indices: list[int] = []
    for entry in models_dir.glob("round_*"):
        if not entry.is_dir():
            continue
        try:
            round_indices.append(int(entry.name.split("_", 1)[1]))
        except Exception:
            continue
    round_indices = sorted(set(round_indices))
    if not round_indices:
        return

    if every <= 0:
        raise ValueError("posterior_convergence_every must be positive.")
    selected = round_indices[::every]
    if selected[-1] != round_indices[-1]:
        selected.append(round_indices[-1])

    outdir = analysis_dir / "posterior_convergence"
    outdir.mkdir(parents=True, exist_ok=True)
    mcmc_root = outdir / "MCMC"
    mcmc_root.mkdir(parents=True, exist_ok=True)

    def _samples_for_round(r: int) -> np.ndarray:
        result = sample_lnl_surrogate(
            lnl_model_path=str(models_dir),
            outdir=str(mcmc_root / f"round_{r}"),
            verbose=False,
            truths=truths,
            mcmc_kwargs=dict(mcmc_settings),
            uncertainty_beta=float(uncertainty_beta),
            model_round_idx=int(r),
        )
        if result is None or getattr(result, "posterior", None) is None:
            raise RuntimeError(f"Missing posterior for round {r}")
        return result.posterior[list(parameters)].to_numpy(dtype=float)

    ref_round = int(round_indices[-1])
    ref_samples = _samples_for_round(ref_round)

    series: list[dict] = []
    for r in selected:
        r = int(r)
        samples = ref_samples if r == ref_round else _samples_for_round(r)
        jsd_by_param: dict[str, float] = {}
        skl_by_param: dict[str, float] = {}
        for idx, param in enumerate(parameters):
            jsd_by_param[param] = _jsd_1d_hist(samples[:, idx], ref_samples[:, idx], bins=bins)
            skl_by_param[param] = _skl_1d_hist(samples[:, idx], ref_samples[:, idx], bins=bins)

        jsd_vals = [v for v in jsd_by_param.values() if np.isfinite(v)]
        skl_vals = [v for v in skl_by_param.values() if np.isfinite(v)]
        n_train = int(initial_points + (r + 1) * steps_per_round)
        series.append(
            {
                "round": r,
                "n_train": n_train,
                "reference_round": ref_round,
                "uncertainty_beta": float(uncertainty_beta),
                "jsd_by_param": jsd_by_param,
                "jsd_mean": float(np.mean(jsd_vals)) if jsd_vals else float("nan"),
                "skl_by_param": skl_by_param,
                "skl_mean": float(np.mean(skl_vals)) if skl_vals else float("nan"),
            }
        )

    (outdir / "divergence_vs_round.json").write_text(json.dumps(series, indent=2), encoding="utf-8")

    x = [item["n_train"] for item in series]
    y_jsd = [item["jsd_mean"] for item in series]
    y_skl = [item["skl_mean"] for item in series]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, y_jsd, marker="o", label="mean JSD (1D marginals)")
    ax.plot(x, y_skl, marker="o", label="mean sym KL (1D marginals)")
    ax.set_xlabel("Training points")
    ax.set_ylabel("Divergence vs final-round posterior")
    ax.set_title("Posterior convergence (checkpointed GPs)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "divergence_vs_train_points.png", dpi=200)
    plt.close(fig)


def local_peak_diagnostics_vs_round(
    analysis_dir: str | Path,
    models_dir: str | Path,
    lnl_computer,
    *,
    truths: dict,
    initial_points: int,
    steps_per_round: int,
    every: int,
    n_points: int,
    fraction: float,
    seed: int,
    uncertainty_beta: float,
    parameters: Sequence[str] = PARAMETERS,
    bounds: np.ndarray = BOUNDS,
) -> None:
    analysis_dir = Path(analysis_dir)
    models_dir = Path(models_dir)
    if not models_dir.exists():
        logging.getLogger("simulation_study").warning(
            "Skipped local-peak diagnostics; models directory %s not found.", models_dir
        )
        return

    round_indices: list[int] = []
    for entry in models_dir.glob("round_*"):
        if not entry.is_dir():
            continue
        try:
            round_indices.append(int(entry.name.split("_", 1)[1]))
        except Exception:
            continue
    round_indices = sorted(set(round_indices))
    if not round_indices:
        return

    if every <= 0:
        raise ValueError("local_peak_every must be positive.")
    selected = round_indices[::every]
    if selected[-1] != round_indices[-1]:
        selected.append(round_indices[-1])

    center = _load_posterior_mode_params(analysis_dir, parameters=parameters)
    center_label = "posterior_mode"
    if center is None:
        center = np.array([float(truths[p]) for p in parameters], dtype=float)
        center_label = "truth"

    bounds_arr = np.asarray(bounds, dtype=float)
    widths = (bounds_arr[1] - bounds_arr[0]).astype(float)
    scale = float(fraction) * widths
    if n_points <= 0:
        raise ValueError("n_points must be positive.")
    if float(fraction) <= 0.0:
        raise ValueError("fraction must be positive.")

    rng = np.random.default_rng(int(seed) + 991)
    points = center + rng.normal(size=(int(n_points), len(parameters))) * scale
    points = np.clip(points, bounds_arr[0], bounds_arr[1])
    points[0, :] = center

    true_lnls = np.array(
        [
            lnl_computer(
                alpha=float(row[0]),
                sigma=float(row[1]),
                sfr_a=float(row[2]),
                sfr_d=float(row[3]),
            )
            for row in tqdm(points, desc="True LnL near peak", unit="pt")
        ],
        dtype=float,
    )

    outdir = analysis_dir / "local_peak_diagnostics"
    outdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        outdir / "evaluation_points.npz",
        points=points,
        true_lnls=true_lnls,
        center=center,
        center_label=center_label,
        bounds=bounds_arr,
        fraction=float(fraction),
    )

    series: list[dict] = []
    for r in selected:
        r = int(r)
        surrogate = LnLSurrogate.load(str(models_dir), uncertainty_beta=float(uncertainty_beta), round_idx=r)

        import tensorflow as tf

        x_tf = tf.convert_to_tensor(points, dtype=tf.float64)
        neg_mu, neg_var = surrogate.gp_model.predict_f(x_tf)
        neg_mu = np.asarray(neg_mu.numpy().reshape(-1), dtype=float)
        neg_var = np.asarray(neg_var.numpy().reshape(-1), dtype=float)
        transformed_mu = -neg_mu
        transformed_std = np.sqrt(np.maximum(neg_var, 0.0))

        pred_mu_lnls = np.asarray(surrogate.scaler.inverse_transform(transformed_mu), dtype=float)
        pred_sigma_lnls = _sigma_lnl_from_transformed(
            transformed_mu=transformed_mu,
            transformed_std=transformed_std,
            scaler=surrogate.scaler,
        )

        err = pred_mu_lnls - true_lnls
        rmse = float(np.sqrt(np.mean(err**2)))
        mae = float(np.mean(np.abs(err)))
        max_abs = float(np.max(np.abs(err)))

        denom = np.where(pred_sigma_lnls > 0.0, pred_sigma_lnls, np.nan)
        z = err / denom
        abs_z = np.abs(z[np.isfinite(z)])
        mean_abs_z = float(np.mean(abs_z)) if abs_z.size else float("nan")

        coverage_1 = float(np.mean(np.abs(err) <= pred_sigma_lnls)) if pred_sigma_lnls.size else float("nan")
        coverage_2 = float(np.mean(np.abs(err) <= 2.0 * pred_sigma_lnls)) if pred_sigma_lnls.size else float("nan")

        n_train = int(initial_points + (r + 1) * steps_per_round)
        series.append(
            {
                "round": r,
                "n_train": n_train,
                "center": center_label,
                "fraction": float(fraction),
                "n_points": int(n_points),
                "uncertainty_beta": float(uncertainty_beta),
                "rmse": rmse,
                "mae": mae,
                "max_abs_err": max_abs,
                "mean_pred_sigma_lnl": float(np.mean(pred_sigma_lnls)),
                "mean_pred_sigma_transformed": float(np.mean(transformed_std)),
                "coverage_1sigma": coverage_1,
                "coverage_2sigma": coverage_2,
                "mean_abs_z": mean_abs_z,
            }
        )

    (outdir / "local_peak_metrics_vs_round.json").write_text(
        json.dumps(series, indent=2), encoding="utf-8"
    )

    x = [item["n_train"] for item in series]
    y_rmse = [item["rmse"] for item in series]
    y_mae = [item["mae"] for item in series]
    y_cov1 = [item["coverage_1sigma"] for item in series]
    y_cov2 = [item["coverage_2sigma"] for item in series]

    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    axes[0].plot(x, y_rmse, marker="o", label="RMSE")
    axes[0].plot(x, y_mae, marker="o", label="MAE")
    axes[0].set_ylabel("Error")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].plot(x, y_cov1, marker="o", label="1σ coverage")
    axes[1].plot(x, y_cov2, marker="o", label="2σ coverage")
    axes[1].set_xlabel("Training points")
    axes[1].set_ylabel("Coverage")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(outdir / "local_peak_metrics_vs_train_points.png", dpi=200)
    plt.close(fig)


def _load_posterior_mode_params(
    analysis_dir: Path,
    *,
    parameters: Sequence[str],
) -> Optional[np.ndarray]:
    posterior_mode_path = analysis_dir / "MCMC" / "posterior_mode.json"
    if not posterior_mode_path.exists():
        return None
    try:
        payload = json.loads(posterior_mode_path.read_text(encoding="utf-8"))
        best = payload.get("best_parameters", {})
        vec = np.array([float(best[p]) for p in parameters], dtype=float)
        if np.all(np.isfinite(vec)):
            return vec
    except Exception:
        return None
    return None


def _jsd_1d_hist(a: np.ndarray, b: np.ndarray, *, bins: int = 60, eps: float = 1e-12) -> float:
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    combined = np.concatenate([a, b])
    if combined.size == 0:
        return float("nan")
    lo = float(np.min(combined))
    hi = float(np.max(combined))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return 0.0

    edges = np.linspace(lo, hi, int(bins) + 1)
    p, _ = np.histogram(a, bins=edges, density=False)
    q, _ = np.histogram(b, bins=edges, density=False)
    p = (p.astype(float) + eps)
    q = (q.astype(float) + eps)
    p /= float(np.sum(p))
    q /= float(np.sum(q))
    m = 0.5 * (p + q)
    return 0.5 * float(np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m)))


def _skl_1d_hist(a: np.ndarray, b: np.ndarray, *, bins: int = 60, eps: float = 1e-12) -> float:
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    combined = np.concatenate([a, b])
    if combined.size == 0:
        return float("nan")
    lo = float(np.min(combined))
    hi = float(np.max(combined))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return 0.0

    edges = np.linspace(lo, hi, int(bins) + 1)
    p, _ = np.histogram(a, bins=edges, density=False)
    q, _ = np.histogram(b, bins=edges, density=False)
    p = (p.astype(float) + eps)
    q = (q.astype(float) + eps)
    p /= float(np.sum(p))
    q /= float(np.sum(q))
    return float(np.sum(p * np.log(p / q)) + np.sum(q * np.log(q / p)))


def _sigma_lnl_from_transformed(
    *,
    transformed_mu: np.ndarray,
    transformed_std: np.ndarray,
    scaler,
) -> np.ndarray:
    transformed_mu = np.asarray(transformed_mu, dtype=float).reshape(-1)
    transformed_std = np.asarray(transformed_std, dtype=float).reshape(-1)
    if not getattr(scaler, "soft_clipping", False):
        deriv = float(scaler.scale) * np.ones_like(transformed_mu)
        return np.abs(deriv) * transformed_std

    clip_factor = float(scaler.clip_factor)
    scaled = transformed_mu / clip_factor
    scaled = np.clip(scaled, -0.999, 0.999)
    dstandardized = np.ones_like(transformed_mu)
    neg_mask = transformed_mu <= 0.0
    dstandardized[neg_mask] = 1.0 / (1.0 - scaled[neg_mask] ** 2)
    deriv = float(scaler.scale) * dstandardized
    return np.abs(deriv) * transformed_std

