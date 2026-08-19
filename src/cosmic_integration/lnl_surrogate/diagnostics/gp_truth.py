from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from ..lnl_surrogate import BOUNDS, PARAMETERS, LnLSurrogate


def _sigma_lnl_from_transformed(
    *,
    transformed_mu: np.ndarray,
    transformed_std: np.ndarray,
    scaler,
) -> np.ndarray:
    """
    Approximate predictive sigma in LnL space via linear error propagation:
      sigma_lnl ~= |d inverse_transform / d transformed| * sigma_transformed
    """

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


def gp_accuracy_vs_distance(
    *,
    outdir: str | Path,
    lnl_computer,
    models_dir: str | Path,
    round_idx: int,
    truths: Dict[str, float],
    seed: int = 0,
    n_per_fraction: int = 200,
    fractions: Sequence[float] = (0.01, 0.02, 0.05, 0.1, 0.2, 0.4),
    uncertainty_beta: float = 0.0,
    parameters: Sequence[str] = PARAMETERS,
    bounds: np.ndarray = BOUNDS,
) -> dict:
    """
    Diagnose GP accuracy as a function of distance from the truth.

    Writes:
      - metrics_vs_distance.json
      - evaluation_points.npz (points, true_lnls, pred_lnls, pred_sigma_lnls, dist)
      - plots/gp_accuracy_vs_distance.png
    """

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "plots").mkdir(parents=True, exist_ok=True)

    truth_vec = np.array([float(truths[p]) for p in parameters], dtype=float)
    bounds_arr = np.asarray(bounds, dtype=float)
    widths = (bounds_arr[1] - bounds_arr[0]).astype(float)
    widths = np.where(widths > 0.0, widths, 1.0)

    surrogate = LnLSurrogate.load(
        str(Path(models_dir)),
        uncertainty_beta=float(uncertainty_beta),
        round_idx=int(round_idx),
    )

    rng = np.random.default_rng(int(seed) + 12345 + 11 * int(round_idx))
    all_points: list[np.ndarray] = []
    all_frac: list[float] = []

    for frac in fractions:
        frac = float(frac)
        if frac <= 0.0:
            continue
        scale = frac * widths
        pts = truth_vec + rng.normal(size=(int(n_per_fraction), len(parameters))) * scale
        pts = np.clip(pts, bounds_arr[0], bounds_arr[1])
        pts[0, :] = truth_vec
        all_points.append(pts)
        all_frac.extend([frac] * int(pts.shape[0]))

    points = np.vstack(all_points) if all_points else np.zeros((0, len(parameters)))
    fractions_per_point = np.asarray(all_frac, dtype=float).reshape(-1)

    if points.size == 0:
        payload = {
            "round": int(round_idx),
            "n_points": 0,
            "fractions": list(fractions),
            "uncertainty_beta": float(uncertainty_beta),
        }
        (outdir / "metrics_vs_distance.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    dist = np.sqrt(np.sum(((points - truth_vec) / widths) ** 2, axis=1)).astype(float)

    true_lnls = np.array(
        [lnl_computer(*row) for row in tqdm(points, desc="True LnL vs distance", unit="pt")],
        dtype=float,
    )

    mu_neg, var_neg = surrogate.gp_model.predict_f(points)
    mu_neg = np.asarray(mu_neg, dtype=float).reshape(-1)
    var_neg = np.asarray(var_neg, dtype=float).reshape(-1)
    std_neg = np.sqrt(np.maximum(var_neg, 0.0))

    mu_transformed = -mu_neg
    std_transformed = std_neg
    pred_lnls = np.asarray(surrogate.scaler.inverse_transform(mu_transformed), dtype=float)
    pred_sigma_lnls = _sigma_lnl_from_transformed(
        transformed_mu=mu_transformed,
        transformed_std=std_transformed,
        scaler=surrogate.scaler,
    )

    err = pred_lnls - true_lnls
    eps = 1e-12
    z = err / np.maximum(pred_sigma_lnls, eps)

    np.savez_compressed(
        outdir / "evaluation_points.npz",
        points=points,
        truth=truth_vec,
        fractions=fractions_per_point,
        dist=dist,
        true_lnls=true_lnls,
        pred_lnls=pred_lnls,
        pred_sigma_lnls=pred_sigma_lnls,
    )

    series: list[dict] = []
    for frac in sorted(set(float(f) for f in fractions_per_point)):
        mask = np.isclose(fractions_per_point, frac)
        if not np.any(mask):
            continue
        e = err[mask]
        s = pred_sigma_lnls[mask]
        zz = z[mask]
        series.append(
            {
                "fraction": float(frac),
                "n_points": int(np.sum(mask)),
                "mean_dist": float(np.mean(dist[mask])),
                "bias": float(np.mean(e)),
                "scatter": float(np.std(e)),
                "rmse": float(np.sqrt(np.mean(e**2))),
                "mean_pred_sigma_lnl": float(np.mean(s)),
                "coverage_1sigma": float(np.mean(np.abs(e) <= np.maximum(s, eps))),
                "coverage_2sigma": float(np.mean(np.abs(e) <= 2.0 * np.maximum(s, eps))),
                "mean_abs_z": float(np.mean(np.abs(zz))),
            }
        )

    payload = {
        "round": int(round_idx),
        "uncertainty_beta": float(uncertainty_beta),
        "n_points": int(points.shape[0]),
        "n_per_fraction": int(n_per_fraction),
        "metrics": series,
    }
    (outdir / "metrics_vs_distance.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    xvals = [item["mean_dist"] for item in series]
    rmse = [item["rmse"] for item in series]
    bias = [item["bias"] for item in series]
    scatter = [item["scatter"] for item in series]
    cov1 = [item["coverage_1sigma"] for item in series]
    cov2 = [item["coverage_2sigma"] for item in series]

    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    axes[0].plot(xvals, rmse, marker="o", label="RMSE")
    axes[0].plot(xvals, np.abs(bias), marker="o", label="|bias|")
    axes[0].plot(xvals, scatter, marker="o", label="scatter")
    axes[0].set_ylabel("Error (LnL)")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].plot(xvals, cov1, marker="o", label="1σ coverage")
    axes[1].plot(xvals, cov2, marker="o", label="2σ coverage")
    axes[1].set_xlabel("Distance from truth (normalized)")
    axes[1].set_ylabel("Coverage")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(outdir / "plots" / "gp_accuracy_vs_distance.png", dpi=200)
    plt.close(fig)

    return payload
