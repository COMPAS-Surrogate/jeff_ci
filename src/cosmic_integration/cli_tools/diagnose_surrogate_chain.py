import json
from pathlib import Path
from typing import Iterable, Optional, Sequence

import click
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tqdm.auto import tqdm

from cosmic_integration.lnl_computer import LnLComputer
from cosmic_integration.lnl_surrogate.lnl_surrogate import PARAMETERS, LnLSurrogate


def _metrics(true_lnls: np.ndarray, pred_lnls: np.ndarray) -> dict:
    residuals = pred_lnls - true_lnls
    rmse = float(np.sqrt(np.mean(residuals**2)))
    mae = float(np.mean(np.abs(residuals)))
    max_abs_err = float(np.max(np.abs(residuals)))
    r2 = 1.0 - float(
        np.sum(residuals**2) / (np.sum((true_lnls - np.mean(true_lnls)) ** 2) + 1e-12)
    )
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "max_abs_err": max_abs_err,
    }


def _compute_iteration_index(walkers: np.ndarray) -> np.ndarray:
    """Assign a per-walker iteration number in the order rows appear."""
    walkers = np.asarray(walkers, dtype=int).reshape(-1)
    counters: dict[int, int] = {}
    it = np.empty_like(walkers, dtype=int)
    for idx, w in enumerate(walkers):
        current = counters.get(int(w), 0)
        it[idx] = current
        counters[int(w)] = current + 1
    return it


def _read_chain(chain_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    data = np.genfromtxt(
        chain_path,
        names=True,
        delimiter="\t",
        dtype=None,
        encoding=None,
        autostrip=True,
    )
    if data.size == 0:
        raise ValueError(f"No rows found in chain file: {chain_path}")

    if "walker" not in data.dtype.names:
        raise ValueError(f"Chain file is missing required 'walker' column: {chain_path}")

    walkers = np.asarray(data["walker"], dtype=int).reshape(-1)
    points = np.vstack([np.asarray(data[name], dtype=float) for name in PARAMETERS]).T

    if "log_l" in data.dtype.names:
        surrogate_logl = np.asarray(data["log_l"], dtype=float).reshape(-1)
    else:
        surrogate_logl = np.full(points.shape[0], np.nan, dtype=float)

    unique_walkers = np.unique(walkers)
    n_walkers = int(unique_walkers.size)
    it = _compute_iteration_index(walkers)
    n_steps = int(np.max(it)) + 1 if it.size else 0
    return points, walkers, surrogate_logl, n_walkers, n_steps


def _parse_float_list(values: str) -> list[float]:
    tokens = []
    for part in str(values).replace(",", " ").split():
        if part.strip():
            tokens.append(float(part))
    return tokens


def _infer_paths_from_analysis_dir(analysis_dir: Path) -> tuple[str, str, str, str]:
    """
    Infer (observation_file, compas_h5, model_dir, chain_path) from a simulation-study analysis directory.

    Expected layout:
      analysis_dir/
        gp_model/models/round_*/
        MCMC/emcee_lnl_surrogate/chain.dat
      analysis_dir/../mock_observation.h5
      analysis_dir/../../../h5out_5M.h5
    """

    analysis_dir = Path(analysis_dir)
    if not analysis_dir.exists():
        raise FileNotFoundError(f"analysis_dir does not exist: {analysis_dir}")

    observation_file = analysis_dir.parent / "mock_observation.h5"
    model_dir = analysis_dir / "gp_model" / "models"

    chain_candidates = [
        analysis_dir / "MCMC" / "emcee_lnl_surrogate" / "chain.dat",
    ]
    if not chain_candidates[0].exists():
        # Fallback: pick the first emcee_* chain we can find.
        chain_candidates = list(analysis_dir.glob("MCMC/emcee_*/chain.dat"))

    if not chain_candidates:
        raise FileNotFoundError(f"Could not find chain.dat under {analysis_dir}/MCMC/")
    chain_path = chain_candidates[0]

    # Walk up to the simulation_study directory (outputs/noise-or-no_noise/dur_*/analysis -> simulation_study).
    compas_h5 = analysis_dir.parents[3] / "h5out_5M.h5"

    missing = []
    for path in [observation_file, compas_h5, model_dir, chain_path]:
        if not Path(path).exists():
            missing.append(str(path))
    if missing:
        raise FileNotFoundError("Inferred paths do not exist:\n- " + "\n- ".join(missing))

    return str(observation_file), str(compas_h5), str(model_dir), str(chain_path)


def run_chain_diagnostics(
    *,
    observation_file: str,
    compas_h5: str,
    model_dir: str,
    chain_path: str,
    outdir: str,
    max_points: int = 2000,
    seed: int = 0,
    burnin: int = 0,
    burnin_fraction: Optional[float] = None,
    tail_fraction: float = 1.0,
    uncertainty_betas: Sequence[float] = (0.0, 1.0),
    top_fractions: Sequence[float] = (0.05, 0.01, 0.001),
) -> dict:
    """Compare surrogate vs true LnL on an existing MCMC chain (post-burn-in)."""

    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    points, walkers, surrogate_logl_chain, n_walkers, n_steps = _read_chain(Path(chain_path))
    n_total = int(points.shape[0])

    if max_points <= 0:
        raise ValueError("max_points must be positive.")
    if tail_fraction <= 0.0 or tail_fraction > 1.0:
        raise ValueError("tail_fraction must be in (0, 1].")

    if burnin_fraction is not None:
        burnin_fraction = float(burnin_fraction)
        if burnin_fraction < 0.0 or burnin_fraction >= 1.0:
            raise ValueError("burnin_fraction must be in [0, 1).")
        burnin = int(np.floor(burnin_fraction * n_steps))

    it = _compute_iteration_index(walkers)
    keep = it >= int(burnin)
    if tail_fraction < 1.0 and n_steps > 0:
        min_iter = int(np.floor((1.0 - float(tail_fraction)) * n_steps))
        keep = keep & (it >= min_iter)

    points = points[keep]
    surrogate_logl_chain = surrogate_logl_chain[keep]
    walkers = walkers[keep]
    it = it[keep]

    n_after = int(points.shape[0])
    if n_after == 0:
        raise ValueError("No chain points remain after applying burn-in/tail filters.")

    rng = np.random.default_rng(int(seed))
    if n_after > max_points:
        idx = rng.choice(n_after, size=int(max_points), replace=False)
        points = points[idx]
        surrogate_logl_chain = surrogate_logl_chain[idx]
        walkers = walkers[idx]
        it = it[idx]

    lnl_computer = LnLComputer.load(
        observation_file=observation_file,
        compas_h5=compas_h5,
    )
    surrogate = LnLSurrogate.load(model_dir, uncertainty_beta=0.0)

    x = tf.convert_to_tensor(points, dtype=tf.float64)
    mu, var = surrogate.gp_model.predict_f(x)
    mu = np.asarray(mu.numpy().reshape(-1), dtype=float)
    var = np.asarray(var.numpy().reshape(-1), dtype=float)
    std = np.sqrt(np.maximum(var, 0.0))

    true_lnls = np.array(
        [lnl_computer(*row) for row in tqdm(points, desc="True LnL", unit="pt")],
        dtype=float,
    )

    warnings: list[str] = []
    top_fractions = [float(f) for f in top_fractions]
    for f in top_fractions:
        if f <= 0.0 or f >= 1.0:
            raise ValueError("top_fractions entries must be in (0, 1).")
        if int(np.ceil(f * len(true_lnls))) < 10:
            warnings.append(
                f"top fraction {f:g} yields <10 points for n_evaluated={len(true_lnls)}; "
                "increase --max-points if you want stable estimates."
            )

    per_beta: dict[str, dict] = {}
    for beta in uncertainty_betas:
        beta = float(beta)
        mu_eff = mu + beta * std if beta else mu
        pred_lnls = np.asarray(surrogate.scaler.inverse_transform(-mu_eff), dtype=float)

        payload = {
            "uncertainty_beta": beta,
            "all": _metrics(true_lnls, pred_lnls),
        }

        for frac in top_fractions:
            threshold = float(np.quantile(true_lnls, max(0.0, 1.0 - float(frac))))
            mask = true_lnls >= threshold
            key = f"top{100*frac:g}%"
            payload[f"{key}_true_lnl_threshold"] = threshold
            payload[f"{key}_n"] = int(np.sum(mask))
            if int(np.sum(mask)) >= 2:
                payload[key] = _metrics(true_lnls[mask], pred_lnls[mask])

        per_beta[f"{beta:g}"] = payload

        beta_dir = out_path / f"beta_{str(beta).replace('-', 'm').replace('.', 'p')}"
        beta_dir.mkdir(parents=True, exist_ok=True)
        (beta_dir / "surrogate_chain_metrics.json").write_text(
            json.dumps(
                {
                    "n_chain_total": n_total,
                    "n_chain_after_filter": n_after,
                    "n_evaluated": int(points.shape[0]),
                    "n_walkers": int(n_walkers),
                    "n_steps": int(n_steps),
                    "burnin": int(burnin),
                    "burnin_fraction": float(burnin_fraction) if burnin_fraction is not None else None,
                    "tail_fraction": float(tail_fraction),
                    "warnings": warnings,
                    **payload,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(true_lnls, pred_lnls, s=10, alpha=0.6)
        lims = [
            float(min(np.min(true_lnls), np.min(pred_lnls))),
            float(max(np.max(true_lnls), np.max(pred_lnls))),
        ]
        ax.plot(lims, lims, "k--", linewidth=1)
        ax.set_xlabel("True LnL")
        ax.set_ylabel("Surrogate LnL")
        ax.set_title(f"Surrogate vs true on chain (beta={beta:g})")
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        fig.savefig(beta_dir / "surrogate_chain_scatter.png", dpi=180)
        plt.close(fig)

        residuals = pred_lnls - true_lnls
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.hist(residuals, bins=60, alpha=0.85)
        ax.axvline(0.0, color="k", linestyle="--", linewidth=1)
        ax.set_xlabel("Residual (pred - true)")
        ax.set_ylabel("Count")
        ax.set_title(f"Surrogate residuals on chain (beta={beta:g})")
        fig.tight_layout()
        fig.savefig(beta_dir / "surrogate_chain_residuals.png", dpi=180)
        plt.close(fig)

    summary = {
        "n_chain_total": n_total,
        "n_chain_after_filter": n_after,
        "n_evaluated": int(points.shape[0]),
        "n_walkers": int(n_walkers),
        "n_steps": int(n_steps),
        "burnin": int(burnin),
        "burnin_fraction": float(burnin_fraction) if burnin_fraction is not None else None,
        "tail_fraction": float(tail_fraction),
        "top_fractions": [float(f) for f in top_fractions],
        "warnings": warnings,
        "betas": {k: v for k, v in per_beta.items()},
    }
    (out_path / "surrogate_chain_metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


@click.command()
@click.argument("paths", nargs=-1, type=click.Path(exists=True))
@click.option("--outdir", default="surrogate_chain_diagnostics", show_default=True)
@click.option("--max-points", default=2000, show_default=True, type=int)
@click.option("--seed", default=0, show_default=True, type=int)
@click.option("--burnin", default=0, show_default=True, type=int, help="Burn-in iterations to discard per walker.")
@click.option(
    "--burnin-fraction",
    default=None,
    type=float,
    help="Burn-in as a fraction of iterations (overrides --burnin).",
)
@click.option(
    "--tail-fraction",
    default=1.0,
    show_default=True,
    type=float,
    help="Keep only the last fraction of iterations (after burn-in).",
)
@click.option(
    "--observation-file",
    default=None,
    type=click.Path(exists=True),
    help="Optional override when using the 1-arg analysis_dir form.",
)
@click.option(
    "--compas-h5",
    default=None,
    type=click.Path(exists=True),
    help="Optional override when using the 1-arg analysis_dir form.",
)
@click.option(
    "--model-dir",
    "model_dir_opt",
    default=None,
    type=click.Path(exists=True),
    help="Optional override when using the 1-arg analysis_dir form.",
)
@click.option(
    "--chain-path",
    default=None,
    type=click.Path(exists=True),
    help="Optional override when using the 1-arg analysis_dir form.",
)
@click.option(
    "--uncertainty-beta",
    default=None,
    type=float,
    help="Deprecated: use --uncertainty-betas instead.",
)
@click.option(
    "--uncertainty-betas",
    default="0,1",
    show_default=True,
    type=str,
    help="Comma/space-separated list of betas to evaluate (runs all in one go).",
)
@click.option(
    "--top-fractions",
    default="0.05,0.01,0.001",
    show_default=True,
    type=str,
    help="Comma/space-separated fractions for 'top' metrics (e.g. 0.001 = top0.1%).",
)
def main(
    paths: Sequence[str],
    outdir: str,
    max_points: int,
    seed: int,
    burnin: int,
    burnin_fraction: Optional[float],
    tail_fraction: float,
    observation_file: Optional[str],
    compas_h5: Optional[str],
    model_dir_opt: Optional[str],
    chain_path: Optional[str],
    uncertainty_beta: Optional[float],
    uncertainty_betas: str,
    top_fractions: str,
) -> None:
    """
    Compare surrogate vs true LnL on an existing MCMC chain.

    Usage:
      1) Simple (simulation study): pass just the analysis directory
         diagnose_surrogate_chain.py path/to/.../analysis --burnin-fraction 0.3

      2) Explicit (backwards compatible): pass 4 paths
         diagnose_surrogate_chain.py OBS.h5 COMPAS.h5 MODEL_DIR CHAIN.dat
    """

    if len(paths) == 1:
        analysis_dir = Path(paths[0])
        inferred_obs, inferred_compas, inferred_model, inferred_chain = _infer_paths_from_analysis_dir(analysis_dir)
        observation_file = observation_file or inferred_obs
        compas_h5 = compas_h5 or inferred_compas
        model_dir = model_dir_opt or inferred_model
        chain_path = chain_path or inferred_chain
        if outdir == "surrogate_chain_diagnostics":
            outdir = str(analysis_dir / "surrogate_chain_diagnostics")
    elif len(paths) == 4:
        observation_file, compas_h5, model_dir, chain_path = paths  # type: ignore[misc]
    else:
        raise click.BadParameter(
            "Provide either 1 path (analysis_dir) or 4 paths "
            "(observation_file compas_h5 model_dir chain_path)."
        )

    assert observation_file is not None and compas_h5 is not None and chain_path is not None
    assert model_dir_opt is None or isinstance(model_dir_opt, str)
    assert isinstance(model_dir, str)

    betas: Sequence[float]
    if uncertainty_beta is not None:
        betas = [float(uncertainty_beta)]
    else:
        betas = _parse_float_list(uncertainty_betas)

    tops = _parse_float_list(top_fractions)
    run_chain_diagnostics(
        observation_file=observation_file,
        compas_h5=compas_h5,
        model_dir=model_dir,
        chain_path=chain_path,
        outdir=outdir,
        max_points=int(max_points),
        seed=int(seed),
        burnin=int(burnin),
        burnin_fraction=burnin_fraction,
        tail_fraction=float(tail_fraction),
        uncertainty_betas=betas,
        top_fractions=tops,
    )


if __name__ == "__main__":
    main()
