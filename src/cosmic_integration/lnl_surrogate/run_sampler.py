import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, List, Optional, Sequence

import bilby
import corner
import matplotlib.pyplot as plt
import numpy as np
from bilby.core.prior import DeltaFunction, PriorDict, Uniform
from matplotlib.lines import Line2D

from .lnl_surrogate import BOUNDS, PARAMETERS, LnLSurrogate


def _to_native(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _beta_tag(beta: float) -> str:
    tag = f"{float(beta):g}"
    tag = tag.replace("-", "m").replace(".", "p")
    return f"beta_{tag}"


def _normalize_uncertainty_beta(uncertainty_beta: float | Sequence[float]) -> List[float]:
    if isinstance(uncertainty_beta, (list, tuple, np.ndarray)):
        return [float(b) for b in uncertainty_beta]
    return [float(uncertainty_beta)]


def _choose_primary_beta(betas: Sequence[float]) -> float:
    for beta in betas:
        if abs(float(beta) - 1.0) < 1e-12:
            return float(beta)
    return float(betas[-1])


def _jsd_1d_from_samples(
    samples_a: np.ndarray,
    samples_b: np.ndarray,
    *,
    bins: int = 50,
) -> float:
    samples_a = np.asarray(samples_a, dtype=float).reshape(-1)
    samples_b = np.asarray(samples_b, dtype=float).reshape(-1)

    combined = np.concatenate([samples_a, samples_b])
    if combined.size == 0:
        return float("nan")

    lo = float(np.min(combined))
    hi = float(np.max(combined))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return 0.0

    edges = np.linspace(lo, hi, int(bins) + 1)
    p, _ = np.histogram(samples_a, bins=edges, density=False)
    q, _ = np.histogram(samples_b, bins=edges, density=False)

    p = p.astype(float)
    q = q.astype(float)
    p_sum = float(np.sum(p))
    q_sum = float(np.sum(q))
    if p_sum <= 0.0 or q_sum <= 0.0:
        return float("nan")

    p /= p_sum
    q /= q_sum
    m = 0.5 * (p + q)

    def _kl_div(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))

    return 0.5 * (_kl_div(p, m) + _kl_div(q, m))


def _overlay_corner_with_jsd(
    *,
    outdir: Path,
    samples_by_beta: dict[float, np.ndarray],
    truths: Optional[dict],
    jsd_by_param: dict[str, float] | None,
) -> None:
    plots_dir = outdir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    betas = list(samples_by_beta.keys())
    if len(betas) < 2:
        return

    labels = [param.replace("_", " ") for param in PARAMETERS]
    if truths:
        corner_truths = [truths.get(param) for param in PARAMETERS]
    else:
        corner_truths = None

    colors = ["C0", "C1", "C2", "C3"]
    fig = None
    for idx, beta in enumerate(betas):
        fig = corner.corner(
            samples_by_beta[beta],
            fig=fig,
            labels=labels,
            show_titles=True,
            truths=corner_truths,
            color=colors[idx % len(colors)],
            plot_density=False,
            plot_datapoints=False,
            fill_contours=False,
            levels=(0.68, 0.95),
        )

    handles = [
        Line2D([0], [0], color=colors[idx % len(colors)], lw=2, label=f"uncertainty_beta={beta:g}")
        for idx, beta in enumerate(betas)
    ]
    fig.legend(handles=handles, loc="upper right", frameon=False)

    if jsd_by_param:
        finite = [v for v in jsd_by_param.values() if np.isfinite(v)]
        jsd_mean = float(np.mean(finite)) if finite else float("nan")
        jsd_str = ", ".join(f"{k}={jsd_by_param[k]:.3g}" for k in PARAMETERS)
        fig.suptitle(f"JSD (mean={jsd_mean:.3g}): {jsd_str}", y=1.02)

    fig.savefig(outdir / "lnl_surrogate_corner.png", dpi=200, bbox_inches="tight")
    fig.savefig(plots_dir / "corner_overlay_uncertainty_beta.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _copy_if_exists(src: Path, dst: Path) -> None:
    try:
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    except Exception:
        return


def get_prior(parameters: List[str] = PARAMETERS, truth: np.ndarray = None) -> PriorDict:
    """
    Get the prior distribution for the parameters.
    """
    prior = {}

    for i, param_name in enumerate(PARAMETERS):

        if param_name in parameters:
            prior[param_name] = Uniform(*BOUNDS.T[i])
        else:
            if truth is not None:
                prior[param_name] = DeltaFunction(truth[i])
            else:
                prior[param_name] = DeltaFunction(float(np.mean(BOUNDS.T[i])))

    return PriorDict(prior)


def sample_lnl_surrogate(
    lnl_model_path: str,
    outdir: str,
    verbose: bool = False,
    truths: Optional[dict] = None,
    mcmc_kwargs: Optional[dict] = None,
    *,
    uncertainty_beta: float | Sequence[float] = (0.0, 1.0),
    model_round_idx: int | None = None,
):
    bilby_logger = logging.getLogger("bilby")

    bilby_logger.setLevel(logging.ERROR)
    if verbose:
        bilby_logger.setLevel(logging.INFO)

    prior = get_prior()

    logger = logging.getLogger(__name__)
    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)

    betas = _normalize_uncertainty_beta(uncertainty_beta)
    logger.info("Sampling from LnLSurrogate model at %s", lnl_model_path)
    logger.info("uncertainty_beta values: %s", betas)
    logger.info("Using prior: %s", prior)

    mcmc_kwargs = dict(mcmc_kwargs or {})
    mcmc_kwargs["nwalkers"] = mcmc_kwargs.get("nwalkers", 32)
    mcmc_kwargs["iterations"] = mcmc_kwargs.get("iterations", 2000)

    def _sample_once(*, beta: float, run_outdir: Path):
        lnl_surrogate = LnLSurrogate.load(
            lnl_model_path,
            uncertainty_beta=beta,
            round_idx=model_round_idx,
        )
        sampler_kwargs = dict(
            priors=prior,
            sampler="emcee",
            injection_parameters=truths or {},
            outdir=str(run_outdir),
            clean=True,
            verbose=verbose,
            plot=True,
            **mcmc_kwargs,
        )
        result = bilby.run_sampler(
            likelihood=lnl_surrogate,
            label="lnl_surrogate",
            **sampler_kwargs,
        )
        _post_process(result, run_outdir, truths)
        return result

    if len(betas) == 1:
        return _sample_once(beta=betas[0], run_outdir=outdir_path)

    results_by_beta: dict[float, object] = {}
    samples_by_beta: dict[float, np.ndarray] = {}
    for beta in betas:
        run_outdir = outdir_path / _beta_tag(beta)
        run_outdir.mkdir(parents=True, exist_ok=True)
        results_by_beta[float(beta)] = _sample_once(beta=float(beta), run_outdir=run_outdir)

        result = results_by_beta[float(beta)]
        if result is not None and getattr(result, "posterior", None) is not None:
            try:
                samples_by_beta[float(beta)] = result.posterior[PARAMETERS].to_numpy(dtype=float)
            except Exception:
                pass

    jsd_by_param: dict[str, float] | None = None
    if len(samples_by_beta) >= 2:
        beta_a = 0.0 if any(abs(float(b) - 0.0) < 1e-12 for b in betas) else float(betas[0])
        beta_b = 1.0 if any(abs(float(b) - 1.0) < 1e-12 for b in betas) else float(betas[1])
        if beta_a in samples_by_beta and beta_b in samples_by_beta:
            a = samples_by_beta[beta_a]
            b = samples_by_beta[beta_b]
            jsd_by_param = {
                param: _jsd_1d_from_samples(a[:, idx], b[:, idx])
                for idx, param in enumerate(PARAMETERS)
            }

    if samples_by_beta:
        _overlay_corner_with_jsd(
            outdir=outdir_path,
            samples_by_beta=samples_by_beta,
            truths=truths,
            jsd_by_param=jsd_by_param,
        )

    comparison = {
        "betas": [float(b) for b in betas],
        "jsd_by_param": jsd_by_param,
        "jsd_mean": (
            float(np.mean([v for v in jsd_by_param.values() if np.isfinite(v)]))
            if jsd_by_param
            else None
        ),
        "runs": {f"{beta:g}": str((outdir_path / _beta_tag(beta)).resolve()) for beta in betas},
    }
    (outdir_path / "uncertainty_beta_comparison.json").write_text(
        json.dumps(comparison, indent=2),
        encoding="utf-8",
    )

    primary_beta = _choose_primary_beta(betas)
    primary_dir = outdir_path / _beta_tag(primary_beta)
    _copy_if_exists(primary_dir / "lnl_surrogate_result.json", outdir_path / "lnl_surrogate_result.json")
    if not (outdir_path / "lnl_surrogate_corner.png").exists():
        _copy_if_exists(primary_dir / "lnl_surrogate_corner.png", outdir_path / "lnl_surrogate_corner.png")

    primary_result = results_by_beta.get(float(primary_beta))
    return primary_result


def _post_process(result, outdir: Path, truths: Optional[dict]) -> None:
    """Create convenience artefacts (corner plots, summary JSON)."""

    if result is None or result.posterior is None:
        return

    def _find_chain_dat(root: Path) -> Optional[Path]:
        candidates = list(root.glob("emcee_*/chain.dat"))
        if candidates:
            return candidates[0]
        candidates = list(root.glob("**/chain.dat"))
        return candidates[0] if candidates else None

    def _compute_iteration_index(walkers: np.ndarray) -> np.ndarray:
        walkers = np.asarray(walkers, dtype=int).reshape(-1)
        counters: dict[int, int] = {}
        it = np.empty_like(walkers, dtype=int)
        for idx, w in enumerate(walkers):
            current = counters.get(int(w), 0)
            it[idx] = current
            counters[int(w)] = current + 1
        return it

    def _ess_from_chain(chain_path: Path) -> dict | None:
        """
        Estimate per-parameter ESS from an emcee `chain.dat` file (Bilby output).

        Returns a dict with keys: n_steps, n_walkers, tau, ess.
        """
        try:
            data = np.genfromtxt(
                chain_path,
                names=True,
                delimiter="\t",
                dtype=None,
                encoding=None,
                autostrip=True,
            )
        except Exception:
            return None

        if getattr(data, "size", 0) == 0:
            return None

        if "walker" not in data.dtype.names:
            return None

        walkers = np.asarray(data["walker"], dtype=int).reshape(-1)
        it = _compute_iteration_index(walkers)
        unique_walkers = np.unique(walkers)
        n_walkers = int(unique_walkers.size)
        n_steps = int(np.max(it)) + 1 if it.size else 0
        if n_steps <= 1 or n_walkers <= 0:
            return None

        points = np.vstack([np.asarray(data[name], dtype=float) for name in PARAMETERS]).T
        ndim = int(points.shape[1])
        walker_index = {int(w): idx for idx, w in enumerate(unique_walkers)}

        chain = np.full((n_steps, n_walkers, ndim), np.nan, dtype=float)
        for row_idx in range(points.shape[0]):
            wi = walker_index.get(int(walkers[row_idx]))
            si = int(it[row_idx])
            if wi is None or si < 0 or si >= n_steps:
                continue
            chain[si, wi, :] = points[row_idx, :]

        if not np.isfinite(chain).all():
            # Best-effort: drop any steps with missing values across walkers.
            ok = np.isfinite(chain).all(axis=(1, 2))
            chain = chain[ok, :, :]
            n_steps = int(chain.shape[0])
            if n_steps <= 1:
                return None

        try:
            import emcee

            tau = emcee.autocorr.integrated_time(chain, quiet=True)
            tau = np.asarray(tau, dtype=float).reshape(-1)
        except Exception:
            return None

        ess = (float(n_steps * n_walkers) / np.maximum(tau, 1.0)).astype(float)
        return {
            "n_steps": int(n_steps),
            "n_walkers": int(n_walkers),
            "tau": {param: float(tau[i]) for i, param in enumerate(PARAMETERS) if i < tau.size},
            "ess": {param: float(ess[i]) for i, param in enumerate(PARAMETERS) if i < ess.size},
        }

    plots_dir = outdir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    posterior_df = result.posterior[PARAMETERS]
    try:
        samples = posterior_df.to_numpy(dtype=float)
        np.save(outdir / "posterior_samples.npy", samples)
    except Exception:
        samples = None
    corner_kwargs = {}
    if truths:
        corner_truths = [truths.get(param) for param in PARAMETERS]
        corner_kwargs["truths"] = corner_truths
    fig = corner.corner(
        posterior_df,
        labels=[param.replace("_", " ") for param in PARAMETERS],
        show_titles=True,
        **corner_kwargs,
    )
    fig.savefig(plots_dir / "corner_truth.png", dpi=200)
    plt.close(fig)

    loglike = result.posterior["log_likelihood"] if "log_likelihood" in result.posterior else None
    best_parameters = None
    if loglike is not None:
        best_idx = loglike.idxmax()
        best_series = posterior_df.loc[best_idx]
        best_parameters = {param: float(best_series[param]) for param in PARAMETERS}

    summary = {
        "n_samples": int(len(posterior_df)),
        "log_evidence": getattr(result, "log_evidence", None),
        "log_evidence_err": getattr(result, "log_evidence_err", None),
        "mcmc_kwargs": {k: _to_native(v) for k, v in (result.sampler_kwargs or {}).items()},
        "max_log_likelihood": float(loglike.max()) if loglike is not None else None,
        "posterior_means": {param: float(posterior_df[param].mean()) for param in PARAMETERS},
        "posterior_stds": {param: float(posterior_df[param].std()) for param in PARAMETERS},
        "best_parameters": best_parameters,
    }
    if samples is not None and getattr(samples, "size", 0) and samples.shape[1] == len(PARAMETERS):
        try:
            cov = np.cov(samples, rowvar=False)
            summary["posterior_covariance"] = cov.tolist()
        except Exception:
            pass

    chain_path = _find_chain_dat(outdir)
    if chain_path is not None:
        ess_payload = _ess_from_chain(chain_path)
        if ess_payload is not None:
            summary["ess"] = ess_payload.get("ess")
            summary["autocorr_time"] = ess_payload.get("tau")
            summary["n_steps"] = ess_payload.get("n_steps")
            summary["n_walkers"] = ess_payload.get("n_walkers")

    sampler = getattr(result, "sampler", None)
    if sampler is not None:
        try:
            acc = getattr(sampler, "acceptance_fraction", None)
            if acc is not None:
                acc = np.asarray(acc, dtype=float).reshape(-1)
                summary["acceptance_fraction_mean"] = float(np.mean(acc))
                summary["acceptance_fraction_min"] = float(np.min(acc))
                summary["acceptance_fraction_max"] = float(np.max(acc))
        except Exception:
            pass

    summary_path = outdir / "lnl_surrogate_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
