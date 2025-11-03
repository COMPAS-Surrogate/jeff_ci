import json
import logging
import os
from pathlib import Path
from typing import Any, List, Optional

import bilby
import corner
import matplotlib.pyplot as plt
import numpy as np
from bilby.core.prior import DeltaFunction, PriorDict, Uniform

from .lnl_surrogate import BOUNDS, PARAMETERS, LnLSurrogate


def _to_native(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value



def get_prior(parameters:List[str]=PARAMETERS, truth:np.ndarray=None) -> PriorDict:
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
                prior[param_name] = Uniform(np.mean(BOUNDS.T[i]))

    return PriorDict(prior)


def sample_lnl_surrogate(
    lnl_model_path: str,
    outdir: str,
    verbose: bool = False,
    truths: Optional[dict] = None,
    mcmc_kwargs: Optional[dict] = None,
):
    bilby_logger = logging.getLogger("bilby")

    bilby_logger.setLevel(logging.ERROR)
    if verbose:
        bilby_logger.setLevel(logging.INFO)

    lnl_surrogate = LnLSurrogate.load(lnl_model_path)
    prior = get_prior()

    logger = logging.getLogger(__name__)
    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)

    logger.info("Sampling from LnLSurrogate model at %s", lnl_model_path)
    logger.info("Using prior: %s", prior)

    mcmc_kwargs = dict(mcmc_kwargs or {})
    mcmc_kwargs["nwalkers"] = mcmc_kwargs.get("nwalkers", 32)
    mcmc_kwargs["iterations"] = mcmc_kwargs.get("iterations", 2000)

    sampler_kwargs = dict(
        priors=prior,
        sampler="emcee",
        injection_parameters=truths or {},
        outdir=str(outdir_path),
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

    _post_process(result, outdir_path, truths)

    return result


def _post_process(result, outdir: Path, truths: Optional[dict]) -> None:
    """Create convenience artefacts (corner plots, summary JSON)."""

    if result is None or result.posterior is None:
        return

    plots_dir = outdir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    posterior_df = result.posterior[PARAMETERS]
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
    summary_path = outdir / "lnl_surrogate_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
