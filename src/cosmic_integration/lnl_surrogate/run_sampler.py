"""Sample the GP surrogate posterior with NUTS.

Replaces the previous bilby/emcee stack. The surrogate is a GPJax posterior, so
its log-density is differentiable and NUTS can use gradients directly; that
removes the affine-invariance limitation of emcee (which cannot follow curved
degeneracies however many steps it takes) and gives trustworthy R-hat/ESS.

Public API is unchanged: :func:`sample_lnl_surrogate` returns an object with a
``.posterior`` DataFrame and writes ``posterior_samples.npy``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Sequence

import corner
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .lnl_surrogate import BOUNDS, PARAMETERS
from .nuts_sampler import run_nuts

logger = logging.getLogger(__name__)


class PriorBounds:
    """Uniform prior over :data:`BOUNDS`, with the small bilby-like API the
    CLI tools relied on (``.sample()``)."""

    def __init__(self, parameters: Sequence[str] = PARAMETERS):
        self.parameters = list(parameters)
        self.low = np.asarray(BOUNDS[0], dtype=float)
        self.high = np.asarray(BOUNDS[1], dtype=float)

    def sample(self, size: int | None = None, rng=None) -> dict:
        rng = rng or np.random.default_rng()
        shape = (len(self.parameters),) if size is None else (size, len(self.parameters))
        draw = rng.uniform(self.low, self.high, size=shape)
        if size is None:
            return {p: float(draw[i]) for i, p in enumerate(self.parameters)}
        return {p: draw[:, i] for i, p in enumerate(self.parameters)}

    def __repr__(self) -> str:
        rng = ", ".join(
            f"{p}=U({lo:g},{hi:g})" for p, lo, hi in zip(self.parameters, self.low, self.high)
        )
        return f"PriorBounds({rng})"


def get_prior(parameters: List[str] = PARAMETERS, truth: np.ndarray = None) -> PriorBounds:
    """Kept for the CLI tools; the surrogate priors are just the box bounds."""
    return PriorBounds(parameters)


@dataclass
class SurrogateResult:
    """Minimal stand-in for a bilby Result: what downstream code actually uses."""

    posterior: pd.DataFrame
    summary: dict = field(default_factory=dict)

    @property
    def converged(self) -> bool:
        return bool(self.summary.get("converged", False))


def sample_lnl_surrogate(
    lnl_model_path: str,
    outdir: str,
    verbose: bool = False,
    truths: Optional[dict] = None,
    mcmc_kwargs: Optional[dict] = None,
    *,
    uncertainty_beta: float | Sequence[float] = 0.0,
    model_round_idx: int | None = None,
) -> Optional[SurrogateResult]:
    """Sample the surrogate posterior with NUTS.

    ``mcmc_kwargs`` accepts ``num_warmup``, ``num_samples``, ``num_chains``,
    ``seed`` and ``target``. Legacy emcee keys (``nwalkers``, ``iterations``)
    are translated so existing callers keep working.

    ``uncertainty_beta`` is accepted for backwards compatibility. A non-zero
    value selects the ``"marginal"`` target, which folds in the GP uncertainty
    exactly (``mu + sigma^2/2``) rather than by the old ``mu - beta*sigma``
    heuristic; see :mod:`.nuts_sampler`.
    """
    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)

    kwargs = dict(mcmc_kwargs or {})
    # Translate the legacy emcee configuration.
    iterations = kwargs.pop("iterations", None)
    kwargs.pop("nwalkers", None)
    kwargs.pop("nburn", None)
    num_samples = int(kwargs.pop("num_samples", iterations or 2000))
    num_warmup = int(kwargs.pop("num_warmup", max(500, num_samples // 4)))
    num_chains = int(kwargs.pop("num_chains", 2))
    seed = int(kwargs.pop("seed", 0))

    betas = uncertainty_beta if isinstance(uncertainty_beta, (list, tuple, np.ndarray)) else [uncertainty_beta]
    target = kwargs.pop("target", "marginal" if any(float(b) for b in betas) else "mean")
    if kwargs:
        logger.warning("Ignoring unknown sampler kwargs: %s", sorted(kwargs))

    logger.info(
        "Sampling surrogate at %s with NUTS (target=%s, chains=%d, samples=%d)",
        lnl_model_path, target, num_chains, num_samples,
    )

    try:
        summary = run_nuts(
            lnl_model_path=lnl_model_path,
            outdir=outdir_path,
            target=target,
            round_idx=model_round_idx,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            seed=seed,
            truths=truths,
        )
    except Exception:
        logger.exception("NUTS sampling failed for %s", lnl_model_path)
        return None

    samples = np.load(outdir_path / "posterior_samples.npy")
    posterior = pd.DataFrame(samples, columns=list(PARAMETERS))
    result = SurrogateResult(posterior=posterior, summary=summary)

    _write_artefacts(result, outdir_path, truths)
    return result


def _write_artefacts(result: SurrogateResult, outdir: Path, truths: Optional[dict]) -> None:
    """Corner plot and a summary JSON, under the historical filenames."""
    samples = result.posterior[list(PARAMETERS)].to_numpy(dtype=float)
    corner_truths = [truths.get(p) for p in PARAMETERS] if truths else None

    try:
        fig = corner.corner(
            samples,
            labels=[p.replace("_", " ") for p in PARAMETERS],
            show_titles=True,
            truths=corner_truths,
            levels=(0.68, 0.95),
            plot_datapoints=False,
            fill_contours=True,
        )
        if not result.converged:
            fig.suptitle("UNCONVERGED - see nuts_summary.json", color="red", y=1.02)
        plots_dir = outdir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(outdir / "lnl_surrogate_corner.png", dpi=200, bbox_inches="tight")
        fig.savefig(plots_dir / "corner_truth.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
    except Exception:
        logger.exception("Failed to write corner plot in %s", outdir)

    payload: dict[str, Any] = dict(result.summary)
    payload["quantiles"] = {
        p: {
            "median": float(np.median(samples[:, i])),
            "q16": float(np.percentile(samples[:, i], 16)),
            "q84": float(np.percentile(samples[:, i], 84)),
        }
        for i, p in enumerate(PARAMETERS)
    }
    (outdir / "lnl_surrogate_result.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
