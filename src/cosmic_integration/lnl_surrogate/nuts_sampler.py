"""Gradient-based sampling of the GP surrogate likelihood with NUTS.

The surrogate is a GPJax posterior, so its predictive mean and variance are
already differentiable JAX functions. This module samples them with numpyro's
NUTS instead of emcee.

Why not emcee: emcee's stretch move is affine-invariant, so it is untroubled by
*linear* parameter correlations however strong. Observed autocorrelation times
of ~10^4 steps therefore point at something affine invariance cannot fix -- a
curved degeneracy, multimodality, or a rough surrogate surface. NUTS follows the
local geometry and handles the first two; :func:`surface_roughness` diagnoses
the third.

Why not sample the GP uncertainty: it is tempting to draw
``lnL ~ Normal(mu, sigma)`` afresh at each call, but that makes the target
density stochastic -- it breaks Hamiltonian trajectories outright, and leaves
even a random-walk sampler converging to nothing well defined. The
marginalisation has a closed form instead: if ``lnL ~ Normal(mu, sigma^2)`` then

    E[L] = E[exp(lnL)] = exp(mu + sigma^2 / 2)

so ``target="marginal"`` uses ``mu + sigma^2/2``. That is exactly what the
random draws would give in expectation, but deterministic and differentiable,
and it widens the posterior where the surrogate is uncertain.

Comparing the ``"mean"`` and ``"marginal"`` posteriors is the surrogate
convergence test: if they agree, the GP uncertainty no longer matters.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal, Optional

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

from .adaptive_robust_scalar import AdaptiveRobustScaler
from .jax_active_learner import FittedJaxGP, JaxActiveLearner
from .lnl_surrogate import BOUNDS, PARAMETERS

logger = logging.getLogger(__name__)

TargetName = Literal["mean", "marginal"]

# Cap on the GP standard deviation, in lnL units, used by the "marginal"
# target. Outside the training envelope sigma grows without bound and
# exp(mu + sigma^2/2) would drag the sampler into unexplored regions; the cap
# keeps that pull finite. Chosen well above the Delta lnL ~ 0.5 that sets a
# 1-sigma credible interval, so it never bites inside the posterior bulk.
DEFAULT_SIGMA_CAP = 20.0


def _inverse_transform_jax(t, scaler: AdaptiveRobustScaler):
    """JAX port of :meth:`AdaptiveRobustScaler.inverse_transform`."""
    if scaler.soft_clipping:
        scaled = jnp.clip(t / scaler.clip_factor, -0.999, 0.999)
        neg_branch = jnp.arctanh(scaled) * scaler.clip_factor
        standardized = jnp.where(t <= 0.0, neg_branch, t)
    else:
        standardized = t
    return standardized * scaler.scale + scaler.median + scaler.reference_value


def make_log_likelihood(
    model: FittedJaxGP,
    scaler: AdaptiveRobustScaler,
    *,
    target: TargetName = "mean",
    sigma_cap: float = DEFAULT_SIGMA_CAP,
):
    """Build a differentiable lnL(theta) from the surrogate.

    The GP is trained on the *negated, scaled* log-likelihood, so we undo both.
    The scaler is affine (slope ``scale``) on the high-lnL side where the
    posterior lives -- soft clipping only compresses the low tail -- so the GP
    standard deviation maps to lnL units as ``scale * sigma``.
    """

    def log_likelihood(theta):
        neg_mean, neg_var = model.predict_f_jax(theta.reshape(1, -1))
        transformed = -neg_mean[0]
        lnl = _inverse_transform_jax(transformed, scaler)
        if target == "mean":
            return lnl
        sigma = jnp.minimum(jnp.sqrt(neg_var[0]) * scaler.scale, sigma_cap)
        return lnl + 0.5 * sigma**2

    return log_likelihood


def _init_points(log_likelihood, model: FittedJaxGP, num_chains: int, *, seed: int) -> np.ndarray:
    """Starting points for each chain: local maxima of the surrogate lnL.

    Chains must start from *different* points for R-hat to mean anything -- if
    they all start together, R-hat can look excellent while the chains sit in
    the same corner of a multimodal or degenerate posterior. We take the best
    training point plus the best of several random restarts, hill-climbed with
    gradient ascent, and keep the most widely separated candidates.
    """
    low, high = np.asarray(BOUNDS[0], float), np.asarray(BOUNDS[1], float)
    rng = np.random.default_rng(seed)

    grad = jax.jit(jax.grad(lambda t: log_likelihood(t).sum()))
    starts = [np.asarray(model.data.query_points, float)[
        int(np.argmin(np.asarray(model.data.observations).reshape(-1)))
    ]]
    starts += list(rng.uniform(low, high, size=(4 * num_chains, len(low))))

    climbed = []
    span = high - low
    for s in starts:
        theta = jnp.asarray(s, dtype=jnp.float64)
        for _ in range(60):
            g = grad(theta)
            step = 0.01 * jnp.asarray(span) * jnp.sign(g)
            theta = jnp.clip(theta + step, jnp.asarray(low), jnp.asarray(high))
        climbed.append((float(log_likelihood(theta)), np.asarray(theta, float)))

    climbed.sort(key=lambda kv: -kv[0])
    chosen = [climbed[0][1]]
    for _, cand in climbed[1:]:
        if len(chosen) >= num_chains:
            break
        # Prefer candidates that are not on top of an already-chosen start.
        if min(np.max(np.abs((cand - c) / span)) for c in chosen) > 0.02:
            chosen.append(cand)
    while len(chosen) < num_chains:
        chosen.append(rng.uniform(low, high))

    logger.info(
        "NUTS init points (lnL): %s",
        [round(float(log_likelihood(jnp.asarray(c))), 2) for c in chosen],
    )
    return np.asarray(chosen[:num_chains], dtype=float)


def _numpyro_model(log_likelihood, low, high):
    theta = numpyro.sample("theta", dist.Uniform(low, high).to_event(1))
    numpyro.factor("surrogate", log_likelihood(theta))


def surface_roughness(
    model: FittedJaxGP,
    centre: np.ndarray,
    *,
    n: int = 400,
    span: float = 0.25,
) -> dict:
    """Measure how smooth the GP mean is along each parameter axis.

    A well-behaved surrogate is locally near-quadratic, so its second difference
    changes slowly. We report the ratio of the RMS second difference to the
    total variation along each 1D slice; values much greater than ~1/n indicate
    small-scale structure that no sampler can integrate cleanly.
    """
    low, high = np.asarray(BOUNDS[0], float), np.asarray(BOUNDS[1], float)
    centre = np.asarray(centre, dtype=float).reshape(-1)
    out: dict[str, float] = {}
    for i, name in enumerate(PARAMETERS):
        half = span * (high[i] - low[i]) / 2.0
        grid = np.linspace(
            max(centre[i] - half, low[i]), min(centre[i] + half, high[i]), n
        )
        pts = np.repeat(centre[None, :], n, axis=0)
        pts[:, i] = grid
        mean, _ = model.predict_f(pts)
        y = np.asarray(mean).reshape(-1)
        spread = float(np.ptp(y))
        if spread <= 0:
            out[name] = 0.0
            continue
        d2 = np.diff(y, n=2)
        out[name] = float(np.sqrt(np.mean(d2**2)) / spread)
    return out


def run_nuts(
    *,
    lnl_model_path: str | Path,
    outdir: str | Path,
    target: TargetName = "mean",
    round_idx: Optional[int] = None,
    num_warmup: int = 1000,
    num_samples: int = 4000,
    num_chains: int = 2,
    seed: int = 0,
    sigma_cap: float = DEFAULT_SIGMA_CAP,
    truths: Optional[dict] = None,
) -> dict:
    """Sample the surrogate posterior with NUTS and record convergence.

    Writes ``posterior_samples.npy`` (the format the KL diagnostics expect) and
    ``nuts_summary.json``. Never raises on poor convergence: the summary carries
    a ``converged`` flag so a bad round is recorded rather than lost.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    model_dir = Path(lnl_model_path)
    gp = JaxActiveLearner.load_model(model_dir, round_idx)
    scaler = AdaptiveRobustScaler.load(str(model_dir / ".."))

    low = jnp.asarray(BOUNDS[0], dtype=jnp.float64)
    high = jnp.asarray(BOUNDS[1], dtype=jnp.float64)
    log_likelihood = make_log_likelihood(
        gp, scaler, target=target, sigma_cap=sigma_cap
    )

    init_theta = _init_points(
        log_likelihood, gp, num_chains, seed=seed
    )
    kernel = NUTS(_numpyro_model, target_accept_prob=0.9)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method="sequential",
        progress_bar=False,
    )
    mcmc.run(
        jax.random.PRNGKey(int(seed)),
        log_likelihood,
        low,
        high,
        init_params={"theta": jnp.asarray(init_theta)},
        extra_fields=("diverging",),
    )

    samples = np.asarray(mcmc.get_samples()["theta"], dtype=float)
    np.save(outdir / "posterior_samples.npy", samples)

    # Convergence diagnostics -- reported, never fatal.
    grouped = np.asarray(
        mcmc.get_samples(group_by_chain=True)["theta"], dtype=float
    )
    r_hat, ess = _rhat_ess(grouped)
    divergences = int(np.sum(np.asarray(
        mcmc.get_extra_fields().get("diverging", np.zeros(1))
    )))
    best = samples[int(np.argmax([log_likelihood(jnp.asarray(s)) for s in samples[:512]]))]
    roughness = surface_roughness(gp, best)

    converged = bool(
        np.all(r_hat < 1.01)
        and np.all(ess > 400)
        and divergences <= 0.01 * num_samples * num_chains
    )
    summary = {
        "sampler": "numpyro-NUTS",
        "target": target,
        "n_samples": int(samples.shape[0]),
        "num_chains": int(num_chains),
        "converged": converged,
        "r_hat": {p: float(r) for p, r in zip(PARAMETERS, r_hat)},
        "ess": {p: float(e) for p, e in zip(PARAMETERS, ess)},
        "divergences": divergences,
        "gp_surface_roughness": roughness,
        "median": {p: float(np.median(samples[:, i])) for i, p in enumerate(PARAMETERS)},
        "truths": truths or {},
    }
    (outdir / "nuts_summary.json").write_text(json.dumps(summary, indent=2))

    if not converged:
        logger.warning(
            "NUTS did not meet convergence targets (r_hat=%s, ess=%s, divergences=%d). "
            "Samples were still written; see nuts_summary.json.",
            summary["r_hat"], summary["ess"], divergences,
        )
    rough = {k: v for k, v in roughness.items() if v > 0.05}
    if rough:
        logger.warning(
            "GP mean surface looks rough along %s; a sampler cannot fix this -- "
            "check the GP kernel/target scaling.", rough,
        )
    return summary


def _rhat_ess(grouped: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split-R-hat and bulk ESS for an array of shape (chains, draws, dim)."""
    n_chains, n_draws, n_dim = grouped.shape
    if n_chains < 2:
        return np.ones(n_dim), np.full(n_dim, float(n_draws))

    chain_mean = grouped.mean(axis=1)
    chain_var = grouped.var(axis=1, ddof=1)
    W = chain_var.mean(axis=0)
    B = n_draws * chain_mean.var(axis=0, ddof=1)
    var_hat = (n_draws - 1) / n_draws * W + B / n_draws
    r_hat = np.sqrt(np.where(W > 0, var_hat / W, 1.0))

    # ESS from the summed positive autocorrelations, averaged over chains.
    ess = np.empty(n_dim)
    for d in range(n_dim):
        taus = []
        for c in range(n_chains):
            x = grouped[c, :, d] - grouped[c, :, d].mean()
            denom = np.dot(x, x)
            if denom <= 0:
                taus.append(1.0)
                continue
            acf = np.correlate(x, x, mode="full")[n_draws - 1:] / denom
            cut = np.argmax(acf < 0.05)
            cut = len(acf) if cut == 0 and acf[0] >= 0.05 else max(cut, 1)
            taus.append(1.0 + 2.0 * float(np.sum(acf[1:cut])))
        ess[d] = n_chains * n_draws / max(float(np.mean(taus)), 1.0)
    return r_hat, ess
