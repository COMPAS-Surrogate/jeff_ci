import json
from pathlib import Path

import click

from cosmic_integration.lnl_surrogate.workflow import (
    SurrogateWorkflowConfig,
    run_surrogate_workflow,
)


@click.command()
@click.argument("compas_h5", type=click.Path(exists=True, dir_okay=False))
@click.argument("observation_file", type=click.Path(exists=True, dir_okay=False))
@click.option("--outdir", type=click.Path(dir_okay=True), default="workflow_out", show_default=True)
@click.option("--initial-points", type=int, default=50, show_default=True)
@click.option("--total-steps", type=int, default=300, show_default=True)
@click.option("--steps-per-round", type=int, default=30, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--force-dataset", is_flag=True, help="Recompute the initial dataset even if it exists.")
@click.option("--postprocess-every", type=int, default=1, show_default=True)
@click.option(
    "--postprocess-during-bo/--postprocess-after-bo",
    default=True,
    show_default=True,
    help="Run per-round MCMC/diagnostics during BO (slower) or only after BO completes.",
)
@click.option("--mcmc-nwalkers", type=int, default=32, show_default=True)
@click.option("--mcmc-iterations", type=int, default=2000, show_default=True)
@click.option(
    "--mcmc-uncertainty-beta",
    default="0.0",
    show_default=True,
    help="Float or comma-separated list (e.g. '0,1').",
)
@click.option("--gp-truth-n-per-fraction", type=int, default=200, show_default=True)
@click.option(
    "--gp-truth-fractions",
    default="0.01,0.02,0.05,0.1,0.2,0.4",
    show_default=True,
    help="Comma-separated list of distance scales (fractions of parameter widths).",
)
@click.option("--scaler-soft-clipping/--no-scaler-soft-clipping", default=True, show_default=True)
@click.option("--scaler-clip-factor", type=float, default=3.0, show_default=True)
@click.option(
    "--scaler-lower-clip-percentile",
    default="auto",
    show_default=True,
    help="Float, 'auto', or 'none'.",
)
@click.option("--callback-fail-fast/--no-callback-fail-fast", default=True, show_default=True)
def main(
    compas_h5: str,
    observation_file: str,
    outdir: str,
    initial_points: int,
    total_steps: int,
    steps_per_round: int,
    seed: int,
    force_dataset: bool,
    postprocess_every: int,
    postprocess_during_bo: bool,
    mcmc_nwalkers: int,
    mcmc_iterations: int,
    mcmc_uncertainty_beta: str,
    gp_truth_n_per_fraction: int,
    gp_truth_fractions: str,
    scaler_soft_clipping: bool,
    scaler_clip_factor: float,
    scaler_lower_clip_percentile: str,
    callback_fail_fast: bool,
) -> None:
    """Run surrogate BO workflow with per-round checkpoints and diagnostics."""

    def _parse_floats(text: str) -> list[float]:
        return [float(x) for x in str(text).replace(",", " ").split() if x.strip()]

    betas = _parse_floats(mcmc_uncertainty_beta)
    beta_value = betas[0] if len(betas) == 1 else betas

    fractions = _parse_floats(gp_truth_fractions)
    if not fractions:
        raise click.UsageError("--gp-truth-fractions must contain at least one value.")

    lower_clip: float | None | str
    if scaler_lower_clip_percentile.strip().lower() in {"none", "null"}:
        lower_clip = None
    elif scaler_lower_clip_percentile.strip().lower() == "auto":
        lower_clip = "auto"
    else:
        lower_clip = float(scaler_lower_clip_percentile)

    config = SurrogateWorkflowConfig(
        compas_h5=str(Path(compas_h5)),
        observation_file=str(Path(observation_file)),
        outdir=str(Path(outdir)),
        initial_points=int(initial_points),
        total_steps=int(total_steps),
        steps_per_round=int(steps_per_round),
        seed=int(seed),
        force_dataset=bool(force_dataset),
        postprocess_every=int(postprocess_every),
        postprocess_during_bo=bool(postprocess_during_bo),
        mcmc_kwargs={"nwalkers": int(mcmc_nwalkers), "iterations": int(mcmc_iterations)},
        mcmc_uncertainty_beta=beta_value,
        gp_truth_n_per_fraction=int(gp_truth_n_per_fraction),
        gp_truth_fractions=tuple(float(x) for x in fractions),
        scaler_soft_clipping=bool(scaler_soft_clipping),
        scaler_clip_factor=float(scaler_clip_factor),
        scaler_lower_clip_percentile=lower_clip,
        callback_fail_fast=bool(callback_fail_fast),
    )

    summary = run_surrogate_workflow(config)
    click.echo(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
