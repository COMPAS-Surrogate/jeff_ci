import os

import click

from cosmic_integration.lnl_computer import LnLComputer, Observation
from cosmic_integration.lnl_surrogate.lnl_surrogate import LnLSurrogate
from cosmic_integration.lnl_surrogate.run_sampler import sample_lnl_surrogate

N_INIT = 200

@click.command()
@click.argument("compas_h5", type=click.Path(exists=True))
@click.option("observation-file", type=click.Path(exists=True))
@click.option("model-cache", type=click.Path(exists=True), default="model_cache.csv", show_default=True)
@click.option("--outdir", default="outdir", show_default=True)
def main(compas_h5, observation_file, model_cache, outdir, ):
    """Train and sample from a log-likelihood surrogate model.

    COMPAS_H5: Path to COMPAS .h5 file
    """

    os.makedirs(outdir, exist_ok=True)

    click.echo(f"Using COMPAS h5 file: {compas_h5}, observation file: {observation_file}, model cache: {model_cache}")

    lnl_computer = LnLComputer.load(
        observation_file=observation_file,
        compas_h5=compas_h5,
        cache_fn=f"{outdir}/lnl_cache.csv"
    )

    observation = Observation.from_ilya(observation_file)

    lnls = lnl_computer.compute_via_cache(model_cache)
    lnl, params = lnls[:, 0], lnls[:, 1:]
    click.echo(f"Computed {len(lnls)} log likelihoods from cache.")

    top_indices = lnl.argsort()[-N_INIT:][::-1]
    top_lnls = lnl[top_indices]
    top_params = params[top_indices]

    best_lnl = top_lnls.max()
    worst_lnl = top_lnls.min()
    threshold_lnl = -(worst_lnl - best_lnl)

    click.echo(f"Best log likelihood: {best_lnl:.2f}")
    click.echo(f"Threshold log likelihood: {threshold_lnl:.2f}")

    lnl_at_true = lnl_computer(*observation.params)
    click.echo(f"Log likelihood at true params: {lnl_at_true - best_lnl:.2f}")

    lnl_surrogate = LnLSurrogate.train(
        observation_file=observation_file,
        compas_h5=compas_h5,
        outdir=outdir,
        total_steps=3,
        steps_per_round=3,
        truth=observation.params,
        inital_samples=top_params,
        initial_lnls=top_lnls,
        threshold=threshold_lnl,
    )

    lnl_surrogate.parameters = observation.param_dict
    lnl = lnl_surrogate.log_likelihood()
    click.echo(f"lnl at true params: {lnl:.2f}")

    sample_lnl_surrogate(
        lnl_model_path=f"{outdir}/gp_model/models",
        outdir=f"{outdir}/MCMC",
        verbose=True,
        truths=observation.param_dict,
        mcmc_kwargs={"nwalkers": 10, "iterations": 1000}
    )


if __name__ == "__main__":
    main()
