import click

from cosmic_integration.lnl_computer import LnLComputer, Observation
import matplotlib.pyplot  as plt


@click.command()
@click.argument("observation_file", type=click.Path(exists=True))
@click.argument("compas_h5", type=click.Path(exists=True))
@click.argument("cache_fn", type=click.Path(exists=True))
def main(observation_file: str, compas_h5: str, cache_fn: str, outdir: str = "."):
    """
    Load the LnLSurrogate model and compute the log likelihood from cached data.

    :param observation_file: Path to the observation file.
    :param compas_h5: Path to the COMPAS h5 file.
    :param cache_fn: Path to the cache file containing precomputed log likelihoods.
    """
    # Load the LnLSurrogate model

    lnl_computer = LnLComputer.load(
        observation_file=observation_file,
        compas_h5=compas_h5,
        cache_fn=f"{outdir}/lnl_cache.csv"  # Cache file for storing results
    )

    observation = Observation.from_jeff(observation_file)

    # Compute log likelihoods using cached data
    lnls = lnl_computer.compute_via_cache(cache_fn)
    print(lnls.shape)
    lnl, params = lnls[:, 0], lnls[:, 1:]
    print(f"Computed {len(lnls)} log likelihoods from cache.")
    # best lnl + param
    best_idx = lnl.argmax()
    best_lnl = lnl[best_idx]
    best_params = params[best_idx]
    print(f"Best log likelihood: {best_lnl:.2f} for params {best_params}")
    print(f"True: {observation.params}")

    lnl_at_true = lnl_computer(*observation.params)


    plt.hist(lnl, bins=50, alpha=0.7, label='Log Likelihoods')
    plt.axvline(best_lnl, color='red', linestyle='--', label='Best Log Likelihood')
    plt.axvline(lnl_at_true, color='green', linestyle='--', label='Log Likelihood at True Params')
    plt.xlabel('Log Likelihood')
    plt.ylabel('Frequency')
    plt.savefig(f"{outdir}/lnl_histogram.png")



if __name__ == "__main__":
    main()
