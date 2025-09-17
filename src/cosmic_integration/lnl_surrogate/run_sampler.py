import logging
import bilby
from bilby.core.prior import PriorDict, Uniform, DeltaFunction
import numpy as np
from typing import List
from .lnl_surrogate import LnLSurrogate, PARAMETERS, BOUNDS



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
    verbose=False,
    truths: dict = {},
    mcmc_kwargs={},
):
    bilby_logger = logging.getLogger("bilby")

    bilby_logger.setLevel(logging.ERROR)
    if verbose:
        bilby_logger.setLevel(logging.INFO)

    lnl_surrogate = LnLSurrogate.load(lnl_model_path)
    prior = get_prior()

    print(f"Sampling from LnLSurrogate model at {lnl_model_path}")
    print(f"Using prior: {prior}")
    mcmc_kwargs["nwalkers"] = mcmc_kwargs.get("nwalkers", 10)
    mcmc_kwargs["iterations"] = mcmc_kwargs.get("iterations", 1000)

    sampler_kwargs = dict(
        priors=prior,
        sampler="emcee",
        injection_parameters=truths,
        outdir=outdir,
        clean=True,
        verbose=verbose,
        plot=True,
        **mcmc_kwargs,
    )

    bilby.run_sampler(
        likelihood=lnl_surrogate,
        label="lnl_surrogate",
        **sampler_kwargs,
    )

    # plot_dir = f"{outdir}/plots"
    # os.makedirs(plot_dir, exist_ok=True)
    #
    # # posterior to numpy array from pd.DataFrame
    # posterior_df = result.posterior[PARAMETERS]
    # # keep only the params that have dynamic range
    #
    # posterior = posterior_df.to_numpy() # shape (n_samples, n_params)
    # corner_fig = corner.corner(
    #     posterior,
    #     labels=PARAMETERS,
    #     truths=[truths.get(param, None) for param in PARAMETERS],
    #     plot_density=False
    # )
    # corner_fig.savefig(f"{plot_dir}/corner.png")




