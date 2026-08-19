import os

from cosmic_integration.lnl_surrogate.lnl_surrogate import LnLSurrogate
from cosmic_integration.lnl_surrogate.run_sampler import sample_lnl_surrogate
from cosmic_integration.observation import load_observation
import pytest

ON_GITHUB = os.getenv("GITHUB_ACTIONS", "false").lower() == "true"
FAST_TESTS = os.getenv("COSMIC_INTEGRATION_FAST_TESTS", "1").lower() not in {"0", "false", "no"}

TRAIN_TOTAL_STEPS = 10 if FAST_TESTS else 300
TRAIN_STEPS_PER_ROUND = 3
MCMC_ITERATIONS = 200 if FAST_TESTS else 3000



@pytest.mark.skipif(ON_GITHUB, reason="Skip test on GitHub Actions due to resource constraints.")
def test_lnl_surrogate(outdir, test_compas_h5, observation_file):
    """
    Test the LnLSurrogate class.

    1. create class
    2. train it
    3. check that it can predict
    4. check that it can be saved
    5. check that the loaded model can predict

    """
    outdir = os.path.join(outdir, "lnl_surrogate")
    os.makedirs(outdir, exist_ok=True)
    obs = load_observation(observation_file)

    # create class + train
    lnl_surrogate = LnLSurrogate.train(
        observation_file=observation_file,
        compas_h5=test_compas_h5,
        outdir=outdir,
        initial_points=4,
        total_steps=TRAIN_TOTAL_STEPS,
        steps_per_round=TRAIN_STEPS_PER_ROUND,
        truth=obs.params
    )

    lnl_surrogate.parameters = obs.param_dict
    # check that it can predict (load some parameter)
    lnl = lnl_surrogate.log_likelihood()

    assert isinstance(lnl, float), "Log likelihood should be a float."

    sample_lnl_surrogate(
        lnl_model_path=f"{outdir}/gp_model/models",
        outdir=f"{outdir}/MCMC",
        verbose=True,
        truths=obs.param_dict,
        mcmc_kwargs={"nwalkers": 10, "iterations": MCMC_ITERATIONS}
    )
    assert os.path.exists(f"{outdir}/MCMC/lnl_surrogate_result.json"), "MCMC samples were not saved."
    assert os.path.exists(f"{outdir}/MCMC/lnl_surrogate_corner.png"), "Corner plot was not saved."

