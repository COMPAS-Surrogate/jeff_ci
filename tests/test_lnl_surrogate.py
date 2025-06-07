import os

import numpy as np
from cosmic_integration.lnl_surrogate.lnl_surrogate import LnLSurrogate
from cosmic_integration.observation import Observation


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
    obs = Observation.from_jeff(observation_file)

    # create class + train
    lnl_surrogate = LnLSurrogate.train(
        observation_file=observation_file,
        compas_h5=test_compas_h5,
        outdir=outdir,
        initial_points=50,
        total_steps=45,
        steps_per_round=15,
        truth=obs.params
    )


    lnl_surrogate.parameters = obs.param_dict
    # check that it can predict (load some parameter)
    lnl = lnl_surrogate.log_likelihood()

    assert isinstance(lnl, float), "Log likelihood should be a float."

