import os

import matplotlib.pyplot as plt
import numpy as np
import pytest

from cosmic_integration.lnl_computer import LnLComputer, Observation
from cosmic_integration import ratesSampler 

from click.testing import CliRunner
from cosmic_integration.cli_tools.run_1d_lnl_check import run_1d_lnl_check


def test_lnl(test_compas_h5, observation_file, outdir):
    np.random.seed(42)  # For reproducibility
    lnl_computer = LnLComputer.load(
        observation_file=observation_file,
        compas_h5=test_compas_h5
    )
    lnl = lnl_computer(
        alpha=0.5,
        sigma=0.3,
        sfr_a=1.0,
        sfr_d=2.0,
        cache_fn=f"{outdir}/lnl_cache.csv"
    )
    assert not np.isnan(lnl), "Log likelihood should not be NaN"
    expected_lnl = 8323455.23  # Example expected value, adjust as necessary
    lnl_err = abs(lnl - expected_lnl)
    np.testing.assert_almost_equal(lnl_err, 0, decimal=2,
                                   err_msg=f"LnL diff: {abs(lnl - expected_lnl):.2f}")


# puytest that only runs on manual invocation
@pytest.mark.skipif(
    os.environ.get("GITHUB_ACTIONS") == "true",
    reason="Skipped on GitHub CI"
)
def test_lnl_1d(test_compas_h5, observation_file, outdir):
    """
    Test the LnLComputer with a 1D observation file.
    """
    np.random.seed(42)
    click_runner = CliRunner()
    result = click_runner.invoke(
        run_1d_lnl_check,
        [
            observation_file,
            test_compas_h5,
            "10",  # n grid points
            "--outdir", outdir,
        ]
    )
