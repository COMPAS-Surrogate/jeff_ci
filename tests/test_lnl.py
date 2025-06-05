import os

import matplotlib.pyplot as plt
import numpy as np
import pytest

from cosmic_integration.lnl_computer import LnLComputer, Observation
from cosmic_integration import ratesSampler 


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
    np.random.seed(42)  # For reproducibility

    test_failed = False
    lnl_computer = LnLComputer.load(
        observation_file=observation_file,
        compas_h5=test_compas_h5
    )
    true_params = Observation.from_jeff(fname=observation_file).param_dict
    lnl_at_true = lnl_computer(**true_params)

    param_ranges = dict(
        alpha=(min(ratesSampler.ALPHA_VALUES), max(ratesSampler.ALPHA_VALUES)),
        sigma=(min(ratesSampler.SIGMA_VALUES), max(ratesSampler.SIGMA_VALUES)),
        sfr_a=(min(ratesSampler.SFR_A_VALUES), max(ratesSampler.SFR_A_VALUES)),
        sfr_d=(min(ratesSampler.SFR_D_VALUES), max(ratesSampler.SFR_D_VALUES))
    )
    param_lnls = {}
    for param, (min_val, max_val) in param_ranges.items():
        p_vals = np.linspace(min_val, max_val, 5)
        lnls = np.zeros(len(p_vals))
        for i, p_val in enumerate(p_vals):
            params = true_params.copy()
            params[param] = p_val
            lnls[i] = lnl_computer(**params)
            if lnls[i] < lnl_at_true:
                test_failed = True

        param_lnls[param] = (p_vals, lnls)

    # Plot the results
    fig, ax = plt.subplots(4, 1, figsize=(4, 10))
    for i, (param, (p_vals, lnls)) in enumerate(param_lnls.items()):
        ax[i].plot(p_vals, lnls)
        ax[i].axhline(lnl_at_true, color='red', linestyle='--', label='True LnL')
        ax[i].axvline(true_params[param], color='red', linestyle='--', label='True Param')
        ax[i].plot(true_params[param], lnl_at_true, 'ro', label='True Point')
        ax[i].set_xlabel(param)
        ax[i].set_ylabel('Log Likelihood')
    plt.tight_layout()
    plt.savefig(f"{outdir}/lnl_1d_test.png")

    if test_failed:
        raise AssertionError(
            "Some parameters produced a lower log likelihood than the true parameters.. Check the plot for details.")
