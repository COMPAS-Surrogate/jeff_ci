from cosmic_integration.observation import generate_posterior_samples, plot_gw_posteriors
from cosmic_integration.ratesSampler import CosmicIntegration
import os
import pytest



GITHUB_ACTION = os.getenv("GITHUB_ACTIONS", "false").lower() == "true"




@pytest.mark.skipif(GITHUB_ACTION, reason="Skip test on GitHub Actions due to resource constraints.")
def test_generate_observation(test_compas_h5, outdir, mock_sys_argv):
    """
    Test that the rate file can be generated without errors.
    """
    outdir = f"{outdir}/test_generate_observation"
    os.makedirs(outdir, exist_ok=True)

    ci = CosmicIntegration.from_compas_h5(*os.path.split(test_compas_h5))
    rates, chirp_masses = ci.FindDetectionRate()


    # scale rates to 0.1 year
    rates = rates * 0.1
    generate_posterior_samples(
        rates=rates,
        chirp_masses=chirp_masses,
        output_dir=outdir,
    )
    plot_gw_posteriors(
        output_dir=outdir,
        figsize=(6, 5),
        save_plots=True,
        show_plots=False,
        dpi=150
    )
