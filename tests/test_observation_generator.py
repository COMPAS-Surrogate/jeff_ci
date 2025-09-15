from cosmic_integration.observation.mock_observation import MockObservation
from cosmic_integration.ratesSampler import CosmicIntegration, DEFAULT_PARAMS
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

    params = dict(
        p_Alpha=DEFAULT_PARAMS[0],
        p_Sigma=DEFAULT_PARAMS[1],
        p_SFRa=DEFAULT_PARAMS[2],
        p_SFRd=DEFAULT_PARAMS[3],
    )
    rates, chirp_masses = ci.FindDetectionRate(**params)


    # scale rates to 0.1 year
    duration  = 0.05
    rates = rates * duration
    fname = f"{outdir}/mock_observation.h5"

    if os.path.exists(fname):
        obs = MockObservation.load_h5(fname)
    else:
        obs = MockObservation.generate_from_rates(
            rates=rates,
            chirp_masses=chirp_masses,
            output_file=f"{outdir}/mock_observation.h5",
            duration=duration,
            params=list(params.values()),
        )

    obs.plot_event_summaries(fname=f"{outdir}/mock_observation_event_summaries.png")
    obs.plot(fname=f"{outdir}/mock_observation.png")


