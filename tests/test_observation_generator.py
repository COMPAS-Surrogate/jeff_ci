from cosmic_integration.observation.mock_observation import MockObservation
from cosmic_integration.ratesSampler import BinnedCosmicIntegrator, CosmicIntegration
from cosmic_integration.ratesSampler.ratesSampler import DEFAULT_PARAMS
import os
import pytest
import numpy as np




@pytest.mark.slow
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
    rates, _ = ci.FindDetectionRate(**params)

    bci = BinnedCosmicIntegrator.from_compas_h5(*os.path.split(test_compas_h5))
    binned_rates = bci.FindBinnedDetectionRate(**params)
    print(f"Expected detections: {np.sum(binned_rates):.2f}")

    duration  = 0.2
    params = np.array(list(params.values()))

    obs = MockObservation.generate_from_rates(
        rates=binned_rates,
        output_file=f"{outdir}/mock_observation.h5",
        duration=duration,
        params=params,
        measurement_uncertainty=False,
    )
    obs.plot(fname=f"{outdir}/mock_observation_no_uncertainty.png")

    # obs = MockObservation.generate_from_rates(
    #     rates=rates,
    #     chirp_masses=chirp_masses,
    #     output_file=f"{outdir}/mock_observation.h5",
    #     duration=duration,
    #     params=params,
    # )
    # obs = MockObservation.load_h5(fname)
    #
    #
    # obs.plot_event_summaries(fname=f"{outdir}/mock_observation_event_summaries.png")
    # obs.plot(fname=f"{outdir}/mock_observation.png")

