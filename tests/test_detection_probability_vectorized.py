import os
import time

import numpy as np

from cosmic_integration.ratesSampler.binned_cosmic_integrator import (
    BinnedCosmicIntegrator,
)
from cosmic_integration.ratesSampler.ratesSampler import Mc_STEP, ETA_STEP, SNR_STEP


def _baseline_detection_probability(
    mc,
    eta,
    redshifts,
    distances,
    n_redshifts_detection,
    snr_grid_at_1mpc,
    detection_prob_from_snr,
    mc_step,
    eta_step,
    snr_step,
):
    n_binaries = mc.shape[0]
    detection_probability = np.ones((n_binaries, n_redshifts_detection), dtype=float)

    max_eta_index = snr_grid_at_1mpc.shape[0] - 1
    max_mc_index = snr_grid_at_1mpc.shape[1] - 1

    redshifts_det = redshifts[:n_redshifts_detection]
    distances_det = distances[:n_redshifts_detection]

    for i in range(n_binaries):
        mc_shifted = mc[i] * (1.0 + redshifts_det)

        eta_index = int(np.round(eta[i] / eta_step)) - 1
        if eta_index < 0:
            eta_index = 0
        elif eta_index > max_eta_index:
            eta_index = max_eta_index

        snr = np.ones(n_redshifts_detection, dtype=float) * 1.0e-5
        mc_index = np.round(mc_shifted / mc_step).astype(int) - 1

        valid = (mc_index >= 0) & (mc_index <= max_mc_index)
        snr[valid] = snr_grid_at_1mpc[eta_index, mc_index[valid]]
        snr /= distances_det

        det_index = np.round(snr / snr_step).astype(int) - 1
        below_min = det_index < 0
        valid_det = (det_index >= 0) & (det_index < detection_prob_from_snr.shape[0])

        detection_probability[i, below_min] = 0.0
        detection_probability[i, valid_det] = detection_prob_from_snr[det_index[valid_det]]

    return detection_probability


def test_detection_probability_vectorization_matches_baseline(test_compas_h5):
    directory, filename = os.path.split(test_compas_h5)
    directory = directory or "."
    integrator = BinnedCosmicIntegrator.from_compas_h5(directory, filename)
    integrator.CalculateRedshiftRelatedParams()

    mass1 = integrator.compas.mass1
    mass2 = integrator.compas.mass2
    chirp_masses = (mass1 * mass2) ** (3.0 / 5.0) / (mass1 + mass2) ** (1.0 / 5.0)
    etas = mass1 * mass2 / (mass1 + mass2) ** 2

    rng = np.random.default_rng(0)
    subset_size = min(200, chirp_masses.shape[0])
    subset = rng.choice(chirp_masses.shape[0], size=subset_size, replace=False)

    mc_subset = chirp_masses[subset]
    eta_subset = etas[subset]

    redshifts = integrator.redshifts
    distances = integrator.distances
    n_det = integrator.nRedshiftsDetection
    snr_grid = integrator.SE.SNRgridAt1Mpc
    det_prob_grid = integrator.SE.detectionProbabilityFromSNR

    start_vec = time.perf_counter()
    vec = integrator.FindDetectionProbability(
        mc_subset,
        eta_subset,
        redshifts,
        distances,
        n_det,
        subset_size,
        snr_grid,
        det_prob_grid,
        Mc_STEP,
        ETA_STEP,
        SNR_STEP,
    )
    vec_time = time.perf_counter() - start_vec

    start_baseline = time.perf_counter()
    baseline = _baseline_detection_probability(
        mc_subset,
        eta_subset,
        redshifts,
        distances,
        n_det,
        snr_grid,
        det_prob_grid,
        Mc_STEP,
        ETA_STEP,
        SNR_STEP,
    )
    baseline_time = time.perf_counter() - start_baseline

    print(f"Vectorized detection probability time: {vec_time:.3f} s")
    print(f"Baseline detection probability time:   {baseline_time:.3f} s")

    np.testing.assert_allclose(vec, baseline, rtol=1e-12, atol=1e-12)
