import os
import numpy as np
import matplotlib.pyplot as plt
from cosmic_integration.lnl_computer import LnLComputer, Observation

compas_h5 = '/home/avaj040/Documents/projects/COSMIC_INTEGRATOR/jeff_ci/tests/test_data/test_compas.h5'
observation_file = '/home/avaj040/Documents/projects/COSMIC_INTEGRATOR/jeff_ci/docs/studies/load_cached/out_512M.csv'

obs  = Observation.from_jeff(observation_file, idx=100)

lnl_computer = LnLComputer.load(
    observation_file=observation_file,
    compas_h5=compas_h5,
    row_idx=100
)
data = lnl_computer.compute_via_cache(observation_file)
lnls = data[:, 0]
params = data[:, 1:]




plt.hist(lnls, bins=50, alpha=0.7, label='Log Likelihoods')
plt.axvline(lnls.max(), color='red', linestyle='--', label='Best Log Likelihood')
plt.show()