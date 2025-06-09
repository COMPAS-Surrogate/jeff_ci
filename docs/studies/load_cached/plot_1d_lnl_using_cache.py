import os
import numpy as np
import matplotlib.pyplot as plt
from cosmic_integration.lnl_computer import LnLComputer, Observation
import pandas as pd
from cosmic_integration.lnl_surrogate.lnl_surrogate import LnLSurrogate
from cosmic_integration.lnl_surrogate.run_sampler import sample_lnl_surrogate



compas_h5 = '/fred/oz101/avajpeyi/COMPAS_DATA/h5out_512M.h5'
lnl_cache ='/fred/oz101/avajpeyi/code/jeff_ci/docs/studies/generate_data/out_512M_FULL.csv'
observation_file = lnl_cache
MODEL_CACHE = '/fred/oz101/avajpeyi/code/jeff_ci/docs/studies/generate_data/out_512M_FULL.csv'

COMPAS_H5 = '/fred/oz101/avajpeyi/COMPAS_DATA/h5out_512M.h5'

OUTDIR = 'out_sim'
IDX = 100

os.makedirs(OUTDIR, exist_ok=True)

# compas_h5 = '/home/avaj040/Documents/projects/COSMIC_INTEGRATOR/jeff_ci/tests/test_data/test_compas.h5'
# observation_file = '/home/avaj040/Documents/projects/COSMIC_INTEGRATOR/jeff_ci/docs/studies/load_cached/out_512M.csv'

obs  = Observation.from_jeff(observation_file, idx=IDX)

lnl_computer = LnLComputer.load(
    observation_file=observation_file,
    compas_h5=compas_h5,
    row_idx=IDX,
    cache_fn=f"{OUTDIR}/lnl_cache.csv"  # Cache file for storing results
)
data = lnl_computer.compute_via_cache(observation_file)
data_headers = ["lnl", "Alpha", "Sigma", "SFRa", "SFRd"]
df = pd.DataFrame(data, columns=data_headers)

lnls = data[:, 0]
params = data[:, 1:]

# only keep params that are unique
unique_params = np.unique(params, axis=0)
lnls = lnls[np.unique(params, axis=0, return_index=True)[1]]


# lnl at true
# lnl_at_true = lnl_computer(*obs.params)

plt.hist(lnls, bins=50, alpha=0.7)
plt.savefig("lnl_histogram.png")
# plt.axvline(lnl_at_true, color='green', linestyle='--', label='Log Likelihood at True Params')
plt.legend()
plt.xlabel('Log Likelihood')
plt.savefig("lnl_histogram.png")




#
# import corner
#
# # normalize lnl (bw 0 and 1)
# w = lnls-lnls.min()
# w /= w.max()
#
#
# # downsample ppionts based on weights
# n_samples = 1000
# indices = np.random.choice(
#     np.arange(len(lnls)),
#     size=n_samples,
#     p=w / w.sum()
# )
# params = params[indices]
#
# fig = corner.corner(
#     params,
#     labels=["Alpha", "Sigma", "SFRa", "SFRd"],
#     truths=obs.params,
#     quantiles=[0.16, 0.5, 0.84],
#     show_titles=True,
#     title_kwargs={"fontsize": 12},
# )
#
# fig.savefig("corner_plot.png")
#
#




lnls = lnl_computer.compute_via_cache(MODEL_CACHE)
print(lnls.shape)
lnl, params = lnls[:, 0], lnls[:, 1:]

# only keep params that are unique
unique_params = np.unique(params, axis=0)
lnls = lnls[np.unique(params, axis=0, return_index=True)[1]]

print(f"Computed {len(lnls)} log likelihoods from cache.")

# filter out to only keep top 200 log likelihoods + params
top_indices = lnl.argsort()[-200:][::-1]
top_lnls = lnl[top_indices]
top_params = params[top_indices]



best_lnl = top_lnls.max()
worst_lnl = top_lnls.min()
print(f"Best log likelihood: {best_lnl:.2f} (this will be the reference for normalization)")

threshold_lnl = -(worst_lnl - best_lnl)
print(f"Threshold log likelihood: {threshold_lnl:.2f}")


# lnl at true
# lnl_at_true = lnl_computer(*obs.params)
# print(f"Log likelihood at true params: {lnl_at_true-best_lnl:.2f}")


lnl_surrogate = LnLSurrogate.train(
    observation_file=observation_file,
    compas_h5=COMPAS_H5,
    outdir=OUTDIR,
    initial_points=50,
    total_steps=300,
    steps_per_round=30,
    truth=obs.params,
    inital_samples=top_params,  # Initial samples
    initial_lnls=top_lnls,  # Initial log likelihoods
    threshold=threshold_lnl,  # Threshold for negative log likelihood
)


lnl_surrogate.parameters = obs.param_dict
# check that it can predict (load some parameter)
lnl = lnl_surrogate.log_likelihood()
print(f"lnl at true params: {lnl:.2f}")


sample_lnl_surrogate(
    lnl_model_path=f"{OUTDIR}/gp_model/models",
    outdir=f"{OUTDIR}/MCMC",
    verbose=True,
    truths=obs.param_dict,
    mcmc_kwargs={"nwalkers": 10, "iterations": 4000}
)

