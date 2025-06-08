import os
import sys

from cosmic_integration.lnl_computer import LnLComputer, Observation
from cosmic_integration.lnl_surrogate.lnl_surrogate import LnLSurrogate
from cosmic_integration.lnl_surrogate.run_sampler import sample_lnl_surrogate

COMPAS_H5 = sys.argv[-1]
observation_file = "binned_rates_alpha-0.325_sigma0.213_asf0.012_dsf4.253.csv"
OUTDIR = "outdir"
MODEL_CACHE = "../generate_data/out_512M.csv"

# check all files exist
if not os.path.exists(COMPAS_H5):
    raise FileNotFoundError(f"COMPAS h5 file {COMPAS_H5} does not exist.")
if not os.path.exists(observation_file):
    raise FileNotFoundError(f"Observation file {observation_file} does not exist.")
if not os.path.exists(MODEL_CACHE):
    raise FileNotFoundError(f"Model cache file {MODEL_CACHE} does not exist.")



os.makedirs(OUTDIR, exist_ok=True)

print(f"Using COMPAS h5 file: {COMPAS_H5}")

lnl_computer = LnLComputer.load(
    observation_file=observation_file,
    compas_h5=COMPAS_H5,
    cache_fn=f"{OUTDIR}/lnl_cache.csv"  # Cache file for storing results
)

observation = Observation.from_jeff(observation_file)


lnls = lnl_computer.compute_via_cache(MODEL_CACHE)
print(lnls.shape)
lnl, params = lnls[:, 0], lnls[:, 1:]
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
lnl_at_true = lnl_computer(*observation.params)
print(f"Log likelihood at true params: {lnl_at_true-best_lnl:.2f}")


lnl_surrogate = LnLSurrogate.train(
    observation_file=observation_file,
    compas_h5=COMPAS_H5,
    outdir=OUTDIR,
    initial_points=50,
    total_steps=3,
    steps_per_round=3,
    truth=observation.params,
    inital_samples=top_params,  # Initial samples
    initial_lnls=top_lnls,  # Initial log likelihoods
    threshold=threshold_lnl,  # Threshold for negative log likelihood
)


lnl_surrogate.parameters = observation.param_dict
# check that it can predict (load some parameter)
lnl = lnl_surrogate.log_likelihood()
print(f"lnl at true params: {lnl:.2f}")


sample_lnl_surrogate(
    lnl_model_path=f"{OUTDIR}/gp_model/models",
    outdir=f"{OUTDIR}/MCMC",
    verbose=True,
    truths=observation.param_dict,
    mcmc_kwargs={"nwalkers": 10, "iterations": 1000}
)

