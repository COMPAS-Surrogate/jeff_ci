"""Run analysis for one specific observation using COMPAS h5 obtained from the CLI 



CLI args:
[observation_fpath]
[compas_h5_fpath]
[n]: number of gridpoints for each parameter.

"""


import click
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm.auto import tqdm
import h5py

from ..lnl_computer import LnLComputer
from ..ratesSampler import ALPHA_VALUES, SIGMA_VALUES, SFR_A_VALUES, SFR_D_VALUES



@click.command()
@click.argument("observation_fpath", type=click.Path(exists=True))
@click.argument("compas_h5_fpath", type=click.Path(exists=True))
@click.argument("n", type=int, default=10,)
@click.option("--outdir", type=click.Path(exists=False), default=".", help="Output directory for plots")
@click.option("--true_params", type=str, default=None, help="True param dict, eg alpha:0 sigma:0.3 sfr_a:1.0 sfr_d:2.0")
def run_1d_lnl_check(observation_fpath, compas_h5_fpath, n, outdir, true_params):
    """
    Run analysis for one specific observation using COMPAS h5 obtained from the CLI.
    
    Args:
        observation_fpath (str): Path to the observation file.
        compas_h5_fpath (str): Path to the COMPAS h5 file.
        n (int): Number of grid points for each parameter.
    """
    lnl_computer = LnLComputer.load(
        observation_file=observation_fpath,
        compas_h5=compas_h5_fpath
    )

    # If true_params is provided, parse it into a dictionary
    if true_params:
        true_params = dict(param.split(":") for param in true_params.split())
        true_params = {k: float(v) for k, v in true_params.items()}
        
    elif lnl_computer.observation.param_dict is not None:
        true_params = lnl_computer.observation.param_dict
    else:
        raise ValueError("No true parameters provided and observation does not have param_dict.")
    

    lnl_cache_fn = f"{outdir}/lnl_cache.csv"
    lnl_at_true = lnl_computer(**true_params, cache_fn=lnl_cache_fn) # type: ignore



    # Perform analysis with the specified number of grid points
    param_ranges = dict(
        alpha=(min(ALPHA_VALUES), max(ALPHA_VALUES)),
        sigma=(min(SIGMA_VALUES), max(SIGMA_VALUES)),
        sfr_a=(min(SFR_A_VALUES), max(SFR_A_VALUES)),
        sfr_d=(min(SFR_D_VALUES), max(SFR_D_VALUES))
    )

    param_lnls = {}
    for param, (min_val, max_val) in param_ranges.items():
        p_vals = np.linspace(min_val, max_val, n)
        lnls = np.zeros(len(p_vals))
        for i, p_val in enumerate(tqdm(p_vals, desc=f"Processing {param}")):
            params = {**lnl_computer.observation.param_dict, param: p_val}
            lnls[i] = lnl_computer(**params, cache_fn=lnl_cache_fn)  # type: ignore
            
        param_lnls[param] = {param: p_vals, "lnl": lnls}


    # Plot the results
    fig, axs = plt.subplots(4, 1, figsize=(4, 6))
    for i, (param, data) in enumerate(param_lnls.items()):
        axs[i].plot(data[param], data["lnl"], label=f"{param} vs LnL")
        axs[i].axhline(lnl_at_true, color='red', linestyle='--', label='LnL at true params')
        axs[i].axvline(true_params[param], color='red', linestyle='--', label=f'True {param}')
        axs[i].set_xlabel(param)
        axs[i].set_ylabel('LnL')
    plt.tight_layout()
    plt.savefig(f"{outdir}/lnl_1d.png")






    







