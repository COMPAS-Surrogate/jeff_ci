import os
import sys
import time

import numpy as np
import click

from cosmic_integration.lnl_surrogate.run_sampler import get_prior
from cosmic_integration.ratesSampler import BinnedCosmicIntegrator
from cosmic_integration.utils import _cache_results, _param_str

PRIOR = get_prior()


@click.command()
@click.argument('compas_h5', type=click.Path(exists=True, dir_okay=False))
@click.option('-s', '--seed', default=None, type=int, help='Random seed for sampling')
@click.option('-o', '--out_csv', default=None, type=click.Path(dir_okay=False))
def main(compas_h5, seed: int = None, out_csv: str = None):
    if seed is None:
        seed = np.random.randint(0, 2**31 - 1)

    if out_csv is None:
        out_csv = f"rates_out_{os.path.basename(compas_h5)}_seed{seed}.csv"
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)



    print("__ RUNNING BINNED COSMIC INTEGRATOR __")
    print(f">>> Using COMPAS H5 file: {compas_h5}")
    print(f">>> Output CSV file: {out_csv}")
    print(f">>> Random seed: {seed}")
    print("--------------------------------------")


    ci_runner = BinnedCosmicIntegrator.from_compas_fpath(compas_h5)
    np.random.seed(seed)

    total_time = 0.0
    count = 0

    while True:
        t0 = time.time()
        params = PRIOR.sample()

        param_list = [
            params['alpha'],
            params['sigma'],
            params['sfr_a'],
            params['sfr_d']
        ]

        matrix = ci_runner.FindBinnedDetectionRate(*param_list)

        _cache_results(
            cache_fn=out_csv,
            data=[*param_list, *matrix.shape, *matrix.ravel().tolist()]
        )

        t1 = time.time()

        itr_time = t1 - t0
        total_time += itr_time
        avg_time = total_time / (count + 1)
        # format time into "HH:MM:SS" format
        tf = time.strftime("%H:%M:%S", time.gmtime(total_time))

        param_str = _param_str(param_list)
        param_str = param_str.replace("_", " ")

        print(
            f"\r{tf} | Itr{count:04d} | "
            f"{param_str} | "
            f"{avg_time:.2f}s/itr",
            flush=True,
            end="",
        )

        count += 1
