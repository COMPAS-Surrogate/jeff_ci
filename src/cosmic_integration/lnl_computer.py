import csv
import math
import os.path
from dataclasses import dataclass
from typing import List, Optional
from tqdm.auto import tqdm

import numpy as np

from .observation import Observation
from .ratesSampler import BinnedCosmicIntegrator
from .utils import row_to_matrix_params_lnl, _param_str
from .plot_rate import plot_matrix


def ln_poisson_likelihood(
        n_obs: float, n_model: float, ignore_factorial: bool = True
) -> float:
    """
    Computes LnL(N_obs | N_model) = N_obs * ln(N_model) - N_model - ln(N_obs!)

    :param n_obs: number of observed events
    :param n_model: number of events predicted by the model
    :param ignore_factorial: ignore the factorial term in the likelihood

    # TODO: Why are we ignoring the factorial term?? It was in Ilya's notes from 2023, but unsure why...

    :return: the log likelihood
    """
    if n_model <= 0:
        return -np.inf
    lnl = n_obs * np.log(n_model) - n_model

    if ignore_factorial is False:
        lnl += -np.log(math.factorial(int(n_obs)))
    return lnl


def ln_mcz_grid_likelihood_weights(
        obs_weights: np.ndarray, model_prob_grid: np.ndarray
) -> float:
    """
    Computes LnL(mc, z | model)
        =  sum_n  ln sum_i  p(mc_i, z_i | model) * wi,n
        (for N_obs events, and i {mc,z} bins)


    for even_idx in range(n_events):
        p_event = 0
        for mc_idx in range(n_mc):
            for z_idx in range(n_z):
                w = obs_weights[even_idx, mc_idx, z_idx]
                p_event += w * model_prob_grid[mc_idx, z_idx]
        lnl += np.log(p_event)

    """
    p_events = np.einsum("nij,ij->n", obs_weights, model_prob_grid)
    return np.nansum(np.log(p_events))


def ln_mcz_grid_likelihood(
        obs_matrix: np.ndarray, model_matrix: np.ndarray) -> float:
    """
    Computes LnL(mc, z | model)
     = for i in rows:
          for j in columns:
            p_event += obs_matrix[i, j] * model_matrix[i, j]


    :param obs_matrix:
    :param model_matrix:
    :return:
    """
    p_event_matrix = obs_matrix * model_matrix
    ln_prob = np.nansum(np.log(p_event_matrix[p_event_matrix > 0]))
    return ln_prob


def core_ln_likelihood(
        model_matrix: np.ndarray,
        duration: float, # in years
        obs_matrix: np.ndarray,
        obs_weights: Optional[np.ndarray] = None
) -> float:
    """
    Compute the log likelihood of the model given the observation

    :param obs: the observation
    :param model: the McZ-grid model
    :param detailed: return detailed likelihood components

    :return: the log likelihood
    if detailed:
        return [lnl, poisson_lnl, mcz_lnl, model_n_detections]

    """

    # unpack the model into the grid and the number of detections
    n_obs = np.nansum(obs_matrix) * duration
    model_n_obs = np.nansum(model_matrix) * duration

    # compute the likelihood
    poisson_lnl = ln_poisson_likelihood(n_obs, model_n_obs)

    if obs_weights is None:
        mcz_lnl = ln_mcz_grid_likelihood(obs_matrix, model_matrix)
    else:
        mcz_lnl = ln_mcz_grid_likelihood_weights(obs_weights, model_matrix)
    lnl = poisson_lnl + mcz_lnl
    return float(lnl)


@dataclass
class LnLComputer:
    """
    A class to compute the log likelihood of a model given an observation.
    """

    observation: Observation
    model: BinnedCosmicIntegrator
    cache_fn: Optional[str] = None

    def __call__(self, alpha: float, sigma: float, sfr_a: float, sfr_d: float) -> float:
        """
        Compute the log likelihood of the model given the observation.

        :param alpha: alpha parameter for the model
        :param sigma: sigma parameter for the model
        :param sfr_a: SFR amplitude parameter for the model
        :param sfr_d: SFR decay parameter for the model

        :return: log likelihood of the model given the observation
        """
        # Generate the model matrix using the parent class method
        model_matrix = self.model.FindBinnedDetectionRate(
            p_Alpha=alpha,
            p_Sigma=sigma,
            p_SFRa=sfr_a,
            p_SFRd=sfr_d
        )

        # Compute the log likelihood
        lnl = core_ln_likelihood(
            model_matrix=model_matrix,
            obs_matrix=self.observation.rate_matrix,
            duration=self.observation.duration,
            obs_weights=self.observation.weights
        )

        # If cache_fn, store the model params, matrix-shape, matrix, and lnl
        if self.cache_fn:
            _cache_results(self.cache_fn,
                           [alpha, sigma, sfr_a, sfr_d, *model_matrix.shape, *model_matrix.ravel().tolist(), lnl]),

        return lnl


    def plot(self, params, outdir: str = "."):
        """
        Plot the model matrix for the given parameters.

        :param params: List of parameters [alpha, sigma, sfr_a, sfr_d]
        :param outdir: Output directory for the plot
        """
        import matplotlib.pyplot as plt

        model_matrix = self.model.FindBinnedDetectionRate(
            p_Alpha=params[0],
            p_Sigma=params[1],
            p_SFRa=params[2],
            p_SFRd=params[3]
        )
        lnl = core_ln_likelihood(
            model_matrix=model_matrix,
            obs_matrix=self.observation.rate_matrix,
            duration=self.observation.duration,
            obs_weights=self.observation.weights
        )

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        # plt the data matrix on the left, model matrix on the right
        plot_matrix(
            self.observation.rate_matrix,
            params=self.observation.params,
            ax=axes[0],
            label="DATA"
        )
        plot_matrix(
            model_matrix,
            params=params,
            ax=axes[1],
            label="MODEL"
        )
        p_string = _param_str(params)
        fig.suptitle(f"Log Likelihood: {lnl:,.2f}")
        fig.tight_layout()
        fname = os.path.join(outdir, f"lnl_matrix_{p_string}.png")
        fig.savefig(fname)
        plt.close(fig)
        return fname


    @classmethod
    def load(
            cls,
            observation_file: str,
            compas_h5: str,
            cache_fn: Optional[str] = None
    ):
        return cls(
            observation=Observation.from_jeff(observation_file),
            model=BinnedCosmicIntegrator.from_compas_h5(
                inputPath=os.path.dirname(compas_h5),
                inputName=os.path.basename(compas_h5)
            ),
            cache_fn=cache_fn
        )

    def compute_via_cache(self, cache_fn: str) -> List[float]:
        """
        Compute the log likelihood using cached binned detection rates.

        :param cache_fn: Path to the cache file.
        :return: List of log likelihood values.
        """
        if not os.path.exists(cache_fn):
            raise FileNotFoundError(f"Cache file {cache_fn} does not exist.")

        lnls = []
        data = np.genfromtxt(cache_fn, delimiter=',')
        for row in tqdm(data):
            matrix, param, _ = row_to_matrix_params_lnl(row)
            lnl = core_ln_likelihood(
                model_matrix=matrix,
                obs_matrix=self.observation.rate_matrix,
                duration=self.observation.duration,
                obs_weights=self.observation.weights
            )
            lnls.append(np.array([lnl, *param]).flatten())
        return np.array(lnls)


def _cache_results(cache_fn: str, data: List[float]):
    with open(cache_fn, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(data)  # Write the data as a single row
