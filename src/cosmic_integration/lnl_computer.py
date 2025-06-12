import csv
import math
import os.path
from dataclasses import dataclass
from typing import List, Optional
from tqdm.auto import tqdm

import numpy as np

from .observation import Observation
from .ratesSampler import BinnedCosmicIntegrator
from .utils import row_to_matrix_params_lnl, _param_str, _cache_results
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
    """
    n_events, n_mc_bins, n_z_bins = obs_weights.shape

    p_events = np.zeros(n_events)  # Initialize an array to store the probabilities for each event
    for event_idx in range(n_events):
        w = obs_weights[event_idx, :, :]
        # normalize the weights for this event
        w = w / np.nansum(w) if np.nansum(w) > 0 else np.zeros_like(w)
        p_event = np.nansum(w * model_prob_grid)
        p_events[event_idx] = p_event

    return np.nansum(np.log(p_events))




def core_ln_likelihood(
        model_matrix: np.ndarray,
        duration: float, # in years
        obs_weights: np.ndarray = None
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

    # scale the model matrix by the duration (atm duartion is 1 year)
    model_matrix = model_matrix * duration


    # unpack the model into the grid and the number of detections
    n_obs = obs_weights.shape[0]  # number of events

    model_n_obs = np.nansum(model_matrix)

    # compute the likelihood
    poisson_lnl = ln_poisson_likelihood(n_obs, model_n_obs)

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
            duration=self.observation.duration,
            obs_weights=self.observation.weights
        )

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        if self.observation.rate_matrix is None:
            obs_matrix = np.nansum(self.observation.weights, axis=0)
            n_events = self.observation.weights.shape[0]
        else:
            obs_matrix = self.observation.rate_matrix
            n_events = np.nansum(obs_matrix)


        # plt the data matrix on the left, model matrix on the right
        plot_matrix(
            obs_matrix,
            params=self.observation.params,
            ax=axes[0],
            label="DATA",
            n_events=n_events
        )
        plot_matrix(
            model_matrix * self.observation.duration,  # scale by duration
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
            cache_fn: Optional[str] = None,
    ):
        return cls(
            observation=Observation.from_ilya(observation_file),
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


