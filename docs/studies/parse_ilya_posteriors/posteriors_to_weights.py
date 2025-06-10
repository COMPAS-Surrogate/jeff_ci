import glob
import os
from dataclasses import dataclass

import bilby
import matplotlib.pyplot as plt
import numpy as np
from bilby.core.prior import Constraint, PowerLaw
from bilby.gw.prior import UniformInComponentsChirpMass
from tqdm.auto import tqdm

from cosmic_integration.ratesSampler import ChirpMassBin as get_mc_bin_idx
from cosmic_integration.ratesSampler import MAX_DETECTION_REDSHIFT, REDSHIFT_STEP
from cosmic_integration.ratesSampler import MakeChirpMassBins, MAX_CHIRPMASS, MIN_CHIRPMASS
from plotting import plot_weights, add_cntr

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "posteriors")

# from jeff's code
MC_BINS, _ = MakeChirpMassBins()
Z_BINS = np.arange(0.0, MAX_DETECTION_REDSHIFT + REDSHIFT_STEP, REDSHIFT_STEP)

prior = bilby.prior.PriorDict(
    dict(
        chirp_mass=UniformInComponentsChirpMass(name='chirp_mass', minimum=1, maximum=200),
        mass_1=Constraint(name='mass_1', minimum=2, maximum=2000),
        mass_2=Constraint(name='mass_2', minimum=0.1, maximum=2000),
        redshift=PowerLaw(name='redshift', alpha=2, minimum=0.01, maximum=1.5)
    )
)


def get_weights(fname: str, mc_bins: np.array = MC_BINS, z_bins: np.array = Z_BINS) -> np.ndarray:
    posterior_samples = np.loadtxt(fname, usecols=(0, 1))

    n_z_bins, n_mc_bins = len(z_bins), len(mc_bins)
    weights = np.zeros((n_z_bins, n_mc_bins))

    for mc, z in posterior_samples:
        # check if the mc and z are within the bins
        in_mbin = MIN_CHIRPMASS <= mc <= MAX_CHIRPMASS
        in_zbin = z_bins[0] <= z <= z_bins[-1]
        if not (in_mbin and in_zbin):
            continue

        mc_bin = get_mc_bin_idx(mc, mc_bins)
        z_bin = np.argmin(np.abs(z_bins - z))
        w = 1 / prior.prob(dict(chirp_mass=mc, redshift=z))
        if not np.isnan(w):
            weights[z_bin, mc_bin] += 1 / prior.prob(dict(chirp_mass=mc, redshift=z))

    weights /= len(posterior_samples)

    return weights


@dataclass
class MockPop:
    weights: np.ndarray  # shape (n_events, n_z_bins, n_mc_bins)
    mc_bins = MC_BINS
    z_bins = Z_BINS

    @classmethod
    def load(cls, ):

        CACHE = os.path.join(DATA_DIR, "mock_population_weights.npy")

        if os.path.exists(CACHE):
            # load the cached weights
            weights = np.load(CACHE)
            return cls(weights=weights)

        else:


            files = glob.glob(os.path.join(DATA_DIR, "posterior*.dat"))
            n_events = len(files)
            weights = np.zeros((n_events, len(Z_BINS), len(MC_BINS)))
            for i, fname in tqdm(enumerate(files), total=n_events, desc="Loading mock population"):
                weights[i] = get_weights(fname, mc_bins=MC_BINS, z_bins=Z_BINS)

            # CAHCE
            # save the weights to a file for future use
            np.save(os.path.join(DATA_DIR, "mock_population_weights.npy"), weights)



            return cls(weights=weights)

    def plot(self):
        weights = self.weights.copy()
        # compress the weights to 2D by summing over the 0th axis
        for i in range(len(weights)):  # normlise each event
            weights[i] = weights[i] / np.sum(weights[i])
        ax = plot_weights(
            np.nansum(weights, axis=0), self.mc_bins, self.z_bins
        )
        Z, MC = np.meshgrid(self.z_bins, self.mc_bins)
        for i in range(len(weights)):
            add_cntr(ax, Z, MC, weights[i])
        fig = ax.get_figure()

        fig.suptitle(
            f"N events {len(weights)}"
        )
        return ax


pop = MockPop.load()
pop.plot()
plt.savefig("mock_population_weights.png", dpi=300, bbox_inches='tight')
