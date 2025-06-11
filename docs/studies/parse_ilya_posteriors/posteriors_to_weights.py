import glob
import os
from dataclasses import dataclass

import h5py
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


CLEAN = True  # set to False to keep the data directory clean
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "posteriors")

# from jeff's code
MC_BINS, _ = MakeChirpMassBins()
Z_BINS = np.arange(0.0, MAX_DETECTION_REDSHIFT + REDSHIFT_STEP, REDSHIFT_STEP)[:15]

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

    n_z_bins, n_mc_bins = len(z_bins), len(mc_bins)+1
    weights = np.zeros((n_mc_bins, n_z_bins))

    for mc, z in posterior_samples:
        # # check if the mc and z are within the bins
        # in_mbin = MIN_CHIRPMASS <= mc <= MAX_CHIRPMASS
        # in_zbin = z_bins[0] <= z <= z_bins[-1]
        # if not (in_mbin and in_zbin):
        #     continue

        mc_bin = get_mc_bin_idx(mc, mc_bins)
        z_bin = np.argmin(np.abs(z_bins - z))
        w = 1 / prior.prob(dict(chirp_mass=mc, redshift=z))
        if not np.isnan(w):
            weights[mc_bin, z_bin] += 1 / prior.prob(dict(chirp_mass=mc, redshift=z))

        # set weights to zero if they are not finite
        if not np.isfinite(weights[mc_bin, z_bin]):
            weights[mc_bin, z_bin] = 0

    weights /= len(posterior_samples)

    return weights


@dataclass
class MockPop:
    weights: np.ndarray  # shape (n_events, n_z_bins, n_mc_bins)
    mc_bins = MC_BINS + [MAX_CHIRPMASS+10]  # add the last bin edge
    z_bins = Z_BINS


    def __post_init__(self):
        print("Loaded mock population with shape:", self.weights.shape)

    @classmethod
    def load(cls, ):


        CACHE = os.path.join(DATA_DIR, "mock_population_weights.h5")

        if os.path.exists(CACHE) and not CLEAN:
            with h5py.File(CACHE, 'r') as f:
                weights = f['weights'][:]
            return cls(weights=weights)

        else:


            files = glob.glob(os.path.join(DATA_DIR, "posterior*.dat"))
            n_events = len(files)
            weights = np.zeros((n_events, len(MC_BINS)+1, len(Z_BINS)))
            for i, fname in tqdm(enumerate(files), total=n_events, desc="Loading mock population"):
                weights[i] = get_weights(fname, mc_bins=MC_BINS, z_bins=Z_BINS)

            # CAHCE
            # save the weights to a file for future use
            with h5py.File(CACHE, 'w') as f:
                f.create_dataset('weights', data=weights)





            return cls(weights=weights)

    def plot(self):
        weights = self.weights.copy()
        # compress the weights to 2D by summing over the 0th axis
        for i in range(len(weights)):  # normlise each event
            weights[i] = weights[i] / np.sum(weights[i])
        ax = plot_weights(
            np.nansum(weights, axis=0).T, self.mc_bins, self.z_bins
        )
        Z, MC = np.meshgrid(self.z_bins, self.mc_bins)

        ax.set_yscale('log')

        # for i in range(len(weights)):
        #     add_cntr(ax, Z, MC, weights[i].T)
        fig = ax.get_figure()

        fig.suptitle(
            f"N events {len(weights)}"
        )
        return ax


pop = MockPop.load()
pop.plot()
plt.savefig("mock_population_weights.png", dpi=300, bbox_inches='tight')


# check num of non-zero weights
n_total = np.shape(pop.weights)[0] * np.shape(pop.weights)[1] * np.shape(pop.weights)[2]
num_non_finite_weights = n_total -  np.sum(np.isfinite(pop.weights))
print("Number of non-finite weights:", num_non_finite_weights)

# check values of non-finite weights
non_finite_weights = pop.weights[~np.isfinite(pop.weights)]

print(np.nansum(pop.weights))
