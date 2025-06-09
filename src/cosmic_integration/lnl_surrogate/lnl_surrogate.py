import numpy as np
from bilby.core.likelihood import Likelihood
from bilby.core.prior import PriorDict, Uniform, DeltaFunction
from tqdm.auto import tqdm

from typing import List
from .active_learner import ActiveLearner
from ..lnl_computer import LnLComputer
from ..ratesSampler import ALPHA_VALUES, SIGMA_VALUES, SFR_A_VALUES, SFR_D_VALUES

BOUNDS = np.array([
    [np.min(ALPHA_VALUES), np.min(SIGMA_VALUES), np.min(SFR_A_VALUES), np.min(SFR_D_VALUES)],
    [np.max(ALPHA_VALUES), np.max(SIGMA_VALUES), np.max(SFR_A_VALUES), np.max(SFR_D_VALUES)]
])



PARAMETERS = ["alpha", "sigma", "sfr_a", "sfr_d"]  # Parameters to train on

class LnLSurrogate(Likelihood):
    def __init__(
            self,
            gp_model,
            reference_lnl: float  = 0  # Reference log likelihood for normalization
    ):
        super().__init__(parameters={param: 0.0 for param in PARAMETERS})  # Initialize with dummy parameters
        self.gp_model = gp_model
        self.reference_lnl = reference_lnl

    @classmethod
    def train(
            cls,
            observation_file: str = None,  # Path to the observation file
            compas_h5: str = None,  # Path to the COMPAS h5 file
            outdir: str = ".",  # Output directory for the learner
            initial_points: int = 50,  # Number of initial points for active learning
            total_steps: int = 300,  # Total number of points to sample
            steps_per_round: int = 30,  # Number of steps per round
            parameters:List[str] = PARAMETERS,  # Parameters to train on
            truth: np.ndarray  = None,  # True minima for helping with visualization
            threshold: float = 10.0,  # Threshold for negative log likelihood
            inital_samples: np.ndarray = None,  # Initial samples for the active learner
            initial_lnls: np.ndarray = None  # Initial log likelihoods for the active learner
    ) -> "LnLSurrogate":
        """
        Train the LnLSurrogate model.
        """

        # 1. create the LnlComputer instance
        lnl_computer = LnLComputer.load(
            observation_file=observation_file,
            compas_h5=compas_h5,
            cache_fn=f"{outdir}/lnl_cache.csv"  # Cache file for storing results
        )

        # 2. sample initial points
        if inital_samples is None or initial_lnls is None:
            inital_samples = sample_points(initial_points, parameters)
            initial_lnls = np.array([lnl_computer(*s) for s in tqdm(inital_samples, desc="Computing initial log likelihoods")])
        reference_lnl = max(initial_lnls)  # Reference log likelihood for normalization
        print(f"Reference log likelihood: {reference_lnl:,.2f}")



        # 3. trainable function for minimizing the log likelihood

        def neg_lnl_computer(*params):
            """
            Compute the negative log likelihood for the given parameters.
            This is used to train the surrogate model.
            """
            params = np.array(params).flatten()
            neg_lnl = -(lnl_computer(*params) - reference_lnl)

            # threshold
            if neg_lnl > threshold:
                print(f"Negative log likelihood is negative: {neg_lnl:.2f} for params {params}")
                neg_lnl = threshold

            return neg_lnl



        _, model = ActiveLearner(
            trainable_function=neg_lnl_computer,
            bounds=BOUNDS,  # Assuming bounds are defined in LnlComputer
            outdir=f"{outdir}/gp_model",  # Output directory for the learner
            initial_data_x=inital_samples,  # Initial samples
            initial_data_y=initial_lnls,  # Initial log likelihoods
            true_minima=truth,  # True minima for visualization
        ).run(total_steps=total_steps, steps_per_round=steps_per_round)

        # best param
        best_params = model.get_best_params()
        lnl_computer.plot(best_params, outdir=f"{outdir}/gp_model")

        return cls(model.model, reference_lnl)

    @classmethod
    def load(cls, model_dir: str ):
        """
        Load the LnLSurrogate model from a saved state.
        """
        model = ActiveLearner.load_model(model_dir)
        return cls(model)


    def log_likelihood(self) -> float:
        params = np.array([list(self.parameters.values())])
        neg_lnl, _ = self.gp_model.predict_f(params)
        neg_lnl = neg_lnl.numpy().flatten()[0]
        # need to add the reference_lnl to the negative log likelihood
        lnl = neg_lnl + self.reference_lnl
        return lnl



def get_prior(parameters:List[str]=PARAMETERS, truth:np.ndarray=None) -> PriorDict:
    """
    Get the prior distribution for the parameters.
    """
    prior = {}

    for i, param_name in enumerate(PARAMETERS):

        if param_name in parameters:
            prior[param_name] = Uniform(*BOUNDS.T[i])
        else:
            if truth is not None:
                prior[param_name] = DeltaFunction(truth[i])
            else:
                prior[param_name] = Uniform(np.mean(BOUNDS.T[i]))

    return PriorDict(prior)


def sample_points(n: int = 10, parameters:List[str]=PARAMETERS, truth:np.ndarray=None) -> np.ndarray:
    """
    Sample points from the prior distribution.

    """
    samples = get_prior(parameters=parameters, truth=truth).sample(n)
    samples = np.array([s for s in samples.values()]).T
    return samples