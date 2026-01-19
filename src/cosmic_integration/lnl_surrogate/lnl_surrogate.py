import logging
import numpy as np
from bilby.core.likelihood import Likelihood
from tqdm.auto import tqdm
from scipy.stats import qmc

from typing import List
from .active_learner import ActiveLearner
from ..lnl_computer import LnLComputer
from ..ratesSampler.ratesSampler import ALPHA_VALUES, SIGMA_VALUES, SFR_A_VALUES, SFR_D_VALUES
from .adaptive_robust_scalar import robust_neg_lnl_computer_factory, AdaptiveRobustScaler

BOUNDS = np.array([
    [np.min(ALPHA_VALUES), np.min(SIGMA_VALUES), np.min(SFR_A_VALUES), np.min(SFR_D_VALUES)],
    [np.max(ALPHA_VALUES), np.max(SIGMA_VALUES), np.max(SFR_A_VALUES), np.max(SFR_D_VALUES)]
])

PARAMETERS = ["alpha", "sigma", "sfr_a", "sfr_d"]  # Parameters to train on


class LnLSurrogate(Likelihood):
    def __init__(
            self,
            gp_model,
            scaler: AdaptiveRobustScaler
    ):
        super().__init__(parameters={param: 0.0 for param in PARAMETERS})  # Initialize with dummy parameters
        self.gp_model = gp_model
        self.scaler = scaler

    @classmethod
    def train(
            cls,
            observation_file: str = None,  # Path to the observation file
            compas_h5: str = None,  # Path to the COMPAS h5 file
            outdir: str = ".",  # Output directory for the learner
            initial_points: int = 50,  # Number of initial points for active learning
            total_steps: int = 300,  # Total number of points to sample
            steps_per_round: int = 30,  # Number of steps per round
            parameters: List[str] = PARAMETERS,  # Parameters to train on
            truth: np.ndarray = None,  # True minima for helping with visualization
            inital_samples: np.ndarray = None,  # Initial samples for the active learner
            initial_lnls: np.ndarray = None,  # Initial log likelihoods for the active learner
            scaler_soft_clipping: bool = True,  # Whether to soft-clip transformed lnL values
            scaler_clip_factor: float = 3.0,  # Clip factor passed to AdaptiveRobustScaler
            scaler_lower_clip_percentile: float | None | str = "auto",  # Floor poor LnL regions
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
            initial_lnls = np.array(
                [lnl_computer(*s) for s in tqdm(inital_samples, desc="Computing initial log likelihoods")])

        stats_msg = f"""Initial LnL statistics:
  Min: {np.min(initial_lnls):,.2f}
  Max: {np.max(initial_lnls):,.2f}
  Median: {np.median(initial_lnls):,.2f}
  Range: {np.max(initial_lnls) - np.min(initial_lnls):,.2f}"""

        logger = logging.getLogger(__name__)
        logger.info(stats_msg)

        # 3. Create negative log-likelihood computer
        lower_clip_value = None
        lower_clip_percentile = None
        if isinstance(scaler_lower_clip_percentile, str):
            if scaler_lower_clip_percentile.lower() == "auto":
                lower_clip_value = float(np.percentile(initial_lnls, 5.0))
            else:
                raise ValueError(f"Unknown scaler_lower_clip_percentile string: {scaler_lower_clip_percentile}")
        else:
            lower_clip_percentile = scaler_lower_clip_percentile

        neg_lnl_computer = robust_neg_lnl_computer_factory(
            lnl_computer,
            initial_lnls,
            soft_clipping=scaler_soft_clipping,
            clip_factor=scaler_clip_factor,
            lower_clip_percentile=lower_clip_percentile,
            lower_clip_value=lower_clip_value,
        )

        # Store reference for later use
        reference_lnl = neg_lnl_computer.scaler.reference_value

        # 4. Bootstrap with best initial point(s) for better starting quality
        log_filename = f"{outdir}/training.log"

        # Set up file handler for this training session
        file_handler = logging.FileHandler(log_filename, mode='a')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        logger.addHandler(file_handler)

        # Log initial statistics
        logger.info("Training started")

        bootstrap_msg = f"Bootstrap: Using {len(initial_lnls)} evaluated points as initial data"
        logger.info(bootstrap_msg)

        # Find the best initial point to ensure we start with high LnL reference
        best_idx = np.argmax(initial_lnls)
        best_point = inital_samples[best_idx]
        best_lnl = initial_lnls[best_idx]
        best_point_msg = f"Best initial point: LnL={best_lnl:.2f} at {best_point}"
        logger.info(best_point_msg)

        # 4. Run active learning
        model_dir = f"{outdir}/gp_model"
        _, model = ActiveLearner(
            trainable_function=neg_lnl_computer,
            bounds=BOUNDS,
            outdir=model_dir,
            initial_data_x=inital_samples,
            initial_data_y=np.array([neg_lnl_computer(*s) for s in inital_samples]),
            true_minima=truth,
        ).run(total_steps=total_steps, steps_per_round=steps_per_round)

        # 5. Save diagnostics
        neg_lnl_computer.scaler.save(model_dir)

        # Log completion
        logger.info("Training completed successfully")
        logger.info(f"Logs saved to {log_filename}")

        return cls(model.model, neg_lnl_computer.scaler)

    @classmethod
    def load(cls, model_dir: str):
        """
        Load the LnLSurrogate model from a saved state.
        """
        model = ActiveLearner.load_model(model_dir)
        return cls(model, AdaptiveRobustScaler.load(f"{model_dir}/../"))

    def log_likelihood(self) -> float:
        params = np.array([list(self.parameters.values())])

        # Get prediction from GP (this is the negative transformed value)
        neg_transformed_lnl, _ = self.gp_model.predict_f(params)
        neg_transformed_lnl = neg_transformed_lnl.numpy().flatten()[0]

        # Convert back to positive transformed value
        transformed_lnl = -neg_transformed_lnl

        # Inverse transform to get back to original log-likelihood space
        original_lnl = self.scaler.inverse_transform(transformed_lnl)

        return original_lnl


def sample_points(n: int = 10, parameters: List[str] = PARAMETERS) -> np.ndarray:
    sampler = qmc.LatinHypercube(d=len(PARAMETERS))
    lhc_samples = sampler.random(n=n // 2)

    # Scale to parameter bounds
    scaled_samples = qmc.scale(lhc_samples, BOUNDS[0], BOUNDS[1])

    # Stage 2: Add some corner/edge cases
    corners = []
    for i in range(min(n // 4, 2 ** len(PARAMETERS))):
        corner = []
        for j, (low, high) in enumerate(BOUNDS.T):
            corner.append(low if (i >> j) & 1 else high)
        corners.append(corner)

    # Stage 3: Add some random samples
    remaining = n - len(scaled_samples) - len(corners)
    if remaining > 0:
        random_samples = np.random.uniform(BOUNDS[0], BOUNDS[1], size=(remaining, len(PARAMETERS)))
        all_samples = np.vstack([scaled_samples, corners, random_samples])
    else:
        all_samples = np.vstack([scaled_samples, corners])
    return all_samples[:n]  # Ensure we only return n samples
