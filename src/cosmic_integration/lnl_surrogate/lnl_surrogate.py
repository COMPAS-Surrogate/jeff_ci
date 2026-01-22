import logging
import numpy as np
from bilby.core.likelihood import Likelihood
from tqdm.auto import tqdm
from scipy.stats import qmc

from typing import List
from ..lnl_computer import LnLComputer
from ..ratesSampler.parameter_grid import ALPHA_VALUES, SIGMA_VALUES, SFR_A_VALUES, SFR_D_VALUES
from .adaptive_robust_scalar import (
    robust_neg_lnl_computer_factory,
    AdaptiveRobustScaler,
    suggest_lower_clip_value,
)

BOUNDS = np.array([
    [np.min(ALPHA_VALUES), np.min(SIGMA_VALUES), np.min(SFR_A_VALUES), np.min(SFR_D_VALUES)],
    [np.max(ALPHA_VALUES), np.max(SIGMA_VALUES), np.max(SFR_A_VALUES), np.max(SFR_D_VALUES)]
])

PARAMETERS = ["alpha", "sigma", "sfr_a", "sfr_d"]  # Parameters to train on


class LnLSurrogate(Likelihood):
    def __init__(
            self,
            gp_model,
            scaler: AdaptiveRobustScaler,
            *,
            uncertainty_beta: float = 0.0,
    ):
        super().__init__(parameters={param: 0.0 for param in PARAMETERS})  # Initialize with dummy parameters
        self.gp_model = gp_model
        self.scaler = scaler
        self.uncertainty_beta = float(uncertainty_beta)

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
        logger.setLevel(logging.INFO)
        logger.info(stats_msg)

        # 3. Create negative log-likelihood computer
        lower_clip_value = None
        lower_clip_percentile = None
        if isinstance(scaler_lower_clip_percentile, str):
            if scaler_lower_clip_percentile.lower() == "auto":
                try:
                    n_events = int(lnl_computer.observation.population_weights.shape[0])
                except Exception:
                    n_events = 0
                # The MC/Z term can produce extremely poor LnL values when the model assigns
                # near-zero probability to some events. A too-shallow floor (e.g. max-200)
                # makes those regions appear artificially plausible to the surrogate.
                #
                # Use an event-count-scaled floor to keep truly-bad regions far below the peak,
                # while relying on the scaler's soft-clipping to keep the transformed space stable.
                min_delta = max(2.0e4, 200.0 * float(max(n_events, 1)))
                max_delta = max(2.0e5, 900.0 * float(max(n_events, 1)))
                lower_clip_value = suggest_lower_clip_value(
                    initial_lnls,
                    best_fraction=0.05,
                    delta_factor=50.0,
                    min_delta=min_delta,
                    max_delta=max_delta,
                )
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
            focus_fraction=0.05,
            max_scale=10.0,
        )

        # Store reference for later use
        reference_lnl = neg_lnl_computer.scaler.reference_value

        # 4. Bootstrap with best initial point(s) for better starting quality
        log_filename = f"{outdir}/training.log"

        # Set up file handler for this training session
        file_handler = logging.FileHandler(log_filename, mode='a')
        file_handler.setLevel(logging.INFO)
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
        logger.info("Scaler diagnostics: %s", neg_lnl_computer.scaler.get_diagnostics())

        # 4. Run active learning
        model_dir = f"{outdir}/gp_model"
        from .active_learner import ActiveLearner
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

        logger.removeHandler(file_handler)
        file_handler.close()

        return cls(model.model, neg_lnl_computer.scaler)

    @classmethod
    def load(
        cls,
        model_dir: str,
        *,
        uncertainty_beta: float = 0.0,
        round_idx: int | None = None,
    ):
        """
        Load the LnLSurrogate model from a saved state.
        """
        from .active_learner import ActiveLearner
        model = ActiveLearner.load_model(model_dir, round_idx=round_idx)
        return cls(
            model,
            AdaptiveRobustScaler.load(f"{model_dir}/../"),
            uncertainty_beta=uncertainty_beta,
        )

    def log_likelihood(self) -> float:
        params = np.array([list(self.parameters.values())])

        # Get prediction from GP (this is the negative transformed value)
        neg_transformed_lnl, neg_transformed_var = self.gp_model.predict_f(params)
        neg_transformed_lnl = float(neg_transformed_lnl.numpy().reshape(-1)[0])
        neg_transformed_var = float(neg_transformed_var.numpy().reshape(-1)[0])
        neg_transformed_std = float(np.sqrt(max(neg_transformed_var, 0.0)))

        # Optionally penalize predictions in high-uncertainty regions (helps prevent
        # spurious high-LnL modes when sampling with the surrogate).
        if self.uncertainty_beta:
            neg_transformed_lnl = neg_transformed_lnl + self.uncertainty_beta * neg_transformed_std

        # Convert back to positive transformed value
        transformed_lnl = -neg_transformed_lnl

        # Inverse transform to get back to original log-likelihood space
        original_lnl = self.scaler.inverse_transform(transformed_lnl)

        return original_lnl


def sample_points(n: int = 10, parameters: List[str] = PARAMETERS, *, seed: int | None = None) -> np.ndarray:
    indices = [PARAMETERS.index(name) for name in parameters]
    bounds = BOUNDS[:, indices]

    rng = np.random.default_rng(seed)
    sampler = qmc.LatinHypercube(d=len(parameters), seed=seed)
    lhc_samples = sampler.random(n=n // 2)

    # Scale to parameter bounds
    scaled_samples = qmc.scale(lhc_samples, bounds[0], bounds[1])

    # Stage 2: Add some corner/edge cases
    corners = []
    for i in range(min(n // 4, 2 ** len(parameters))):
        corner = []
        for j, (low, high) in enumerate(bounds.T):
            corner.append(low if (i >> j) & 1 else high)
        corners.append(corner)

    # Stage 3: Add some random samples
    remaining = n - len(scaled_samples) - len(corners)
    if remaining > 0:
        random_samples = rng.uniform(bounds[0], bounds[1], size=(remaining, len(parameters)))
        all_samples = np.vstack([scaled_samples, corners, random_samples])
    else:
        all_samples = np.vstack([scaled_samples, corners])
    return all_samples[:n]  # Ensure we only return n samples
