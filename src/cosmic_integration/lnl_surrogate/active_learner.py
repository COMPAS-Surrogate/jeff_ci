# active_learner.py

import os
import shutil
from typing import Callable
import glob

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import gpflow
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tqdm import tqdm
from trieste.acquisition import (
    ExpectedImprovement,
    PredictiveVariance,
)
from trieste.acquisition.rule import EfficientGlobalOptimization
from trieste.bayesian_optimizer import BayesianOptimizer, OptimizationResult
from trieste.data import Dataset
from trieste.models.gpflow import GaussianProcessRegression
from trieste.models.utils import get_module_with_variables
from trieste.objectives import mk_observer
from trieste.space import Box

from .plotting import plot_regret, plot_scatter


class ActiveLearner:
    """
    ActiveLearner wraps a Trieste 4.x workflow to build a GP surrogate for a black‐box function `foo`
    in D dimensions, alternating between exploration (PredictiveVariance) and exploitation (ExpectedImprovement).
    After each “round,” it saves diagnostic plots and persists the current GPflow model to disk.

    Example usage:
        def foo(a, b, c, d):
            return np.sin(3*np.pi*a)**2 + (b - 0.5)**2 + 0.5*(c - 0.2)**2 + 0.3*(d - 0.8)**2

        bounds = np.array([[0.0, 0.0, 0.0, 0.0],
                           [1.0, 1.0, 1.0, 1.0]])
        learner = ActiveLearner(
            foo=foo,
            bounds=bounds,
            outdir="out_trieste",
            initial_points=5,
            random_seed=123,
        )
        # Run e.g. 60 total steps, 15 steps per “round” → 4 rounds of (10 explore + 5 exploit)
        final_dataset, final_model = learner.run(total_steps=60, steps_per_round=15)
        # After this, `final_dataset` holds all (x,y) pairs, and `final_model.model` is the GPflow GPR.
    """

    def __init__(
            self,
            trainable_function: Callable[..., float],
            bounds: np.ndarray,
            outdir: str,
            initial_data_x: np.ndarray | None = None,
            initial_data_y: np.ndarray | None = None,
            initial_points: int = 5,
            random_seed: int = 42,
            ## for testing
            true_minima: np.ndarray | None = None,
    ):
        """
        Args:
            trainable_function: Python function f(x1, x2, …, xD) → float
            bounds: np.ndarray shape [2, D], row0 = lower, row1 = upper
            outdir: directory to save plots & checkpoints
            initial_data_x: (optional) precomputed NxD array of input points
            initial_data_y: (optional) precomputed Nx1 array of observations f(X)
            initial_points: how many random seeds to draw if no initial_data is given
            random_seed: seed for reproducible Box sampling
        """
        self.trainable_function = trainable_function
        self.bounds = np.asarray(bounds, dtype=np.float64)
        assert (
                self.bounds.ndim == 2 and self.bounds.shape[0] == 2
        ), "bounds must be a [2, D] array."
        self.dim = self.bounds.shape[1]

        self.outdir = outdir
        os.makedirs(self.outdir, exist_ok=True)

        # ─── 1) Wrap trainable_function as a NumPy→TensorFlow observer ─────────────
        def _f(x: np.ndarray) -> np.ndarray:
            """
            x: shape [N, D], dtype float64 or float32
            returns: shape [N, 1], dtype float64
            """
            x = np.asarray(x, dtype=np.float64)
            out = np.array(
                [self.trainable_function(*row) for row in x],
                dtype=np.float64,
            ).reshape(-1, 1)
            return out

        # mk_observer wraps _f so we can call self.observer(tf.Tensor) directly
        self.observer = mk_observer(_f)

        # ─── 2) Build the Trieste search space ────────────────────────────────────
        self.search_space = Box(self.bounds[0], self.bounds[1])

        # ─── 3) Handle “optional initial data” vs. “random sample” ───────────────
        if (initial_data_x is not None and initial_data_y is not None):
            # 3a) User supplied (X0, Y0). We assume they are NumPy arrays.
            X0_np = np.asarray(initial_data_x, dtype=np.float64)
            Y0_np = np.asarray(initial_data_y, dtype=np.float64).reshape(-1, 1)

            assert X0_np.ndim == 2 and X0_np.shape[1] == self.dim, (
                "If you pass initial_data_x, it must be shape [N, D]."
            )
            assert Y0_np.ndim == 2 and Y0_np.shape[0] == X0_np.shape[0] and Y0_np.shape[1] == 1, (
                "initial_data_y must be shape [N, 1]."
            )

            # Convert to tf.Tensor and build a Dataset
            X0 = tf.convert_to_tensor(X0_np, dtype=tf.float64)
            Y0 = tf.convert_to_tensor(Y0_np, dtype=tf.float64)
            self.current_dataset = Dataset(X0, Y0)
            N_init = X0_np.shape[0]
        else:
            # 3b) No user‐provided data → draw `initial_points` random seeds from the Box
            tf.random.set_seed(random_seed)
            X0 = self.search_space.sample(initial_points)  # dtype=float32 by default
            X0 = tf.cast(X0, tf.float64)  # cast to float64
            self.current_dataset = self.observer(X0)
            N_init = initial_points


        # ─── 4) Build an initial GPflow GPR on those seed data ───────────────────

        def build_model(data: Dataset) -> gpflow.models.GPR:
            noise = 1e-5
            variance0 = tf.math.reduce_variance(data.observations)
            k = gpflow.kernels.Matern52(variance=variance0, lengthscales=[0.2] * self.dim)
            prior_scale = tf.constant(1.0, dtype=tf.float64)
            k.variance.prior = tfp.distributions.LogNormal(tf.constant(-2.0, dtype=tf.float64), prior_scale)
            k.lengthscales.prior = tfp.distributions.LogNormal(tf.math.log(k.lengthscales), prior_scale)
            gpr0 = gpflow.models.GPR(data=data.astuple(), kernel=k, noise_variance=noise)
            return gpr0

        gpr = build_model(self.current_dataset)
        opt = gpflow.optimizers.Scipy()
        opt.minimize(
            gpr.training_loss,
            variables=gpr.trainable_variables,
            options={"maxiter": 1000},
        )

        # Wrap in Trieste model
        self.current_model = GaussianProcessRegression(gpr, num_kernel_samples=100)

        # ─── 5) Create the BayesianOptimizer ───────────────────────────────────
        self.bo = BayesianOptimizer(self.observer, self.search_space)
        self.exploration_rule = EfficientGlobalOptimization(PredictiveVariance(jitter=1e-6))
        self.exploitation_rule = EfficientGlobalOptimization(ExpectedImprovement(search_space=self.search_space))
        self.result = None

        # ─── 6) Initialize best‐so‐far history using N_init seeds ────────────────
        y0_np = self.current_dataset.observations
        self.current_best = float(np.min(y0_np))
        self.history_best = [self.current_best] * N_init

        self.true_minima = true_minima  # Store true minima if provided


    def _update_model_and_dataset(self, result: OptimizationResult):
        self.current_dataset = result.try_get_final_dataset()
        self.current_model = result.try_get_final_model()

        # Update best‐so‐far history
        y_new = float(self.current_dataset.observations.numpy()[-1, 0])
        self.current_best = min(self.current_best, y_new)
        self.history_best.append(self.current_best)

    def _one_bo_step(self, i: int, explore: bool = True):
        rule = self.exploration_rule if explore else self.exploitation_rule
        self.result = self.bo.optimize(
            num_steps=1,
            datasets=self.current_dataset,
            models=self.current_model,
            acquisition_rule=rule,
            fit_model=True,
            fit_initial_model=i == 0,  # Only fit the initial model once
        )
        self._update_model_and_dataset(self.result)

    def run(self, total_steps: int, steps_per_round: int):
        """
        Run active learning for `total_steps`, grouped into “rounds” of length `steps_per_round`.
        Within each round:
          •   explore_steps = round( (2/3) * steps_per_round ) calls that use PredictiveVariance
          •   exploit_steps = steps_per_round − explore_steps calls that use ExpectedImprovement

        After each round, we save:
          (a) a diagnostic plot of best‐so‐far vs. step index
          (b) the GPflow model checkpoint under “{outdir}/model_round_{round_idx}”

        Returns:
            final_dataset: trieste.data.Dataset containing all points (initial + acquired)
            final_model: the final Trieste GaussianProcessRegressionModel
        """
        assert total_steps > 0 and steps_per_round > 0
        num_rounds = total_steps // steps_per_round

        pbar = tqdm(total=total_steps, unit="step")
        step_counter = 0

        for r in range(num_rounds):
            # Compute how many explore vs. exploit in this round
            explore_steps = int(round((2.0 / 3.0) * steps_per_round))
            exploit_steps = steps_per_round - explore_steps

            # ── Explore Phase: PredictiveVariance ───────────────────────────────────────
            for _ in range(explore_steps):
                pbar.set_description(f"Exploring (current best: {self.current_best:.4f})")
                self._one_bo_step(step_counter, explore=True)
                step_counter += 1
                pbar.update(1)

            # ── Exploit Phase: ExpectedImprovement ──────────────────────────────────────
            for _ in range(exploit_steps):
                pbar.set_description(f"Exploiting (current best: {self.current_best:.4f})")
                self._one_bo_step(step_counter, explore=False)
                self._update_model_and_dataset(self.result)
                step_counter += 1
                pbar.update(1)

            # ── End of Round: Save diagnostics + checkpoint ─────────────────────────
            pbar.set_description("Plotting/Checkpointing")
            self._plot_diagnostics(round_idx=r)
            self.save_model(round_idx=r)

        pbar.close()
        return self.current_dataset, self.current_model

    def _plot_diagnostics(self, round_idx: int):
        plot_dir = os.path.join(self.outdir, "plots")
        os.makedirs(plot_dir, exist_ok=True)

        plot_regret(
            self.current_dataset.observations.numpy(),
            self.history_best,
            title=f"Best Observed Value vs. Step (Round {round_idx})",
            fname=os.path.join(plot_dir, f"regret_round_{round_idx}.png"),
        )

        # Also save a scatter plot of the current dataset
        X_np = self.current_dataset.query_points.numpy()
        Y_np = self.current_dataset.observations.numpy()
        plot_scatter(
            X_np,
            bounds=self.bounds,
            labels=[f"p{i}" for i in range(self.dim)],
            title=f"Round {round_idx}",
            fname=os.path.join(plot_dir, f"scatter_round_{round_idx}.png"),
            true_minima=self.true_minima,
        )

    def save_model(self, round_idx: int = None):
        """
        Save the underlying GPflow GPR model under "{outdir}/model_round_{round_idx}/".

        The folder will be removed & overwritten if it already exists.
        """

        model_dir = os.path.join(self.outdir, f"models/round_{round_idx}")

        if os.path.isdir(model_dir):
            shutil.rmtree(model_dir)
        os.makedirs(model_dir)

        gpr_model = self.current_model.model
        module = get_module_with_variables(self.result.try_get_final_model())

        module.predict_f = tf.function(
            gpr_model.predict_f,
            input_signature=[tf.TensorSpec(shape=[None, self.dim], dtype=tf.float64)],
        )
        tf.saved_model.save(module, model_dir)



    @staticmethod
    def load_model(model_dir: str, round_idx: int = None) -> tf.Module:
        """
        Checks for all saved models in the directory and loads the latest one

        If `round_idx` is specified, it will load the model from that specific round.
        """

        models = glob.glob(os.path.join(model_dir, "round_*"))
        if len(models) == 0:
            raise FileNotFoundError(f"No models found in {model_dir}.")

        if round_idx is not None:
            model_path = os.path.join(model_dir, f"round_{round_idx}")
            if model_path not in models:
                raise FileNotFoundError(f"Model for round {round_idx} does not exist in {model_dir}. Available models: {models}")
        else:
            model_path = max(models, key=os.path.getmtime)

        # Load the saved model
        module = tf.saved_model.load(model_path)
        return module
