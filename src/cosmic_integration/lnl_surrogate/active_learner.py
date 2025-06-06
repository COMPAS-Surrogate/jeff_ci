# active_learner.py

import os
import numpy as np
import tensorflow as tf
import gpflow
import matplotlib.pyplot as plt
from typing import Callable

import tensorflow_probability as tfp
from tqdm import tqdm

from trieste.space import Box
from trieste.data import Dataset
from trieste.models.gpflow import GaussianProcessRegression
from trieste.acquisition.rule import EfficientGlobalOptimization
from trieste.bayesian_optimizer import BayesianOptimizer
from trieste.objectives import mk_observer

from trieste.models.utils import get_module_with_variables

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
        initial_points: int = 5,
        random_seed: int = 42,
    ):
        """
        Args:
            foo: Callable[[float, …], float], a Python function taking D floats → scalar float.
            bounds: np.ndarray of shape [2, D]. Row 0 = lower bounds, row 1 = upper bounds.
            outdir: str, directory in which to save diagnostic plots & GP‐checkpoints.
            initial_points: int, how many seed points to acquire before starting active learning.
            random_seed: int, for reproducible Box sampling of the initial points.
        """
        self.trainable_function = trainable_function
        self.bounds = np.asarray(bounds, dtype=np.float64)
        assert (
            self.bounds.ndim == 2 and self.bounds.shape[0] == 2
        ), "bounds must be a [2, D] array."
        self.dim = self.bounds.shape[1]

        # Create output folder if it does not exist
        self.outdir = outdir
        os.makedirs(self.outdir, exist_ok=True)

        # ─── 1) Wrap your Python `foo` as an observer ─────────────

        # Define a “NumPy” version that takes a 2D array [N, D] and returns [N, 1]
        def _f(x: np.ndarray) -> np.ndarray:
            """
            x: shape [N, D], dtype float64 or float32 (we’ll cast internally).
            Returns: shape [N, 1], dtype float64
            """
            # Ensure x is float64
            x = np.asarray(x, dtype=np.float64)
            # Apply `foo` row-by-row
            output = np.array([self.trainable_function(*row) for row in x], dtype=np.float64).reshape(-1, 1)
            return output

        # mk_observer takes your NumPy-based f and returns a TF-friendly Observer
        self.observer = mk_observer(_f)

        # ─── 2) Build the Trieste search space: a D‐dimensional Box ─────────────────────
        self.search_space = Box(self.bounds[0], self.bounds[1])

        # ─── 3) Sample `initial_points` uniformly from Box, then call `observer` ────────
        tf.random.set_seed(random_seed)
        # Box.sample returns a Tensor of shape [initial_points, D], dtype float32 by default.
        X0 = self.search_space.sample(initial_points)
        self.current_dataset = self.observer(X0)

        # ─── 4) Build an initial GPflow GPR on the seed data ───────────────────────────
        kernel = gpflow.kernels.Matern52()
        gpr = gpflow.models.GPR(
            data=(self.current_dataset.query_points, self.current_dataset.observations),
            kernel=kernel,
            mean_function=None,
        )

        def build_model(data):
            variance = tf.math.reduce_variance(data.observations)
            kernel = gpflow.kernels.Matern52(variance=variance, lengthscales=[0.2, 0.2])
            prior_scale = tf.cast(1.0, dtype=tf.float64)
            kernel.variance.prior = tfp.distributions.LogNormal(
                tf.cast(-2.0, dtype=tf.float64), prior_scale
            )
            kernel.lengthscales.prior = tfp.distributions.LogNormal(
                tf.math.log(kernel.lengthscales), prior_scale
            )
            gpr = gpflow.models.GPR(data.astuple(), kernel, noise_variance=1e-5)
            # gpflow.set_trainable(gpr.likelihood, False)

            # return GaussianProcessRegression(gpr, num_kernel_samples=100)
            return gpr
        #
        gpr = build_model(self.current_dataset)


        # Optimize hyperparameters once on the seed data
        opt = gpflow.optimizers.Scipy()
        opt.minimize(
            gpr.training_loss,
            variables=gpr.trainable_variables,
            options={"maxiter": 1000},
        )

        # Wrap it for Trieste
        self.current_model = GaussianProcessRegression(gpr, num_kernel_samples=100)

        # ─── 5) Create the Trieste BayesianOptimizer ───────────────────────────────────
        self.bo = BayesianOptimizer(self.observer, self.search_space)

        # ─── 6) Track best‐so‐far values (for plotting diagnostics) ───────────────────
        y0_np = self.current_dataset.observations.flatten()
        self.current_best = float(np.min(y0_np))  # assume minimization
        # Fill history_best with as many copies as there are initial_points,
        # so the x‐axis “step index” starts from 0 through (initial_points−1).
        self.history_best = [self.current_best] * initial_points

        # A small flag so we only skip “fit_initial_model” on the very first call
        self._did_initial_fit = False
        self.result = None  # Will hold the final optimization result

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
        remainder = total_steps % steps_per_round
        if remainder:
            print(
                f"Warning: total_steps={total_steps} not divisible by "
                f"steps_per_round={steps_per_round}. "
                f"Remainder {remainder} will be run as a final “explore‐only” mini‐round."
            )

        pbar = tqdm(total=total_steps, unit="step")
        step_counter = 0

        for r in range(num_rounds):
            # Compute how many explore vs. exploit in this round
            explore_steps = int(round((2.0 / 3.0) * steps_per_round))
            exploit_steps = steps_per_round - explore_steps

            # ── Explore Phase: PredictiveVariance ───────────────────────────────────────
            for _ in range(explore_steps):
                pbar.set_description("Exploring")
                result = self.bo.optimize(
                    num_steps=1,
                    datasets= self.current_dataset,
                    models=self.current_model,
                    acquisition_rule=EfficientGlobalOptimization(),  # PredictiveVariance
                    fit_model=True,
                    fit_initial_model=not self._did_initial_fit,
                )
                # After the first call, we should set this False so that subsequent calls only re‐fit:
                self._did_initial_fit = True

                # Extract the updated dataset & model
                self.current_dataset = result.try_get_final_dataset()
                self.current_model = result.try_get_final_model()

                # Update best‐so‐far history
                y_new = float(self.current_dataset.observations.numpy()[-1, 0])
                self.current_best = min(self.current_best, y_new)
                self.history_best.append(self.current_best)

                step_counter += 1
                pbar.update(1)

            # ── Exploit Phase: ExpectedImprovement ──────────────────────────────────────
            for _ in range(exploit_steps):
                pbar.set_description("Exploiting")
                self.result = self.bo.optimize(
                    num_steps=1,
                    datasets= self.current_dataset,
                    models= self.current_model,
                    acquisition_rule=EfficientGlobalOptimization(),  # ExpectedImprovement
                    fit_model=True,
                    fit_initial_model=False,
                )
                # Extract updated dataset & model
                self.current_dataset = self.result.try_get_final_dataset()
                self.current_model = self.result.try_get_final_model()

                # Update best‐so‐far history
                y_new = float(self.current_dataset.observations.numpy()[-1, 0])
                self.current_best = min(self.current_best, y_new)
                self.history_best.append(self.current_best)

                step_counter += 1
                pbar.update(1)

            # ── End of Round: Save diagnostics + checkpoint ─────────────────────────
            pbar.set_description("Plotting/Checkpointing")
            self._plot_diagnostics(round_idx=r)
            self._save_model(round_idx=r)

        # ── If there is a remainder, treat it as “explore‐only” steps ───────────────
        if remainder:
            for _ in range(remainder):
                pbar.set_description("Exploring (final)")
                self.result = self.bo.optimize(
                    num_steps=1,
                    datasets= self.current_dataset,
                    models=self.current_model,
                    acquisition_rule=EfficientGlobalOptimization(),
                    fit_model=True,
                    fit_initial_model=False,
                )
                self.current_dataset = self.result.try_get_final_dataset()
                self.current_model = self.result.try_get_final_model()

                y_new = float(self.current_dataset.observations.numpy()[-1, 0])
                self.current_best = min(self.current_best, y_new)
                self.history_best.append(self.current_best)

                step_counter += 1
                pbar.update(1)

            pbar.set_description("Plotting/Checkpointing (final)")
            self._plot_diagnostics(round_idx=num_rounds)
            self._save_model(round_idx=num_rounds)

        pbar.close()
        return self.current_dataset, self.current_model

    def _plot_diagnostics(self, round_idx: int):
        """
        Save a plot of “step index vs. best‐so‐far f(x)” up to this point.

        File → "{outdir}/diagnostic_round_{round_idx}.png"
        """
        fig, ax = plt.subplots(figsize=(6, 4))
        x_axis = np.arange(len(self.history_best))
        y_axis = np.array(self.history_best)

        ax.plot(x_axis, y_axis, marker="o", linestyle="-")
        ax.set_title(f"Round {round_idx} — Best Observed Value vs. Step")
        ax.set_xlabel("Step Index")
        ax.set_ylabel("Best Observed f(x)")
        ax.grid(True)

        out_file = os.path.join(self.outdir, f"diagnostic_round_{round_idx}.png")
        fig.tight_layout()
        fig.savefig(out_file)
        plt.close(fig)

    def _save_model(self, round_idx: int):
        """
        Save the underlying GPflow GPR model under "{outdir}/model_round_{round_idx}/".

        The folder will be removed & overwritten if it already exists.
        """
        model_dir = os.path.join(self.outdir, f"model_round_{round_idx}")
        if os.path.isdir(model_dir):
            import shutil

            shutil.rmtree(model_dir)

        os.makedirs(model_dir, exist_ok=True)
        # `self.current_model` is a Trieste GaussianProcessRegressionModel
        gpr_model = self.current_model.model  # this is the gpflow.models.GPR
        # self.result.save(model_dir, )


        module = get_module_with_variables(self.result.try_get_final_model())
        module.predict = tf.function(
            gpr_model.predict_f,
            input_signature=[tf.TensorSpec(shape=[None, 2], dtype=tf.float64)],
        )
        tf.saved_model.save(module, model_dir)






        # ─▶  Later you can reload it via:
        #        # load the results
        # saved_result = trieste.bayesian_optimizer.OptimizationResult.from_path(  # type: ignore
        #     "results_path"
        # )
        # saved_result.try_get_final_model().model

