# active_learner.py

import logging
import os
from typing import Callable, Optional

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import gpflow
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from scipy.stats import normaltest
from trieste.acquisition import (
    ExpectedImprovement,
    PredictiveVariance,
)
from trieste.acquisition.rule import EfficientGlobalOptimization
from trieste.bayesian_optimizer import BayesianOptimizer, OptimizationResult
from trieste.data import Dataset
from trieste.models.gpflow import GaussianProcessRegression
from trieste.objectives import mk_observer
from trieste.space import Box

from .active_learning.gp_model import build_and_optimize_gpr
from .active_learning.persistence import load_round_model, save_round_model
from .active_learning.trieste_loop import run_active_learning
from .diagnostics.plots import plot_diagnostics
from ..tf_compat import patch_tensorflow_nest_protocol_for_py312_tf216

patch_tensorflow_nest_protocol_for_py312_tf216()


class ActiveLearner:
    """
    ActiveLearner wraps a Trieste 4.x workflow to build a GP surrogate for a black‐box function `foo`
    in D dimensions, alternating between exploration (PredictiveVariance) and exploitation (ExpectedImprovement).
    After each "round," it saves diagnostic plots and persists the current GPflow model to disk.
    """

    def __init__(
            self,
            trainable_function: Callable[..., float],
            bounds: np.ndarray,
            outdir: str,
            initial_data_x: Optional[np.ndarray] = None,
            initial_data_y: Optional[np.ndarray] = None,
            initial_points: int = 5,
            random_seed: int = 42,
            true_minima: Optional[np.ndarray] = None,
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

        # Enhanced tracking
        self.convergence_history = []
        self.model_uncertainty_history = []
        self.acquisition_values_history = []
        self.acquisition_history = []

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

        # ─── 3) Handle "optional initial data" vs. "random sample" ───────────────
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
        gpr = build_and_optimize_gpr(self.current_dataset, self.bounds)

        # Kernel diagnostics tracking
        self.kernel_diagnostics = {
            'log_likelihood_history': [],
            'lengthscales_history': [],
            'variance_history': [],
            'residual_patterns': []
        }

        # Flag to track if we should consider kernel upgrade
        self.kernel_warning_issued = False

        # Use basic Trieste model setup to avoid shape issues
        # Wrap in Trieste model - keep it simple to avoid shape conflicts
        self.current_model = GaussianProcessRegression(gpr, num_kernel_samples=250)

        # ─── 5) Create the BayesianOptimizer with available acquisition functions ───
        self.bo = BayesianOptimizer(self.observer, self.search_space)

        # Only use acquisition functions that actually exist in Trieste
        self.exploration_rule = EfficientGlobalOptimization(PredictiveVariance(jitter=1e-6))
        self.exploitation_rule = EfficientGlobalOptimization(ExpectedImprovement(search_space=self.search_space))
        self.result = None

        # ─── 6) Initialize best‐so‐far history using N_init seeds ────────────────
        y0_np = self.current_dataset.observations
        self.current_best = float(np.min(y0_np))
        self.history_best = [self.current_best] * N_init

        # Track inverse-transformed (LnL) best values
        scaler = getattr(self.trainable_function, 'scaler', None)
        if scaler is not None:
            # Objective is y = -transform(lnl), so transform(lnl_best) = -min(y).
            self.current_best_lnl = scaler.inverse_transform(-self.current_best)
        else:
            self.current_best_lnl = self.current_best
        self.history_best_lnl = [self.current_best_lnl] * N_init

        self.true_minima = true_minima  # Store true minima if provided

    def _update_model_and_dataset(self, result: OptimizationResult):
        self.current_dataset = result.try_get_final_dataset()
        self.current_model = result.try_get_final_model()

        # Update best‐so‐far history
        y_new = float(self.current_dataset.observations.numpy()[-1, 0])
        self.current_best = min(self.current_best, y_new)
        self.history_best.append(self.current_best)

        # Update inverse-transformed (LnL) best value
        scaler = getattr(self.trainable_function, 'scaler', None)
        if scaler is not None:
            # Objective is y = -transform(lnl); invert using the corresponding transform value.
            lnl_new_scalar = scaler.inverse_transform(-y_new)
        else:
            lnl_new_scalar = y_new
        self.current_best_lnl = lnl_new_scalar
        self.history_best_lnl.append(self.current_best_lnl)

    @property
    def current_log_likelihood(self):
        """Return the current best inverse-transformed log-likelihood value."""
        return self.current_best_lnl

    def _one_bo_step_with_rule(self, i: int, rule, rule_name: str = ""):
        """Enhanced BO step with custom rule and tracking"""
        self.result = self.bo.optimize(
            num_steps=1,
            datasets=self.current_dataset,
            models=self.current_model,
            acquisition_rule=rule,
            fit_model=True,
            fit_initial_model=i == 0,
        )

        # Track which acquisition function was used
        if rule_name:
            self.acquisition_history.append({
                'step': i,
                'rule': rule_name,
                'best_value': self.current_best
            })

        self._update_model_and_dataset(self.result)

    def _adaptive_acquisition_split(self, round_idx: int, steps_per_round: int) -> tuple:
        """
        Adaptively choose exploration vs exploitation based on progress
        """
        logger = logging.getLogger(__name__)
        if round_idx < 2:
            # Early rounds: more exploration
            explore_ratio = 0.8
            logger.info(f"Early exploration phase: {explore_ratio * 100:.0f}% exploration")
        else:
            # Look at recent improvement rate
            recent_improvements = np.diff(self.history_best[-40:]) if len(self.history_best) > 40 else []

            if len(recent_improvements) > 0:
                avg_improvement = np.mean(recent_improvements)
                if avg_improvement < -0.001:  # Still improving well
                    explore_ratio = 0.6
                    logger.info(
                        f"Good progress: {explore_ratio * 100:.0f}% exploration (avg improvement: {avg_improvement:.4f})")
                elif avg_improvement < -0.0001:  # Slow improvement
                    explore_ratio = 0.4
                    logger.info(
                        f"Slow progress: {explore_ratio * 100:.0f}% exploration (avg improvement: {avg_improvement:.4f})")
                else:  # Very slow/no improvement
                    explore_ratio = 0.7  # Back to more exploration
                    logger.info(
                        f"Stagnation detected: {explore_ratio * 100:.0f}% exploration (avg improvement: {avg_improvement:.4f})")
            else:
                explore_ratio = 0.6  # Default

        explore_steps = int(explore_ratio * steps_per_round)
        exploit_steps = steps_per_round - explore_steps

        return explore_steps, exploit_steps

    def run(self, total_steps: int, steps_per_round: int,
            convergence_warnings: bool = True,
            patience: int = 3,
            kernel_diagnostics: bool = True,
            adaptive_acquisition: bool = True,
            round_callback: Optional[Callable[[int, "ActiveLearner"], None]] = None,
            callback_fail_fast: bool = True):
        """
        Run active learning for `total_steps`, grouped into "rounds" of length `steps_per_round`.
        Enhanced with adaptive acquisition strategy and comprehensive diagnostics.
        """
        return run_active_learning(
            self,
            total_steps=total_steps,
            steps_per_round=steps_per_round,
            convergence_warnings=convergence_warnings,
            patience=patience,
            kernel_diagnostics=kernel_diagnostics,
            adaptive_acquisition=adaptive_acquisition,
            round_callback=round_callback,
            callback_fail_fast=callback_fail_fast,
        )

    def _plot_diagnostics(self, round_idx: int):
        plot_dir = os.path.join(self.outdir, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        fname = os.path.join(plot_dir, f"diagnostics_round_{round_idx}.png")
        # If truth is available, compute f(x_true) in the same objective space
        true_fx = None
        if self.true_minima is not None:
            try:
                true_fx = float(self.trainable_function(*self.true_minima))
            except Exception:
                true_fx = None
        plot_diagnostics(
            all_obs=self.current_dataset.observations.numpy(),
            history_best=self.history_best,
            model_uncertainty_history=self.model_uncertainty_history,
            points=self.current_dataset.query_points.numpy(),
            bounds=self.bounds,
            labels=['alpha', 'sigma', 'sfr_a', 'sfr_d'],
            true_minima=self.true_minima,
            true_fx=true_fx,
            fname=fname,
        )


    def save_model(self, round_idx: Optional[int] = None):
        if round_idx is None:
            round_idx = 0
        save_round_model(
            current_model=self.current_model,
            result=self.result,
            outdir=self.outdir,
            round_idx=int(round_idx),
            dim=int(self.dim),
        )

    @staticmethod
    def load_model(model_dir: str, round_idx: int | None = None) -> tf.Module:
        return load_round_model(model_dir, round_idx=round_idx)

    def _diagnose_kernel_adequacy(self) -> dict:
        gpr_model = self.current_model.model
        current_lengthscales = gpr_model.kernel.lengthscales.numpy()
        current_variance = gpr_model.kernel.variance.numpy()
        current_loglik = gpr_model.log_marginal_likelihood().numpy()

        self.kernel_diagnostics['lengthscales_history'].append(current_lengthscales.copy())
        self.kernel_diagnostics['variance_history'].append(float(current_variance))
        self.kernel_diagnostics['log_likelihood_history'].append(float(current_loglik))

        diagnostics = {
            'current_lengthscales': current_lengthscales,
            'current_variance': float(current_variance),
            'current_loglik': float(current_loglik)
        }

        warnings = []
        if np.any(current_lengthscales < 0.01):
            warnings.append("Some lengthscales very small (<0.01) - may need shorter-range kernel")
        if np.any(current_lengthscales > 2.0):
            warnings.append("Some lengthscales very large (>2.0) - may need longer-range kernel")
        if current_variance > 10.0:
            warnings.append("High kernel variance - data may have large-scale trends")

        if len(self.kernel_diagnostics['log_likelihood_history']) > 5:
            recent_logliks = self.kernel_diagnostics['log_likelihood_history'][-5:]
            if np.std(recent_logliks) < 0.1:
                warnings.append("Log-likelihood plateaued - kernel may be insufficient")

        X_np = self.current_dataset.query_points.numpy()
        Y_np = self.current_dataset.observations.numpy()

        if len(X_np) > 20:
            mean_pred, var_pred = gpr_model.predict_f(X_np)
            residuals = Y_np.flatten() - mean_pred.numpy().flatten()
            residual_std = np.std(residuals)
            if residual_std > 2.0:
                warnings.append(f"Large residuals (std={residual_std:.3f}) - kernel may not capture complexity")

            try:
                _, p_value = normaltest(residuals)
                if p_value < 0.01:
                    warnings.append("Non-normal residuals - may need different kernel/likelihood")
            except:
                pass

        diagnostics['warnings'] = warnings
        return diagnostics

    def _print_kernel_diagnostics(self, round_idx: int):
        diag = self._diagnose_kernel_adequacy()
        logger = logging.getLogger(__name__)

        logger.info(f"Kernel Diagnostics (Round {round_idx}):")
        logger.info(f"Log-likelihood: {diag['current_loglik']:.3f}")
        logger.info(f"Kernel variance: {diag['current_variance']:.3f}")
        logger.info(f"Lengthscales: {diag['current_lengthscales']}")

        if diag['warnings'] and not self.kernel_warning_issued:
            logger.warning("KERNEL WARNINGS:")
            for warning in diag['warnings']:
                logger.warning(f"• {warning}")

            if len(diag['warnings']) >= 3:
                logger.warning("SUGGESTION: Consider upgrading to multi-scale kernel!")
                logger.warning("Current simple kernel may be inadequate for your likelihood surface")
                self.kernel_warning_issued = True

    def _compute_model_uncertainty(self) -> float:
        test_points = self.search_space.sample(100)
        mean, var = self.current_model.model.predict_f(test_points)
        return float(np.mean(np.sqrt(var.numpy())))

    def _check_convergence(self, window_size: int = 20, tolerance: float = 1e-3) -> bool:
        if len(self.history_best) < window_size:
            return False
        recent_best = self.history_best[-window_size:]
        improvement = abs(recent_best[0] - recent_best[-1])
        relative_improvement = improvement / abs(recent_best[0]) if abs(recent_best[0]) > 1e-10 else improvement
        return relative_improvement < tolerance

    def _print_acquisition_summary(self):
        if not self.acquisition_history:
            return
        rule_counts = {}
        for entry in self.acquisition_history:
            rule = entry['rule']
            rule_counts[rule] = rule_counts.get(rule, 0) + 1

        logger = logging.getLogger(__name__)
        logger.info("Acquisition Function Summary:")
        for rule, count in rule_counts.items():
            percentage = 100 * count / len(self.acquisition_history)
            logger.info(f"{rule}: {count} steps ({percentage:.1f}%)")
