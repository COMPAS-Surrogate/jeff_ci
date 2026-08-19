"""JAX/GPJax active-learning surrogate for the COMPAS log-likelihood."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Callable, Literal

import numpy as np
from scipy.special import ndtr

try:
    import equinox as eqx
    import gpjax as gpx
    import jax
    import jax.numpy as jnp
    import optax
except ImportError as exc:  # pragma: no cover - exercised by installation
    raise ImportError(
        "The LnL surrogate requires its GPJax runtime dependencies. "
        "Install cosmic-integration on Python 3.11+."
    ) from exc


jax.config.update("jax_enable_x64", True)

KernelName = Literal["matern12", "matern32", "matern52", "rbf"]
AcquisitionName = Literal["predictive_variance", "expected_improvement"]


@dataclass(frozen=True)
class JaxGPConfig:
    """Explicit experimental choices for the exact-GP baseline.

    The kernel is deliberately configurable rather than presented as a known
    correct model for the likelihood surface.  Comparisons should use the same
    data and configuration, then assess held-out calibration before promoting a
    configuration to the production path.
    """

    kernel: KernelName = "matern52"
    # Keep the tiny observation noise fixed by default. Treat noise fitting as
    # a separate model comparison because it changes uncertainty calibration.
    initial_noise_stddev: float = float(np.sqrt(1e-5))
    optimise_noise: bool = False
    learning_rate: float = 0.03
    optimisation_steps: int = 250
    candidate_count: int = 4096


@dataclass(frozen=True)
class JaxTrainingData:
    """Numpy-owned observations collected from the non-JAX black-box target."""

    query_points: np.ndarray
    observations: np.ndarray

    def __post_init__(self) -> None:
        x = np.asarray(self.query_points, dtype=np.float64)
        y = np.asarray(self.observations, dtype=np.float64).reshape(-1, 1)
        if x.ndim != 2:
            raise ValueError(
                "query_points must have shape (n_observations, dimension)."
            )
        if y.shape[0] != x.shape[0]:
            raise ValueError(
                "query_points and observations must have the same length."
            )
        object.__setattr__(self, "query_points", x)
        object.__setattr__(self, "observations", y)

    @property
    def n(self) -> int:
        return int(self.query_points.shape[0])


@dataclass(frozen=True)
class FittedJaxGP:
    """A fitted GPJax posterior together with its fixed training data."""

    posterior: object
    data: JaxTrainingData
    objective_history: np.ndarray

    def predict_f(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return latent predictive mean and marginal variance at ``x``."""
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.data.query_points.shape[1]:
            raise ValueError("x must have shape (n_points, dimension).")
        mean, variance = self.predict_f_jax(jnp.asarray(x))
        return (
            np.asarray(mean, dtype=np.float64).reshape(-1, 1),
            np.maximum(np.asarray(variance, dtype=np.float64).reshape(-1, 1), 0.0),
        )

    def predict_f_jax(self, x):
        """Differentiable predictive mean and marginal variance.

        Identical to :meth:`predict_f` but stays in JAX, so gradient-based
        samplers (e.g. NUTS) can differentiate through the surrogate. The numpy
        boundary in ``predict_f`` would otherwise discard those gradients.
        """
        distribution = self.posterior.predict(
            x,
            _as_gpjax_data(self.data),
            return_covariance_type="diagonal",
        )
        return (
            distribution.mean.reshape(-1),
            jnp.maximum(distribution.variance.reshape(-1), 0.0),
        )


@dataclass(frozen=True)
class KernelValidationResult:
    """Held-out predictive metrics for one candidate GP kernel."""

    kernel: KernelName
    rmse: float
    mean_negative_log_predictive_density: float
    coverage_68: float


def compare_kernel_candidates(
    data: JaxTrainingData,
    bounds: np.ndarray,
    *,
    kernels: tuple[KernelName, ...] = (
        "matern12",
        "matern32",
        "matern52",
        "rbf",
    ),
    heldout_fraction: float = 0.2,
    seed: int = 0,
    base_config: JaxGPConfig = JaxGPConfig(),
) -> list[KernelValidationResult]:
    """Rank kernel candidates by deterministic held-out predictive evidence.

    This comparison must be run on a fixed set of transformed COMPAS LnL
    evaluations.  It does not choose a production kernel automatically: the
    metrics quantify interpolation accuracy and uncertainty calibration, but
    do not prove that BO will explore equally well in unobserved regions.
    """
    if data.n < 6:
        raise ValueError(
            "at least six observations are required for validation."
        )
    if not 0.0 < heldout_fraction < 0.5:
        raise ValueError("heldout_fraction must lie in (0, 0.5).")
    if not kernels:
        raise ValueError("at least one candidate kernel is required.")

    rng = np.random.default_rng(seed)
    order = rng.permutation(data.n)
    n_test = max(1, int(round(data.n * heldout_fraction)))
    test_idx, train_idx = order[:n_test], order[n_test:]
    train = JaxTrainingData(
        data.query_points[train_idx], data.observations[train_idx]
    )
    test_x = data.query_points[test_idx]
    test_y = data.observations[test_idx, 0]

    results: list[KernelValidationResult] = []
    for kernel in kernels:
        config = JaxGPConfig(
            kernel=kernel,
            initial_noise_stddev=base_config.initial_noise_stddev,
            optimise_noise=base_config.optimise_noise,
            learning_rate=base_config.learning_rate,
            optimisation_steps=base_config.optimisation_steps,
            candidate_count=base_config.candidate_count,
        )
        model = fit_exact_gp(train, bounds, config=config, seed=seed)
        mean, variance = model.predict_f(test_x)
        mean = mean[:, 0]
        variance = np.maximum(variance[:, 0], 1e-12)
        residual = test_y - mean
        stddev = np.sqrt(variance)
        results.append(
            KernelValidationResult(
                kernel=kernel,
                rmse=float(np.sqrt(np.mean(residual**2))),
                mean_negative_log_predictive_density=float(
                    np.mean(
                        0.5 * np.log(2.0 * np.pi * variance)
                        + residual**2 / (2.0 * variance)
                    )
                ),
                coverage_68=float(np.mean(np.abs(residual) <= stddev)),
            )
        )
    return sorted(
        results, key=lambda result: result.mean_negative_log_predictive_density
    )


def _as_gpjax_data(data: JaxTrainingData) -> gpx.Dataset:
    return gpx.Dataset(
        X=jnp.asarray(data.query_points), y=jnp.asarray(data.observations)
    )


def _make_kernel(config: JaxGPConfig, lengthscales: np.ndarray):
    kernels = {
        "matern12": gpx.kernels.Matern12,
        "matern32": gpx.kernels.Matern32,
        "matern52": gpx.kernels.Matern52,
        "rbf": gpx.kernels.RBF,
    }
    return kernels[config.kernel](
        lengthscale=jnp.asarray(lengthscales), variance=1.0
    )


def fit_exact_gp(
    data: JaxTrainingData,
    bounds: np.ndarray,
    *,
    config: JaxGPConfig = JaxGPConfig(),
    seed: int = 0,
) -> FittedJaxGP:
    """Fit an exact ARD Gaussian GP using GPJax marginal-likelihood training."""
    bounds = np.asarray(bounds, dtype=np.float64)
    if bounds.shape != (2, data.query_points.shape[1]):
        raise ValueError("bounds must have shape (2, dimension).")
    if config.initial_noise_stddev <= 0.0:
        raise ValueError("initial_noise_stddev must be strictly positive.")
    if config.optimisation_steps < 1:
        raise ValueError("optimisation_steps must be positive.")

    widths = bounds[1] - bounds[0]
    if np.any(widths <= 0.0):
        raise ValueError(
            "every upper bound must be greater than its lower bound."
        )

    train_data = _as_gpjax_data(data)
    prior = gpx.gps.Prior(
        kernel=_make_kernel(config, 0.3 * widths),
        mean_function=gpx.mean_functions.Constant(
            constant=jnp.asarray(float(np.mean(data.observations)))
        ),
    )
    likelihood = gpx.likelihoods.Gaussian(
        num_datapoints=data.n,
        obs_stddev=config.initial_noise_stddev,
    )
    posterior = prior * likelihood
    if not config.optimise_noise:
        posterior = eqx.tree_at(
            lambda model: model.likelihood.obs_stddev._unconstrained,
            posterior,
            replace_fn=jax.lax.stop_gradient,
        )
    negative_mll = lambda model, train: -gpx.objectives.conjugate_mll(
        model, train
    )
    fitted, history = gpx.fit(
        model=posterior,
        objective=negative_mll,
        train_data=train_data,
        optim=optax.adam(config.learning_rate),
        key=jax.random.key(seed),
        num_iters=config.optimisation_steps,
        verbose=False,
    )
    return FittedJaxGP(
        posterior=fitted,
        data=data,
        objective_history=np.asarray(history, dtype=np.float64),
    )


class JaxActiveLearner:
    """Experimental random-candidate BO loop backed by an exact GPJax GP.

    The target may remain an ordinary Python function: only GP fitting,
    prediction, and acquisition evaluation are JAX-based.  This makes the
    class applicable while COMPAS/cosmic integration remains I/O-bound.
    """

    def __init__(
        self,
        trainable_function: Callable[..., float],
        bounds: np.ndarray,
        *,
        initial_data_x: np.ndarray | None = None,
        initial_data_y: np.ndarray | None = None,
        initial_points: int = 5,
        random_seed: int = 42,
        config: JaxGPConfig = JaxGPConfig(),
        outdir: str | None = None,
    ) -> None:
        self.trainable_function = trainable_function
        self.bounds = np.asarray(bounds, dtype=np.float64)
        if self.bounds.ndim != 2 or self.bounds.shape[0] != 2:
            raise ValueError("bounds must have shape (2, dimension).")
        self.dim = int(self.bounds.shape[1])
        if np.any(self.bounds[1] <= self.bounds[0]):
            raise ValueError(
                "every upper bound must be greater than its lower bound."
            )
        self.config = config
        self.outdir = Path(outdir) if outdir is not None else None
        self.rng = np.random.default_rng(random_seed)
        self._fit_seed = int(random_seed)

        if (initial_data_x is None) != (initial_data_y is None):
            raise ValueError(
                "initial_data_x and initial_data_y must be supplied together."
            )
        if initial_data_x is None:
            if initial_points < 1:
                raise ValueError("initial_points must be positive.")
            x = self._sample_uniform(initial_points)
            y = self._evaluate(x)
        else:
            x = np.asarray(initial_data_x, dtype=np.float64)
            y = np.asarray(initial_data_y, dtype=np.float64)
        self.data = JaxTrainingData(x, y)
        if self.data.query_points.shape[1] != self.dim:
            raise ValueError(
                "initial_data_x has a different dimension to bounds."
            )
        self.model = self._fit()
        self.history_best = [float(np.min(self.data.observations))]
        self.acquisition_history: list[AcquisitionName] = []

    def _sample_uniform(self, n: int) -> np.ndarray:
        return self.rng.uniform(
            self.bounds[0], self.bounds[1], size=(n, self.dim)
        )

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(
            [self.trainable_function(*row) for row in x], dtype=np.float64
        ).reshape(-1, 1)

    def _fit(self) -> FittedJaxGP:
        model = fit_exact_gp(
            self.data,
            self.bounds,
            config=self.config,
            seed=self._fit_seed,
        )
        self._fit_seed += 1
        return model

    def _choose_point(self, acquisition: AcquisitionName) -> np.ndarray:
        candidates = self._sample_uniform(self.config.candidate_count)
        mean, variance = self.model.predict_f(candidates)
        stddev = np.sqrt(np.maximum(variance[:, 0], 1e-18))
        if acquisition == "predictive_variance":
            scores = variance[:, 0]
        elif acquisition == "expected_improvement":
            improvement = float(np.min(self.data.observations)) - mean[:, 0]
            z = improvement / stddev
            scores = improvement * ndtr(z) + stddev * np.exp(
                -0.5 * z**2
            ) / np.sqrt(2.0 * np.pi)
        else:
            raise ValueError(f"Unknown acquisition: {acquisition}")
        best_index = int(np.argmax(scores))
        return candidates[best_index : best_index + 1]

    def step(self, acquisition: AcquisitionName) -> tuple[np.ndarray, float]:
        """Evaluate one BO-selected point, refit the GP, and return it."""
        point = self._choose_point(acquisition)
        value = self._evaluate(point)
        self.data = JaxTrainingData(
            np.vstack((self.data.query_points, point)),
            np.vstack((self.data.observations, value)),
        )
        self.model = self._fit()
        self.history_best.append(float(np.min(self.data.observations)))
        self.acquisition_history.append(acquisition)
        return point[0], float(value[0, 0])

    def run(
        self,
        total_steps: int,
        *,
        exploration_fraction: float = 2.0 / 3.0,
        steps_per_round: int | None = None,
        round_callback: Callable[[int, "JaxActiveLearner"], None] | None = None,
    ) -> tuple[JaxTrainingData, FittedJaxGP]:
        """Run the simple exploration/EI policy used by the current workflow."""
        if total_steps < 1:
            raise ValueError("total_steps must be positive.")
        if not 0.0 <= exploration_fraction <= 1.0:
            raise ValueError("exploration_fraction must lie in [0, 1].")
        if steps_per_round is not None and steps_per_round < 1:
            raise ValueError("steps_per_round must be positive when supplied.")
        exploration_steps = int(round(total_steps * exploration_fraction))
        for step_idx in range(total_steps):
            acquisition: AcquisitionName = (
                "predictive_variance"
                if step_idx < exploration_steps
                else "expected_improvement"
            )
            self.step(acquisition)
            if steps_per_round is not None and (
                (step_idx + 1) % steps_per_round == 0
                or step_idx + 1 == total_steps
            ):
                self.save_model(round_idx=step_idx // steps_per_round)
                if round_callback is not None:
                    round_callback(step_idx // steps_per_round, self)
        return self.data, self.model

    def save_model(self, *, round_idx: int) -> Path:
        """Persist a fitted GPJax posterior for later surrogate evaluation.

        GPJax posteriors are immutable PyTrees and are safely serializable with
        Python's pickle protocol.  The files are versioned by training round;
        they are versioned by round for later surrogate evaluation.
        """
        if self.outdir is None:
            raise ValueError("outdir is required to save a JAX model.")
        model_dir = self.outdir / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f"round_{int(round_idx)}.pkl"
        with model_path.open("wb") as stream:
            pickle.dump(self.model, stream, protocol=pickle.HIGHEST_PROTOCOL)
        return model_path

    @staticmethod
    def load_model(
        model_dir: str | Path, round_idx: int | None = None
    ) -> FittedJaxGP:
        """Load a GPJax round checkpoint written by :meth:`save_model`."""
        root = Path(model_dir)
        models = sorted(
            root.glob("round_*.pkl"), key=lambda path: path.stat().st_mtime
        )
        if not models:
            raise FileNotFoundError(f"No JAX models found in {root}.")
        model_path = (
            root / f"round_{int(round_idx)}.pkl"
            if round_idx is not None
            else models[-1]
        )
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Model for round {round_idx} does not exist in {root}."
            )
        with model_path.open("rb") as stream:
            model = pickle.load(stream)
        if not isinstance(model, FittedJaxGP):
            raise TypeError(f"Unexpected JAX model payload in {model_path}.")
        return model
