from __future__ import annotations

import gpflow
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from trieste.data import Dataset


def _build_gpr_model(data: Dataset, bounds: np.ndarray) -> gpflow.models.GPR:
    # Must be strictly above GPflow's positive transform lower bound (often 1e-6),
    # otherwise the unconstrained parameter becomes -inf and TF will error.
    noise = np.float64(1e-5)

    # Set ARD lengthscales proportional to parameter widths to respect units
    widths = (bounds[1] - bounds[0]).astype(np.float64)
    # Start around ~30% of each dimension's width (heuristic)
    init_ls = np.clip(0.3 * widths, 1e-6, np.inf)

    kernel = gpflow.kernels.Matern52(
        variance=np.float64(1.0),
        lengthscales=init_ls,
    )

    observations = data.observations
    if hasattr(observations, "numpy"):
        observations = observations.numpy()
    mean_init = np.float64(np.mean(np.asarray(observations)))
    mean_function = gpflow.mean_functions.Constant(mean_init)

    # LogNormal priors centered on initial values for stability
    prior_scale = tf.constant(0.75, dtype=tf.float64)
    kernel.variance.prior = tfp.distributions.LogNormal(
        tf.constant(-1.0, dtype=tf.float64), prior_scale
    )
    kernel.lengthscales.prior = tfp.distributions.LogNormal(
        tf.math.log(tf.convert_to_tensor(init_ls, dtype=tf.float64)), prior_scale
    )

    # Standard Gaussian likelihood with tiny jitter (deterministic objective)
    x = tf.convert_to_tensor(data.query_points, dtype=tf.float64)
    y = tf.convert_to_tensor(data.observations, dtype=tf.float64)
    model = gpflow.models.GPR(
        data=(x, y),
        kernel=kernel,
        noise_variance=noise,
        mean_function=mean_function,
    )
    gpflow.set_trainable(model.likelihood.variance, False)
    return model


def build_and_optimize_gpr(data: Dataset, bounds: np.ndarray, *, maxiter: int = 1000) -> gpflow.models.GPR:
    model = _build_gpr_model(data, bounds)
    opt = gpflow.optimizers.Scipy()
    opt.minimize(
        model.training_loss,
        variables=model.trainable_variables,
        options={"maxiter": int(maxiter)},
    )
    return model

