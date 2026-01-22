import gpflow
import tensorflow as tf
import tensorflow_probability as tfp
from trieste.data import Dataset
from trieste.models.gpflow import GaussianProcessRegression
from trieste.models.optimizer import Optimizer

from cosmic_integration.tf_compat import patch_tensorflow_nest_protocol_for_py312_tf216


def test_tf_nest_protocol_patch_allows_trieste_gpflow_optimization():
    """
    Regression test for TF 2.16 + Python 3.12:
      TypeError: this __dict__ descriptor does not support '_TupleWrapper' objects
    """

    patch_tensorflow_nest_protocol_for_py312_tf216()

    x = tf.random.uniform([8, 4], dtype=tf.float64)
    y = tf.reduce_sum(x, axis=1, keepdims=True)
    dataset = Dataset(x, y)

    kernel = gpflow.kernels.Matern52(lengthscales=tf.ones([4], dtype=tf.float64))
    prior_scale = tf.constant(0.75, dtype=tf.float64)
    kernel.variance.prior = tfp.distributions.LogNormal(tf.constant(-1.0, tf.float64), prior_scale)
    kernel.lengthscales.prior = tfp.distributions.LogNormal(tf.math.log(tf.ones([4], tf.float64)), prior_scale)

    model = gpflow.models.GPR(data=(x, y), kernel=kernel, noise_variance=1e-3)
    optimizer = Optimizer(gpflow.optimizers.Scipy(), minimize_args={"options": {"maxiter": 3}})
    tri_model = GaussianProcessRegression(model, optimizer=optimizer, num_kernel_samples=1)

    tri_model.optimize(dataset)

