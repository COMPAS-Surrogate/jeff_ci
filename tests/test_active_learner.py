# test_active_learner.py
# test_active_learner.py

import os
import numpy as np
import pytest
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from cosmic_integration.lnl_surrogate.active_learner import ActiveLearner


def banana(x: float, y: float, a=1, b=100) -> float:
    """
    A 2D “banana-shaped” (Rosenbrock) function.
    Global minimum f(1,1) = 0.
    """
    return (a - x) ** 2 + b * (y - x ** 2) ** 2


def plot_banana_and_surrogate(
    bounds: np.ndarray,
    out_path: str,
    model=None,
    nx: int = 100,
    ny: int = 100,
):
    """
    Plots the 2D banana (Rosenbrock) function over the rectangle given by `bounds`.
    If `model` is None, produces a single-panel image of the analytic function.
    If `model` is a Trieste GaussianProcessRegressionModel, produces a 1×3 figure:
      [0] true banana
      [1] GP surrogate mean
      [2] squared error between true and surrogate

    - bounds: shape [2,2], [[x_min, y_min], [x_max, y_max]]
    - out_path: filename to save the figure
    - nx, ny: grid resolution in x and y
    """
    x_min, y_min = bounds[0]
    x_max, y_max = bounds[1]

    # Build uniform grid
    xs = np.linspace(x_min, x_max, nx)
    ys = np.linspace(y_min, y_max, ny)
    X_grid, Y_grid = np.meshgrid(xs, ys)  # shape [ny, nx]
    XY_flat = np.vstack([X_grid.ravel(), Y_grid.ravel()]).T  # [nx*ny, 2]

    # True banana values
    Z_true = np.array([banana(xi, yi) for xi, yi in XY_flat], dtype=np.float64).reshape(ny, nx)

    norm = LogNorm(vmin=Z_true.min() + 1e-8, vmax=Z_true.max())

    if model is None:
        # Single-panel plot of the analytic banana function
        fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
        im = ax.pcolormesh(xs, ys, Z_true, shading="auto", norm=norm)
        ax.set_title("True Banana Function")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.savefig(out_path)
        plt.close(fig)
        return

    # If a model is provided, compute predictions
    tf_XY = tf.constant(XY_flat, dtype=tf.float64)
    mean_tf, _ = model.model.predict_f(tf_XY)
    Z_pred = mean_tf.numpy().reshape(ny, nx)
    Z_err2 = (Z_true - Z_pred) ** 2

    # Three-panel plot: true | surrogate | squared error
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

    im0 = axes[0].pcolormesh(xs, ys, Z_true, shading="auto", norm=norm)
    axes[0].set_title("True Banana Function")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].pcolormesh(xs, ys, Z_pred, shading="auto", norm=norm)
    axes[1].set_title("GP Surrogate Mean")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].pcolormesh(xs, ys, Z_err2, shading="auto")
    axes[2].set_title("Squared Error $(f_{true} - f_{GP})^2$")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.savefig(out_path)
    plt.close(fig)


def test_active_learner_runs(outdir):
    outdir = os.path.join(outdir, "test_active_learner_runs")
    os.makedirs(outdir, exist_ok=True)

    # 1) Define 2D bounds for (x, y)
    bounds = np.array([[-3, -3], [3, 3]])  # shape [2, 2]

    # 2) Make an initial plot of just the analytic banana function
    analytic_plot_path = os.path.join(outdir, "banana_true.png")
    plot_banana_and_surrogate(bounds=bounds, out_path=analytic_plot_path)
    assert os.path.exists(analytic_plot_path), "Analytic banana plot was not saved."

    # 3) Instantiate ActiveLearner with the 2D banana function
    n_init = 50
    learner = ActiveLearner(
        trainable_function=lambda x, y: banana(x, y),
        bounds=bounds,
        outdir=str(outdir),
        initial_points=n_init,
        random_seed=0,
    )

    # 4) Run with a small total and per-round budget
    total_steps = 30
    steps_per_round = 10
    final_dataset, final_model = learner.run(total_steps=total_steps, steps_per_round=steps_per_round)

    # 5) Confirm the final dataset has the right shape
    qp = final_dataset.query_points.numpy()
    obs = final_dataset.observations.numpy()
    assert qp.shape == (n_init + total_steps, 2)
    assert obs.shape == (n_init + total_steps, 1)

    # 6) Plot analytic vs. GP vs. error
    comparison_plot_path = os.path.join(outdir, "banana_comparison.png")
    plot_banana_and_surrogate(bounds=bounds, out_path=comparison_plot_path, model=final_model)
    assert os.path.exists(comparison_plot_path), "Comparison plot was not saved."
