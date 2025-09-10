import corner
import matplotlib.pyplot as plt
import numpy as np


def plot_regret(all_obs, history_best, title: str = "Best Observed Value vs. Step", fname: str = "regret.png", ):
    """
    Plot the best point found by the active learner over iterations.

    This function is a placeholder and should be implemented to visualize
    how the best point changes as the active learning process iterates.
    """
    fig, axs = plt.subplots(1,2, figsize=(6, 4))
    x_axis = np.arange(len(history_best))
    y_axis = np.array(history_best)

    ax = axs[0]
    ax.plot(x_axis, y_axis, marker="o", linestyle="-")
    ax.set_title(title)
    ax.set_xlabel("Step Index")
    ax.set_ylabel("Best Observed f(x)")
    ax.grid(True)

    ax = axs[1]
    ax.hist(all_obs, bins=50, density=True, alpha=0.7, color='blue')
    ax.set_title("Distribution of Observed Values")
    ax.set_xlabel("Observed Value")

    fig.tight_layout()
    fig.savefig(fname)
    plt.close(fig)


def plot_scatter(points, bounds, labels=None, title='scatter', fname: str = "scatter_plot.png", true_minima: np.ndarray= None):
    if labels is None:
        labels = [f"p{i}" for i in range(points.shape[1])]
    fig = corner.corner(
        points,
        labels=labels,
        range=[(x[0], x[1]) for x in bounds.T],
        colorbar=True,
        show_titles=True,
        title_kwargs={"fontsize": 12},
        plot_datapoints=True,
        plot_density=True,
        plot_contours=False,
        fill_contours=False,
        truths= true_minima,
    )
    fig.suptitle(title, fontsize=16)
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
