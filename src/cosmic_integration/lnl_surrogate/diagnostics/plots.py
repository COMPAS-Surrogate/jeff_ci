import os
import numpy as np
import matplotlib.pyplot as plt


def scatter_matrix(points,
                   labels,
                   bounds=None,
                   true_minima=None,
                   values=None,
                   ax=None,
                   cmap="viridis"):
    """
    Draw a lower-triangle + diagonal scatter/hist matrix inside `ax`'s cell
    (if ax provided) or as a standalone figure (if ax is None).

    - `points`: (N, D)
    - `labels`: list of length D
    - `bounds`: array-like shape (2, D) giving [lower_row, upper_row]
    - `true_minima`: length-D array of minima (optional)
    - `values`: length-N array used to color scatter points (optional)
    - returns (fig, axes) where axes is (D, D) array of Axes
    """
    points = np.asarray(points)
    n_params = points.shape[1]
    if labels is None:
        labels = [f"p{i}" for i in range(n_params)]

    # Find observed minimum
    observed_minima = None
    if values is not None:
        best_idx = np.argmin(values)
        observed_minima = points[best_idx]

    if ax is None:
        fig, axes = plt.subplots(n_params, n_params, figsize=(3 * n_params, 3 * n_params))
    else:
        fig = ax.figure
        # subdivide the parent axis cell into an n_params x n_params grid
        parent_spec = ax.get_subplotspec()
        gs = parent_spec.subgridspec(n_params, n_params, wspace=0.05, hspace=0.05)
        axes = np.empty((n_params, n_params), dtype=object)
        for i in range(n_params):
            for j in range(n_params):
                axes[i, j] = fig.add_subplot(gs[i, j])

    sc = None
    # draw lower triangle + diagonal
    for i in range(n_params):
        for j in range(n_params):
            axij = axes[i, j]

            # upper triangle: keep free for things like the colorbar
            if i < j:
                axij.axis("off")
                continue

            if i == j:
                # diagonal: histogram with enhanced information
                counts, bins, patches = axij.hist(points[:, i], bins=30, alpha=0.7,
                                                  density=True, color='lightblue')

                # Add vertical lines for minima
                if true_minima is not None:
                    axij.axvline(true_minima[i], color="red", linestyle="--",
                                 linewidth=1.5, label="True min" if i == 0 else "")
                if observed_minima is not None:
                    axij.axvline(observed_minima[i], color="blue", linestyle=":",
                                 linewidth=1.5, label="Best obs" if i == 0 else "")

                # Remove y-ticks for histograms
                axij.set_yticks([])

            else:
                # lower triangle: scatter (colored by `values` if provided)
                if values is not None:
                    sc = axij.scatter(points[:, j], points[:, i],
                                      c=values, cmap=cmap, s=18, alpha=0.8, edgecolors="none")
                else:
                    axij.scatter(points[:, j], points[:, i], s=18, alpha=0.6)

                # Mark true minima with smaller red X
                if true_minima is not None:
                    axij.scatter(true_minima[j], true_minima[i],
                                 marker="x", s=80, c="red", linewidths=2,
                                 label="True minimum" if i == n_params - 1 and j == 0 else "")

                # Mark observed minima with smaller blue X
                if observed_minima is not None:
                    axij.scatter(observed_minima[j], observed_minima[i],
                                 marker="x", s=80, c="blue", linewidths=2,
                                 label="Best observed" if i == n_params - 1 and j == 0 else "")

            # tick label logic: only show bottom x-labels and left y-labels
            show_xlabels = (i == n_params - 1)
            show_ylabels = (j == 0)

            axij.tick_params(axis="x", labelbottom=show_xlabels, bottom=show_xlabels, labelsize=8)
            axij.tick_params(axis="y", labelleft=show_ylabels, left=show_ylabels, labelsize=8)

            # add axis labels only on the outer edges
            if show_xlabels:
                axij.set_xlabel(labels[j], fontsize=9)
            if show_ylabels:
                axij.set_ylabel(labels[i], fontsize=9)

            # ranges
            if bounds is not None:
                try:
                    axij.set_xlim(bounds[0, j], bounds[1, j])
                    axij.set_ylim(bounds[0, i], bounds[1, i])
                except Exception:
                    # ignore if bounds shape is unexpected
                    pass

    # Find available upper triangle cells for legend and colorbar
    available_cells = []
    for i in range(n_params):
        for j in range(n_params):
            if i < j:
                available_cells.append((i, j))

    # Add legend and information panel in separate cells
    if len([x for x in [true_minima, observed_minima, values] if x is not None]) > 0 and available_cells:
        # Create legend in the first available upper triangle cell
        legend_i, legend_j = available_cells[0]
        legend_ax = axes[legend_i, legend_j]
        legend_ax.clear()
        legend_ax.axis('off')

        legend_elements = []
        if values is not None:
            legend_elements.append(plt.scatter([], [], c="gray", s=18, alpha=0.8,
                                               label='Evaluated points'))
        if true_minima is not None:
            legend_elements.append(plt.scatter([], [], marker="x", s=80, c="red",
                                               linewidths=2, label='True minimum'))
        if observed_minima is not None:
            legend_elements.append(plt.scatter([], [], marker="x", s=80, c="blue",
                                               linewidths=2, label='Best observed'))

        legend_ax.legend(handles=legend_elements, loc='center', fontsize=9)

        # Add text info in second available cell if it exists
        if len(available_cells) >= 2:
            info_i, info_j = available_cells[1]
            info_ax = axes[info_i, info_j]
            info_ax.clear()
            info_ax.axis('off')

            info_text = f"Points evaluated: {len(points)}"
            if values is not None:
                info_text += f"\nBest value: {np.min(values):.4f}"
            if true_minima is not None and observed_minima is not None:
                distance = np.linalg.norm(true_minima - observed_minima)
                info_text += f"\nDistance to true min: {distance:.4f}"

            info_ax.text(0.5, 0.5, info_text, transform=info_ax.transAxes,
                         ha='center', va='center', fontsize=9,
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
        else:
            # If only one cell available, add info below legend
            info_text = f"Points: {len(points)}"
            if values is not None:
                info_text += f" | Best: {np.min(values):.4f}"
            if true_minima is not None and observed_minima is not None:
                distance = np.linalg.norm(true_minima - observed_minima)
                info_text += f" | Dist: {distance:.4f}"

            legend_ax.text(0.5, 0.1, info_text, transform=legend_ax.transAxes,
                           ha='center', va='center', fontsize=8,
                           bbox=dict(boxstyle="round,pad=0.2", facecolor="lightgray", alpha=0.5))

    # If we have a colormap, place the colorbar in a remaining upper-triangle cell
    if sc is not None:
        cax = None
        used_cells = set()
        if len([x for x in [true_minima, observed_minima, values] if x is not None]) > 0:
            used_cells.add(available_cells[0])  # legend cell
            if len(available_cells) >= 2:
                used_cells.add(available_cells[1])  # info cell

        for i, j in available_cells:
            if (i, j) not in used_cells:
                cax = axes[i, j]
                break

        if cax is not None:
            # Create a smaller colorbar within the cell
            cax.clear()
            cax.axis('off')

            # Create a smaller inset for the colorbar
            cbar_ax = cax.inset_axes([0.2, 0.2, 0.3, 0.6])  # [x0, y0, width, height]
            cbar = fig.colorbar(sc, cax=cbar_ax)
            cbar.ax.tick_params(labelsize=8)
            cbar.set_label("Objective\nfunction", fontsize=9)
        else:
            # fallback: create a standard colorbar (may overlap)
            fig.colorbar(sc, ax=axes.ravel().tolist(), label="Objective function")

    return fig, axes


def plot_trace(points, param_idx, values=None, true_minima=None, bounds=None,
               label=None, ax=None, cmap="viridis"):
    """
    Plot parameter trace for a single parameter.

    - `points`: (N, D) array of parameter values
    - `param_idx`: index of parameter to plot
    - `values`: (N,) array of objective function values for coloring
    - `true_minima`: true parameter values (optional)
    - `bounds`: parameter bounds for y-axis limits
    - `label`: parameter label
    - `ax`: matplotlib axis to plot on
    """
    points = np.asarray(points)
    steps = np.arange(len(points))

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure

    # Plot trace
    if values is not None:
        sc = ax.scatter(steps, points[:, param_idx], c=values, cmap=cmap,
                        s=20, alpha=0.7, edgecolors="none")
        # also connect with line
        ax.plot(steps, points[:, param_idx], color="gray", alpha=0.3, linewidth=1)
    else:
        ax.scatter(steps, points[:, param_idx], s=20, alpha=0.6)

    # Add horizontal dashed line for true value
    if true_minima is not None:
        ax.axhline(true_minima[param_idx], color="red", linestyle="--",
                   linewidth=2, alpha=0.8, label="True value")
        ax.legend(fontsize=9)

    ax.set_xlabel("Step", fontsize=10)
    if label is not None:
        ax.set_ylabel(label, fontsize=10)
    else:
        ax.set_ylabel(f"Parameter {param_idx}", fontsize=10)

    ax.tick_params(labelsize=9)
    ax.grid(True, alpha=0.25)

    # Set y-axis based on bounds
    if bounds is not None:
        try:
            ax.set_ylim(bounds[0, param_idx], bounds[1, param_idx])
        except Exception:
            pass

    return fig, ax


def plot_diagnostics(all_obs,
                     history_best,
                     model_uncertainty_history,
                     points,
                     bounds,
                     labels=None,
                     true_minima=None,
                     true_fx=None,
                     fname="diagnostics.png",
                     figsize=(14, 16)):
    """
    One-page diagnostics: (4x2 layout)
      [0,0] Regret curve
      [0,1] Histogram of observed values
      [1,0] Model uncertainty evolution
      [1,1] Scatter-matrix (lower-triangle + diagonal, colored by `all_obs`)
      [2,0] Param1 trace
      [2,1] Param2 trace
      [3,0] Param3 trace
      [3,1] Param4 trace
      ... (additional rows as needed for more parameters)
    """
    all_obs = np.asarray(all_obs)
    points = np.asarray(points)
    n_params = points.shape[1]

    if labels is None:
        labels = [f"p{i}" for i in range(n_params)]

    # Calculate number of rows needed: 2 for main diagnostics + ceil(n_params/2) for traces
    n_trace_rows = (n_params + 1) // 2  # ceiling division
    n_rows = 2 + n_trace_rows

    fig, axs = plt.subplots(n_rows, 2, figsize=figsize)

    # Ensure axs is always 2D
    if n_rows == 1:
        axs = axs.reshape(1, -1)
    elif axs.ndim == 1:
        axs = axs.reshape(-1, 1)

    # Regret curve
    x_axis = np.arange(len(history_best))
    axs[0, 0].plot(x_axis, history_best, marker="o", linewidth=1.25)
    if true_fx is not None:
        axs[0, 0].axhline(true_fx, color='red', linestyle='--', linewidth=1.5, label='f(x_true)')
        axs[0, 0].legend(fontsize=9, loc='best')
    axs[0, 0].set_title("Best Observed Value vs. Step")
    axs[0, 0].set_xlabel("Step")
    axs[0, 0].set_ylabel("Best f(x)")
    axs[0, 0].grid(True, alpha=0.25)

    # Histogram of observations
    axs[0, 1].hist(all_obs, bins=50, density=True, alpha=0.75)
    axs[0, 1].set_title("Distribution of Observed Values")
    axs[0, 1].set_xlabel("Observed Value")

    # Model uncertainty
    axs[1, 0].plot(model_uncertainty_history, linewidth=1.5)
    axs[1, 0].set_title("Model Uncertainty Evolution")
    axs[1, 0].set_xlabel("Round")
    axs[1, 0].set_ylabel("Avg. Model Uncertainty")
    axs[1, 0].grid(True, alpha=0.25)

    # Scatter-matrix on [1,1]
    axs[1, 1].remove()  # free up the grid cell
    parent_ax = fig.add_subplot(n_rows, 2, 4)  # position [1,1] in the grid
    scatter_matrix(points, labels=labels, bounds=bounds,
                   true_minima=true_minima, values=all_obs, ax=parent_ax)

    # Parameter trace plots starting from row 2
    for param_idx in range(n_params):
        row = 2 + param_idx // 2
        col = param_idx % 2

        if row < n_rows:
            plot_trace(points, param_idx, values=all_obs, true_minima=true_minima,
                       bounds=bounds, label=labels[param_idx], ax=axs[row, col])
            axs[row, col].set_title(f"{labels[param_idx]} Trace")

    # Hide any unused trace plot axes
    for param_idx in range(n_params, n_trace_rows * 2):
        row = 2 + param_idx // 2
        col = param_idx % 2
        if row < n_rows:
            axs[row, col].axis('off')

    fig.tight_layout()
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)

