import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

for chain_file in Path("outputs").glob("**/MCMC*/**/chain.dat"):
    # Read header
    with open(chain_file) as f:
        header = f.readline().strip().split()
    
    data = np.loadtxt(chain_file, skiprows=1)
    if data.ndim == 1:
        data = data[:, None]
    
    walkers = data[:, 0].astype(int)
    params = data[:, 1:]
    param_labels = header[1:]  # Skip 'walker' label
    
    if params.ndim == 1:
        params = params[:, None]
    
    n_walkers = len(np.unique(walkers))
    cmap = plt.colormaps['tab20'](np.linspace(0, 1, n_walkers))
    
    fig, axes = plt.subplots(params.shape[1], 1, figsize=(8, 2*params.shape[1]), sharex=True)
    if params.shape[1] == 1:
        axes = [axes]
    
    for idx, ax in enumerate(axes):
        for walker_id in np.unique(walkers):
            mask = walkers == walker_id
            steps = np.where(mask)[0]
            ax.plot(steps, params[mask, idx], lw=0.8, color=cmap[int(walker_id)])
        ax.set_ylabel(param_labels[idx])
    
    axes[-1].set_xlabel("step")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = chain_file.with_name("trace.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
