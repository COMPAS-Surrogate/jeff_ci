"""Small figures for the wiki write-up of the finite-COMPAS-sample study.

Caches the per-system rate weights for each COMPAS run (small npz) so the plots
can be regenerated without re-running the cosmic integrator.

Run:  python make_wiki_plots.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from finite_sample_bias import bias, log_bias

REPO = Path(__file__).resolve().parents[3]
ASSETS = REPO / "wiki/.gitbook/assets"
CACHE = Path(__file__).resolve().parent / "weight_cache.npz"

TRUE = dict(p_Alpha=-0.325, p_Sigma=0.213, p_SFRa=0.012, p_SFRd=4.253)
RUNS = {
    "5M": REPO / "tests/large_test_data/h5out_5M.h5",
    "32M": REPO / "tests/large_test_data/h5out_32M_reduced.h5",
}
# N_eff measured at the fiducial parameters (see effective_sample_size.py).
N_EFF = {"5M": 162.9, "32M": 1018.0, "512M (extrapolated)": 1018.0 * 16}

DPI = 110


def build_cache() -> dict:
    if CACHE.exists():
        return dict(np.load(CACHE))

    from cosmic_integration.ratesSampler import BinnedCosmicIntegrator
    from cosmic_integration.ratesSampler.binned_cosmic_integrator import (
        ChirpMassBin,
        MakeChirpMassBins,
    )

    out = {}
    mc_edges, _ = MakeChirpMassBins()
    for name, path in RUNS.items():
        ci = BinnedCosmicIntegrator.from_compas_fpath(str(path))
        rate, chirp = ci.FindDetectionRate(p_BinaryFraction=0.7, **TRUE)

        # Per-system total weight (summed over z) -> weight concentration.
        out[f"{name}_sys"] = rate.sum(axis=1)

        # Per-bin Kish N_eff -> how many systems back each grid pixel.
        s1 = np.zeros((len(mc_edges) + 1, rate.shape[1]))
        s2 = np.zeros_like(s1)
        for col in range(rate.shape[0]):
            b = ChirpMassBin(chirp[col], mc_edges)
            w = rate[col, :]
            s1[b] += w
            s2[b] += w * w
        with np.errstate(divide="ignore", invalid="ignore"):
            neff = np.where(s2 > 0, s1 * s1 / s2, 0.0)
        out[f"{name}_binneff"] = neff[s1 > 0]
        print(f"{name}: {rate.shape[0]} systems, {out[f'{name}_binneff'].size} occupied bins")

    np.savez_compressed(CACHE, **out)
    return out


def plot_bias_vs_t() -> None:
    t = np.geomspace(0.03, 300, 70)
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.plot(t, [bias(750, 750.0, x)[0] for x in t], lw=2, label="what we do now")
    ax.plot(t, [bias(750, 750.0, x)[1] for x in t], lw=2, ls="--",
            label="Ilya's marginalised version")
    ax.axhspan(-0.05, 0.05, color="tab:green", alpha=0.15)
    ax.text(60, -0.28, "safe", color="tab:green", fontsize=9)
    for name, ne in N_EFF.items():
        ax.axvline(ne / 750, color="k", ls=":", lw=1, alpha=0.6)
        ax.text(ne / 750, 0.15, name.split()[0], rotation=90, fontsize=7,
                ha="right", va="bottom")
    ax.set_xscale("log")
    ax.set_ylim(-2.2, 0.6)
    ax.set_xlabel("t  =  effective COMPAS detections / observed detections")
    ax.set_ylabel("bias of the averaged likelihood (lnL)")
    ax.set_title("Ilya's question: averaging likelihoods over many COMPAS runs\n"
                 "(dotted lines: our runs, for a 1 yr / ~750 event mock)", fontsize=9)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ASSETS / "fsb_bias_vs_t.png", dpi=DPI)
    plt.close(fig)


def plot_weight_concentration(cache: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.5))

    ax = axes[0]
    ref = np.geomspace(1e-3, 100, 200)
    ax.plot(ref, ref, "k:", lw=1, label="if all binaries counted equally")
    for name in RUNS:
        w = np.sort(cache[f"{name}_sys"])[::-1]
        frac_sys = 100 * np.arange(1, w.size + 1) / w.size
        frac_rate = 100 * np.cumsum(w) / w.sum()
        ax.plot(frac_sys, frac_rate, lw=2, label=f"{name} run")
        half = frac_sys[np.searchsorted(frac_rate, 50)]
        print(f"  {name}: {half:.2f}% of binaries carry 50% of the rate")
    ax.set_xscale("log")
    ax.set_xlim(1e-3, 100)
    ax.set_ylim(0, 105)
    ax.set_xlabel("% of merging BBHs (most important first)")
    ax.set_ylabel("% of the total detection rate")
    ax.set_title("A tiny minority of binaries carries the rate", fontsize=9)
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.3)

    ax = axes[1]
    bins = np.geomspace(0.8, 60, 30)
    for name in RUNS:
        ax.hist(cache[f"{name}_binneff"], bins=bins, alpha=0.6, label=f"{name} run")
    ax.axvline(10, color="k", ls=":", lw=1)
    ax.text(10.5, ax.get_ylim()[1] * 0.8, "~10 needed", fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("effective binaries per McZ grid bin")
    ax.set_ylabel("number of bins")
    ax.set_title("Each grid pixel rests on very few binaries", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(ASSETS / "fsb_weight_concentration.png", dpi=DPI)
    plt.close(fig)


def plot_which_analysis_safe() -> None:
    n_obs = np.geomspace(20, 4000, 60)
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    for name, ne in N_EFF.items():
        b = [log_bias(int(round(n)), ne) for n in n_obs]
        ax.plot(n_obs, b, lw=2, label=f"{name}  (N_eff={ne:.0f})")
    ax.axhspan(-0.05, 0.05, color="tab:green", alpha=0.15)
    for x, lab in [(50, "LVK O3\n~50 events"), (750, "mock 1 yr\n~750"), (2250, "mock 3 yr\n~2250")]:
        ax.axvline(x, color="grey", ls=":", lw=1)
        ax.text(x, 0.35, lab, fontsize=7, ha="center")
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=0.1)
    ax.set_ylim(-8, 1.2)
    ax.set_xlabel("number of observed events")
    ax.set_ylabel("lnL bias of a single run")
    ax.set_title("Which analyses are safe with which COMPAS run", fontsize=10)
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ASSETS / "fsb_which_analysis_safe.png", dpi=DPI)
    plt.close(fig)


if __name__ == "__main__":
    ASSETS.mkdir(parents=True, exist_ok=True)
    cache = build_cache()
    plot_bias_vs_t()
    plot_weight_concentration(cache)
    plot_which_analysis_safe()
    for p in sorted(ASSETS.glob("fsb_*.png")):
        print(f"{p.name:36s} {p.stat().st_size / 1024:6.1f} KB")
