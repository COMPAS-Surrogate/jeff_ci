"""Paper figure: population-synthesis sample-size requirement.

Panel (a): the cosmic-integration weights are extremely concentrated, so the
           effective sample size is ~1% of the raw merging-BBH count.
Panel (b): the resulting lnL bias vs the number of observed events, for each
           COMPAS run, with the |bias| < 0.05 requirement band.

Writes overleaf/figures/sample_size.pdf

Run:  python plot_sample_size.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from finite_sample_bias import log_bias
from make_wiki_plots import build_cache

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "overleaf/figures/sample_size.pdf"

# Per-system Kish N_eff at the fiducial parameters (effective_sample_size.py).
N_EFF = {"5M": 162.9, "32M": 1018.0, "512M": 1018.0 * 16}
N_RAW = {"5M": 13019, "32M": 83145, "512M": 83145 * 16}

plt.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.7,
})


def main() -> None:
    cache = build_cache()
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.7))

    # ---- (a) weight concentration ------------------------------------
    ax = axes[0]
    ref = np.geomspace(1e-3, 100, 200)
    ax.plot(ref, ref, ":", color="0.4", lw=0.9, label="equal weights")
    for name, c in [("5M", "tab:blue"), ("32M", "tab:orange")]:
        w = np.sort(cache[f"{name}_sys"])[::-1]
        fs = 100 * np.arange(1, w.size + 1) / w.size
        fr = 100 * np.cumsum(w) / w.sum()
        ax.plot(fs, fr, lw=1.6, color=c, label=f"{name} run")
        half = fs[np.searchsorted(fr, 50)]
        print(f"  {name}: {half:.2f}% of binaries carry 50% of the rate")
    ax.axhline(50, color="0.7", lw=0.6, ls="--")
    ax.annotate("0.45% of binaries\ncarry 50% of the rate",
                xy=(0.45, 50), xytext=(0.004, 72), fontsize=6.5,
                arrowprops=dict(arrowstyle="->", lw=0.6, color="0.3"))
    ax.set_xscale("log")
    ax.set_xlim(1e-3, 100)
    ax.set_ylim(0, 103)
    ax.set_xlabel(r"per cent of merging BBHs (ranked by weight)")
    ax.set_ylabel(r"per cent of total detection rate")
    ax.legend(loc="lower right", frameon=False)
    ax.text(0.03, 0.93, "(a)", transform=ax.transAxes, fontweight="bold")

    # ---- (b) requirement, in terms of the TILT -----------------------
    # A constant bias cancels on normalisation; only its variation across the
    # prior distorts the posterior. Measured along alpha (the steepest
    # direction) at N_obs = 750; the tilt scales linearly with N_obs.
    tilt_750 = {"5M": 4.637, "32M": 0.704, "512M": 0.042}
    ax = axes[1]
    n_obs = np.geomspace(20, 4000, 80)
    for (name, t750), c in zip(tilt_750.items(), ["tab:blue", "tab:orange", "tab:green"]):
        ax.plot(n_obs, t750 * n_obs / 750.0, lw=1.6, color=c,
                label=rf"{name}  ($N_{{\rm eff}}={N_EFF[name]:.0f}$)")
    ax.axhspan(1e-3, 0.5, color="tab:green", alpha=0.13, lw=0)
    ax.text(1500, 0.0016, r"$T<0.5$", fontsize=6.5, color="0.25")
    for x in (50, 750, 2250):   # LVK O3, 1 yr mock, 3 yr mock (see caption)
        ax.axvline(x, color="0.6", ls=":", lw=0.7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 60)
    ax.set_xlabel(r"number of observed events $N_{\rm obs}$")
    ax.set_ylabel(r"likelihood tilt $T$ across the prior")
    ax.legend(loc="upper left", frameon=False, handlelength=1.4)
    ax.text(0.90, 0.93, "(b)", transform=ax.transAxes, fontweight="bold")

    for a in axes:
        a.grid(alpha=0.25, lw=0.5)

    fig.tight_layout(pad=0.5)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT}  ({OUT.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
