"""Does the effective sample size vary across parameter space?

This is the load-bearing question. A finite-sample likelihood bias that is
*constant* across lambda is harmless -- it shifts lnL by a constant and cancels
in the posterior. A bias that *varies* with lambda distorts the shape of the
likelihood surface, moves the peak, and would show up exactly as the spurious
sfr_a--sfr_d ridge seen in the Jan 2026 slides.

The single-run bias is  E[ln L_hat] - ln L_true ~ -N_obs / (2 N_eff(lambda)),
so we need N_eff(lambda). Note this depends on N_eff ONLY -- the mu(lambda)
dependence cancels exactly, so it does not matter where we sit relative to the
likelihood peak.


Run:  python neff_vs_params.py [path/to/COMPAS.h5]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from cosmic_integration.ratesSampler import BinnedCosmicIntegrator

from finite_sample_bias import log_bias

TRUE = dict(p_Alpha=-0.325, p_Sigma=0.213, p_SFRa=0.012, p_SFRd=4.253)

# Prior ranges, from the 1D scans in the Jan 21 2026 slides.
SCANS = {
    "p_Alpha": [-0.50, -0.40, -0.325, -0.25, -0.15],
    "p_Sigma": [0.10, 0.15, 0.213, 0.27, 0.33],
    "p_SFRa": [0.006, 0.009, 0.012, 0.015, 0.018],
    "p_SFRd": [2.0, 3.0, 4.253, 5.5, 6.5],
}

DEFAULT_H5 = Path(__file__).resolve().parents[3] / "tests/large_test_data/h5out_5M.h5"


def kish(w: np.ndarray) -> float:
    s1, s2 = w.sum(), (w * w).sum()
    return float(s1 * s1 / s2) if s2 > 0 else 0.0


def main(h5: str, n_obs: float = 750.0) -> None:
    ci = BinnedCosmicIntegrator.from_compas_fpath(h5)

    def probe(**kw):
        params = dict(TRUE, **kw)
        rate, _ = ci.FindDetectionRate(p_BinaryFraction=0.7, **params)
        return kish(rate.sum(axis=1)), float(rate.sum())

    n_eff_0, rate_0 = probe()
    print(f"at truth: N_eff = {n_eff_0:.1f}, rate = {rate_0:.1f}/yr\n")

    print("N_eff and the single-run lnL bias across the prior:")
    print(f"  N_obs = {n_obs:g}   (bias = E[ln L_hat] - ln L_true ~ -N_obs / (2 N_eff))\n")
    n = int(round(n_obs))
    for name, values in SCANS.items():
        print(f"  {name}")
        print(f"    {'value':>10} {'N_eff':>10} {'rate/yr':>10} {'bias':>10}")
        biases = []
        for v in values:
            n_eff, rate = probe(**{name: v})
            b = log_bias(n, n_eff)
            biases.append(b)
            print(f"    {v:>10g} {n_eff:>10.1f} {rate:>10.1f} {b:>10.3f}")
        print(f"    -> tilt across this direction: {max(biases) - min(biases):.3f} lnL\n")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(DEFAULT_H5))
