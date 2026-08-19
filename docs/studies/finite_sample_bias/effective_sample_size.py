"""How many *effective* COMPAS systems back each bin of the McZ grid?

The finite-sample bias derived in `finite_sample_bias.py` is controlled by

    t = (effective model detections) / (data detections)

Raw COMPAS counts overstate this badly, because each system enters the rate
matrix with a very non-uniform cosmic-integration weight. The honest number is
Kish's effective sample size,

    N_eff = (sum_i w_i)^2 / sum_i w_i^2 ,

computed per bin (and for the grid total). This script reports it at the
fiducial parameters so we can say whether the bias matters for:

  * the Poisson normalisation term  -> uses N_eff of the whole grid
  * the per-event McZ grid term     -> uses N_eff of the occupied bins

Run:  python effective_sample_size.py [path/to/COMPAS.h5]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from cosmic_integration.ratesSampler import BinnedCosmicIntegrator
from cosmic_integration.ratesSampler.binned_cosmic_integrator import (
    ChirpMassBin,
    MakeChirpMassBins,
)

# Jeff's "true" values, from the wiki minutes (Jan 20 2026).
TRUE = dict(p_Alpha=-0.325, p_Sigma=0.213, p_SFRa=0.012, p_SFRd=4.253)

DEFAULT_H5 = Path(__file__).resolve().parents[3] / "tests/large_test_data/h5out_5M.h5"


def kish(w: np.ndarray, axis=None) -> np.ndarray:
    """Kish effective sample size of a set of weights."""
    s1 = np.sum(w, axis=axis)
    s2 = np.sum(w * w, axis=axis)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(s2 > 0, s1 * s1 / s2, 0.0)


def main(h5: str) -> None:
    print(f"COMPAS file: {h5}")
    ci = BinnedCosmicIntegrator.from_compas_fpath(h5)

    rate, chirp_masses = ci.FindDetectionRate(p_BinaryFraction=0.7, **TRUE)
    print(f"detectionRate shape (systems x z): {rate.shape}")

    mc_edges, _ = MakeChirpMassBins()
    n_mc = len(mc_edges) + 1
    n_z = rate.shape[1]

    # Accumulate sum(w) and sum(w^2) per (Mc bin, z bin).
    s1 = np.zeros((n_mc, n_z))
    s2 = np.zeros((n_mc, n_z))
    raw = np.zeros((n_mc, n_z), dtype=int)
    for col in range(rate.shape[0]):
        b = ChirpMassBin(chirp_masses[col], mc_edges)
        w = rate[col, :]
        s1[b] += w
        s2[b] += w * w
        raw[b] += w > 0

    with np.errstate(divide="ignore", invalid="ignore"):
        n_eff_bin = np.where(s2 > 0, s1 * s1 / s2, 0.0)

    total_rate = s1.sum()
    # Systems are the Monte Carlo draws; the z axis is a deterministic
    # integration grid, so its cells are NOT independent samples.
    n_eff_total = kish(rate.sum(axis=1))

    print(f"\ngrid                     : {n_mc} Mc bins x {n_z} z bins = {n_mc * n_z} bins")
    print(f"total detection rate     : {total_rate:.2f} / yr")
    print(f"N_eff of the whole grid  : {n_eff_total:.1f}   (raw systems: {rate.shape[0]})")
    print(f"  -> weight efficiency   : {100 * n_eff_total / rate.shape[0]:.2f}%")

    occupied = n_eff_bin[s1 > 0]
    print(f"\noccupied bins            : {occupied.size} / {n_mc * n_z}")
    print("per-bin N_eff percentiles:")
    for p in [1, 5, 25, 50, 75, 95, 99]:
        print(f"  {p:>3d}%  {np.percentile(occupied, p):10.1f}")

    # Bins that actually carry the likelihood: those holding most of the rate.
    flat = s1.ravel()
    order = np.argsort(flat)[::-1]
    csum = np.cumsum(flat[order]) / flat.sum()
    for frac in (0.5, 0.9, 0.99):
        k = int(np.searchsorted(csum, frac)) + 1
        idx = order[:k]
        print(
            f"\nbins holding {frac:.0%} of the rate: {k}"
            f"   median N_eff = {np.median(n_eff_bin.ravel()[idx]):.1f}"
            f"   min N_eff = {n_eff_bin.ravel()[idx].min():.1f}"
        )

    print("\n--- implied likelihood bias  (bias ~ -1/(2t), t = N_eff / N_obs) ---")
    print(f"  {'N_obs':>8} {'t (Poisson term)':>18} {'bias':>10}")
    for n_obs in (50, 750, 2250):
        t = n_eff_total / n_obs
        print(f"  {n_obs:>8} {t:>18.1f} {-0.5 / t:>10.4f}")

    med_eff = float(np.median(occupied))
    print(f"\n  per-bin median N_eff = {med_eff:.1f}"
          f"  -> fractional noise on p_grid ~ {1 / np.sqrt(med_eff):.3f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(DEFAULT_H5))
