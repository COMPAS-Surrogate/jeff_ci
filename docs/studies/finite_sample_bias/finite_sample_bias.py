"""Does marginalising over finite model size remove the downward likelihood bias?

Answers Ilya's open question (COMPAS slack, 2026-05-10), following his toy problem
from 2026-01-29.

Setup (Ilya's "how many Avi's" problem, generalised to n observed events):
  - The data contain ``n`` detections; the true model predicts ``mu`` detections.
  - We do not evaluate ``mu`` analytically. Instead a forward-model run of size
    ``S`` yields ``k_S ~ Poisson(t * mu)`` predicted detections, where

        t = S / (exposure of the observation)
          = (detections the model run produces) / (detections in the data)

    so ``t`` is the only thing that matters: how many times more model
    detections than data detections a single run gives us.

Two estimators of L = p(data | mu):

  naive       L_hat = Poisson(n | k_S / t)
              -- what we do today. k_S = 0 gives L_hat = 0, which annihilates
              the average over runs. This is Ilya's systematic underestimate.

  marginal    L_hat = \\int dk p(n | k) p(k | k_S, t)
              -- Ilya's proposed fix. With prior p(k) ~ k^(a-1) the posterior on
              k is Gamma(k_S + a, t) and the integral is closed-form:

                  L_marg = Gamma(c+n) / (Gamma(c) n!) * t^c / (1+t)^(c+n),
                  c = k_S + a

Because a surrogate is trained on lnL and posterior widths are set by
Delta lnL ~ O(1), we report the bias in ln-likelihood units. A bias >> 1 is fatal.

Run:  python finite_sample_bias.py
"""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln
from scipy.stats import poisson


def true_lnl(n: int, mu: float) -> float:
    """Exact Poisson lnL of observing n events when the model predicts mu."""
    return -mu + n * np.log(mu) - gammaln(n + 1)


def _k_grid(mean: float, tol: float = 1e-14) -> np.ndarray:
    """Support of a Poisson(mean) wide enough that the neglected tail < tol."""
    hi = int(poisson.ppf(1.0 - tol, mean)) + 10
    return np.arange(0, max(hi, 50))


def expected_naive_l(n: int, mu: float, t: float) -> float:
    """E[L_hat] over runs, for the naive plug-in estimator. Exact summation."""
    k = _k_grid(t * mu)
    pmf = poisson.pmf(k, t * mu)

    k_m = k / t
    with np.errstate(divide="ignore", invalid="ignore"):
        lnl = -k_m + n * np.log(k_m) - gammaln(n + 1)
    l_hat = np.where(k > 0, np.exp(lnl), 0.0)  # k_S = 0 -> L_hat = 0 exactly
    return float(np.sum(pmf * l_hat))


def expected_marginal_l(n: int, mu: float, t: float, a: float = 1.0) -> float:
    """E[L_hat] over runs, for Ilya's marginalised estimator. Exact summation.

    a is the prior exponent: p(k) ~ k^(a-1). a=1 is flat, a=0.5 is Jeffreys.
    """
    k = _k_grid(t * mu)
    pmf = poisson.pmf(k, t * mu)

    c = k + a
    ln_l = (
        gammaln(c + n) - gammaln(c) - gammaln(n + 1)
        + c * np.log(t) - (c + n) * np.log1p(t)
    )
    return float(np.sum(pmf * np.exp(ln_l)))


def log_bias(n: int, n_eff: float) -> float:
    """Bias of a SINGLE run's lnL:  E[ln L_hat] - ln L_true.

    THIS is the quantity our pipeline cares about. We run COMPAS once and train
    the GP on lnL, so we need the mean of the *log* likelihood -- not the log of
    the mean likelihood, which is what `bias()` returns and what Ilya's
    average-over-runs question needs. By Jensen the two differ, badly.

    With k_S ~ Poisson(N_eff) the mu-dependence cancels exactly and

        E[ln L_hat] - ln L_true = n * ( E[ln k_S] - ln E[k_S] )
                                ~ -n / (2 N_eff)

    so the bias depends only on the effective sample size -- not on where in
    parameter space we are relative to the likelihood peak. It is always
    negative (Jensen), i.e. we always under-estimate lnL.
    """
    lo = max(1, int(n_eff - 12 * np.sqrt(n_eff)))
    hi = int(n_eff + 12 * np.sqrt(n_eff)) + 20
    k = np.arange(lo, hi)
    p = poisson.pmf(k, n_eff)
    p = p / p.sum()  # renormalise: we dropped k_S = 0, where ln L_hat = -inf
    return float(n * (np.sum(p * np.log(k)) - np.log(n_eff)))


def bias(n: int, mu: float, t: float, a: float = 1.0) -> tuple[float, float]:
    """Bias in lnL units: ln(E[L_hat] / L_true) for (naive, marginal)."""
    ref = true_lnl(n, mu)
    naive = expected_naive_l(n, mu, t)
    marg = expected_marginal_l(n, mu, t, a=a)
    with np.errstate(divide="ignore"):
        return np.log(naive) - ref, np.log(marg) - ref


def t_required(n: int, mu: float, tol: float = 1.0, a: float = 1.0) -> tuple[float, float]:
    """Smallest t at which each estimator's |lnL bias| drops below tol."""

    def solve(idx: int) -> float:
        grid = np.geomspace(1e-2, 1e7, 400)
        for t in grid:
            if abs(bias(n, mu, t, a=a)[idx]) < tol:
                return float(t)
        return float("inf")

    return solve(0), solve(1)


def demo() -> None:
    # --- Ilya's exact toy: 1 Avi in a million, mu = 1 ---------------------
    n, mu = 1, 1.0
    print(f"Ilya's toy problem: n={n} observed, mu={mu} predicted")
    print(f"  true lnL = {true_lnl(n, mu):+.4f}   (L = {np.exp(true_lnl(n, mu)):.4f})\n")
    print(f"  {'t = S/exposure':>16} {'naive bias':>14} {'marginal bias':>15}")
    for t in [0.01, 0.1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e6]:
        b_naive, b_marg = bias(n, mu, t)
        print(f"  {t:>16g} {b_naive:>+14.4f} {b_marg:>+15.4f}")

    # Reproduces Ilya's plot: R runs of S=N/R each, N fixed.
    print("\n  Ilya's R-split (N fixed, t_total = 1e3, split into R runs):")
    print(f"  {'R':>10} {'t per run':>12} {'naive bias':>14} {'marginal bias':>15}")
    t_total = 1e3
    for r in [1, 10, 100, 1_000, 10_000, 100_000]:
        t = t_total / r
        b_naive, b_marg = bias(n, mu, t)
        print(f"  {r:>10} {t:>12g} {b_naive:>+14.4f} {b_marg:>+15.4f}")

    # --- COMPAS-like: many events ----------------------------------------
    print("\n\nCOMPAS-like regimes (evaluated at the likelihood peak, mu = n):")
    print(f"  {'n_events':>10} {'t':>10} {'naive bias':>14} {'marginal bias':>15}")
    for n_ev in [50, 750]:
        for t in [1.0, 10.0, 100.0, 1e3, 1e4]:
            b_naive, b_marg = bias(n_ev, float(n_ev), t)
            print(f"  {n_ev:>10} {t:>10g} {b_naive:>+14.4f} {b_marg:>+15.4f}")
        print()

    print("Smallest t for |lnL bias| < 1  (the run-size requirement):")
    print(f"  {'n_events':>10} {'naive':>12} {'marginal':>12}")
    for n_ev in [1, 50, 750]:
        tn, tm = t_required(n_ev, float(n_ev))
        print(f"  {n_ev:>10} {tn:>12.3g} {tm:>12.3g}")

    # --- does the prior choice help? --------------------------------------
    priors = [0.0, 0.5, 1.0, 1.5, 2.0]
    print("\n\nPrior-exponent scan, p(k) ~ k^(a-1), n=1, mu=1:")
    header = f"  {'t':>8} {'naive':>10}" + "".join(f"{'a=' + str(a):>10}" for a in priors)
    print(header)
    for t in [0.01, 0.1, 0.5, 1.0, 3.0, 10.0, 100.0]:
        row = [bias(1, 1.0, t, a=a)[1] for a in priors]
        print(f"  {t:>8g} {bias(1, 1.0, t)[0]:>10.4f}" + "".join(f"{v:>10.4f}" for v in row))

    print("\nAsymptotic bias * t  (large t):")
    for t in [100.0, 1e3, 1e4]:
        b_naive, b_marg = bias(50, 50.0, t)
        print(f"  t={t:>8g}   naive*t = {b_naive * t:+.4f}   marginal*t = {b_marg * t:+.4f}")
    print("  -> naive -> -1/(2t),  marginal -> -1/t   (marginal is 2x worse asymptotically)")

    # --- self-checks ------------------------------------------------------
    # Both estimators must converge to the truth as the run size -> infinity.
    b_naive, b_marg = bias(1, 1.0, 1e7)
    assert abs(b_naive) < 1e-3, b_naive
    assert abs(b_marg) < 1e-3, b_marg

    # Closed form must match brute-force numerical integration of Ilya's integral.
    for n_ev, k_s, t, a in [(1, 0, 0.5, 1.0), (1, 3, 2.0, 1.0), (7, 12, 3.0, 0.5)]:
        c = k_s + a
        grid = np.linspace(1e-9, c / t + 60 * np.sqrt(c + 1) / t, 400_000)
        post = np.exp((c - 1) * np.log(grid) - t * grid + c * np.log(t) - gammaln(c))
        lik = np.exp(-grid + n_ev * np.log(grid) - gammaln(n_ev + 1))
        numeric = np.trapezoid(post * lik, grid)
        closed = np.exp(
            gammaln(c + n_ev) - gammaln(c) - gammaln(n_ev + 1)
            + c * np.log(t) - (c + n_ev) * np.log1p(t)
        )
        assert np.isclose(numeric, closed, rtol=1e-4), (n_ev, k_s, t, numeric, closed)

    # The bias is downward (never optimistic) for both estimators.
    for t in [0.1, 1.0, 10.0, 100.0]:
        assert all(b < 1e-9 for b in bias(1, 1.0, t)), t

    # Asymptotes: naive -> -1/(2t), marginal -> -1/t.
    b_naive, b_marg = bias(50, 50.0, 1e4)
    assert np.isclose(b_naive * 1e4, -0.5, atol=1e-3), b_naive * 1e4
    assert np.isclose(b_marg * 1e4, -1.0, atol=1e-3), b_marg * 1e4

    # Marginalising rescues the catastrophic small-run regime.
    b_naive, b_marg = bias(1, 1.0, 0.01, a=0.5)
    assert b_naive < -50 and b_marg > -5, (b_naive, b_marg)

    # Single-run log bias: matches -n/(2 N_eff), and is always downward.
    for n_ev, n_eff in [(750, 200.4), (750, 1266.0), (50, 200.5)]:
        lb = log_bias(n_ev, n_eff)
        assert lb < 0, (n_ev, n_eff, lb)
        assert np.isclose(lb, -n_ev / (2 * n_eff), rtol=0.02), (lb, -n_ev / (2 * n_eff))

    print("\nself-checks passed")


def plot(fname: str = "finite_sample_bias.png") -> None:
    import matplotlib.pyplot as plt

    t = np.geomspace(0.02, 1e3, 60)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)

    for ax, (n, mu) in zip(axes, [(1, 1.0), (50, 50.0)]):
        ax.plot(t, [bias(n, mu, x)[0] for x in t], label="naive plug-in", lw=2)
        for a in (0.5, 1.0):
            ax.plot(t, [bias(n, mu, x, a=a)[1] for x in t], ls="--",
                    label=f"marginalised, $a$={a}")
        ax.plot(t, -0.5 / t, color="k", ls=":", lw=1, label=r"$-1/(2t)$")
        ax.axhspan(-0.1, 0.1, color="green", alpha=0.12)
        ax.set_xscale("log")
        ax.set_ylim(-6, 0.5)
        ax.set_xlabel(r"$t$ = model detections / data detections")
        ax.set_title(rf"$N_{{\rm obs}}={n}$")
        ax.grid(alpha=0.3)

    axes[0].set_ylabel(r"bias  $\ln\,\mathbb{E}[\hat{L}]/L_{\rm true}$")
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Finite model-sample bias in the likelihood (green band: $|$bias$|<0.1$)")
    fig.tight_layout()
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"\nwrote {fname}")


if __name__ == "__main__":
    demo()
    plot()
