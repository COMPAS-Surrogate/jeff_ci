from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from ..lnl_surrogate import PARAMETERS


def _kl_1d_hist(
    samples_p: np.ndarray,
    samples_q: np.ndarray,
    *,
    bins: int = 60,
    eps: float = 1e-12,
) -> float:
    samples_p = np.asarray(samples_p, dtype=float).reshape(-1)
    samples_q = np.asarray(samples_q, dtype=float).reshape(-1)
    combined = np.concatenate([samples_p, samples_q])
    if combined.size == 0:
        return float("nan")

    lo = float(np.min(combined))
    hi = float(np.max(combined))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return 0.0

    edges = np.linspace(lo, hi, int(bins) + 1)
    p, _ = np.histogram(samples_p, bins=edges, density=False)
    q, _ = np.histogram(samples_q, bins=edges, density=False)
    p = p.astype(float) + float(eps)
    q = q.astype(float) + float(eps)
    p /= float(np.sum(p))
    q /= float(np.sum(q))
    return float(np.sum(p * np.log(p / q)))


def consecutive_posterior_kl(
    *,
    postprocess_root: str | Path,
    rounds: Sequence[int],
    outdir: str | Path,
    bins: int = 60,
    parameters: Sequence[str] = PARAMETERS,
) -> list[dict]:
    """
    Compute KL(P_i || P_{i+1}) between consecutive per-round posteriors.

    Expects `posterior_samples.npy` under:
      postprocess_root/round_<r>/MCMC/posterior_samples.npy
    """

    postprocess_root = Path(postprocess_root)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ordered = [int(r) for r in rounds]
    ordered = sorted(set(ordered))
    if len(ordered) < 2:
        (outdir / "kl_consecutive.json").write_text("[]", encoding="utf-8")
        return []

    samples_by_round: dict[int, np.ndarray] = {}
    for r in ordered:
        samples_path = postprocess_root / f"round_{r}" / "MCMC" / "posterior_samples.npy"
        if not samples_path.exists():
            raise FileNotFoundError(f"Missing posterior samples for round {r}: {samples_path}")
        samples = np.load(samples_path)
        samples = np.asarray(samples, dtype=float)
        if samples.ndim != 2 or samples.shape[1] != len(parameters):
            raise ValueError(f"Unexpected samples shape for round {r}: {samples.shape}")
        samples_by_round[r] = samples

    series: list[dict] = []
    for r0, r1 in zip(ordered[:-1], ordered[1:]):
        a = samples_by_round[r0]
        b = samples_by_round[r1]
        kl_by_param = {}
        skl_by_param = {}
        for idx, param in enumerate(parameters):
            kl_ab = _kl_1d_hist(a[:, idx], b[:, idx], bins=bins)
            kl_ba = _kl_1d_hist(b[:, idx], a[:, idx], bins=bins)
            kl_by_param[param] = float(kl_ab)
            skl_by_param[param] = float(0.5 * (kl_ab + kl_ba))

        kl_vals = [v for v in kl_by_param.values() if np.isfinite(v)]
        skl_vals = [v for v in skl_by_param.values() if np.isfinite(v)]
        series.append(
            {
                "round": int(r0),
                "next_round": int(r1),
                "kl_by_param": kl_by_param,
                "kl_mean": float(np.mean(kl_vals)) if kl_vals else float("nan"),
                "skl_by_param": skl_by_param,
                "skl_mean": float(np.mean(skl_vals)) if skl_vals else float("nan"),
            }
        )

    (outdir / "kl_consecutive.json").write_text(json.dumps(series, indent=2), encoding="utf-8")

    x = [item["round"] for item in series]
    y = [item["kl_mean"] for item in series]
    y_skl = [item["skl_mean"] for item in series]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, y, marker="o", label="mean KL(P_i || P_{i+1}) (1D marginals)")
    ax.plot(x, y_skl, marker="o", label="mean sym KL (1D marginals)")
    ax.set_xlabel("Round i")
    ax.set_ylabel("Divergence")
    ax.set_title("Consecutive posterior divergence")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "kl_consecutive.png", dpi=200)
    plt.close(fig)

    return series


def posterior_kl_vs_reference(
    *,
    postprocess_root: str | Path,
    rounds: Sequence[int],
    reference_round: int,
    outdir: str | Path,
    bins: int = 60,
    parameters: Sequence[str] = PARAMETERS,
) -> list[dict]:
    """
    Compute KL(P_round || P_reference_round) for each round in `rounds`.

    Expects `posterior_samples.npy` under:
      postprocess_root/round_<r>/MCMC/posterior_samples.npy
    """

    postprocess_root = Path(postprocess_root)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ordered = [int(r) for r in rounds]
    ordered = sorted(set(ordered))
    if not ordered:
        (outdir / "kl_vs_reference.json").write_text("[]", encoding="utf-8")
        return []

    reference_round = int(reference_round)
    ref_path = postprocess_root / f"round_{reference_round}" / "MCMC" / "posterior_samples.npy"
    if not ref_path.exists():
        raise FileNotFoundError(f"Missing posterior samples for reference round {reference_round}: {ref_path}")
    ref_samples = np.asarray(np.load(ref_path), dtype=float)
    if ref_samples.ndim != 2 or ref_samples.shape[1] != len(parameters):
        raise ValueError(f"Unexpected reference samples shape for round {reference_round}: {ref_samples.shape}")

    series: list[dict] = []
    for r in ordered:
        samples_path = postprocess_root / f"round_{r}" / "MCMC" / "posterior_samples.npy"
        if not samples_path.exists():
            raise FileNotFoundError(f"Missing posterior samples for round {r}: {samples_path}")
        samples = np.asarray(np.load(samples_path), dtype=float)
        if samples.ndim != 2 or samples.shape[1] != len(parameters):
            raise ValueError(f"Unexpected samples shape for round {r}: {samples.shape}")

        kl_by_param = {}
        skl_by_param = {}
        for idx, param in enumerate(parameters):
            kl_pr = _kl_1d_hist(samples[:, idx], ref_samples[:, idx], bins=bins)
            kl_rp = _kl_1d_hist(ref_samples[:, idx], samples[:, idx], bins=bins)
            kl_by_param[param] = float(kl_pr)
            skl_by_param[param] = float(0.5 * (kl_pr + kl_rp))

        kl_vals = [v for v in kl_by_param.values() if np.isfinite(v)]
        skl_vals = [v for v in skl_by_param.values() if np.isfinite(v)]
        series.append(
            {
                "round": int(r),
                "reference_round": int(reference_round),
                "kl_by_param": kl_by_param,
                "kl_mean": float(np.mean(kl_vals)) if kl_vals else float("nan"),
                "skl_by_param": skl_by_param,
                "skl_mean": float(np.mean(skl_vals)) if skl_vals else float("nan"),
            }
        )

    (outdir / "kl_vs_reference.json").write_text(json.dumps(series, indent=2), encoding="utf-8")

    x = [item["round"] for item in series]
    y = [item["kl_mean"] for item in series]
    y_skl = [item["skl_mean"] for item in series]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, y, marker="o", label=f"mean KL(P_round || P_round={reference_round}) (1D marginals)")
    ax.plot(x, y_skl, marker="o", label="mean sym KL (1D marginals)")
    ax.set_xlabel("Round")
    ax.set_ylabel("Divergence")
    ax.set_title("Posterior divergence vs final-round surrogate")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "kl_vs_reference.png", dpi=200)
    plt.close(fig)

    return series
