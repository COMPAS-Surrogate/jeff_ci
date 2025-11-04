from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import corner


FIDUCIAL_PARAMS = np.array([-0.325, 0.213, 0.012, 4.253], dtype=float)


def _default_labels(dims: int) -> List[str]:
    try:
        # Try to use package labels if available
        from cosmic_integration.lnl_surrogate.lnl_surrogate import PARAMETERS  # type: ignore
        if isinstance(PARAMETERS, (list, tuple)) and len(PARAMETERS) == dims:
            return list(PARAMETERS)
    except Exception:
        pass
    return [f"p{i}" for i in range(dims)]


def _default_truths(labels: List[str]) -> Optional[List[float]]:
    """Return hard-coded fiducial truths mapped to labels if possible."""
    truth_map = {
        "alpha": float(FIDUCIAL_PARAMS[0]),
        "sigma": float(FIDUCIAL_PARAMS[1]),
        "sfr_a": float(FIDUCIAL_PARAMS[2]),
        "sfr_d": float(FIDUCIAL_PARAMS[3]),
    }
    try:
        vec = [truth_map[lbl] for lbl in labels]
        return vec
    except KeyError:
        return None


def load_offline_npz(npz_path: Path):
    data = np.load(npz_path)
    # Support both new and old key names
    if "train_points" in data and "train_lnls" in data:
        X = np.asarray(data["train_points"], dtype=float)
        y = np.asarray(data["train_lnls"], dtype=float)
    elif "final_train_points" in data and "final_train_lnls" in data:
        X = np.asarray(data["final_train_points"], dtype=float)
        y = np.asarray(data["final_train_lnls"], dtype=float)
    else:
        raise KeyError("Could not find train_points/train_lnls in npz file.")
    return X, y


def _parse_truths_arg(arg: Optional[str], labels: List[str]) -> Optional[List[float]]:
    if arg is None:
        return None
    # Try JSON/dict-like: key1:val1,key2:val2
    if ":" in arg:
        parts = [p.strip() for p in arg.split(",") if p.strip()]
        mapping: Dict[str, float] = {}
        for p in parts:
            k, v = p.split(":", 1)
            mapping[k.strip()] = float(v)
        return [float(mapping.get(lbl)) for lbl in labels]
    # Else assume comma-separated floats in order
    vals = [float(v) for v in arg.split(",")]
    if len(vals) != len(labels):
        raise ValueError("Number of --truths values must match dimensionality")
    return vals


def weighted_corner(
    npz_path: str,
    *,
    out: Optional[str] = None,
    labels: Optional[List[str]] = None,
    temperature: float = 500.0,
    max_points: Optional[int] = None,
    truths: Optional[str] = None,
) -> Path:
    npz_path = Path(npz_path)
    out_path = Path(out) if out is not None else npz_path.with_name("weighted_corner.png")

    X, y = load_offline_npz(npz_path)
    n, d = X.shape
    if labels is None:
        labels = _default_labels(d)

    # Optional subsample to keep plotting responsive
    if max_points is not None and n > max_points:
        idx = np.random.default_rng(0).choice(n, size=max_points, replace=False)
        X = X[idx]
        y = y[idx]

    # Stabilised weights: w ∝ exp((LnL - max)/T)
    delta = y - np.max(y)
    T = max(float(temperature), 1e-6)
    w = np.exp(delta / T)

    truth_vec = _parse_truths_arg(truths, labels) if truths is not None else _default_truths(labels)
    fig = corner.corner(X, labels=labels, weights=w, show_titles=True, truths=truth_vec)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Make a weighted corner plot from offline_samples.npz")
    p.add_argument("npz", help="Path to offline_samples.npz")
    p.add_argument("--out", default=None, help="Output image path (default next to npz)")
    p.add_argument("--labels", default=None, help="Comma separated labels (optional)")
    p.add_argument("--temperature", type=float, default=500.0, help="Softmax temperature for weights")
    p.add_argument("--truths", default=None, help="Comma-separated floats or name:value pairs matching labels (default: hard-coded fiducial if labels match)")
    p.add_argument("--max-points", type=int, default=None, help="Optional subsample for plotting")
    args = p.parse_args(argv)

    labels = args.labels.split(",") if args.labels else None
    out = weighted_corner(
        args.npz,
        out=args.out,
        labels=labels,
        temperature=args.temperature,
        max_points=args.max_points,
        truths=args.truths,
    )
    print(f"Saved weighted corner to {out}")


if __name__ == "__main__":
    main()
