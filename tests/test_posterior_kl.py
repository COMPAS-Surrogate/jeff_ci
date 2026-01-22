from pathlib import Path

import numpy as np

from cosmic_integration.lnl_surrogate.diagnostics.posterior_kl import consecutive_posterior_kl


def test_consecutive_posterior_kl_writes_outputs(tmp_path: Path):
    root = tmp_path / "rounds"
    (root / "round_0" / "MCMC").mkdir(parents=True)
    (root / "round_1" / "MCMC").mkdir(parents=True)

    rng = np.random.default_rng(0)
    a = rng.normal(size=(500, 4))
    b = rng.normal(loc=0.1, size=(500, 4))
    np.save(root / "round_0" / "MCMC" / "posterior_samples.npy", a)
    np.save(root / "round_1" / "MCMC" / "posterior_samples.npy", b)

    outdir = tmp_path / "final"
    series = consecutive_posterior_kl(
        postprocess_root=root,
        rounds=[0, 1],
        outdir=outdir,
        bins=40,
    )

    assert len(series) == 1
    assert series[0]["round"] == 0
    assert series[0]["next_round"] == 1
    assert (outdir / "kl_consecutive.json").exists()
    assert (outdir / "kl_consecutive.png").exists()

