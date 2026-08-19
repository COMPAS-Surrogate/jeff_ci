import numpy as np
import pytest


def test_run_chain_diagnostics_respects_burnin(tmp_path, monkeypatch):
    from cosmic_integration.cli_tools import diagnose_surrogate_chain as dsc

    # NUTS writes a flat (n_samples, n_params) array of posterior samples.
    chain_path = tmp_path / "posterior_samples.npy"
    samples = np.array(
        [
            [-0.1 * (i + 1), 0.2 + 0.01 * i, 0.01 + 0.0001 * i, 4.0 + 0.1 * i]
            for i in range(10)
        ],
        dtype=float,
    )
    np.save(chain_path, samples)

    class DummyLnLComputer:
        def __call__(self, alpha, sigma, sfr_a, sfr_d):
            return float(alpha + sigma + sfr_a + sfr_d)

        @classmethod
        def load(cls, *args, **kwargs):
            return cls()

    class DummyScaler:
        def inverse_transform(self, x):
            return np.asarray(x, dtype=float)

    class DummyGP:
        def predict_f(self, x):
            x = np.asarray(x, dtype=float)
            lnl = np.sum(x, axis=1, keepdims=True)
            mu = -lnl  # negative-transformed convention used by the surrogate
            var = np.zeros_like(mu)
            return mu, var

    class DummySurrogate:
        def __init__(self):
            self.gp_model = DummyGP()
            self.scaler = DummyScaler()

        @classmethod
        def load(cls, *args, **kwargs):
            return cls()

    monkeypatch.setattr(dsc, "LnLComputer", DummyLnLComputer)
    monkeypatch.setattr(dsc, "LnLSurrogate", DummySurrogate)

    summary = dsc.run_chain_diagnostics(
        observation_file="obs.h5",
        compas_h5="compas.h5",
        model_dir="model_dir",
        chain_path=str(chain_path),
        outdir=str(tmp_path / "out"),
        max_points=100,
        burnin=2,
        uncertainty_betas=(0.0,),
        top_fractions=(0.05, 0.01, 0.001),
    )

    assert summary["n_chain_total"] == 10
    # NUTS samples are a single flat chain: burnin=2 drops the first 2 samples
    assert summary["n_chain_after_filter"] == 8
    assert summary["n_evaluated"] == 8
    beta_payload = summary["betas"]["0"]
    assert beta_payload["all"]["rmse"] == pytest.approx(0.0)
