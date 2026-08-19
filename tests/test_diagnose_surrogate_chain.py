import numpy as np
import pytest


def test_run_chain_diagnostics_respects_burnin(tmp_path, monkeypatch):
    from cosmic_integration.cli_tools import diagnose_surrogate_chain as dsc

    # Create a tiny synthetic chain file with 2 walkers and 5 iterations each.
    chain_path = tmp_path / "chain.dat"
    header = "walker\talpha\tsigma\tsfr_a\tsfr_d\tlog_l\tlog_p\n"
    rows = []
    for step in range(5):
        for walker in range(2):
            alpha = -0.1 * (step + 1) + 0.01 * walker
            sigma = 0.2 + 0.01 * step
            sfr_a = 0.01 + 0.001 * walker
            sfr_d = 4.0 + 0.1 * step
            log_l = 0.0
            log_p = 0.0
            rows.append(f"{walker}\t{alpha}\t{sigma}\t{sfr_a}\t{sfr_d}\t{log_l}\t{log_p}\n")
    chain_path.write_text(header + "".join(rows), encoding="utf-8")

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
    # burnin=2 leaves 3 iterations per walker -> 6 rows
    assert summary["n_chain_after_filter"] == 6
    assert summary["n_evaluated"] == 6
    beta_payload = summary["betas"]["0"]
    assert beta_payload["all"]["rmse"] == pytest.approx(0.0)
