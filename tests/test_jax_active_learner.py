import numpy as np
import pytest
from types import SimpleNamespace

gpjax = pytest.importorskip("gpjax")

from cosmic_integration.lnl_surrogate.jax_active_learner import (
    JaxActiveLearner,
    JaxGPConfig,
    JaxTrainingData,
    compare_kernel_candidates,
    fit_exact_gp,
)
from cosmic_integration.lnl_surrogate.adaptive_robust_scalar import (
    AdaptiveRobustScaler,
)
from cosmic_integration.lnl_surrogate import lnl_surrogate as lnl_module
from cosmic_integration.lnl_surrogate.lnl_surrogate import (
    BOUNDS,
    LnLSurrogate,
    PARAMETERS,
)


def test_exact_gp_predicts_finite_mean_and_variance():
    x = np.linspace(-1.0, 1.0, 9).reshape(-1, 1)
    y = (x**2).reshape(-1, 1)
    data = JaxTrainingData(x, y)
    model = fit_exact_gp(
        data,
        np.array([[-1.0], [1.0]]),
        config=JaxGPConfig(optimisation_steps=30, candidate_count=32),
    )

    mean, variance = model.predict_f(np.array([[0.0], [0.5]]))
    assert mean.shape == variance.shape == (2, 1)
    assert np.isfinite(mean).all()
    assert np.isfinite(variance).all()
    assert np.all(variance >= 0.0)


def test_fixed_hyperparameter_fit_is_deterministic():
    bounds = np.array([[-1.0, -2.0], [1.0, 2.0]])
    x = np.array([[-0.8, -1.4], [-0.3, 0.2], [0.1, -0.5], [0.7, 1.3]])
    y = np.array([[0.7], [-0.2], [0.1], [1.2]])
    test_x = np.array([[-0.5, 0.5], [0.4, -1.0]])
    config = JaxGPConfig(
        optimisation_steps=1,
        learning_rate=0.0,
        initial_noise_stddev=float(np.sqrt(1e-5)),
        optimise_noise=False,
    )
    first = fit_exact_gp(
        JaxTrainingData(x, y),
        bounds,
        config=config,
    )
    second = fit_exact_gp(JaxTrainingData(x, y), bounds, config=config)
    np.testing.assert_allclose(first.predict_f(test_x)[0], second.predict_f(test_x)[0])
    np.testing.assert_allclose(first.predict_f(test_x)[1], second.predict_f(test_x)[1])


def test_active_learner_is_seed_reproducible_and_improves_best_value():
    def bowl(x, y):
        return (x - 0.25) ** 2 + 2.0 * (y + 0.4) ** 2

    bounds = np.array([[-1.0, -1.0], [1.0, 1.0]])
    config = JaxGPConfig(optimisation_steps=25, candidate_count=64)
    first = JaxActiveLearner(
        bowl, bounds, initial_points=6, random_seed=8, config=config
    )
    second = JaxActiveLearner(
        bowl, bounds, initial_points=6, random_seed=8, config=config
    )

    initial_best = first.history_best[-1]
    first_data, first_model = first.run(
        total_steps=2, exploration_fraction=0.5
    )
    second_data, _ = second.run(total_steps=2, exploration_fraction=0.5)

    assert first_data.query_points.shape == (8, 2)
    assert first_data.observations.shape == (8, 1)
    assert first.history_best[-1] <= initial_best
    assert first.acquisition_history == [
        "predictive_variance",
        "expected_improvement",
    ]
    np.testing.assert_allclose(
        first_data.query_points, second_data.query_points
    )
    np.testing.assert_allclose(
        first_data.observations, second_data.observations
    )
    mean, variance = first_model.predict_f(first_data.query_points[:2])
    assert np.isfinite(mean).all() and np.isfinite(variance).all()


def test_kernel_comparison_reports_finite_heldout_metrics():
    x = np.linspace(-1.0, 1.0, 15).reshape(-1, 1)
    data = JaxTrainingData(x, np.sin(2.0 * x))
    results = compare_kernel_candidates(
        data,
        np.array([[-1.0], [1.0]]),
        kernels=("matern32", "matern52"),
        base_config=JaxGPConfig(optimisation_steps=20, candidate_count=32),
        seed=13,
    )

    assert [result.kernel for result in results] == sorted(
        ("matern32", "matern52"),
        key=lambda kernel: next(
            result.mean_negative_log_predictive_density
            for result in results
            if result.kernel == kernel
        ),
    )
    for result in results:
        assert np.isfinite(result.rmse)
        assert np.isfinite(result.mean_negative_log_predictive_density)
        assert 0.0 <= result.coverage_68 <= 1.0


def test_round_checkpoint_loads_through_lnl_surrogate(tmp_path):
    bounds = np.array([[-1.0] * 4, [1.0] * 4])
    initial_x = np.array(
        [
            [-0.8, -0.4, 0.2, 0.7],
            [-0.2, 0.5, -0.6, 0.4],
            [0.1, -0.3, 0.8, -0.5],
            [0.6, 0.2, -0.1, -0.7],
            [0.3, 0.7, 0.4, -0.2],
        ]
    )

    def raw_lnl(point):
        return -float(np.sum((point - 0.2) ** 2))

    raw_initial_lnls = np.array([raw_lnl(point) for point in initial_x])
    scaler = AdaptiveRobustScaler(soft_clipping=False)
    scaler.initialize_with_data(raw_initial_lnls)

    def objective(*point):
        return -float(scaler.transform(raw_lnl(np.asarray(point))))

    model_root = tmp_path / "gp_model"
    learner = JaxActiveLearner(
        objective,
        bounds,
        initial_data_x=initial_x,
        initial_data_y=np.array([objective(*point) for point in initial_x]),
        outdir=str(model_root),
        random_seed=6,
        config=JaxGPConfig(optimisation_steps=20, candidate_count=48),
    )
    _, fitted = learner.run(total_steps=2, steps_per_round=1)
    scaler.save(str(model_root))

    loaded = LnLSurrogate.load(str(model_root / "models"), round_idx=1)
    loaded.parameters = {name: 0.2 for name in PARAMETERS}
    expected_mean, _ = fitted.predict_f(np.full((1, 4), 0.2))

    assert (model_root / "models" / "round_0.pkl").is_file()
    assert (model_root / "models" / "round_1.pkl").is_file()
    np.testing.assert_allclose(
        loaded.gp_model.predict_f(np.full((1, 4), 0.2))[0], expected_mean
    )
    assert np.isfinite(loaded.log_likelihood())


def test_lnl_surrogate_train_uses_gpjax_with_compas_like_objective(
    tmp_path, monkeypatch
):
    center = np.mean(BOUNDS, axis=0)
    widths = BOUNDS[1] - BOUNDS[0]
    initial_x = np.vstack(
        (
            center - 0.25 * widths,
            center - 0.10 * widths,
            center,
            center + 0.15 * widths,
            center + 0.30 * widths,
        )
    )

    def raw_lnl(*point):
        normalized = (np.asarray(point) - center) / widths
        return -float(np.sum(normalized**2))

    initial_lnls = np.array([raw_lnl(*point) for point in initial_x])
    scaler = AdaptiveRobustScaler(soft_clipping=False)
    scaler.initialize_with_data(initial_lnls)

    class DummyLnLComputer:
        observation = SimpleNamespace(population_weights=np.ones(2))

        def __call__(self, *point):
            return raw_lnl(*point)

    class TransformedObjective:
        def __init__(self):
            self.scaler = scaler

        def __call__(self, *point):
            return -float(self.scaler.transform(raw_lnl(*point)))

    monkeypatch.setattr(
        lnl_module.LnLComputer,
        "load",
        lambda **_: DummyLnLComputer(),
    )
    monkeypatch.setattr(
        lnl_module,
        "robust_neg_lnl_computer_factory",
        lambda *_, **__: TransformedObjective(),
    )

    outdir = tmp_path / "trained"
    outdir.mkdir()
    trained = LnLSurrogate.train(
        observation_file="observation.json",
        compas_h5="compas.h5",
        outdir=str(outdir),
        inital_samples=initial_x,
        initial_lnls=initial_lnls,
        total_steps=2,
        steps_per_round=1,
        jax_config=JaxGPConfig(optimisation_steps=20, candidate_count=48),
    )
    trained.parameters = {
        name: float(value) for name, value in zip(PARAMETERS, center)
    }
    restored = LnLSurrogate.load(
        str(outdir / "gp_model" / "models"), round_idx=1
    )
    restored.parameters = dict(trained.parameters)

    assert (outdir / "gp_model" / "models" / "round_1.pkl").is_file()
    assert np.isfinite(trained.log_likelihood())
    np.testing.assert_allclose(
        trained.log_likelihood(),
        restored.log_likelihood(),
        rtol=0.0,
        atol=1e-12,
    )
