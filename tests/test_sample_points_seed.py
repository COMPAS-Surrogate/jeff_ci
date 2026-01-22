import numpy as np

from cosmic_integration.lnl_surrogate.lnl_surrogate import sample_points


def test_sample_points_is_reproducible_with_seed():
    pts_a = sample_points(25, seed=123)
    pts_b = sample_points(25, seed=123)
    assert np.allclose(pts_a, pts_b)

