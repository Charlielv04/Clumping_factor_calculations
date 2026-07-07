import numpy as np

from clumping_factor.clumping import clumping_factor_sweep, clumping_factor_sweep_with_mask


def test_uniform_density_has_unit_clumping():
    thresholds = np.array([-1.0, 0.5, 1.0])
    factors, _ = clumping_factor_sweep(thresholds, np.ones((2, 2, 2)))
    assert np.isnan(factors[0])
    assert factors[1] == 1.0
    assert factors[2] == 1.0


def test_zero_mean_density_returns_nan():
    factors, _ = clumping_factor_sweep(np.array([-1.0, 1.0]), np.zeros((2, 2, 2)))
    assert np.all(np.isnan(factors))


def test_threshold_sweep_shape_and_order():
    thresholds = np.array([-1.0, 0.0, 2.0])
    factors, _ = clumping_factor_sweep(thresholds, np.array([1.0, 2.0, 3.0, 4.0]))
    assert factors.shape == thresholds.shape
    assert np.isnan(factors[0])
    assert np.isfinite(factors[1])
    assert np.isfinite(factors[2])


def test_masked_clumping_matches_direct_legacy_definition():
    mask_density = np.array([1.0, 2.0, 4.0, 8.0])
    target_density = np.array([2.0, 3.0, 5.0, 11.0])
    threshold = np.array([0.5])
    factors, _, diagnostics = clumping_factor_sweep_with_mask(threshold, mask_density, target_density)
    selected = target_density[(mask_density / mask_density.mean() - 1.0) < threshold[0]]
    expected = np.mean(selected**2) / np.mean(selected) ** 2
    assert np.allclose(factors, [expected])
    assert diagnostics["selected_cell_counts"] == [selected.size]
