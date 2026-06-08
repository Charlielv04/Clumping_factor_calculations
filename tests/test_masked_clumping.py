import numpy as np

from clumping_factor.clumping import clumping_factor_sweep_with_mask


def test_mask_field_can_differ_from_target_field():
    thresholds = np.array([-0.5, 0.5])
    mask_density = np.array([1.0, 2.0, 3.0, 4.0])
    target_density = np.array([10.0, 10.0, 20.0, 20.0])

    factors, _, diagnostics = clumping_factor_sweep_with_mask(
        thresholds,
        mask_density,
        target_density,
    )

    assert np.isclose(factors[0], 1.0)
    assert np.isfinite(factors[1])
    assert diagnostics["overdensity_definition"] == "mask_density / mean(mask_density) - 1"


def test_mask_and_target_shapes_must_match():
    thresholds = np.array([0.0])
    try:
        clumping_factor_sweep_with_mask(thresholds, np.ones(4), np.ones(3))
    except ValueError as exc:
        assert "same shape" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
