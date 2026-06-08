import numpy as np

from clumping_factor.raw_gas import raw_gas_clumping_sweep, raw_gas_volume_weighted_clumping_sweep


def test_raw_gas_uniform_density_has_unit_clumping():
    factors, _, diagnostics = raw_gas_clumping_sweep(
        np.array([-0.5, 0.5]),
        np.ones(4),
        rho_mean=1.0,
    )
    assert np.isnan(factors[0])
    assert factors[1] == 1.0
    assert diagnostics["overdensity_definition"] == "Density / (sum(Masses) / Lbox**3) - 1"


def test_raw_gas_volume_weighted_uniform_density_has_unit_clumping():
    factors, _, diagnostics = raw_gas_volume_weighted_clumping_sweep(
        np.array([-0.5, 0.5]),
        np.ones(4),
        np.array([1.0, 2.0, 3.0, 4.0]),
        rho_mean=1.0,
    )
    assert np.isnan(factors[0])
    assert factors[1] == 1.0
    assert "volume" in diagnostics["clumping_definition"]


def test_raw_gas_rejects_empty_density():
    try:
        raw_gas_clumping_sweep(np.array([0.0]), np.array([]), rho_mean=1.0)
    except ValueError as exc:
        assert "at least one cell" in str(exc)
    else:
        raise AssertionError("raw_gas_clumping_sweep should reject empty density arrays")


def test_raw_gas_rejects_zero_mean_density():
    try:
        raw_gas_clumping_sweep(np.array([0.0]), np.ones(2), rho_mean=0.0)
    except ValueError as exc:
        assert "rho_mean" in str(exc)
    else:
        raise AssertionError("raw_gas_clumping_sweep should reject zero rho_mean")


def test_raw_gas_volume_weighted_rejects_bad_volumes():
    try:
        raw_gas_volume_weighted_clumping_sweep(np.array([0.0]), np.ones(2), np.array([1.0, 0.0]), rho_mean=1.0)
    except ValueError as exc:
        assert "cell_volume" in str(exc)
    else:
        raise AssertionError("raw_gas_volume_weighted_clumping_sweep should reject non-positive volumes")
