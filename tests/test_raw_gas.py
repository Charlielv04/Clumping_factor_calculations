import numpy as np

from clumping_factor.raw_gas import raw_gas_clumping_sweep, raw_gas_volume_weighted_clumping_sweep


def test_raw_gas_uniform_density_has_unit_clumping():
    factors, _, diagnostics = raw_gas_clumping_sweep(
        np.array([0.5, 1.5]),
        np.ones(4),
        rho_mean=1.0,
    )
    assert np.isnan(factors[0])
    assert factors[1] == 1.0
    assert diagnostics["overdensity_definition"] == "Density / (sum(Masses) / Lbox**3), no minus one"


def test_raw_gas_volume_weighted_uniform_density_has_unit_clumping():
    factors, _, diagnostics = raw_gas_volume_weighted_clumping_sweep(
        np.array([0.5, 1.5]),
        np.ones(4),
        np.array([1.0, 2.0, 3.0, 4.0]),
        rho_mean=1.0,
    )
    assert np.isnan(factors[0])
    assert factors[1] == 1.0
    assert "volume" in diagnostics["clumping_definition"]
