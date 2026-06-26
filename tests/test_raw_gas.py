import numpy as np

from clumping_factor.raw_gas import raw_gas_clumping_sweep, raw_gas_clumping_sweep_chunked, raw_gas_volume_weighted_clumping_sweep


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


def test_raw_volume_hii_density_mode_uses_ionized_hydrogen_fraction():
    def chunks():
        yield {
            "density": np.array([1.0, 2.0, 4.0]),
            "masses": np.array([1.0, 2.0, 4.0]),
            "cell_volume": np.ones(3),
            "hi_fraction": np.array([0.0, 0.5, 0.75]),
            "hydrogen_mass_fraction": np.ones(3),
            "input_count": 3,
            "valid_count": 3,
            "dropped_count": 0,
        }

    factors, _, diagnostics = raw_gas_clumping_sweep_chunked(
        np.array([10.0]),
        chunks,
        lbox=3.0 ** (1.0 / 3.0),
        chunk_size=3,
        volume_weighted=True,
        clumping_mode="hii-density",
        hii_source="hi-fraction",
    )
    assert np.isclose(factors[0], 1.0)
    assert diagnostics["raw_clumping_mode"] == "hii-density"


def test_raw_volume_electron_hii_mode_uses_electron_abundance():
    def chunks():
        yield {
            "density": np.array([1.0, 2.0]),
            "masses": np.array([1.0, 2.0]),
            "cell_volume": np.ones(2),
            "hii_fraction": np.ones(2),
            "electron_abundance": np.array([1.0, 2.0]),
            "hydrogen_mass_fraction": np.ones(2),
            "input_count": 2,
            "valid_count": 2,
            "dropped_count": 0,
        }

    factors, _, diagnostics = raw_gas_clumping_sweep_chunked(
        np.array([10.0]),
        chunks,
        lbox=3.0 ** (1.0 / 3.0),
        chunk_size=2,
        volume_weighted=True,
        clumping_mode="electron-hii",
        hii_source="hii-fraction",
        electron_source="electron-abundance",
    )
    ne = np.array([1.0, 4.0])
    nhii = np.array([1.0, 2.0])
    expected = np.mean(ne * nhii) / (np.mean(ne) * np.mean(nhii))
    assert np.isclose(factors[0], expected)
    assert diagnostics["raw_electron_source"] == "electron-abundance"
