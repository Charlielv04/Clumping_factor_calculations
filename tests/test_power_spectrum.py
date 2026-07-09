import numpy as np

from clumping_factor.power_spectrum import density_power_spectrum


def test_uniform_density_has_zero_power():
    result = density_power_spectrum(np.ones((8, 8, 8)), 10.0, bin_count=6)
    assert np.allclose(result.power, 0.0)
    assert result.diagnostics["mean_density"] == 1.0


def test_single_cosine_mode_has_expected_total_power():
    grid_size = 16
    box_size = 8.0
    amplitude = 0.2
    x = np.arange(grid_size) * box_size / grid_size
    density = 1.0 + amplitude * np.cos(2.0 * np.pi * x / box_size)[:, None, None]
    density = np.broadcast_to(density, (grid_size, grid_size, grid_size))
    fundamental = 2.0 * np.pi / box_size
    edges = np.array([0.5 * fundamental, 1.1 * fundamental])

    result = density_power_spectrum(density, box_size, k_edges=edges)

    assert result.mode_counts.tolist() == [6]
    expected_shell_average = (2.0 * box_size**3 * amplitude**2 / 4.0) / 6.0
    assert np.allclose(result.power[0], expected_shell_average)


def test_rejects_non_cubic_grid():
    try:
        density_power_spectrum(np.ones((4, 4, 2)), 1.0)
    except ValueError as exc:
        assert "cubic" in str(exc)
    else:
        raise AssertionError("non-cubic grids should fail")
