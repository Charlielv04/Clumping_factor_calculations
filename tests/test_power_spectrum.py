import numpy as np

from clumping_factor.power_spectrum import density_power_spectrum, density_power_spectrum_pylians


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


def test_pylians_engine_uses_pk_library(monkeypatch):
    calls = {}

    class FakePk:
        k3D = np.array([1.0, 2.0])
        Pk = np.array([[10.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
        Nmodes3D = np.array([6, 12])

    class FakePkLibrary:
        @staticmethod
        def Pk(delta, box_size, axis, mas, threads, verbose):
            calls["mean"] = float(np.mean(delta, dtype=np.float64))
            calls["box_size"] = box_size
            calls["axis"] = axis
            calls["mas"] = mas
            calls["threads"] = threads
            calls["verbose"] = verbose
            return FakePk()

    monkeypatch.setitem(__import__("sys").modules, "Pk_library", FakePkLibrary)

    result = density_power_spectrum_pylians(
        np.full((4, 4, 4), 2.0),
        50.0,
        mas="TSC",
        threads=3,
        axis=0,
        verbose=True,
    )

    assert np.isclose(calls["mean"], 0.0)
    assert calls["box_size"] == 50.0
    assert calls["mas"] == "TSC"
    assert calls["threads"] == 3
    assert calls["verbose"] is True
    assert result.power.tolist() == [10.0, 20.0]
    assert result.diagnostics["engine"] == "pylians"
