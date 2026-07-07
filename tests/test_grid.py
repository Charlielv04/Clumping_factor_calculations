import numpy as np

from clumping_factor.grid import build_density_grid_scipy, spherical_tophat_kernel
from clumping_factor.models import ParticleData


def synthetic_particles():
    return ParticleData(
        coords=np.array(
            [
                [0.1, 0.1, 0.1],
                [0.6, 0.6, 0.6],
            ],
            dtype=np.float32,
        ),
        radii=np.array([0.25, 0.25], dtype=np.float32),
        masses=np.array([2.0, 3.0], dtype=np.float32),
        lbox=1.0,
        particle_type="gas",
    )


def test_spherical_kernel_normalizes_to_one():
    kernel = spherical_tophat_kernel(0.3, 0.25)
    assert np.isclose(kernel.sum(dtype=np.float64), 1.0)


def test_cube_backend_mass_conservation_and_shape():
    result = build_density_grid_scipy(synthetic_particles(), grid_size=4, radius_bins=2, backend="cube")
    assert result.density_grid.shape == (4, 4, 4)
    assert result.density_grid.dtype == np.float64
    assert np.all(np.isfinite(result.density_grid))
    assert abs(result.diagnostics["relative_mass_error"]) < 1e-6


def test_sphere_backend_mass_conservation():
    result = build_density_grid_scipy(synthetic_particles(), grid_size=4, radius_bins=2, backend="sphere")
    assert abs(result.diagnostics["relative_mass_error"]) < 1e-6


def test_tsc_assignment_conserves_mass_for_scipy_backends():
    for backend in ("sphere", "cube"):
        result = build_density_grid_scipy(synthetic_particles(), grid_size=4, radius_bins=2, backend=backend, mas="TSC")
        assert abs(result.diagnostics["relative_mass_error"]) < 1e-6
        assert result.backend_metadata["mas"] == "TSC"


def test_tsc_spreads_mass_over_more_cells_than_cic():
    particles = ParticleData(
        coords=np.array([[0.37, 0.42, 0.58]], dtype=np.float32),
        radii=np.array([0.0], dtype=np.float32),
        masses=np.array([1.0], dtype=np.float32),
        lbox=1.0,
        particle_type="dm",
    )
    cic = build_density_grid_scipy(particles, grid_size=8, radius_bins=1, backend="cube", mas="CIC")
    tsc = build_density_grid_scipy(particles, grid_size=8, radius_bins=1, backend="cube", mas="TSC")
    assert np.count_nonzero(cic.density_grid) <= 8
    assert np.count_nonzero(tsc.density_grid) > np.count_nonzero(cic.density_grid)


def test_periodic_deposition_is_translation_invariant():
    left = ParticleData(
        coords=np.array([[0.125, 0.5, 0.5]], dtype=np.float32),
        radii=np.array([0.0], dtype=np.float32), masses=np.array([2.0], dtype=np.float32),
        lbox=1.0, particle_type="dm",
    )
    wrapped = ParticleData(
        coords=np.array([[1.125, 0.5, 0.5]], dtype=np.float32),
        radii=left.radii, masses=left.masses, lbox=1.0, particle_type="dm",
    )
    for mas in ("CIC", "TSC"):
        first = build_density_grid_scipy(left, 8, 1, "cube", mas=mas)
        second = build_density_grid_scipy(wrapped, 8, 1, "cube", mas=mas)
        assert np.allclose(first.density_grid, second.density_grid, rtol=0, atol=1e-6)


def test_grid_size_above_limit_is_rejected(monkeypatch):
    monkeypatch.setenv("CLUMPING_MAX_GRID_CELLS", "7")
    try:
        build_density_grid_scipy(synthetic_particles(), grid_size=2, radius_bins=1, backend="cube")
    except ValueError as exc:
        assert "supported maximum" in str(exc)
    else:
        raise AssertionError("build_density_grid_scipy should reject oversized grids")
