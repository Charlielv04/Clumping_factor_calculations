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
    assert np.all(np.isfinite(result.density_grid))
    assert abs(result.diagnostics["relative_mass_error"]) < 1e-6


def test_sphere_backend_mass_conservation():
    result = build_density_grid_scipy(synthetic_particles(), grid_size=4, radius_bins=2, backend="sphere")
    assert abs(result.diagnostics["relative_mass_error"]) < 1e-6

