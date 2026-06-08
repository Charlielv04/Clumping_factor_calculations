import numpy as np

from clumping_factor.preprocess import make_radius_groups, particle_flat_indices


def test_identical_radius_grouping_uses_single_group():
    group_ids, group_radii, metadata = make_radius_groups(np.ones(5), 10)
    assert np.all(group_ids == 0)
    assert group_radii.tolist() == [1.0]
    assert metadata["radius_binning"] == "single"


def test_linear_radius_grouping():
    radii = np.linspace(1.0, 2.0, 10)
    group_ids, group_radii, metadata = make_radius_groups(radii, 4)
    assert group_ids.shape == radii.shape
    assert np.count_nonzero(np.isfinite(group_radii)) == 4
    assert metadata["radius_binning"] == "linear"


def test_geometric_radius_grouping():
    radii = np.geomspace(1.0, 100.0, 20)
    _, group_radii, metadata = make_radius_groups(radii, 5)
    assert np.count_nonzero(np.isfinite(group_radii)) == 5
    assert metadata["radius_binning"] == "geometric"


def test_particle_flat_indices_wrap_periodic_boundaries():
    coords = np.array(
        [
            [1.0, 0.0, 0.0],
            [-0.25, 0.0, 0.0],
        ]
    )
    assert particle_flat_indices(coords, lbox=1.0, grid_size=4).tolist() == [0, 3 * 4**2]
