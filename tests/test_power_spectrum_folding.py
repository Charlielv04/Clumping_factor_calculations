import json

import numpy as np

from clumping_factor.infrastructure.models import ParticleData
from clumping_factor.methods.power_spectrum.folding import fold_coordinates, fold_particle_data, validate_fold_factors
from clumping_factor.methods.power_spectrum.estimator import density_power_spectrum
from clumping_factor.visualization.power_spectrum import plot_power_spectrum_files


def test_modulo_folding_is_periodic_and_coordinatewise():
    coords = np.array([[-0.1, 2.1, 4.9], [1.0, 3.0, 5.0]])
    folded = fold_coordinates(coords, 4.0, 2)
    np.testing.assert_allclose(folded, [[1.9, 0.1, 0.9], [1.0, 1.0, 1.0]])
    assert np.all((folded >= 0) & (folded < 2.0))


def test_uniform_particles_remain_uniform_under_folding():
    points = np.array(np.meshgrid(np.arange(4) + 0.25, np.arange(4) + 0.25, np.arange(4) + 0.25)).reshape(3, -1).T
    particles = ParticleData(points, np.ones(len(points)), np.ones(len(points)), 4.0, "dm")
    folded = fold_particle_data(particles, 2)
    assert folded.lbox == 2.0
    assert folded.count == particles.count
    assert np.all((folded.coords >= 0) & (folded.coords < folded.lbox))


def test_folded_box_has_expected_k_scaling_for_a_mode():
    n = 16
    box = 8.0
    effective_box = box / 2
    x = np.arange(n) * effective_box / n
    density = 1.0 + 0.2 * np.cos(2 * np.pi * x / effective_box)[:, None, None]
    density = np.broadcast_to(density, (n, n, n))
    fundamental = 2 * np.pi / effective_box
    result = density_power_spectrum(density, effective_box, k_edges=[0.9 * fundamental, 1.1 * fundamental])
    np.testing.assert_allclose(result.k[0], fundamental, rtol=0.1)


def test_fold_factor_validation_and_metadata():
    assert validate_fold_factors([1, 2, 4], 75.0) == (1, 2, 4)
    try:
        validate_fold_factors([0], 75.0)
    except ValueError:
        pass
    else:
        raise AssertionError("non-positive folds must fail")


def test_plot_all_fold_blocks_and_json_round_trip(tmp_path):
    result = {
        "statistic": "density_power_spectrum",
        "primary_spectrum_engine": "numpy",
        "parameters": {"simulation_name": "toy", "snapshot": 1, "grid_size": 8, "smoothing": "none"},
        "spectra": {"numpy": {"k": [1, 2], "dimensionless_power": [1, 2]}},
        "folded_spectra": {
            "1": {"fold_factor": 1, "effective_box_size": 8, "nominal_nyquist": 3, "spectra": {"numpy": {"k": [1, 2], "dimensionless_power": [1, 2]}}},
            "2": {"fold_factor": 2, "effective_box_size": 4, "nominal_nyquist": 6, "spectra": {"numpy": {"k": [2, 4], "dimensionless_power": [1, 2]}}},
        },
    }
    source = tmp_path / "folded.json"
    source.write_text(json.dumps(result), encoding="utf-8")
    assert json.loads(source.read_text(encoding="utf-8"))["folded_spectra"]["2"]["fold_factor"] == 2
    output = plot_power_spectrum_files([source], tmp_path / "folded.png", fold_factor="all", average_bins=2)
    assert output.exists() and output.stat().st_size > 0
