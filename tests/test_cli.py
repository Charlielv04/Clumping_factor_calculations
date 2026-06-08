import json
from argparse import Namespace

import numpy as np

from clumping_factor.cli import build_compute_parser, plot_main, run_compute
from clumping_factor.models import GridResult, ParticleData
from clumping_factor.results import default_output_path, resolve_simulation_name


def test_compute_help():
    parser = build_compute_parser()
    help_text = parser.format_help()
    assert "--particle-type" in help_text
    assert "--backend" in help_text
    assert "raw" in help_text
    assert "raw-volume" in help_text
    assert "--radius-mode" in help_text
    assert "--simulation-name" in help_text
    assert "--load-mode" in help_text
    assert "--chunk-size" in help_text


def test_simulation_name_inferred_from_base_path():
    assert resolve_simulation_name("../Thesan-1") == "Thesan-1"
    assert resolve_simulation_name("./tng100-3/output") == "tng100-3"
    assert resolve_simulation_name("./anything", "Thesan 1!") == "Thesan-1"


def test_default_output_path_uses_simulation_subdirectory():
    output = default_output_path("results", "gas", "sphere", 81, 256, "Thesan-1")
    assert output.as_posix() == "results/Thesan-1/gas_sphere_snapshot081_grid256.json"


def test_run_compute_writes_json_with_mock_loader_and_grid(monkeypatch, tmp_path):
    particles = ParticleData(
        coords=np.array([[0.1, 0.1, 0.1]], dtype=np.float32),
        radii=np.array([0.25], dtype=np.float32),
        masses=np.array([1.0], dtype=np.float32),
        lbox=1.0,
        particle_type="dm",
        metadata={"valid_count": 1},
    )

    def fake_loader(*args, **kwargs):
        return particles, {"load_data": 0.0}

    def fake_grid(*args, **kwargs):
        return GridResult(
            density_grid=np.ones((2, 2, 2), dtype=np.float32),
            diagnostics={"input_mass": 1.0, "grid_mass": 1.0, "relative_mass_error": 0.0},
            timings={"build_density_grid": 0.0},
            backend_metadata={"backend": "cube"},
        )

    monkeypatch.setattr("clumping_factor.cli._load_tng_particles", fake_loader)
    monkeypatch.setattr("clumping_factor.cli._build_density_grid_scipy", fake_grid)

    output = tmp_path / "result.json"
    args = Namespace(
        base_path="./data",
        snapshot=98,
        particle_type="dm",
        backend="cube",
        grid_size=2,
        radius_bins=1,
        radius_mode="sphere",
        threshold_min=-1.0,
        threshold_max=1.0,
        threshold_count=3,
        output=str(output),
        output_dir=str(tmp_path),
        mas="CIC",
        filter_type="Top-Hat",
        threads=1,
        verbose=False,
    )
    written = run_compute(args)
    assert written == output
    document = json.loads(output.read_text())
    assert document["simulation"]["name"] == "data"
    assert document["particle_type"] == "dm"
    assert document["backend"]["backend"] == "cube"
    assert document["clumping_factors"][2] == 1.0


def test_run_compute_rejects_invalid_threshold_range(tmp_path):
    args = Namespace(
        base_path="./data",
        snapshot=98,
        particle_type="dm",
        backend="cube",
        grid_size=2,
        radius_bins=1,
        radius_mode="sphere",
        threshold_min=1.0,
        threshold_max=1.0,
        threshold_count=3,
        output=str(tmp_path / "result.json"),
        output_dir=str(tmp_path),
        mas="CIC",
        filter_type="Top-Hat",
        threads=1,
        verbose=False,
    )
    try:
        run_compute(args)
    except ValueError as exc:
        assert "--threshold-min" in str(exc)
    else:
        raise AssertionError("run_compute should reject an invalid threshold range")


def test_raw_backend_default_output_path_has_no_grid(monkeypatch, tmp_path):
    def fake_loader(*args, **kwargs):
        return {
            "density": np.ones(4, dtype=np.float32),
            "rho_mean": 1.0,
            "metadata": {"valid_count": 4},
        }, {"load_data": 0.0}

    monkeypatch.setattr("clumping_factor.cli._load_tng_gas_cells", fake_loader)

    args = Namespace(
        base_path="./data",
        snapshot=98,
        particle_type="gas",
        backend="raw",
        grid_size=256,
        radius_bins=10,
        radius_mode="sphere",
        threshold_min=-1.0,
        threshold_max=1.0,
        threshold_count=3,
        output=None,
        output_dir=str(tmp_path),
        mas="CIC",
        filter_type="Top-Hat",
        threads=1,
        verbose=False,
    )
    written = run_compute(args)
    assert written.name == "gas_raw_snapshot098.json"
    assert written.parent.name == "data"
    assert json.loads(written.read_text())["parameters"]["grid_size"] is None


def test_gas_radius_mode_is_passed_independently_from_backend(monkeypatch, tmp_path):
    seen_radius_modes = []
    particles = ParticleData(
        coords=np.array([[0.1, 0.1, 0.1]], dtype=np.float32),
        radii=np.array([0.25], dtype=np.float32),
        masses=np.array([1.0], dtype=np.float32),
        lbox=1.0,
        particle_type="gas",
        metadata={"valid_count": 1},
    )

    def fake_loader(_base_path, _snapshot, _particle_type, radius_mode, **_kwargs):
        seen_radius_modes.append(radius_mode)
        return particles, {"load_data": 0.0}

    def fake_grid(*args, **kwargs):
        return GridResult(
            density_grid=np.ones((2, 2, 2), dtype=np.float64),
            diagnostics={"input_mass": 1.0, "grid_mass": 1.0, "relative_mass_error": 0.0},
            timings={"build_density_grid": 0.0},
            backend_metadata={"backend": "pylians"},
        )

    monkeypatch.setattr("clumping_factor.cli._load_tng_particles", fake_loader)
    monkeypatch.setattr("clumping_factor.cli._build_density_grid_pylians", fake_grid)

    args = Namespace(
        base_path="./data",
        snapshot=98,
        particle_type="gas",
        backend="pylians",
        grid_size=2,
        radius_bins=1,
        radius_mode="cube",
        threshold_min=-1.0,
        threshold_max=1.0,
        threshold_count=3,
        output=str(tmp_path / "result.json"),
        output_dir=str(tmp_path),
        mas="CIC",
        filter_type="Top-Hat",
        threads=1,
        verbose=False,
    )
    run_compute(args)
    assert seen_radius_modes == ["cube"]


def test_plot_command_reads_json_and_writes_plot(tmp_path):
    result_json = tmp_path / "result.json"
    result_json.write_text(
        json.dumps(
            {
                "particle_type": "gas",
                "backend": {"backend": "sphere"},
                "thresholds": [-1.0, 0.0, 1.0],
                "clumping_factors": [None, 1.0, 1.2],
            }
        )
    )
    output = tmp_path / "plot.png"
    plot_main([str(result_json), "--output", str(output)])
    assert output.exists()
    assert output.stat().st_size > 0


def test_plot_command_rejects_malformed_result(tmp_path):
    result_json = tmp_path / "bad.json"
    result_json.write_text(json.dumps({"thresholds": [0.0]}))
    output = tmp_path / "plot.png"
    try:
        plot_main([str(result_json), "--output", str(output)])
    except ValueError as exc:
        assert "clumping_factors" in str(exc)
    else:
        raise AssertionError("plot_main should reject malformed result JSON")


def test_plot_command_rejects_all_nan_result(tmp_path):
    result_json = tmp_path / "nan.json"
    result_json.write_text(
        json.dumps(
            {
                "particle_type": "gas",
                "backend": {"backend": "sphere"},
                "thresholds": [0.0],
                "clumping_factors": [None],
            }
        )
    )
    output = tmp_path / "plot.png"
    try:
        plot_main([str(result_json), "--output", str(output)])
    except ValueError as exc:
        assert "No finite" in str(exc)
    else:
        raise AssertionError("plot_main should reject all-NaN plot inputs")
