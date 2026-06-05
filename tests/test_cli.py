import json
from argparse import Namespace

import numpy as np

from clumping_factor.cli import build_compute_parser, plot_main, run_compute
from clumping_factor.models import GridResult, ParticleData


def test_compute_help():
    parser = build_compute_parser()
    help_text = parser.format_help()
    assert "--particle-type" in help_text
    assert "--backend" in help_text
    assert "raw" in help_text


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
    assert document["particle_type"] == "dm"
    assert document["backend"]["backend"] == "cube"
    assert document["clumping_factors"][2] == 1.0


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
