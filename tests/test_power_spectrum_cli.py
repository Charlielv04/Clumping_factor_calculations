import json

import numpy as np

from clumping_factor.models import GridResult, ParticleData
from clumping_factor.power_spectrum_cli import build_power_spectrum_parser, run_power_spectrum


class Metadata:
    lbox = 1.0
    particle_counts = [1, 1, 0, 0, 0, 0]


def test_power_spectrum_help_mentions_smoothing():
    help_text = build_power_spectrum_parser().format_help()
    assert "--smoothing" in help_text
    assert "none" in help_text
    assert "sphere" in help_text


def test_run_power_spectrum_defaults_to_mas_only(monkeypatch, tmp_path):
    particles = ParticleData(
        coords=np.array([[0.1, 0.1, 0.1]], dtype=np.float32),
        radii=np.array([0.25], dtype=np.float32),
        masses=np.array([1.0], dtype=np.float32),
        lbox=1.0,
        particle_type="dm",
        metadata={"valid_count": 1},
    )

    monkeypatch.setattr("clumping_factor.power_spectrum_cli.read_snapshot_metadata", lambda *_args: Metadata())
    monkeypatch.setattr("clumping_factor.power_spectrum_cli.estimate_full_load_bytes", lambda *_args: 1)
    monkeypatch.setattr(
        "clumping_factor.power_spectrum_cli.load_tng_particles",
        lambda *_args, **_kwargs: (particles, {"load_data": 0.0}),
    )

    output = tmp_path / "pk.json"
    args = build_power_spectrum_parser().parse_args(
        [
            "--base-path", "./data",
            "--particle-type", "dm",
            "--grid-size", "4",
            "--bin-count", "4",
            "--output", str(output),
        ]
    )
    written = run_power_spectrum(args)
    document = json.loads(written.read_text())

    assert written == output
    assert document["parameters"]["smoothing"] == "none"
    assert document["grid"]["backend"]["backend"] == "mass-assignment"
    assert document["grid"]["backend"]["smoothing"] == "none"
    assert len(document["k"]) > 0


def test_run_power_spectrum_can_request_smoothed_grid(monkeypatch, tmp_path):
    particles = ParticleData(
        coords=np.array([[0.1, 0.1, 0.1]], dtype=np.float32),
        radii=np.array([0.25], dtype=np.float32),
        masses=np.array([1.0], dtype=np.float32),
        lbox=1.0,
        particle_type="gas",
    )

    def fake_smoothed_grid(*_args, **_kwargs):
        return GridResult(
            density_grid=np.ones((4, 4, 4), dtype=np.float64),
            diagnostics={"grid_mass": 1.0},
            timings={"build_density_grid": 0.0},
            backend_metadata={"backend": "sphere", "smoothing": "periodic scipy tophat", "mas": "CIC"},
        )

    monkeypatch.setattr("clumping_factor.power_spectrum_cli.read_snapshot_metadata", lambda *_args: Metadata())
    monkeypatch.setattr("clumping_factor.power_spectrum_cli.estimate_full_load_bytes", lambda *_args: 1)
    monkeypatch.setattr(
        "clumping_factor.power_spectrum_cli.load_tng_particles",
        lambda *_args, **_kwargs: (particles, {"load_data": 0.0}),
    )
    monkeypatch.setattr("clumping_factor.grid.build_density_grid_scipy", fake_smoothed_grid)

    output = tmp_path / "pk-smoothed.json"
    args = build_power_spectrum_parser().parse_args(
        [
            "--particle-type", "gas",
            "--smoothing", "sphere",
            "--grid-size", "4",
            "--output", str(output),
        ]
    )
    written = run_power_spectrum(args)
    document = json.loads(written.read_text())

    assert document["parameters"]["smoothing"] == "sphere"
    assert document["grid"]["backend"]["backend"] == "sphere"
