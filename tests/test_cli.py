import json
from argparse import Namespace
from pathlib import Path

import numpy as np

from clumping_factor.cli import build_campaign_plot_parser, build_compute_parser, evolution_plot_main, plot_main, run_compute
from clumping_factor.models import GridResult, ParticleData
from clumping_factor.plotting import _auto_plot_context, _campaign_simulation_name, _plot_label
from clumping_factor.results import canonical_thesan_result_path, default_output_path, resolve_simulation_name


def test_compute_help():
    parser = build_compute_parser()
    help_text = parser.format_help()
    assert "--particle-type" in help_text
    assert "--backend" in help_text
    assert "raw" in help_text
    assert "raw-volume" in help_text
    assert "raw-transmission" in help_text
    assert "--sigma-bar-ion-cm2" in help_text
    assert "--radius-mode" in help_text
    assert "--simulation-name" in help_text
    assert "--load-mode" in help_text
    assert "--chunk-size" in help_text
    assert "--radius-bin-batch-size" in help_text
    assert "--progress-interval" in help_text
    assert "--threads" in help_text
    assert "--memory-limit" in help_text
    assert "--memory-safety-fraction" in help_text
    assert "--summary-cache" in help_text
    assert "--work-partition" in help_text
    assert "--max-file-readers" in help_text
    assert "TSC" in help_text


def test_campaign_plot_help():
    parser = build_campaign_plot_parser()
    help_text = parser.format_help()
    assert "--batch" in help_text
    assert "--grid" in help_text
    assert "--particle" in help_text
    assert "--baseline-batch" in help_text
    assert "--analysis-root" in help_text


def test_campaign_simulation_name_prefers_physical_thesan_name():
    document = {
        "simulation": {
            "name": "Thesan-2-grid-campaign",
            "base_path": "/lustre/work/carlos.lopez/Thesan-2/output",
        },
        "parameters": {"simulation_name": "Thesan-2-grid-campaign"},
    }
    path = Path(
        "results/thesan/Thesan-2/gas/pylians/"
        "snapshot080_grid256/threads8_batch2_run001.json"
    )

    assert _campaign_simulation_name(path, document) == "Thesan-2"


def test_compute_parser_accepts_tsc_for_pylians():
    args = build_compute_parser().parse_args(["--particle-type", "gas", "--backend", "pylians", "--mas", "TSC"])
    assert args.mas == "TSC"


def test_compute_parser_accepts_tsc_for_scipy():
    args = build_compute_parser().parse_args(["--particle-type", "gas", "--backend", "sphere", "--mas", "TSC"])
    assert args.mas == "TSC"


def test_raw_transmission_requires_sigma_and_provenance(tmp_path):
    args = build_compute_parser().parse_args(["--particle-type", "gas", "--backend", "raw-transmission"])
    args.output = str(tmp_path / "result.json")
    try:
        run_compute(args)
    except ValueError as exc:
        assert "sigma-bar-ion-cm2" in str(exc)
    else:
        raise AssertionError("raw-transmission should require sigma")

    args.sigma_bar_ion_cm2 = 1e-18
    try:
        run_compute(args)
    except ValueError as exc:
        assert "sigma-bar-ion-source" in str(exc)
    else:
        raise AssertionError("raw-transmission should require sigma provenance")


def test_raw_transmission_rejects_dm():
    args = build_compute_parser().parse_args(
        [
            "--particle-type", "dm",
            "--backend", "raw-transmission",
            "--sigma-bar-ion-cm2", "1e-18",
            "--sigma-bar-ion-source", "test",
        ]
    )
    try:
        run_compute(args)
    except ValueError as exc:
        assert "gas" in str(exc)
    else:
        raise AssertionError("raw-transmission should reject dark matter")


def test_plot_help_includes_quantity():
    from clumping_factor.cli import build_plot_parser

    help_text = build_plot_parser().format_help()
    assert "--quantity" in help_text
    assert "cell-count" in help_text


def test_evolution_plot_help_includes_threshold():
    from clumping_factor.cli import build_evolution_plot_parser

    help_text = build_evolution_plot_parser().format_help()
    assert "--threshold" in help_text
    assert "redshift" in help_text


def test_simulation_name_inferred_from_base_path():
    assert resolve_simulation_name("../Thesan-1") == "Thesan-1"
    assert resolve_simulation_name("./tng100-3/output") == "tng100-3"
    assert resolve_simulation_name("./anything", "Thesan 1!") == "Thesan-1"


def test_default_output_path_uses_simulation_subdirectory():
    output = default_output_path("results", "gas", "sphere", 81, 256, "Thesan-1")
    assert output.as_posix() == "results/Thesan-1/gas_sphere_snapshot081_grid256.json"


def test_canonical_thesan_result_path():
    output = canonical_thesan_result_path("results", "Thesan-1", "dm", "pylians", 81, 512, 16, 10, 1)
    assert output.as_posix() == "results/thesan/Thesan-1/dm/pylians/snapshot081_grid512/threads16_batch10_run001.json"


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


def test_plot_command_writes_selected_cell_count_plot(tmp_path):
    result_json = tmp_path / "result.json"
    result_json.write_text(
        json.dumps(
            {
                "particle_type": "gas",
                "backend": {"backend": "sphere"},
                "thresholds": [-1.0, 0.0, 1.0],
                "clumping_factors": [None, 1.0, 1.2],
                "diagnostics": {"clumping": {"selected_cell_counts": [0, 4, 8]}},
            }
        )
    )
    output = tmp_path / "cell-counts.png"
    plot_main([str(result_json), "--quantity", "cell-count", "--output", str(output)])
    assert output.exists()
    assert output.stat().st_size > 0


def test_plot_command_writes_relative_to_baseline_plot(tmp_path):
    baseline = tmp_path / "baseline.json"
    comparison = tmp_path / "comparison.json"
    baseline.write_text(
        json.dumps(
            {
                "particle_type": "gas",
                "backend": {"backend": "pylians"},
                "thresholds": [-1.0, 0.0, 1.0],
                "clumping_factors": [1.0, 2.0, 4.0],
            }
        )
    )
    comparison.write_text(
        json.dumps(
            {
                "particle_type": "gas",
                "backend": {"backend": "pylians"},
                "thresholds": [-1.0, 0.0, 1.0],
                "clumping_factors": [1.5, 1.0, 8.0],
            }
        )
    )
    output = tmp_path / "relative.png"
    plot_main(
        [
            str(baseline),
            str(comparison),
            "--relative-to-baseline",
            str(baseline),
            "--output",
            str(output),
        ]
    )
    assert output.exists()
    assert output.stat().st_size > 0


def test_relative_to_baseline_rejects_cell_count_quantity(tmp_path):
    result_json = tmp_path / "result.json"
    result_json.write_text(
        json.dumps(
            {
                "thresholds": [-1.0, 0.0],
                "clumping_factors": [1.0, 2.0],
                "diagnostics": {"clumping": {"selected_cell_counts": [1, 2]}},
            }
        )
    )
    try:
        plot_main(
            [
                str(result_json),
                "--quantity",
                "cell-count",
                "--relative-to-baseline",
                str(result_json),
                "--output",
                str(tmp_path / "bad.png"),
            ]
        )
    except ValueError as exc:
        assert "relative-to-baseline" in str(exc)
    else:
        raise AssertionError("relative baseline plotting should reject cell-count plots")


def test_cell_count_plot_rejects_missing_diagnostics(tmp_path):
    result_json = tmp_path / "result.json"
    result_json.write_text(json.dumps({"thresholds": [-1.0, 0.0], "clumping_factors": [None, 1.0]}))
    output = tmp_path / "cell-counts.png"
    try:
        plot_main([str(result_json), "--quantity", "cell-count", "--output", str(output)])
    except ValueError as exc:
        assert "cell-count" in str(exc)
    else:
        raise AssertionError("cell-count plotting should reject missing diagnostics")


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


def test_threshold_plot_rejects_scalar_transmission_result(tmp_path):
    result_json = tmp_path / "scalar.json"
    result_json.write_text(
        json.dumps(
            {
                "particle_type": "gas",
                "backend": {"backend": "raw-transmission"},
                "clumping_factor": 2.0,
            }
        )
    )
    try:
        plot_main([str(result_json), "--output", str(tmp_path / "bad.png")])
    except ValueError as exc:
        assert "scalar" in str(exc)
    else:
        raise AssertionError("threshold plotting should reject scalar transmission results")


def _write_evolution_result(path, redshift, factors, grid_size=256):
    path.write_text(
        json.dumps(
            {
                "simulation": {"name": "sim", "snapshot": int(redshift * 10), "redshift": redshift},
                "particle_type": "gas",
                "parameters": {
                    "grid_size": grid_size,
                    "radius_bins": 10,
                    "target": {"particle_type": "gas", "backend": "sphere", "radius_mode": "sphere"},
                    "mask": {"particle_type": "gas", "backend": "sphere", "radius_mode": "sphere"},
                },
                "backend": {"backend": "sphere"},
                "thresholds": [0.0, 10.0, 20.0],
                "clumping_factors": factors,
            }
        )
    )


def test_threshold_plot_labels_same_method_by_redshift(tmp_path):
    first = tmp_path / "snapshot040_grid256" / "threads8_batch2_run001.json"
    second = tmp_path / "snapshot080_grid256" / "threads8_batch2_run001.json"
    first.parent.mkdir()
    second.parent.mkdir()
    _write_evolution_result(first, 10.0, [1.0, 2.0, 3.0])
    _write_evolution_result(second, 6.0, [1.5, 2.5, 3.5])
    documents = [(first, json.loads(first.read_text())), (second, json.loads(second.read_text()))]

    label_mode, legend_title, title = _auto_plot_context(documents, "clumping-factor")
    seen = set()
    labels = [_plot_label(document, path, seen, label_mode) for path, document in documents]

    assert label_mode == "redshift"
    assert legend_title == "Redshift"
    assert labels == ["z = 10.00", "z = 6.00"]
    assert "sim gas sphere grid 256" in title
    assert "threads" not in " ".join(labels)


def test_threshold_plot_labels_cross_method_by_method(tmp_path):
    sphere = tmp_path / "sphere.json"
    pylians = tmp_path / "pylians.json"
    _write_evolution_result(sphere, 6.0, [1.0, 2.0, 3.0])
    _write_evolution_result(pylians, 6.0, [1.5, 2.5, 3.5])
    pylians_doc = json.loads(pylians.read_text())
    pylians_doc["backend"] = {"backend": "pylians"}
    pylians.write_text(json.dumps(pylians_doc))
    documents = [(sphere, json.loads(sphere.read_text())), (pylians, json.loads(pylians.read_text()))]

    label_mode, legend_title, _ = _auto_plot_context(documents, "clumping-factor")
    seen = set()
    labels = [_plot_label(document, path, seen, label_mode) for path, document in documents]

    assert label_mode == "method"
    assert legend_title == "Method"
    assert labels == ["gas sphere, grid 256", "gas pylians, grid 256"]


def test_evolution_plot_combines_snapshots_and_interpolates(tmp_path):
    high_z = tmp_path / "high.json"
    low_z = tmp_path / "low.json"
    _write_evolution_result(high_z, 8.0, [1.0, 2.0, 3.0])
    _write_evolution_result(low_z, 4.0, [2.0, 4.0, 6.0])
    output = tmp_path / "evolution.png"

    evolution_plot_main(
        [str(high_z), str(low_z), "--threshold", "5", "--threshold", "20", "--output", str(output)]
    )

    assert output.exists()
    assert output.stat().st_size > 0


def test_evolution_plot_rejects_mixed_grid_configuration(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    _write_evolution_result(first, 8.0, [1.0, 2.0, 3.0], grid_size=256)
    _write_evolution_result(second, 4.0, [2.0, 4.0, 6.0], grid_size=512)

    try:
        evolution_plot_main([str(first), str(second), "--output", str(tmp_path / "bad.png")])
    except ValueError as exc:
        assert "configuration" in str(exc)
    else:
        raise AssertionError("evolution plotting should reject mixed configurations")


def test_evolution_plot_accepts_scalar_transmission_results(monkeypatch, tmp_path):
    first = tmp_path / "first_scalar.json"
    second = tmp_path / "second_scalar.json"
    for path, redshift, factor in [(first, 8.0, 2.0), (second, 6.0, 3.0)]:
        path.write_text(
            json.dumps(
                {
                    "simulation": {"name": "sim", "redshift": redshift},
                    "particle_type": "gas",
                    "parameters": {"grid_size": 256, "mas": "CIC"},
                    "backend": {"backend": "raw-transmission", "method": "test"},
                    "clumping_factor": factor,
                }
            )
        )

    monkeypatch.setattr("matplotlib.figure.Figure.savefig", lambda self, path, **kwargs: path.write_bytes(b"plot"))
    output = tmp_path / "scalar_evolution.png"
    evolution_plot_main([str(first), str(second), "--output", str(output)])
    assert output.read_bytes() == b"plot"
