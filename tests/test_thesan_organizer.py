import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "organize_thesan_results.py"
SPEC = importlib.util.spec_from_file_location("organize_thesan_results", SCRIPT_PATH)
organizer = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(organizer)


def write_result(path: Path, *, simulation="Thesan-1", particle="dm", backend="pylians", snapshot=81, grid=512, threads=16):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "simulation": {"name": simulation, "base_path": f"../{simulation}/output", "snapshot": snapshot},
                "particle_type": particle,
                "parameters": {
                    "simulation_name": simulation,
                    "base_path": f"../{simulation}/output",
                    "snapshot": snapshot,
                    "grid_size": grid,
                    "radius_bin_batch_size": 10,
                    "mas": "CIC",
                    "filter_type": "Top-Hat",
                },
                "backend": {"backend": backend},
                "thresholds": [0.0, 1.0],
                "clumping_factors": [1.0, 2.0],
                "timings": {"total": 12.0},
            }
        ),
        encoding="utf-8",
    )


def test_parse_filename_supports_old_new_and_canonical_formats():
    old = organizer.parse_filename(Path("gas_cube_grid256_threads8_batch10.json"))
    assert old["particle"] == "gas"
    assert old["backend"] == "cube"
    assert old["grid"] == "256"
    assert old["threads"] == "8"

    new = organizer.parse_filename(Path("dm_pylians_snapshot081_grid512_threads16_batch10_run1.json"))
    assert new["particle"] == "dm"
    assert new["snapshot"] == "081"
    assert new["run"] == "1"

    canonical = organizer.parse_filename(
        Path("results/thesan/Thesan-1/dm/pylians/snapshot081_grid512/threads16_batch10_run001.json")
    )
    assert canonical["snapshot"] == "081"
    assert canonical["grid"] == "512"
    assert canonical["threads"] == "16"
    assert canonical["batch"] == "10"
    assert canonical["run"] == "001"


def test_build_reports_assigns_unique_canonical_runs(tmp_path):
    results = tmp_path / "results"
    first = results / "Thesan-1-dm-pylians-grid-thread-scaling" / "dm_pylians_snapshot081_grid512_threads16_batch10_run1.json"
    second = results / "Thesan-1-dm-pylians-grid-thread-scaling" / "dm_pylians_snapshot081_grid512_threads16_batch10_run2.json"
    write_result(first)
    write_result(second)

    manifest, duplicates, moves = organizer.build_reports(results)

    assert len(manifest) == 2
    destinations = {row["destination_path"] for row in moves if row["file_type"] == "json"}
    assert len(destinations) == 2
    assert any("threads16_batch10_run001.json" in path for path in destinations)
    assert any("threads16_batch10_run002.json" in path for path in destinations)
    assert {row["status"] for row in duplicates} == {"byte-identical"}


def test_canonical_runs_are_unique_even_when_metadata_differs(tmp_path):
    results = tmp_path / "results"
    first = results / "Thesan-2-scaling" / "gas_cube_grid256_threads8_batch10.json"
    second = results / "Thesan-2-new-pipeline" / "gas_cube_grid256_threads8_batch10_run1.json"
    write_result(first, simulation="Thesan-2", particle="gas", backend="cube", snapshot=80, grid=256, threads=8)
    write_result(second, simulation="Thesan-2", particle="gas", backend="cube", snapshot=80, grid=256, threads=8)
    document = json.loads(second.read_text(encoding="utf-8"))
    document["parameters"]["filter_type"] = "Different"
    second.write_text(json.dumps(document), encoding="utf-8")

    _manifest, _duplicates, moves = organizer.build_reports(results)

    destinations = sorted(row["destination_path"] for row in moves if row["file_type"] == "json")
    assert destinations == [
        str(results / "thesan" / "Thesan-2" / "gas" / "cube" / "snapshot080_grid256" / "threads8_batch10_run001.json"),
        str(results / "thesan" / "Thesan-2" / "gas" / "cube" / "snapshot080_grid256" / "threads8_batch10_run002.json"),
    ]


def test_numeric_duplicate_report_marks_different_curves(tmp_path):
    results = tmp_path / "results"
    first = results / "Thesan-1-scaling" / "gas_cube_grid256_threads8_batch10_run1.json"
    second = results / "Thesan-1-scaling" / "gas_cube_grid256_threads8_batch10_run2.json"
    write_result(first, particle="gas", backend="cube", grid=256, threads=8)
    write_result(second, particle="gas", backend="cube", grid=256, threads=8)
    document = json.loads(second.read_text(encoding="utf-8"))
    document["clumping_factors"] = [1.0, 3.0]
    second.write_text(json.dumps(document), encoding="utf-8")

    _manifest, duplicates, _moves = organizer.build_reports(results)

    assert "numerically-different" in {row["status"] for row in duplicates}


def test_plot_destination_includes_snapshot_level(tmp_path):
    results = tmp_path / "results"
    plot = results / "analysis" / "old" / "Thesan-2_snapshot080_dm_pylians_performance_dashboard.png"

    destination = organizer.infer_plot_destination(results, plot)

    assert destination == (
        results
        / "analysis"
        / "performance"
        / "thesan"
        / "Thesan-2"
        / "snapshot080"
        / "dm"
        / "pylians"
        / plot.name
    )


def test_default_scan_excludes_existing_canonical_outputs(tmp_path):
    results = tmp_path / "results"
    legacy = results / "Thesan-1-scaling" / "gas_cube_grid256_threads8_batch10.json"
    canonical = results / "thesan" / "Thesan-1" / "gas" / "cube" / "snapshot081_grid256" / "threads8_batch10_run001.json"
    write_result(legacy, particle="gas", backend="cube", grid=256, threads=8)
    write_result(canonical, particle="gas", backend="cube", grid=256, threads=8)

    assert organizer.thesan_json_paths(results) == [legacy]
    assert set(organizer.thesan_json_paths(results, include_canonical=True)) == {legacy, canonical}


def test_plot_scan_excludes_existing_canonical_analysis_outputs(tmp_path):
    results = tmp_path / "results"
    legacy = results / "analysis" / "old-thesan" / "Thesan-1_performance.png"
    canonical = results / "analysis" / "performance" / "thesan" / "Thesan-1" / "snapshot081" / "gas" / "cube" / "Thesan-1_performance.png"
    legacy.parent.mkdir(parents=True, exist_ok=True)
    canonical.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_text("legacy", encoding="utf-8")
    canonical.write_text("canonical", encoding="utf-8")

    assert organizer.thesan_plot_paths(results) == [legacy]
