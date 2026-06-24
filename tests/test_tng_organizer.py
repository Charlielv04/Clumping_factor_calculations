import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "organize_tng_results.py"
SPEC = importlib.util.spec_from_file_location("organize_tng_results", SCRIPT_PATH)
organizer = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(organizer)


def write_result(path: Path, *, simulation="tng100-3", particle="gas", backend="cube", snapshot=98, grid=512, threads=4):
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


def test_parse_filename_supports_tng_legacy_and_canonical_formats():
    short = organizer.parse_filename(Path("gas_cube_512.json"))
    assert short["particle"] == "gas"
    assert short["backend"] == "cube"
    assert short["grid"] == "512"

    benchmark = organizer.parse_filename(Path("gas_pylians_grid256_threads8_batch10_run2.json"))
    assert benchmark["particle"] == "gas"
    assert benchmark["threads"] == "8"
    assert benchmark["batch"] == "10"
    assert benchmark["run"] == "2"

    diagnostic = organizer.parse_filename(Path("gas_cube_256_diag.json"))
    assert diagnostic["particle"] == "gas"
    assert diagnostic["backend"] == "cube"
    assert diagnostic["grid"] == "256"

    raw = organizer.parse_filename(Path("gas_raw_volume.json"))
    assert raw["particle"] == "gas"
    assert raw["backend"] == "raw_volume"

    canonical = organizer.parse_filename(
        Path("results/tng/tng100-3/gas/cube/snapshot098_grid256/threads8_batch10_run001.json")
    )
    assert canonical["snapshot"] == "098"
    assert canonical["grid"] == "256"
    assert canonical["threads"] == "8"


def test_build_reports_assigns_unique_tng_canonical_runs(tmp_path):
    results = tmp_path / "results"
    first = results / "tng100-3-benchmark" / "gas_cube_grid512_threads4_batch10_run1.json"
    second = results / "tng100-3-batching" / "gas_cube_grid512_threads4_batch10_run1.json"
    write_result(first)
    write_result(second)

    manifest, duplicates, moves = organizer.build_reports(results)

    assert len(manifest) == 2
    destinations = {row["destination_path"] for row in moves if row["file_type"] == "json"}
    assert destinations == {
        str(results / "tng" / "tng100-3" / "gas" / "cube" / "snapshot098_grid512" / "threads4_batch10_run001.json"),
        str(results / "tng" / "tng100-3" / "gas" / "cube" / "snapshot098_grid512" / "threads4_batch10_run002.json"),
    }
    assert {row["status"] for row in duplicates} == {"byte-identical"}


def test_default_scan_excludes_existing_tng_canonical_outputs(tmp_path):
    results = tmp_path / "results"
    legacy = results / "tng100-3" / "gas_cube_512.json"
    canonical = results / "tng" / "tng100-3" / "gas" / "cube" / "snapshot098_grid512" / "threads1_batch1_run001.json"
    write_result(legacy, threads=1)
    write_result(canonical, threads=1)

    assert organizer.tng_json_paths(results) == [legacy]
    assert set(organizer.tng_json_paths(results, include_canonical=True)) == {legacy, canonical}


def test_root_level_tng_outputs_are_scanned(tmp_path):
    results = tmp_path / "results"
    loose = results / "gas_raw_volume.json"
    write_result(loose, backend="raw-volume", grid=None)

    manifest, _duplicates, moves = organizer.build_reports(results)

    assert len(manifest) == 1
    assert moves[0]["destination_path"] == str(
        results / "tng" / "tng100-3" / "gas" / "raw-volume" / "snapshot098_nogrid" / "threads1_batch10_run001.json"
    )
