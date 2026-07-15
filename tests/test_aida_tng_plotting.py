import csv
import json
from pathlib import Path

from clumping_factor.aida_tng_plotting import (
    archive_aida_plots,
    canonical_plot_path,
    discover_aida_tng_results,
    generate_aida_tng_plots,
)


def _write_clumping(path: Path, simulation: str, snapshot: int, backend: str, grid: int = 256) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "simulation": {"name": simulation, "snapshot": snapshot, "redshift": 10.0 - snapshot / 100},
                "particle_type": "gas",
                "backend": {"backend": backend},
                "parameters": {"simulation_name": simulation, "snapshot": snapshot, "grid_size": grid},
                "thresholds": [0.0, 20.0],
                "clumping_factors": [1.0, 2.0],
            }
        ),
        encoding="utf-8",
    )


def test_canonical_plot_path_is_deterministic(tmp_path):
    assert canonical_plot_path(
        tmp_path,
        "clumping/method-comparison",
        "L35n1080_CDM",
        "snapshot017",
        "gas",
        "grid256",
        "backend_comparison.png",
    ) == tmp_path / "clumping" / "aida-tng" / "method-comparison" / "L35n1080_CDM" / "snapshot017" / "gas" / "grid256" / "backend_comparison.png"


def test_discover_groups_clumping_results_and_skips_malformed_json(tmp_path):
    root = tmp_path / "results" / "aida-tng"
    good = root / "L35n1080_CDM" / "gas" / "sphere" / "snapshot017_grid256" / "run.json"
    _write_clumping(good, "L35n1080_CDM", 17, "sphere")
    (root / "broken.json").write_text("not json", encoding="utf-8")

    results = discover_aida_tng_results(root)

    assert len(results) == 1
    assert results[0].simulation == "L35n1080_CDM"
    assert results[0].snapshot == 17
    assert results[0].grid == 256
    assert results[0].kind == "clumping"


def test_dry_run_writes_manifest_and_plans_method_and_evolution_outputs(tmp_path):
    root = tmp_path / "results" / "aida-tng"
    for snapshot in (17, 99):
        _write_clumping(root / "L35n1080_CDM" / "gas" / "sphere" / f"snapshot{snapshot:03d}_grid256" / "run.json", "L35n1080_CDM", snapshot, "sphere")
        _write_clumping(root / "L35n1080_CDM" / "gas" / "cube" / f"snapshot{snapshot:03d}_grid256" / "run.json", "L35n1080_CDM", snapshot, "cube")

    analysis = tmp_path / "results" / "analysis"
    outputs = generate_aida_tng_plots(root, analysis, dry_run=True)
    manifest = analysis / "manifests" / "aida-tng-plots.csv"
    rows = list(csv.DictReader(manifest.open(encoding="utf-8")))

    assert any("backend_comparison.png" in str(path) for path in outputs)
    assert any("clumping_vs_redshift.png" in str(path) for path in outputs)
    assert {row["status"] for row in rows} == {"planned"}
    assert all(row["inputs"] for row in rows)


def test_archive_existing_dry_run_does_not_move_files(tmp_path):
    source = tmp_path / "analysis" / "clumping" / "aida-tng" / "old"
    source.mkdir(parents=True)
    plot = source / "old.png"
    plot.write_bytes(b"png")

    archive = archive_aida_plots(tmp_path / "analysis", dry_run=True)

    assert archive is not None
    assert plot.exists()
