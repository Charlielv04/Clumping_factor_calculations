import csv
import json
from pathlib import Path

from clumping_factor.infrastructure.artifacts import (
    analysis_directory,
    validate_analysis_manifest,
    write_analysis_manifest,
)


def test_preflight_inventory_has_only_final_canonical_destinations():
    report = Path(__file__).parents[1] / "reports" / "migrations" / "results-consolidation-preflight.csv"
    rows = list(csv.DictReader(report.open(encoding="utf-8")))
    assert len(rows) == 4428
    final = {row["destination"] for row in rows if row["action"] != "deduplicate-alias"}
    forbidden = ("results/forest/", "results/analysis-od", "results/analysis-raw", "results/aida-tng/", "results/unknown/")
    for row in rows:
        if row["action"] != "delete-cache":
            assert row["destination"]
        assert "\\" not in row["destination"]
        assert not any(value in row["destination"] for value in forbidden)
        if row["action"] == "deduplicate-alias":
            assert row["destination"] in final
        if row["action"] == "move-analysis-artifact":
            assert "/analysis-" in row["destination"] and "/artifacts/" in row["destination"]


def test_analysis_preflight_specs_recompute_canonical_destinations():
    root = Path(__file__).parents[1]
    specs = list(csv.DictReader((root / "reports/migrations/results-consolidation-analysis-specs.csv").open(encoding="utf-8")))
    assert len(specs) == 1794
    allowed_families = {"aida-tng", "thesan", "combined", "legacy"}
    known_kinds = (
        ("/equation-story/", "equation-story"),
        ("/model-comparison/", "model-comparison"),
        ("/method-comparison/", "method-comparison"),
        ("/grid-comparison/", "grid-comparison"),
        ("/ionization/", "ionization"),
        ("/igm/", "igm"),
        ("/equations/", "equations"),
        ("/benchmark/", "benchmark"),
        ("/performance/", "performance"),
        ("campaign", "campaign-manifest"),
        ("/power-comparison/", "power-comparison"),
        ("/forest-comparison/", "forest-comparison"),
        ("/evolution/", "evolution"),
    )
    topology_kinds = (
        ("/analysis/equations/", "equations"),
        ("/analysis/performance/", "performance"),
        ("/analysis/power-spectra/", "power-comparison"),
        ("/analysis/forest/", "forest-comparison"),
    )
    filename_suffixes = (".png", ".csv", ".json", ".hdf5", ".txt", ".dat", ".md")
    for row in specs:
        labels = (row["domain"], row["family"], row["analysis_kind"], row["subject"], row["method_label"])
        assert all(value and not value.lower().endswith(filename_suffixes) for value in labels)
        assert row["family"] in allowed_families
        source = f"/{row['source'].lower()}"
        expected_kind = next((kind for token, kind in topology_kinds if token in source), None)
        expected_kind = expected_kind or next((kind for token, kind in known_kinds if token in source), None)
        if expected_kind is not None:
            assert row["analysis_kind"] == expected_kind
        directory = analysis_directory(root / "results", domain=row["domain"], family=row["family"], analysis_kind=row["analysis_kind"],
                                       subject=row["subject"], method_label=row["method_label"], options=json.loads(row["options_json"]), inputs=json.loads(row["inputs_json"]))
        assert row["destination"].endswith((directory / "artifacts" / Path(row["source"]).name).relative_to(root).as_posix())

    alternative = next(row for row in specs if row["source"] == "Thesan-2/alternative_clumping/eq13_clumping_vs_redshift.png")
    equations = next(row for row in specs if row["source"] == "Thesan-2/gas/equation-tests/snapshot080.csv")
    assert (alternative["domain"], alternative["family"], alternative["analysis_kind"], alternative["subject"], alternative["method_label"]) == (
        "clumping", "thesan", "evolution", "Thesan-2", "alternative.raw-volume",
    )
    assert (equations["domain"], equations["family"], equations["analysis_kind"], equations["subject"], equations["method_label"]) == (
        "diagnostics", "thesan", "equations", "Thesan-2", "diagnostics.equations",
    )
    assert any(row["subject"] == "thesan-mini-4-128-sl" for row in specs)
    assert any(row["subject"] == "thesan-mini-4-128-rsl" for row in specs)


def test_representative_preflight_analysis_manifests_validate(tmp_path):
    root = Path(__file__).parents[1]
    specs = list(csv.DictReader((root / "reports/migrations/results-consolidation-analysis-specs.csv").open(encoding="utf-8")))
    representatives = (
        next(row for row in specs if len(row["source"].split("/")) >= 9),
        next(row for row in specs if row["analysis_kind"] == "legacy-analysis"),
        next(row for row in specs if row["source"] == "Thesan-2/gas/equation-tests/snapshot080.csv"),
        next(row for row in specs if "output_4_128_sl/" in row["source"]),
    )
    # Keep the materialized canonical layout below the repository's deliberately
    # short pytest base directory; Windows still enforces the traditional path
    # limit for the full deeply nested analysis contract.
    materialized_root = tmp_path.parent / "m"
    for row in representatives:
        directory = analysis_directory(
            materialized_root / "results", domain=row["domain"], family=row["family"], analysis_kind=row["analysis_kind"],
            subject=row["subject"], method_label=row["method_label"], options=json.loads(row["options_json"]), inputs=[],
        )
        artifact = directory / "artifacts" / Path(row["source"]).name
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(b"preflight artifact")
        manifest = write_analysis_manifest(
            directory, domain=row["domain"], family=row["family"], analysis_kind=row["analysis_kind"],
            subject=row["subject"], method_label=row["method_label"], options=json.loads(row["options_json"]),
            inputs=[], artifacts=[artifact], generator="results-consolidation-preflight",
            legacy_sources=[row["source"]],
        )
        assert validate_analysis_manifest(manifest) == []
