import json

import numpy as np

from clumping_factor.infrastructure.artifacts import (
    analysis_directory, attach_artifacts, companion_artifact_path, validate_analysis_manifest, validate_archive_manifest,
    write_analysis_manifest, write_archive_manifest,
)
from clumping_factor.infrastructure.artifacts import validate_external_analysis_sidecar, write_explicit_analysis_sidecar
from clumping_factor.infrastructure.models import GridResult, ParticleData
from clumping_factor.infrastructure.results import build_result_document, normalize_simulation_identity, read_json_result, write_json_result
from clumping_factor.infrastructure.validation import validate_paths


def _document():
    particles = ParticleData(
        coords=np.zeros((1, 3)), radii=np.ones(1), masses=np.ones(1),
        lbox=1.0, particle_type="dm",
    )
    grid = GridResult(np.ones((1, 1, 1)), {}, {}, {"name": "fixture"})
    return build_result_document(
        particles, grid, np.array([1.0]), np.array([1.0]),
        {"base_path": "missing-fixture", "snapshot": 0, "load_mode": "full"}, {},
    )


def test_schema_two_records_reproducibility_metadata():
    document = _document()
    assert document["schema_version"] == 2
    assert document["provenance"]["estimator"]
    assert document["provenance"]["units"]["clumping_factor"] == "dimensionless"
    assert "numpy" in document["provenance"]["runtime"]["dependencies"]


def test_atomic_write_replaces_existing_result_and_cleans_temporary_file(tmp_path):
    output = tmp_path / "result.json"
    output.write_text("old", encoding="utf-8")
    write_json_result(_document(), output, method_id="clumping.cube")
    assert read_json_result(output)["schema_version"] == 2
    assert list(tmp_path.glob(".result.json.*.tmp")) == []


def test_reader_rejects_legacy_and_unknown_schemas(tmp_path):
    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps({"schema_version": 1, "thresholds": []}), encoding="utf-8")
    with np.testing.assert_raises_regex(ValueError, "only schema version 2"):
        read_json_result(legacy)

    future = tmp_path / "future.json"
    future.write_text(json.dumps({"schema_version": 999}), encoding="utf-8")
    try:
        read_json_result(future)
    except ValueError as exc:
        assert "Unsupported" in str(exc)
    else:
        raise AssertionError("unknown schemas must be rejected")


def test_companion_artifact_index_is_checksum_backed(tmp_path):
    result = tmp_path / "result.json"
    companion = companion_artifact_path(result, "equation-table", "csv")
    companion.parent.mkdir(parents=True)
    companion.write_text("threshold,value\n1,2\n", encoding="utf-8")
    indexed = attach_artifacts(_document(), result, [(companion, "equation-table")])
    assert indexed["artifacts"][0]["path"] == "result.artifacts/equation-table.csv"
    assert indexed["artifacts"][0]["sha256"]


def test_analysis_paths_and_manifests_are_content_addressed(tmp_path):
    input_a = tmp_path / "science-a.json"
    input_b = tmp_path / "science-b.json"
    input_a.write_text("a", encoding="utf-8")
    input_b.write_text("b", encoding="utf-8")
    directory = analysis_directory(
        tmp_path, domain="clumping", family="thesan", analysis_kind="evolution", subject="Thesan-1",
        method_label="cube", options={"threshold": 20}, inputs=[input_a, input_b],
    )
    artifact = directory / "artifacts" / "plot.png"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"plot")
    manifest = write_analysis_manifest(
        directory, domain="clumping", family="thesan", analysis_kind="evolution", subject="Thesan-1",
        method_label="cube", options={"threshold": 20}, inputs=[input_a, input_b], artifacts=[artifact],
        generator="tests", legacy_sources=["legacy/plot.png"],
    )
    assert directory.name.startswith("analysis-")
    assert validate_analysis_manifest(manifest) == []


def test_archive_manifest_rejects_paths_outside_archive(tmp_path):
    archive = tmp_path / "archive" / "import-1"
    artifact = archive / "old" / "plot.png"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"plot")
    manifest = write_archive_manifest(archive, import_id="import-1", files=[(artifact, "old/plot.png")])
    assert validate_archive_manifest(manifest) == []
    document = json.loads(manifest.read_text(encoding="utf-8"))
    document["files"][0]["path"] = "../../outside.png"
    manifest.write_text(json.dumps(document), encoding="utf-8")
    assert validate_archive_manifest(manifest)


def test_result_validation_skips_historical_archive_payloads(tmp_path):
    archive = tmp_path / "archive" / "import-1"
    payload = archive / "legacy-manifest.json"
    payload.parent.mkdir(parents=True)
    payload.write_text(json.dumps({"workflow_version": 1, "status": "historical"}), encoding="utf-8")
    write_archive_manifest(archive, import_id="import-1", files=[(payload, "old/manifest.json")])
    report = validate_paths([tmp_path])
    assert len(report) == 1
    assert report[0]["kind"] == "archive"
    assert report[0]["valid"]


def test_explicit_analysis_sidecar_is_strictly_validated(tmp_path):
    source = tmp_path / "source.json"
    artifact = tmp_path / "plot.png"
    source.write_text("{}", encoding="utf-8")
    artifact.write_bytes(b"plot")
    sidecar = write_explicit_analysis_sidecar(artifact, domain="plot", family="test", analysis_kind="result", options={}, inputs=[source], generator="tests")
    assert validate_external_analysis_sidecar(sidecar) == []


def test_mini_thesan_identities_are_normalized_from_name_or_base_path():
    assert normalize_simulation_identity("output_4_128_sl") == "thesan-mini-4-128-sl"
    assert normalize_simulation_identity("thesan_in_house", base_path="/work/output_100_256") == "thesan-mini-100-256"
    assert normalize_simulation_identity("Thesan-1-parallelized") == "Thesan-1"

