import json

import numpy as np

from clumping_factor.infrastructure.models import GridResult, ParticleData
from clumping_factor.infrastructure.results import build_result_document, read_json_result, write_json_result


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

