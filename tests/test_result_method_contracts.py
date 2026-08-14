import json

import pytest

from clumping_factor.methods.clumping.alternative import AlternativeClumpingResult, write_alternative_clumping_result
from clumping_factor.diagnostics.density_ratio import write_density_ratio_result


def test_alternative_writer_distinguishes_native_and_grid_variants(tmp_path):
    raw = tmp_path / "raw.json"
    grid = tmp_path / "grid.json"
    write_alternative_clumping_result(
        AlternativeClumpingResult({"parameters": {"backend": "raw-volume"}}),
        raw,
    )
    write_alternative_clumping_result(
        AlternativeClumpingResult({"parameters": {"backend": "grid"}}),
        grid,
    )
    assert json.loads(raw.read_text())["method_spec"]["identifier"] == "alternative.raw-volume"
    assert json.loads(grid.read_text())["method_spec"]["identifier"] == "alternative.grid-masked"


def test_density_ratio_writer_declares_diagnostic_method(tmp_path):
    output, csv_output = write_density_ratio_result(
        {"parameters": {}, "rows": [{"threshold": 1.0, "ratio": 2.0}]},
        tmp_path / "ratio.json",
    )
    assert csv_output.exists()
    assert json.loads(output.read_text())["method_spec"]["identifier"] == "diagnostics.density-ratio"


def test_alternative_writer_rejects_unknown_backend(tmp_path):
    with pytest.raises(ValueError, match="Unknown alternative-clumping backend"):
        write_alternative_clumping_result(
            AlternativeClumpingResult({"parameters": {"backend": "mystery"}}),
            tmp_path / "invalid.json",
        )
