import ast
from pathlib import Path

import pytest

from clumping_factor.infrastructure.results import canonical_result_path, with_result_specs


def test_new_result_specs_preserve_science_payload():
    document = with_result_specs(
        {"thresholds": [1], "clumping_factors": [2], "parameters": {"backend": "cube", "particle_type": "dm"}},
        method_id="clumping.cube",
    )
    assert document["thresholds"] == [1]
    assert document["clumping_factors"] == [2]
    assert document["method_spec"]["identifier"] == "clumping.cube"
    assert {"selection_spec", "execution_spec"} <= document.keys()


def test_canonical_path_is_pure_and_stable():
    method_spec = with_result_specs(
        {"parameters": {"backend": "raw-volume"}}, method_id="clumping.raw-volume-weighted"
    )["method_spec"]
    path = canonical_result_path(
        "results", family="tng", simulation_name="TNG 100", particle_type="gas", snapshot=98,
        method_spec=method_spec, selection_spec={"overdensity": 100}, execution_spec={"threads": 1},
    )
    assert path.parts[:7] == (
        "results", "tng", "TNG-100", "clumping", "clumping.raw-volume-weighted", "gas", "snapshot098"
    )
    assert path.parts[7].startswith("science-")
    assert path.name.startswith("execution-") and path.name.endswith("_run001.json")


@pytest.mark.parametrize(
    ("method_id", "expected"),
    [
        ("alternative.raw-volume", "alternative.raw-volume"),
        ("alternative.grid-masked", "alternative.grid-masked"),
        ("alternative.ionized-sweep", "alternative.ionized-sweep"),
        ("power-spectrum.numpy", "power-spectrum.numpy"),
        ("forest.mfp", "forest.mfp"),
        ("thermodynamics.snapshot-temperature", "thermodynamics.snapshot-temperature"),
        ("diagnostics.density-ratio", "diagnostics.density-ratio"),
        ("diagnostics.equations", "diagnostics.equations"),
    ],
)
def test_explicit_method_ids_are_not_guessed(method_id: str, expected: str):
    document = with_result_specs({"parameters": {}}, method_id=method_id)
    assert document["method_spec"]["identifier"] == expected


def test_method_identity_is_never_guessed_from_legacy_fields():
    with pytest.raises(ValueError, match="explicit registered method_id"):
        with_result_specs({"parameters": {}, "backend": "pylians"})


def test_unknown_explicit_producer_method_is_rejected():
    with pytest.raises(ValueError, match="unregistered method identifier"):
        with_result_specs({"parameters": {}}, method_id="clumping.not-a-method")


def test_explicit_producer_contract_replaces_stale_method_metadata():
    document = with_result_specs(
        {"parameters": {}, "method_spec": {"identifier": "clumping.sphere"}},
        method_id="diagnostics.equations",
    )
    assert document["method_spec"]["identifier"] == "diagnostics.equations"


def test_domain_cli_adapters_do_not_import_scientific_libraries():
    package_root = Path(__file__).parents[1] / "src" / "clumping_factor"
    adapters = sorted(package_root.rglob("cli_adapter.py"))
    assert adapters
    forbidden = {"numpy", "scipy", "h5py"}
    for path in adapters:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported = {
            alias.name.split(".", 1)[0]
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        assert not (imported & forbidden), f"Scientific dependency leaked into {path}"

