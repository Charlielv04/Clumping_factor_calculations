import json

import h5py
import numpy as np

from clumping_factor.forest.ionizing import compute_gamma_hi_result
from clumping_factor.forest.workflow import SnapshotWorkflowConfig, run_snapshot_workflow
from clumping_factor.forest.workflow_cli import build_snapshot_parser
from test_equation_tests import _write_snapshot, _write_tigm
from test_forest import _write_los


def _inputs(tmp_path):
    base = _write_snapshot(tmp_path / "output")
    los = _write_los(tmp_path / "rays_080.hdf5", hi_scale=1e8)
    with h5py.File(los, "a") as handle:
        handle["Header"].attrs["Redshift"] = 4.0
    _write_tigm(base / "snapdir_080" / "Tigm_Thesan1.dat")
    return base, los


def test_chunked_gamma_matches_across_chunk_sizes(tmp_path):
    base, _ = _inputs(tmp_path)
    path = base / "snapdir_080" / "snap_080.0.hdf5"
    whole = compute_gamma_hi_result([path], cross_check=True, chunk_size=100)
    tiny = compute_gamma_hi_result([path], cross_check=True, chunk_size=1)
    assert np.isclose(whole.gamma_hi_s_1, tiny.gamma_hi_s_1, rtol=1e-14)
    assert np.isclose(tiny.gamma_hi_s_1, tiny.reference_gamma_hi_s_1, rtol=1e-12)


def test_threaded_gamma_matches_serial(tmp_path):
    base, _ = _inputs(tmp_path)
    path = base / "snapdir_080" / "snap_080.0.hdf5"
    serial = compute_gamma_hi_result([path, path], cross_check=True, chunk_size=1, workers=1)
    threaded = compute_gamma_hi_result([path, path], cross_check=True, chunk_size=1, workers=2)
    assert np.isclose(serial.gamma_hi_s_1, threaded.gamma_hi_s_1, rtol=1e-14)
    assert serial.selected_cells == threaded.selected_cells


def test_combined_lya_mfp_loads_los_once(tmp_path, monkeypatch):
    base, los = _inputs(tmp_path)
    import clumping_factor.forest.workflow as workflow
    original = workflow.read_thesan_random_los
    calls = []
    def counted(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)
    monkeypatch.setattr(workflow, "read_thesan_random_los", counted)
    result = run_snapshot_workflow(SnapshotWorkflowConfig(
        base, 80, "Thesan-1", ["lya", "mfp"], los_file=los,
        output_root=tmp_path / "results", resolution_kms=25,
        mfp_starts_per_ray=2, mfp_cross_check=True,
    ))
    assert result.succeeded
    assert len(calls) == 1
    assert set(result.document["products"]) == {"lya", "mfp"}


def test_workflow_resume_and_refresh(tmp_path):
    base, _ = _inputs(tmp_path)
    config = SnapshotWorkflowConfig(base, 80, "Thesan-1", ["gamma"], output_root=tmp_path / "results", gamma_chunk_size=1)
    first = run_snapshot_workflow(config)
    second = run_snapshot_workflow(config)
    assert first.document["products"]["gamma"]["status"] == "success"
    assert second.document["products"]["gamma"]["status"] == "reused"
    refreshed = run_snapshot_workflow(SnapshotWorkflowConfig(
        base, 80, "Thesan-1", ["gamma"], output_root=tmp_path / "results",
        gamma_chunk_size=1, refresh_products=True,
    ))
    assert refreshed.document["products"]["gamma"]["status"] == "success"


def test_threaded_workflow_records_worker_counts(tmp_path):
    base, los = _inputs(tmp_path)
    result = run_snapshot_workflow(SnapshotWorkflowConfig(
        base, 80, "Thesan-1", ["mfp", "gamma", "equations"], los_file=los,
        output_root=tmp_path / "results", mfp_starts_per_ray=2, threads=2,
        thresholds=[1e9], ionized_density_thresholds=[1e9], ionized_sweep=True,
        photon_group_tests=["0", "1"], ionized_cut_min=0.9,
        ionized_cut_max=0.99, ionized_cut_count=2,
        equation_chunk_size=1, gamma_chunk_size=1,
    ))
    assert result.succeeded, result.failures
    assert result.document["products"]["mfp"]["details"]["workers"] == 2
    assert result.document["products"]["gamma"]["details"]["workers"] == 2
    assert result.document["products"]["equations"]["details"]["workers"] == 2


def test_workflow_computes_missing_temperature_table(tmp_path):
    base, los = _inputs(tmp_path)
    (base / "snapdir_080" / "Tigm_Thesan1.dat").unlink()
    result = run_snapshot_workflow(SnapshotWorkflowConfig(
        base, 80, "Thesan-1", ["equations"], los_file=los,
        output_root=tmp_path / "results", mfp_starts_per_ray=2,
        thresholds=[1e9], ionized_density_thresholds=[1e9],
        ionized_cuts=[0.7], photon_group_tests=["0"],
        equation_chunk_size=1, gamma_chunk_size=1,
    ))
    assert result.succeeded, result.failures
    assert (base / "snapdir_080" / "Tigm_from_sim.dat").exists()
    root = tmp_path / "results" / "thesan" / "Thesan-1" / "snapshot080"
    equations = json.loads((root / "equations" / "equations.json").read_text())
    assert equations["parameters"]["Tigm_table"].endswith("Tigm_from_sim.dat")


def test_product_only_rerun_preserves_other_manifest_products(tmp_path):
    base, los = _inputs(tmp_path)
    common = dict(base_path=base, snapshot=80, simulation_name="Thesan-1", los_file=los,
                  output_root=tmp_path / "results", mfp_starts_per_ray=2)
    first = run_snapshot_workflow(SnapshotWorkflowConfig(products=["mfp", "gamma"], **common))
    assert first.succeeded
    second = run_snapshot_workflow(SnapshotWorkflowConfig(products=["gamma"], refresh_products=True, **common))
    assert set(second.document["products"]) == {"mfp", "gamma"}


def test_partial_failure_continues_independent_product(tmp_path):
    base, _ = _inputs(tmp_path)
    result = run_snapshot_workflow(SnapshotWorkflowConfig(
        base, 80, "Thesan-1", ["lya", "gamma"], los_file=tmp_path / "missing.hdf5",
        output_root=tmp_path / "results",
    ))
    assert not result.succeeded
    assert result.document["products"]["lya"]["status"] == "failed"
    assert result.document["products"]["gamma"]["status"] == "success"
    assert json.loads(result.manifest_path.read_text())["status"] == "failed"


def test_all_products_write_canonical_snapshot_tree(tmp_path):
    base, los = _inputs(tmp_path)
    result = run_snapshot_workflow(SnapshotWorkflowConfig(
        base, 80, "Thesan-1", ["lya", "mfp", "gamma", "equations"], los_file=los,
        output_root=tmp_path / "results", resolution_kms=25, mfp_starts_per_ray=2,
        thresholds=[1e9], ionized_density_thresholds=[1e9], ionized_sweep=True,
        photon_group_tests=["0", "1", "2", "0+1", "1+2", "0+1+2"],
        ionized_cut_min=0.9, ionized_cut_max=0.99, ionized_cut_count=2,
        equation_chunk_size=2, gamma_chunk_size=2,
    ))
    assert result.succeeded, result.failures
    root = tmp_path / "results" / "thesan" / "Thesan-1" / "snapshot080"
    assert (root / "manifest.json").exists()
    assert (root / "lya" / "rays_080_lya.hdf5").exists()
    assert (root / "mfp912" / "rays_080_mfp912.json").exists()
    assert (root / "gamma_hi" / "gamma_hi.json").exists()
    assert (root / "equations" / "equations.json").exists()
    equations = json.loads((root / "equations" / "equations.json").read_text())
    assert equations["parameters"]["ionized_density_thresholds"] == [1e9]
    assert equations["parameters"]["ionized_cuts"] == [0.9, 0.99]
    assert [row["label"] for row in equations["parameters"]["photon_group_tests"]] == [
        "0", "1", "2", "0+1", "1+2", "0+1+2"
    ]


def test_cluster_style_cli_and_focused_imports():
    args = build_snapshot_parser().parse_args([
        "--base-path", "/lustre/work/example/Thesan-1/output", "--snapshot", "80",
        "--simulation-name", "Thesan-1", "--los-file", "/lustre/work/example/rays_080.hdf5",
        "--products", "lya", "mfp", "gamma", "equations",
        "--photon-group-tests", "0", "1", "2", "0+1", "1+2", "0+1+2",
        "--ionized-sweep", "--ionized-cut-min", "0.9", "--ionized-cut-max", "0.9999",
        "--ionized-cut-count", "200", "--ionized-density-thresholds", "1", "5", "10",
    ])
    assert args.products == ["lya", "mfp", "gamma", "equations"]
    assert args.ionized_sweep is True
    assert args.photon_group_tests == ["0", "1", "2", "0+1", "1+2", "0+1+2"]
    assert args.ionized_density_thresholds == [1.0, 5.0, 10.0]
    from clumping_factor.forest.gamma import GammaHIResult
    from clumping_factor.forest.mfp import MeanFreePathResult
    from clumping_factor.forest.ionizing_cache import validate_ionizing_cache
    assert GammaHIResult is not None and MeanFreePathResult is not None and callable(validate_ionizing_cache)
