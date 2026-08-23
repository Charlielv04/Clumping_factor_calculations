import json
from pathlib import Path

import pytest

from clumping_factor.infrastructure.campaigns import plan_campaign, render_pbs_array, render_pbs_worker, write_manifest


def _write_matrix(
    path: Path,
    *,
    methods='["sphere", "raw-volume"]',
    particles='["gas"]',
    threads=1,
) -> None:
    path.write_text(
        f'''name = "demo"
output_root = "results"
[simulation]
family = "tng"
name = "tng100-3"
base_path = "/data/tng"
[matrix]
snapshots = [17, 98]
particle_types = {particles}
methods = {methods}
grids = [256]
[execution]
threads = {threads}
load_mode = "chunked"
radius_bin_batch_size = 2
[resources]
cpus = 8
memory = "32gb"
walltime = "04:00:00"
queue = "mini"
''',
        encoding="utf-8",
    )


def test_typed_campaign_expands_matrix_and_derives_outputs(tmp_path: Path):
    campaign = tmp_path / "campaign.toml"
    _write_matrix(campaign)
    manifest = plan_campaign(campaign)

    assert len(manifest.tasks) == 4
    assert [task.task_id for task in manifest.tasks] == sorted(task.task_id for task in manifest.tasks)
    raw_tasks = [task for task in manifest.tasks if task.method_id == "clumping.raw-volume-weighted"]
    assert all(task.grid_size is None and "/science-" in task.output.replace("\\", "/") for task in raw_tasks)
    assert all(task.output in task.command for task in manifest.tasks)
    assert all(task.resources.memory == "32gb" for task in manifest.tasks)


def test_campaign_manifest_is_deterministic(tmp_path: Path):
    campaign = tmp_path / "campaign.toml"
    _write_matrix(campaign, methods='["cube", "sphere"]')
    first = write_manifest(plan_campaign(campaign), tmp_path / "first.json")
    second = write_manifest(plan_campaign(campaign), tmp_path / "second.json")
    assert json.loads(first.read_text(encoding="utf-8")) == json.loads(second.read_text(encoding="utf-8"))


def test_campaign_validates_registry_and_particle_compatibility(tmp_path: Path):
    unknown = tmp_path / "unknown.toml"
    _write_matrix(unknown, methods='["invented"]')
    with pytest.raises(KeyError, match="Unknown method"):
        plan_campaign(unknown)

    unsupported = tmp_path / "unsupported.toml"
    _write_matrix(unsupported, methods='["raw-volume"]', particles='["dm"]')
    with pytest.raises(ValueError, match="does not support"):
        plan_campaign(unsupported)


def test_pbs_worker_uses_manifest_resources(tmp_path: Path):
    campaign = tmp_path / "campaign.toml"
    _write_matrix(campaign, methods='["sphere"]', threads=8)
    worker = render_pbs_worker(plan_campaign(campaign).tasks[0])
    assert "#PBS -q mini" in worker
    assert "#PBS -V" in worker
    assert "#PBS -o logs/pbs/" in worker
    assert "#PBS -e logs/pbs/" in worker
    assert "cd \"${PBS_O_WORKDIR:-.}\"" in worker
    assert "conda activate clumping-factor" in worker
    assert "$HOME/.conda/envs/clumping-factor/bin" in worker
    assert "select=1:ncpus=8:mem=32gb" in worker
    assert "#PBS -l walltime=04:00:00" in worker


def test_pbs_array_uses_one_submission_and_selects_each_task(tmp_path: Path):
    campaign = tmp_path / "campaign.toml"
    _write_matrix(campaign, methods='["sphere"]', threads=8)
    manifest = plan_campaign(campaign)

    worker = render_pbs_array(manifest)

    assert "#PBS -J 1-2" in worker
    assert "#PBS -o logs/pbs/" in worker
    assert "#PBS -e logs/pbs/" in worker
    assert 'task_index="${PBS_ARRAY_INDEX:-${PBS_ARRAYID:-}}"' in worker
    assert 'case "$task_index" in' in worker
    assert "    1)" in worker and "    2)" in worker
    assert worker.count("clumping clumping compute") == 2
    assert "Invalid PBS array index" in worker
    assert "#PBS -l select=1:ncpus=8:mem=32gb" in worker


def test_pbs_array_supports_torque_directive(tmp_path: Path):
    campaign = tmp_path / "campaign.toml"
    _write_matrix(campaign, methods='["sphere"]')

    worker = render_pbs_array(plan_campaign(campaign), array_syntax="torque")

    assert "#PBS -t 1-2" in worker


def test_explicit_command_tasks_are_rejected(tmp_path: Path):
    campaign = tmp_path / "legacy.toml"
    campaign.write_text(
        'name = "legacy"\n[defaults]\nvalue = "x"\n[[tasks]]\nid = "b"\ncommand = ["echo", "{value}"]\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Explicit command tasks"):
        plan_campaign(campaign)


def test_campaign_rejects_unsupported_execution_mode(tmp_path: Path):
    campaign = tmp_path / "campaign.toml"
    _write_matrix(campaign, methods='["raw-volume"]', threads=8)
    with pytest.raises(ValueError, match="threaded execution"):
        plan_campaign(campaign)


def test_typed_campaign_expands_multiple_simulations(tmp_path: Path):
    campaign = tmp_path / "multi.toml"
    campaign.write_text(
        '''name = "multi"
output_root = "results"
[[simulations]]
family = "tng"
name = "one"
base_path = "/data/one"
[[simulations]]
family = "tng"
name = "two"
base_path = "/data/two"
[matrix]
snapshots = [98]
particle_types = ["gas"]
methods = ["power-spectrum.combined"]
grids = [128]
[execution]
threads = 1
load_mode = "full"
''',
        encoding="utf-8",
    )
    manifest = plan_campaign(campaign)
    assert len(manifest.tasks) == 2
    assert {task.simulation for task in manifest.tasks} == {"one", "two"}
    assert all(task.method_id == "power-spectrum.combined" for task in manifest.tasks)
    assert all("--spectrum-engine" in task.command and "both" in task.command for task in manifest.tasks)


def test_simulation_snapshot_override_targets_only_requested_tasks(tmp_path: Path):
    campaign = tmp_path / "targeted-retry.toml"
    campaign.write_text(
        '''name = "targeted-retry"
output_root = "results"
[[simulations]]
family = "tng"
name = "one"
base_path = "/data/one"
snapshots = [17, 25]
[[simulations]]
family = "tng"
name = "two"
base_path = "/data/two"
snapshots = [98]
[matrix]
snapshots = "available"
particle_types = ["dm"]
methods = ["power-spectrum.numpy"]
grids = [128]
[execution]
threads = 1
load_mode = "full"
''',
        encoding="utf-8",
    )
    manifest = plan_campaign(campaign)
    assert [(task.simulation, task.snapshot) for task in manifest.tasks] == [("one", 17), ("one", 25), ("two", 98)]


def test_power_spectrum_campaign_forwards_fold_options(tmp_path: Path):
    campaign = tmp_path / "folded.toml"
    campaign.write_text(
        '''name = "folded"
output_root = "results"
[simulation]
family = "tng"
name = "toy"
base_path = "/data/toy"
[matrix]
snapshots = [98]
particle_types = ["dm"]
methods = ["power-spectrum.combined"]
grids = [256, 512, 1024]
[method_options."power-spectrum.combined"]
fold_factors = [1, 2, 4]
[execution]
threads = 1
load_mode = "chunked"
''',
        encoding="utf-8",
    )
    manifest = plan_campaign(campaign)
    assert len(manifest.tasks) == 3
    assert all("--fold-factors" in task.command for task in manifest.tasks)
    assert all("1" in task.command and "2" in task.command and "4" in task.command for task in manifest.tasks)


def test_available_snapshots_are_discovered_per_simulation(tmp_path: Path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    (first / "snapdir_017").mkdir(parents=True)
    (first / "snapdir_099").mkdir()
    (second / "snapdir_025").mkdir(parents=True)
    (first / "snapdir_017" / "snap_017.0.hdf5").touch()
    (first / "snapdir_099" / "snap_099.0.hdf5").touch()
    (second / "snapdir_025" / "snap_025.0.hdf5").touch()
    campaign = tmp_path / "available.toml"
    campaign.write_text(
        f'''name = "available"
[[simulations]]
family = "tng"
name = "first"
base_path = "{first.as_posix()}"
[[simulations]]
family = "tng"
name = "second"
base_path = "{second.as_posix()}"
[matrix]
snapshots = "available"
particle_types = ["dm"]
methods = ["power-spectrum.numpy"]
grids = [256]
''',
        encoding="utf-8",
    )
    manifest = plan_campaign(campaign)
    assert {(task.simulation, task.snapshot) for task in manifest.tasks} == {
        ("first", 17), ("first", 99), ("second", 25)
    }


def test_typed_campaign_plans_both_alternative_compute_methods(tmp_path: Path):
    campaign = tmp_path / "alternative.toml"
    campaign.write_text(
        '''name = "alternative"
output_root = "results"
[[simulations]]
family = "thesan"
name = "Thesan-2"
base_path = "/data/thesan"
[matrix]
snapshots = [80]
particle_types = ["gas"]
methods = ["alternative.raw-volume", "alternative.grid-masked"]
grids = [256]
[method_options."alternative.raw-volume"]
mfp_file = "/data/mfp.json"
[method_options."alternative.grid-masked"]
mfp_file = "/data/mfp.json"
[execution]
threads = 1
load_mode = "chunked"
''',
        encoding="utf-8",
    )
    manifest = plan_campaign(campaign)
    assert {task.method_id for task in manifest.tasks} == {"alternative.raw-volume", "alternative.grid-masked"}
    assert any("--backend" in task.command and "grid" in task.command for task in manifest.tasks)
    assert all("--mfp-file" in task.command for task in manifest.tasks)


def test_typed_campaign_rejects_empty_grid_axis_and_plans_diagnostics(tmp_path: Path):
    empty = tmp_path / "empty.toml"
    _write_matrix(empty)
    empty.write_text(empty.read_text(encoding="utf-8").replace("grids = [256]", "grids = []"), encoding="utf-8")
    with pytest.raises(ValueError, match="matrix.grids must be a non-empty array"):
        plan_campaign(empty)

    workflow = tmp_path / "workflow.toml"
    _write_matrix(workflow, methods='["diagnostics.equations"]')
    manifest = plan_campaign(workflow)
    assert manifest.tasks
    assert manifest.tasks[0].command[:3] == ("clumping", "diagnostics", "equations")


def test_omitted_execution_uses_legacy_safe_defaults_for_raw_methods(tmp_path: Path):
    campaign = tmp_path / "defaults.toml"
    campaign.write_text(
        '''name = "defaults"
output_root = "results"
[[simulations]]
family = "tng"
name = "raw-case"
base_path = "/data/raw"
[matrix]
snapshots = [98]
particle_types = ["gas"]
methods = ["raw-volume"]
grids = [256]
[resources]
cpus = 8
memory = "32gb"
walltime = "01:00:00"
''',
        encoding="utf-8",
    )
    task = plan_campaign(campaign).tasks[0]
    assert "--threads" not in task.command
    assert "--load-mode" not in task.command
    assert task.resources.cpus == 8


def test_duplicate_normalized_simulation_identity_is_rejected(tmp_path: Path):
    campaign = tmp_path / "duplicate-simulations.toml"
    campaign.write_text(
        '''name = "duplicate"
[[simulations]]
family = "TNG"
name = "Same Name"
base_path = "/data/one"
[[simulations]]
family = "tng"
name = "Same-Name"
base_path = "/data/two"
[matrix]
snapshots = [98]
particle_types = ["gas"]
methods = ["sphere"]
grids = [256]
''',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate simulation identity"):
        plan_campaign(campaign)
