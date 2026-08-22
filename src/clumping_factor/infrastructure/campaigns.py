"""Typed campaign matrices, deterministic manifests, and generic PBS workers."""

from __future__ import annotations

import argparse
import itertools
import json
import re
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]

from ..methods.registry import METHOD_REGISTRY, MethodSpec
from clumping_factor.infrastructure.results import canonical_result_path


@dataclass(frozen=True)
class ResourceSpec:
    cpus: int = 1
    memory: str = "4gb"
    walltime: str = "01:00:00"
    queue: str | None = None

    def __post_init__(self) -> None:
        if self.cpus < 1:
            raise ValueError("resources.cpus must be at least 1")
        if not re.fullmatch(r"\d+(?:kb|mb|gb|tb)", self.memory.lower()):
            raise ValueError("resources.memory must look like 32gb")
        if not re.fullmatch(r"\d{1,3}:\d{2}:\d{2}", self.walltime):
            raise ValueError("resources.walltime must use HH:MM:SS")

    def to_dict(self) -> dict[str, object]:
        return {"cpus": self.cpus, "memory": self.memory, "walltime": self.walltime, "queue": self.queue}


@dataclass(frozen=True)
class CampaignTask:
    task_id: str
    command: tuple[str, ...]
    environment: tuple[tuple[str, str], ...] = ()
    method_id: str | None = None
    simulation: str | None = None
    snapshot: int | None = None
    particle_type: str | None = None
    grid_size: int | None = None
    output: str | None = None
    resources: ResourceSpec = field(default_factory=ResourceSpec)

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.task_id,
            "method_id": self.method_id,
            "simulation": self.simulation,
            "snapshot": self.snapshot,
            "particle_type": self.particle_type,
            "grid_size": self.grid_size,
            "output": self.output,
            "command": list(self.command),
            "environment": dict(self.environment),
            "resources": self.resources.to_dict(),
        }


@dataclass(frozen=True)
class CampaignManifest:
    name: str
    source: str
    tasks: tuple[CampaignTask, ...]

    def to_dict(self) -> dict[str, object]:
        return {"campaign": self.name, "source": self.source, "tasks": [task.to_dict() for task in self.tasks]}


def load_campaign(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    with source.open("rb") as handle:
        document = tomllib.load(handle)
    if not isinstance(document, dict):
        raise ValueError(f"Campaign root must be a table: {source}")
    return document


def _resources(document: dict[str, Any]) -> ResourceSpec:
    values = document.get("resources", {})
    return ResourceSpec(
        cpus=int(values.get("cpus", 1)),
        memory=str(values.get("memory", "4gb")),
        walltime=str(values.get("walltime", "01:00:00")),
        queue=str(values["queue"]) if values.get("queue") else None,
    )


def _as_list(table: dict[str, Any], key: str) -> list[Any]:
    value = table.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"matrix.{key} must be a non-empty array")
    return value


def _snapshots_for_simulation(base_path: str, snapshot_spec: Any) -> list[int]:
    """Resolve a static snapshot list or discover every snapshot on disk."""
    if isinstance(snapshot_spec, list):
        if not snapshot_spec:
            raise ValueError("matrix.snapshots must be a non-empty array")
        return [int(value) for value in snapshot_spec]
    if snapshot_spec != "available":
        raise ValueError("matrix.snapshots must be an array or the string 'available'")

    numbers: set[int] = set()
    root = Path(base_path)
    for path in root.glob("snapdir_*"):
        match = re.fullmatch(r"snapdir_(\d+)", path.name)
        if match and path.is_dir() and any(path.glob("snap_*.hdf5")):
            numbers.add(int(match.group(1)))
    for path in root.glob("snap_*.hdf5"):
        match = re.fullmatch(r"snap_(\d+)\.hdf5", path.name)
        if match:
            numbers.add(int(match.group(1)))
    if not numbers:
        raise ValueError(f"No snapshots were found under {base_path!r}.")
    return sorted(numbers)


def _backend(spec: MethodSpec) -> str:
    if spec.command_variant is not None:
        return spec.command_variant
    if spec.legacy_backends:
        return spec.legacy_backends[0]
    raise ValueError(f"Method {spec.identifier} has no compute-backend compatibility mapping")


def _validate_compute_capability(spec: MethodSpec) -> None:
    if spec.command_kind is None:
        raise ValueError(
            f"{spec.identifier} is a diagnostic/workflow method, not a plannable compute method; "
            "use an explicit compatibility task"
        )


def _method_options(document: dict[str, Any], spec: MethodSpec) -> dict[str, Any]:
    values = document.get("method_options", {})
    if not isinstance(values, dict):
        raise ValueError("method_options must be a table")
    selected = values.get(spec.identifier, {})
    if not isinstance(selected, dict):
        raise ValueError(f"method_options.{spec.identifier} must be a table")
    return {str(key): value for key, value in selected.items()}


def _option_tokens(options: dict[str, Any]) -> list[str]:
    tokens: list[str] = []
    for key, value in sorted(options.items()):
        if value is None or value is False:
            continue
        option = "--" + key.replace("_", "-")
        if value is True:
            tokens.append(option)
        elif isinstance(value, list):
            tokens.extend([option, *(str(item) for item in value)])
        else:
            tokens.extend([option, str(value)])
    return tokens


def _task_command(
    spec: MethodSpec,
    *,
    base_path: str,
    simulation: str,
    snapshot: int,
    particle: str,
    grid: int | None,
    execution: dict[str, Any],
    method_options: dict[str, Any],
    output: Path,
) -> tuple[str, ...]:
    common = ["--base-path", base_path, "--simulation-name", simulation, "--snapshot", str(snapshot)]
    if spec.command_kind == "clumping-compute":
        options = dict(method_options)
        backend = _backend(spec)
        if spec.identifier == "clumping.mask-target":
            target_backend = options.pop("target_backend", None)
            mask_backend = options.pop("mask_backend", None)
            if target_backend is None or mask_backend is None:
                raise ValueError("clumping.mask-target requires method_options.target_backend and mask_backend")
            backend = str(target_backend)
            command = [
                "clumping", "clumping", "compute", *common, "--particle-type", particle,
                "--backend", backend, "--target-backend", backend, "--mask-backend", str(mask_backend),
                "--mask-particle-type", str(options.pop("mask_particle_type", particle)),
            ]
        else:
            command = ["clumping", "clumping", "compute", *common, "--particle-type", particle, "--backend", backend]
        if grid is not None:
            command += ["--grid-size", str(grid)]
        command += _option_tokens(options)
    elif spec.command_kind == "power-spectrum-compute":
        engine = spec.command_variant
        if engine is None:
            raise ValueError(f"{spec.identifier} has no power-spectrum command variant")
        command = ["clumping", "power", "compute", *common, "--particle-type", particle, "--spectrum-engine", engine]
        if grid is not None:
            command += ["--grid-size", str(grid)]
        command += _option_tokens(method_options)
    elif spec.command_kind == "alternative-compute":
        alternative_backend = spec.command_variant
        if alternative_backend is None:
            raise ValueError(f"{spec.identifier} has no alternative command variant")
        options = dict(method_options)
        if "mfp_file" not in options and not options.get("compute_missing_mfp"):
            raise ValueError(f"{spec.identifier} requires method_options.mfp_file or compute_missing_mfp = true")
        command = ["clumping", "clumping", "alternative", *common, "--backend", alternative_backend]
        if alternative_backend == "grid":
            options.setdefault("mask_particle_type", particle)
            if grid is not None:
                options.setdefault("grid_size", grid)
        command += _option_tokens(options)
    elif spec.command_kind == "ionized-sweep":
        command = ["clumping", "clumping", "ionized-sweep", *common]
        command += _option_tokens(method_options)
    elif spec.command_kind == "temperature":
        command = ["clumping", "temperature", "compute", "--base-path", base_path, "--snapshot", str(snapshot)]
        command += _option_tokens(method_options)
    elif spec.command_kind == "diagnostics":
        command = ["clumping", "diagnostics", str(spec.command_variant), *common]
        command += _option_tokens(method_options)
    elif spec.command_kind == "forest-spectra":
        options = dict(method_options)
        if "los_file" not in options and "los_dir" not in options:
            raise ValueError("forest.lyman-alpha requires method_options.los_file or los_dir")
        command = ["clumping", "forest", "spectra", "--simulation-name", simulation]
        command += _option_tokens(options)
    elif spec.command_kind == "forest-ionizing":
        options = dict(method_options)
        command = ["clumping", "forest", "ionizing", str(spec.command_variant)]
        if spec.command_variant == "gamma":
            command += ["--base-path", base_path, "--snapshot", str(snapshot)]
        elif "los_file" not in options:
            raise ValueError("forest.mfp requires method_options.los_file")
        command += _option_tokens(options)
    elif spec.command_kind == "forest-snapshot":
        command = [
            "clumping", "forest", "snapshot", "--base-path", base_path,
            "--simulation-name", simulation, "--snapshot", str(snapshot),
        ]
        command += _option_tokens(method_options)
    else:
        raise ValueError(
            f"Typed campaign matrices currently execute compute methods only; {spec.identifier} "
            "is a diagnostic/workflow method and must use an explicit compatibility task"
        )
    option_names = {
        "threads": "--threads",
        "load_mode": "--load-mode",
        "chunk_size": "--chunk-size",
        "mas": "--mas",
        "radius_bin_batch_size": "--radius-bin-batch-size",
    }
    for key, option in option_names.items():
        if key in execution:
            command += [option, str(execution[key])]
    if spec.command_kind == "forest-snapshot":
        command += ["--output-dir", str(output.parent)]
    else:
        command += ["--output", str(output)]
    return tuple(command)


def _simulation_tables(document: dict[str, Any]) -> list[dict[str, Any]]:
    plural = document.get("simulations")
    singular = document.get("simulation")
    if plural is not None and singular is not None:
        raise ValueError("Use either [simulation] or [[simulations]], not both")
    values = plural if plural is not None else [singular]
    if not isinstance(values, list) or not values or not all(isinstance(item, dict) for item in values):
        raise ValueError("Typed campaigns require [simulation] or non-empty [[simulations]] tables")
    identities: set[tuple[str, str]] = set()
    for index, item in enumerate(values, start=1):
        family = _task_component(str(item.get("family") or "").lower())
        name = _task_component(str(item.get("name") or ""))
        if not family or not name or not str(item.get("base_path") or ""):
            raise ValueError(f"simulation {index} requires family, name, and base_path")
        identity = (family.lower(), name.lower())
        if identity in identities:
            raise ValueError(f"Duplicate simulation identity: {family}/{name}")
        identities.add(identity)
    return values


def _task_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")


def _plan_matrix(source: Path, document: dict[str, Any]) -> CampaignManifest:
    matrix = document.get("matrix")
    if not isinstance(matrix, dict):
        raise ValueError("Typed campaigns require a [matrix] table")
    simulations = _simulation_tables(document)
    snapshot_spec = matrix.get("snapshots")
    if snapshot_spec is None:
        raise ValueError("matrix.snapshots must be a non-empty array or 'available'")
    particles = [str(value) for value in _as_list(matrix, "particle_types")]
    methods = [METHOD_REGISTRY.get(str(value)) for value in _as_list(matrix, "methods")]
    grid_values = _as_list(matrix, "grids") if "grids" in matrix else [None]
    grids = [None if value is None or int(value) == 0 else int(value) for value in grid_values]
    execution = document.get("execution", {})
    if not isinstance(execution, dict):
        raise ValueError("execution must be a table")
    resources = _resources(document)
    output_root = Path(str(document.get("output_root", "results")))
    threads = int(execution.get("threads", 1))
    if threads < 1 or threads > resources.cpus:
        raise ValueError("execution.threads must be between 1 and resources.cpus")

    tasks: list[CampaignTask] = []
    seen: set[tuple[str, str, str, str, int, str, int | None]] = set()
    for simulation in simulations:
        simulation_snapshots = _snapshots_for_simulation(str(simulation.get("base_path") or ""), snapshot_spec)
        for snapshot, particle, spec, requested_grid in itertools.product(
            simulation_snapshots, particles, methods, grids
        ):
            family = str(simulation.get("family") or "").lower()
            name = str(simulation.get("name") or "")
            base_path = str(simulation.get("base_path") or "")
            if not family or not name or not base_path:
                raise ValueError("Every simulation requires family, name, and base_path")
            _validate_compute_capability(spec)
            if particle not in spec.supported_particle_types:
                raise ValueError(f"Method {spec.identifier} does not support particle type {particle}")
            load_mode = str(execution.get("load_mode", "auto"))
            if load_mode in {"full", "chunked"} and load_mode not in spec.execution_modes:
                raise ValueError(f"Method {spec.identifier} does not support execution.load_mode={load_mode}")
            if threads > 1 and "threaded" not in spec.execution_modes:
                raise ValueError(f"Method {spec.identifier} does not support threaded execution")
            needs_grid = any(item in {"grid-size", "optional-grid"} for item in spec.grid_requirements)
            grid = requested_grid if needs_grid else None
            if needs_grid and grid is None:
                raise ValueError(f"Method {spec.identifier} requires matrix.grids")
            key = (family, name, base_path, spec.identifier, snapshot, particle, grid)
            if key in seen:
                continue
            seen.add(key)
            options = _method_options(document, spec)
            method_spec = spec.to_dict()
            method_spec["configuration"] = {**options, "grid_size": grid}
            selection_spec = {"particle_type": particle}
            execution_spec = {
                **execution,
                "cpus": resources.cpus,
                "queue": resources.queue,
                "walltime": resources.walltime,
                "campaign": str(document.get("name") or source.stem),
            }
            output = canonical_result_path(
                output_root,
                family=family,
                simulation_name=name,
                particle_type=particle,
                snapshot=snapshot,
                method_spec=method_spec,
                selection_spec=selection_spec,
                execution_spec=execution_spec,
            )
            task_id = "-".join(filter(None, (_task_component(family), _task_component(name), f"s{snapshot:03d}", _task_component(particle), _task_component(spec.identifier), f"g{grid}" if grid else "nogrid")))
            command = _task_command(
                spec,
                base_path=base_path,
                simulation=name,
                snapshot=snapshot,
                particle=particle,
                grid=grid,
                execution=execution,
                method_options=options,
                output=output,
            )
            tasks.append(CampaignTask(task_id, command, method_id=spec.identifier, simulation=name,
                                      snapshot=snapshot, particle_type=particle, grid_size=grid,
                                      output=str(output), resources=resources))
    return CampaignManifest(str(document.get("name") or source.stem), str(source), tuple(sorted(tasks, key=lambda task: task.task_id)))


def plan_campaign(path: str | Path) -> CampaignManifest:
    source = Path(path)
    document = load_campaign(source)
    if "tasks" in document:
        raise ValueError("Explicit command tasks are not supported; use typed simulation and matrix tables")
    return _plan_matrix(source, document)


def write_manifest(manifest: CampaignManifest, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def render_pbs_worker(task: CampaignTask, *, job_name: str | None = None) -> str:
    """Render one generic PBS worker from manifest-owned resources."""

    resources = task.resources
    # qsub does not otherwise inherit an activated conda/venv environment.
    # Preserve the executable PATH used by the activated environment.  This
    # PBS implementation does not support the -d directive, so the worker
    # changes directory from PBS_O_WORKDIR below instead.
    lines = [
        "#!/bin/sh",
        f"#PBS -N {job_name or task.task_id}",
        "#PBS -V",
    ]
    if resources.queue:
        lines.append(f"#PBS -q {resources.queue}")
    lines += [
        f"#PBS -l select=1:ncpus={resources.cpus}:mem={resources.memory}",
        f"#PBS -l walltime={resources.walltime}",
        "set -eu",
        "if [ -f \"$HOME/.conda/etc/profile.d/conda.sh\" ]; then",
        "    . \"$HOME/.conda/etc/profile.d/conda.sh\"",
        "    conda activate clumping-factor",
        "elif [ -x \"$HOME/.conda/envs/clumping-factor/bin/clumping\" ]; then",
        "    export PATH=\"$HOME/.conda/envs/clumping-factor/bin:$PATH\"",
        "    export CONDA_DEFAULT_ENV=clumping-factor",
        "    export CONDA_PREFIX=\"$HOME/.conda/envs/clumping-factor\"",
        "else",
        "    echo 'clumping-factor conda environment was not found' >&2",
        "    exit 127",
        "fi",
        "cd \"${PBS_O_WORKDIR:-.}\"",
    ]
    lines.extend(f"export {key}={shlex.quote(value)}" for key, value in task.environment)
    lines.append("exec " + " ".join(shlex.quote(token) for token in task.command))
    return "\n".join(lines) + "\n"


def render_pbs_array(
    manifest: CampaignManifest,
    *,
    job_name: str | None = None,
    array_syntax: str = "pbspro",
) -> str:
    """Render one PBS job array whose index selects a deterministic campaign task.

    PBS Pro/OpenPBS uses ``-J`` and ``PBS_ARRAY_INDEX``; Torque uses ``-t``
    and ``PBS_ARRAYID``.  The worker accepts either environment variable so
    that only the scheduler directive needs to change between implementations.
    Arrays require uniform resources because PBS allocates one resource shape
    for every index in a single array.  Use one-based indices because the
    target OpenPBS/Torque installations reject a zero-valued array index.
    """
    tasks = manifest.tasks
    if not tasks:
        raise ValueError("Cannot render a PBS array for an empty campaign manifest.")
    resources = tasks[0].resources
    if any(task.resources != resources for task in tasks[1:]):
        raise ValueError("PBS array tasks must have identical resource requests.")
    directive_by_syntax = {"pbspro": "-J", "torque": "-t"}
    try:
        array_directive = directive_by_syntax[array_syntax]
    except KeyError as exc:
        allowed = ", ".join(sorted(directive_by_syntax))
        raise ValueError(f"array_syntax must be one of: {allowed}") from exc

    lines = [
        "#!/bin/sh",
        f"#PBS -N {job_name or manifest.name}",
        "#PBS -V",
        f"#PBS {array_directive} 1-{len(tasks)}",
    ]
    if resources.queue:
        lines.append(f"#PBS -q {resources.queue}")
    lines += [
        f"#PBS -l select=1:ncpus={resources.cpus}:mem={resources.memory}",
        f"#PBS -l walltime={resources.walltime}",
        "set -eu",
        "if [ -f \"$HOME/.conda/etc/profile.d/conda.sh\" ]; then",
        "    . \"$HOME/.conda/etc/profile.d/conda.sh\"",
        "    conda activate clumping-factor",
        "elif [ -x \"$HOME/.conda/envs/clumping-factor/bin/clumping\" ]; then",
        "    export PATH=\"$HOME/.conda/envs/clumping-factor/bin:$PATH\"",
        "    export CONDA_DEFAULT_ENV=clumping-factor",
        "    export CONDA_PREFIX=\"$HOME/.conda/envs/clumping-factor\"",
        "else",
        "    echo 'clumping-factor conda environment was not found' >&2",
        "    exit 127",
        "fi",
        "cd \"${PBS_O_WORKDIR:-.}\"",
        "task_index=\"${PBS_ARRAY_INDEX:-${PBS_ARRAYID:-}}\"",
        "if [ -z \"$task_index\" ]; then",
        "    echo 'PBS array index was not set' >&2",
        "    exit 2",
        "fi",
        "case \"$task_index\" in",
    ]
    for index, task in enumerate(tasks, start=1):
        lines.append(f"    {index})")
        lines.extend(f"        export {key}={shlex.quote(value)}" for key, value in task.environment)
        lines.append("        exec " + " ".join(shlex.quote(token) for token in task.command))
        lines.append("        ;;")
    lines += [
        "    *)",
        "        echo \"Invalid PBS array index: $task_index\" >&2",
        "        exit 2",
        "        ;;",
    ]
    lines += ["esac"]
    return "\n".join(lines) + "\n"


def submit_campaign(manifest: CampaignManifest, *, execute: bool = False) -> list[str]:
    rendered = [render_pbs_worker(task) for task in manifest.tasks]
    if execute:
        for script in rendered:
            subprocess.run(["qsub"], input=script, text=True, check=True)
    return rendered


def submit_campaign_array(
    manifest: CampaignManifest,
    *,
    execute: bool = False,
    array_syntax: str = "pbspro",
) -> str:
    """Render or submit a single PBS job array for every task in a campaign."""
    rendered = render_pbs_array(manifest, array_syntax=array_syntax)
    if execute:
        subprocess.run(["qsub"], input=rendered, text=True, check=True)
    return rendered


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Plan or submit a declarative campaign.")
    parser.add_argument("action", choices=("plan", "submit", "submit-array"))
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--execute", action="store_true", help="Submit rendered workers with qsub.")
    parser.add_argument(
        "--array-syntax",
        choices=("pbspro", "torque"),
        default="pbspro",
        help="PBS array directive: pbspro uses -J; torque uses -t.",
    )
    args = parser.parse_args(argv)
    manifest = plan_campaign(args.campaign)
    if args.action == "plan":
        destination = args.manifest or args.campaign.with_suffix(".manifest.json")
        write_manifest(manifest, destination)
        print(f"Planned {len(manifest.tasks)} tasks: {destination}")
    elif args.action == "submit":
        for index, script in enumerate(submit_campaign(manifest, execute=args.execute), start=1):
            print(f"# worker {index}\n{script}", end="")
    else:
        print(submit_campaign_array(manifest, execute=args.execute, array_syntax=args.array_syntax), end="")

