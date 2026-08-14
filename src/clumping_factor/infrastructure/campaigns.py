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
    snapshots = [int(value) for value in _as_list(matrix, "snapshots")]
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
    for simulation, snapshot, particle, spec, requested_grid in itertools.product(
        simulations, snapshots, particles, methods, grids
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


def _as_command(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(shlex.split(value))
    if isinstance(value, list) and all(isinstance(item, (str, int, float)) for item in value):
        return tuple(str(item) for item in value)
    raise ValueError("Campaign task command must be a string or a list of scalar values")


def _plan_legacy(source: Path, document: dict[str, Any]) -> CampaignManifest:
    """Read the original explicit-command format during the migration window."""

    if not isinstance(document.get("tasks"), list):
        raise ValueError("Campaign must define typed [simulation]/[matrix] tables or [[tasks]]")
    defaults = {str(key): str(value) for key, value in document.get("defaults", {}).items()}
    resources = _resources(document)
    tasks = []
    for index, item in enumerate(document["tasks"], start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Campaign task {index} is not a table")
        task_id = str(item.get("id") or f"task-{index:04d}")
        command = tuple(token.format(**defaults) for token in _as_command(item["command"]))
        environment = tuple(sorted((str(key), str(value)) for key, value in item.get("environment", {}).items()))
        tasks.append(CampaignTask(task_id, command, environment, resources=resources))
    return CampaignManifest(str(document.get("name") or source.stem), str(source), tuple(sorted(tasks, key=lambda task: task.task_id)))


def plan_campaign(path: str | Path) -> CampaignManifest:
    source = Path(path)
    document = load_campaign(source)
    return _plan_matrix(source, document) if "matrix" in document or "simulation" in document or "simulations" in document else _plan_legacy(source, document)


def write_manifest(manifest: CampaignManifest, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def render_pbs_worker(task: CampaignTask, *, job_name: str | None = None) -> str:
    """Render one generic PBS worker from manifest-owned resources."""

    resources = task.resources
    lines = ["#!/bin/sh", f"#PBS -N {job_name or task.task_id}"]
    if resources.queue:
        lines.append(f"#PBS -q {resources.queue}")
    lines += [
        f"#PBS -l select=1:ncpus={resources.cpus}:mem={resources.memory}",
        f"#PBS -l walltime={resources.walltime}",
        "set -eu",
    ]
    lines.extend(f"export {key}={shlex.quote(value)}" for key, value in task.environment)
    lines.append("exec " + " ".join(shlex.quote(token) for token in task.command))
    return "\n".join(lines) + "\n"


def submit_campaign(manifest: CampaignManifest, *, execute: bool = False) -> list[str]:
    rendered = [render_pbs_worker(task) for task in manifest.tasks]
    if execute:
        for script in rendered:
            subprocess.run(["qsub"], input=script, text=True, check=True)
    return rendered


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Plan or submit a declarative campaign.")
    parser.add_argument("action", choices=("plan", "submit"))
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--execute", action="store_true", help="Submit rendered workers with qsub.")
    args = parser.parse_args(argv)
    manifest = plan_campaign(args.campaign)
    if args.action == "plan":
        destination = args.manifest or args.campaign.with_suffix(".manifest.json")
        write_manifest(manifest, destination)
        print(f"Planned {len(manifest.tasks)} tasks: {destination}")
    else:
        for index, script in enumerate(submit_campaign(manifest, execute=args.execute), start=1):
            print(f"# worker {index}\n{script}", end="")

