from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Callable, Sequence

from .ionizing import (
    atomic_write_json, calculate_mean_free_paths, calculate_mean_free_paths_reference,
    compute_and_cache_snapshot_ionizing_inputs, compute_gamma_hi_result,
    gamma_result_document, mfp_result_document,
)
from .los_loader import LosData, read_thesan_random_los
from .spectra import compute_los_spectra, write_spectra_hdf5

WORKFLOW_VERSION = 1
PRODUCTS = ("lya", "mfp", "gamma", "equations")


@dataclass(frozen=True)
class SnapshotWorkflowConfig:
    base_path: str | Path
    snapshot: int
    simulation_name: str
    products: Sequence[str]
    los_file: str | Path | None = None
    output_root: str | Path = "results/forest"
    refresh_products: bool = False
    verbose: bool = False
    threads: int = 1
    equation_workers: int | None = None
    gamma_workers: int | None = None
    mfp_workers: int | None = None
    only_rays: Sequence[int] | None = None
    line: str = "Ly a"
    resolution_kms: float = 1.0
    static: bool = False
    mfp_starts_per_ray: int = 100
    mfp_seed: int | None = 0
    mfp_cross_check: bool = False
    gamma_hi_threshold: float = 0.5
    gamma_cross_check: bool = False
    gamma_chunk_size: int = 1_000_000
    progress_interval: int = 10
    sigma_hi_cm2: float = 6.3e-18
    temperature_file: str | Path | None = None
    compute_missing_temperature: bool = True
    mean_molecular_weight: float = 1.6
    temperature_weighting: str = "volume"
    recombination_temperature_mode: str = "tigm"
    reduced_speed_of_light_fraction: float = 0.2
    photon_groups: Sequence[int] = (0,)
    photon_group_tests: Sequence[str] | None = None
    thresholds: Sequence[float] | None = None
    threshold_min: float = -1.0
    threshold_max: float = 25.0
    threshold_count: int = 200
    ionized_density_thresholds: Sequence[float] | None = None
    ionized_cuts: Sequence[float] | None = None
    ionized_sweep: bool = False
    ionized_cut_min: float = 0.9
    ionized_cut_max: float = 0.9999
    ionized_cut_count: int = 200
    equation_chunk_size: int = 1_000_000
    allow_legacy_ionizing_table: bool = False
    refresh_ionizing_cache: bool = False


@dataclass(frozen=True)
class SnapshotWorkflowResult:
    manifest_path: Path
    document: dict

    @property
    def succeeded(self) -> bool:
        return self.document.get("status") == "success"

    @property
    def failures(self) -> dict:
        return {name: row for name, row in self.document.get("products", {}).items() if row.get("status") == "failed"}


def _family(simulation: str) -> str:
    lowered = simulation.lower()
    return "thesan" if lowered.startswith("thesan") else "tng" if lowered.startswith("tng") else "unknown"


def snapshot_output_dir(config: SnapshotWorkflowConfig) -> Path:
    return Path(config.output_root) / _family(config.simulation_name) / config.simulation_name / f"snapshot{config.snapshot:03d}"


def _signature(path: str | Path) -> dict[str, int | str]:
    resolved = Path(path).resolve()
    stat = resolved.stat()
    return {"path": str(resolved), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _fingerprint(config: SnapshotWorkflowConfig, product: str, inputs: list[dict]) -> str:
    payload = {"workflow_version": WORKFLOW_VERSION, "product": product, "config": asdict(config), "inputs": inputs}
    payload["config"]["products"] = [product]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _existing_manifest(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _outputs_exist(paths: Sequence[str]) -> bool:
    return bool(paths) and all(Path(path).exists() for path in paths)


def run_snapshot_workflow(
    config: SnapshotWorkflowConfig, *, progress: Callable[[str], None] | None = None
) -> SnapshotWorkflowResult:
    requested = list(dict.fromkeys(config.products))
    invalid = sorted(set(requested).difference(PRODUCTS))
    if not requested or invalid:
        raise ValueError(f"products must be a non-empty subset of {PRODUCTS}; invalid={invalid}.")
    if any(product in requested for product in ("lya", "mfp")) and config.los_file is None:
        raise ValueError("lya and mfp products require los_file.")
    if config.resolution_kms <= 0 or config.gamma_chunk_size < 1 or config.equation_chunk_size < 1:
        raise ValueError("resolution and chunk sizes must be positive.")
    if config.threads < 1:
        raise ValueError("threads must be at least 1.")
    equation_workers = int(config.equation_workers or config.threads)
    gamma_workers = int(config.gamma_workers or config.threads)
    mfp_workers = int(config.mfp_workers or config.threads)
    if min(equation_workers, gamma_workers, mfp_workers) < 1:
        raise ValueError("worker counts must be at least 1.")

    output_dir = snapshot_output_dir(config)
    manifest_path = output_dir / "manifest.json"
    previous = _existing_manifest(manifest_path)
    configuration = json.loads(json.dumps(asdict(config), default=str))
    preserved_products = {
        name: row for name, row in previous.get("products", {}).items() if name not in requested
    }
    document = {
        "workflow_version": WORKFLOW_VERSION, "status": "running", "started_at": _now(),
        "updated_at": _now(), "simulation": config.simulation_name, "snapshot": config.snapshot,
        "requested_products": requested, "configuration": configuration, "products": preserved_products,
        "warnings": [], "failures": [],
    }
    atomic_write_json(manifest_path, document)

    from ..loaders import snapshot_file_paths
    snapshot_error: Exception | None = None
    try:
        snapshot_files = snapshot_file_paths(config.base_path, config.snapshot)
        snapshot_signatures = [_signature(path) for path in snapshot_files]
    except Exception as exc:
        snapshot_files = []
        snapshot_signatures = []
        snapshot_error = exc
    if config.los_file is not None:
        try:
            los_signature = [_signature(config.los_file)]
        except OSError:
            los_signature = [{"path": str(Path(config.los_file).resolve()), "missing": True}]
    else:
        los_signature = []
    los_data: LosData | None = None
    los_error: Exception | None = None
    if any(product in requested for product in ("lya", "mfp")):
        try:
            los_data = read_thesan_random_los(config.los_file, only_rays=config.only_rays)
        except Exception as exc:
            los_error = exc

    def update_product(name: str, row: dict) -> None:
        document["products"][name] = row
        document["failures"] = [
            {"product": product, "error": product_row["error"]}
            for product, product_row in document["products"].items()
            if product_row.get("status") == "failed"
        ]
        document["updated_at"] = _now()
        atomic_write_json(manifest_path, document)

    def reusable(name: str, fingerprint: str) -> dict | None:
        row = previous.get("products", {}).get(name, {})
        if not config.refresh_products and row.get("status") in ("success", "reused") and row.get("fingerprint") == fingerprint and _outputs_exist(row.get("outputs", [])):
            return {**row, "status": "reused", "reused_at": _now()}
        return None

    for product in requested:
        started = perf_counter()
        inputs = los_signature if product in ("lya", "mfp") else snapshot_signatures
        fingerprint = _fingerprint(config, product, inputs)
        reused = reusable(product, fingerprint)
        if reused is not None:
            update_product(product, reused)
            if progress:
                progress(f"{product}: reused provenance-matching output")
            continue
        try:
            if product in ("lya", "mfp") and los_error is not None:
                raise los_error
            if product in ("gamma", "equations") and snapshot_error is not None:
                raise snapshot_error
            if product == "lya":
                output = output_dir / "lya" / f"{Path(config.los_file).stem}_lya.hdf5"
                result = compute_los_spectra(los_data, line_name=config.line, resolution_kms=config.resolution_kms,
                                             static=config.static, only_rays=config.only_rays, verbose=config.verbose)
                write_spectra_hdf5(result, output, overwrite=True)
                outputs = [str(output)]
                details = {"line": config.line, "ray_count": len(result.ray_ids)}
            elif product == "mfp":
                output = output_dir / "mfp912" / f"{Path(config.los_file).stem}_mfp912.json"
                result = calculate_mean_free_paths(los_data, only_rays=config.only_rays,
                                                   starts_per_ray=config.mfp_starts_per_ray, seed=config.mfp_seed,
                                                   workers=mfp_workers)
                reference = calculate_mean_free_paths_reference(los_data, result.starting_indices) if config.mfp_cross_check else None
                atomic_write_json(output, mfp_result_document(result, source_los_file=config.los_file,
                                                              simulation=config.simulation_name, snapshot=config.snapshot,
                                                              reference=reference))
                compute_and_cache_snapshot_ionizing_inputs(
                    config.base_path, config.snapshot, mfp_los_file=config.los_file, need_mfp=True,
                    need_gamma=False, starts_per_ray=config.mfp_starts_per_ray, seed=config.mfp_seed,
                    refresh=config.refresh_ionizing_cache, allow_legacy=config.allow_legacy_ionizing_table,
                    progress=progress, mfp_result=result, mfp_los_data=los_data,
                    mfp_workers=mfp_workers,
                )
                outputs = [str(output)]
                details = {**result.summary(), "workers": mfp_workers}
            elif product == "gamma":
                output = output_dir / "gamma_hi" / "gamma_hi.json"
                result = compute_gamma_hi_result(snapshot_files, hi_threshold=config.gamma_hi_threshold,
                                                 cross_check=config.gamma_cross_check,
                                                 chunk_size=config.gamma_chunk_size,
                                                 progress_interval=config.progress_interval,
                                                 workers=gamma_workers)
                atomic_write_json(output, gamma_result_document(result, source_files=snapshot_files))
                compute_and_cache_snapshot_ionizing_inputs(
                    config.base_path, config.snapshot, need_mfp=False, need_gamma=True,
                    hi_threshold=config.gamma_hi_threshold, refresh=config.refresh_ionizing_cache,
                    allow_legacy=config.allow_legacy_ionizing_table, progress=progress,
                    gamma_chunk_size=config.gamma_chunk_size, gamma_result=result,
                    gamma_workers=gamma_workers,
                )
                outputs = [str(output)]
                details = {**result.summary(), "workers": gamma_workers}
            else:
                mfp_table, gamma_table = compute_and_cache_snapshot_ionizing_inputs(
                    config.base_path, config.snapshot, mfp_los_file=config.los_file, need_mfp=True, need_gamma=True,
                    starts_per_ray=config.mfp_starts_per_ray, seed=config.mfp_seed,
                    hi_threshold=config.gamma_hi_threshold, refresh=config.refresh_ionizing_cache,
                    allow_legacy=config.allow_legacy_ionizing_table, progress=progress,
                    gamma_chunk_size=config.gamma_chunk_size, mfp_los_data=los_data,
                    gamma_workers=gamma_workers, mfp_workers=mfp_workers,
                )
                if config.temperature_file:
                    temperature = Path(config.temperature_file)
                else:
                    temperature = snapshot_files[0].parent / "Tigm_Thesan1.dat"
                    if not temperature.exists() and config.compute_missing_temperature:
                        from ..temperature import compute_and_cache_snapshot_temperature

                        temperature = compute_and_cache_snapshot_temperature(
                            config.base_path,
                            config.snapshot,
                            mean_molecular_weight=config.mean_molecular_weight,
                            weighting=config.temperature_weighting,
                            chunk_size=config.equation_chunk_size,
                            workers=equation_workers,
                            progress=progress,
                        )
                from ..equation_tests import compute_equation_tests, write_equation_tests_result
                result = compute_equation_tests(
                    config.base_path, config.snapshot, mfp_table, sigma_hi_cm2=config.sigma_hi_cm2,
                    temperature_file=temperature, gamma_hi_file=gamma_table,
                    reduced_speed_of_light_fraction=config.reduced_speed_of_light_fraction,
                    photon_groups=config.photon_groups, photon_group_tests=config.photon_group_tests,
                    thresholds=config.thresholds,
                    threshold_min=config.threshold_min, threshold_max=config.threshold_max,
                    threshold_count=config.threshold_count,
                    ionized_density_thresholds=config.ionized_density_thresholds,
                    ionized_cuts=config.ionized_cuts, ionized_sweep=config.ionized_sweep,
                    ionized_cut_min=config.ionized_cut_min, ionized_cut_max=config.ionized_cut_max,
                    ionized_cut_count=config.ionized_cut_count, chunk_size=config.equation_chunk_size,
                    recombination_temperature_mode=config.recombination_temperature_mode,
                    mean_molecular_weight=config.mean_molecular_weight,
                    simulation_name=config.simulation_name, progress=progress,
                    progress_interval=config.progress_interval, workers=equation_workers,
                )
                output = output_dir / "equations" / "equations.json"
                json_output, csv_output = write_equation_tests_result(result, output)
                outputs = [str(json_output), str(csv_output)]
                details = {"row_count": len(result.document.get("rows", [])), "workers": equation_workers}
            update_product(product, {"status": "success", "fingerprint": fingerprint, "outputs": outputs,
                                     "duration_seconds": perf_counter() - started, "details": details})
            if progress:
                progress(f"{product}: completed")
        except Exception as exc:
            update_product(product, {"status": "failed", "fingerprint": fingerprint, "outputs": [],
                                     "duration_seconds": perf_counter() - started,
                                     "error": {"type": type(exc).__name__, "message": str(exc)}})
            if progress:
                progress(f"{product}: failed: {exc}")

    document["status"] = "failed" if document["failures"] else "success"
    document["finished_at"] = _now()
    document["updated_at"] = _now()
    atomic_write_json(manifest_path, document)
    return SnapshotWorkflowResult(manifest_path, document)
