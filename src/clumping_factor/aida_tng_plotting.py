from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .plotting import (
    plot_campaign_files,
    plot_evolution_files,
    plot_model_evolution_files,
    plot_result_files,
)
from .power_spectrum_plotting import (
    plot_power_spectrum_files,
    plot_relative_power_spectrum_evolution_files,
)


_SNAPSHOT_RE = re.compile(r"snapshot(?P<snapshot>\d+)_grid(?P<grid>\d+)", re.IGNORECASE)
_MODEL_RE = re.compile(r"^(?P<box>L\d+n\d+)_?(?P<model>CDM|SIDM1|vSIDM|WDM3)$", re.IGNORECASE)


@dataclass(frozen=True)
class AidaResult:
    path: Path
    simulation: str
    kind: str
    particle: str
    backend: str
    snapshot: int | None
    grid: int | None
    method: str


def _metadata(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _snapshot_grid(path: Path, document: dict) -> tuple[int | None, int | None]:
    simulation = document.get("simulation", {})
    parameters = document.get("parameters", {})
    snapshot = simulation.get("snapshot", parameters.get("snapshot"))
    grid = parameters.get("grid_size")
    match = _SNAPSHOT_RE.search(str(path))
    if snapshot is None and match:
        snapshot = int(match.group("snapshot"))
    if grid is None and match:
        grid = int(match.group("grid"))
    return (int(snapshot) if isinstance(snapshot, int) else None, int(grid) if isinstance(grid, int) else None)


def _simulation(path: Path, document: dict) -> str:
    simulation = document.get("simulation", {})
    parameters = document.get("parameters", {})
    value = simulation.get("name") or parameters.get("simulation_name")
    if value:
        text = str(value)
        match = _MODEL_RE.match(text)
        return text if match else text
    return path.parts[path.parts.index("aida-tng") + 1] if "aida-tng" in path.parts else "unknown"


def discover_aida_tng_results(results_root: str | Path = "results/aida-tng") -> list[AidaResult]:
    root = Path(results_root)
    results: list[AidaResult] = []
    for path in sorted(root.rglob("*.json")) if root.exists() else []:
        document = _metadata(path)
        if not document:
            continue
        simulation = _simulation(path, document)
        snapshot, grid = _snapshot_grid(path, document)
        particle = str(document.get("particle_type") or document.get("parameters", {}).get("particle_type") or ("gas" if "ionized-sweep" in path.parts else "unknown"))
        backend = str(document.get("backend", {}).get("backend") or "unknown")
        if document.get("statistic") == "density_power_spectrum":
            smoothing = str(document.get("parameters", {}).get("smoothing") or "unknown")
            method = path.parent.parent.name or smoothing
            results.append(AidaResult(path, simulation, "power-spectrum", particle, backend, snapshot, grid, method))
        elif document.get("calculation") == "ionized_igm_raw_volume_sweep" or "ionized-sweep" in path.parts:
            results.append(AidaResult(path, simulation, "ionization", particle, backend, snapshot, grid, "ionized-sweep"))
        elif "clumping_factors" in document or "clumping_factor" in document:
            results.append(AidaResult(path, simulation, "clumping", particle, backend, snapshot, grid, backend))
    return results


def canonical_plot_path(
    analysis_root: str | Path,
    family: str,
    subject: str,
    period: str,
    particle: str,
    method: str,
    filename: str,
) -> Path:
    if family.startswith("clumping/"):
        category = family.split("/", 1)[1]
        return Path(analysis_root) / "clumping" / "aida-tng" / category / subject / period / particle / method / filename
    return Path(analysis_root) / family / "aida-tng" / subject / period / particle / method / filename


def _group(results: Iterable[AidaResult], *keys: str) -> dict[tuple, list[AidaResult]]:
    grouped: dict[tuple, list[AidaResult]] = defaultdict(list)
    for result in results:
        grouped[tuple(getattr(result, key) for key in keys)].append(result)
    return dict(grouped)


def _unique_paths(results: Iterable[AidaResult]) -> list[Path]:
    return sorted({result.path for result in results})


def _one_result_per_snapshot(results: Iterable[AidaResult]) -> list[Path]:
    selected: dict[int | None, Path] = {}
    for result in sorted(results, key=lambda item: item.path.name):
        selected.setdefault(result.snapshot, result.path)
    return sorted(selected.values())


def _snapshot_name(snapshot: int | None) -> str:
    return f"snapshot{snapshot:03d}" if snapshot is not None else "unknown-snapshot"


def _grid_name(grid: int | None) -> str:
    return f"grid{grid}" if grid is not None else "unknown-grid"


def _box_name(simulations: Iterable[str]) -> str:
    boxes = sorted({match.group("box") for simulation in simulations if (match := _MODEL_RE.match(simulation))})
    return boxes[0] if len(boxes) == 1 else "combined-boxes"


def _write_manifest(rows: list[dict], output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["category", "output", "inputs", "status", "message"]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return output


def _record(rows: list[dict], category: str, output: Path, inputs: Iterable[Path], status: str = "written", message: str = "") -> None:
    rows.append({"category": category, "output": str(output), "inputs": ";".join(str(path) for path in sorted(set(inputs))), "status": status, "message": message})


def archive_aida_plots(analysis_root: str | Path = "results/analysis", *, dry_run: bool = False) -> Path | None:
    source = Path(analysis_root)
    pngs = sorted(path for path in source.rglob("*.png") if "aida-tng" in path.parts)
    if not pngs:
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = Path(analysis_root) / "archive" / "aida-tng" / stamp
    if not dry_run:
        for png in pngs:
            if not png.exists():
                continue
            destination = archive / png.relative_to(source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.move(str(png), str(destination))
            except FileNotFoundError:
                # A prior interrupted archive may have moved this file while
                # the initial recursive scan was still in progress.
                continue
    return archive


def generate_aida_tng_plots(
    results_root: str | Path = "results/aida-tng",
    analysis_root: str | Path = "results/analysis",
    *,
    threshold: float = 20.0,
    dry_run: bool = False,
    archive_existing: bool = False,
) -> list[Path]:
    if archive_existing:
        archive_aida_plots(analysis_root, dry_run=dry_run)
    results = discover_aida_tng_results(results_root)
    rows: list[dict] = []
    outputs: list[Path] = []

    def render(category: str, output: Path, inputs: list[Path], function, **kwargs) -> None:
        if dry_run:
            _record(rows, category, output, inputs, status="planned")
            outputs.append(output)
            return
        try:
            written = function(inputs, output, **kwargs)
        except (ValueError, OSError) as exc:
            _record(rows, category, output, inputs, status="skipped", message=str(exc))
            return
        _record(rows, category, Path(written), inputs)
        outputs.append(Path(written))

    clumping = [result for result in results if result.kind == "clumping" and result.snapshot is not None]
    for (simulation, snapshot, particle, grid), values in sorted(_group(clumping, "simulation", "snapshot", "particle", "grid").items()):
        if len({value.backend for value in values}) < 2:
            continue
        output = canonical_plot_path(analysis_root, "clumping/method-comparison", simulation, _snapshot_name(snapshot), particle, _grid_name(grid), "backend_comparison.png")
        render("clumping/method-comparison", output, _unique_paths(values), plot_result_files, title=f"{simulation} {particle} methods, snapshot {snapshot:03d}")

    for (simulation, particle, backend, grid), values in sorted(_group(clumping, "simulation", "particle", "backend", "grid").items()):
        snapshots = {value.snapshot for value in values}
        if len(snapshots) < 2:
            continue
        output = canonical_plot_path(analysis_root, "clumping/evolution", simulation, "combined", particle, f"{backend}-{_grid_name(grid)}", "clumping_vs_redshift.png")
        render("clumping/evolution", output, _one_result_per_snapshot(values), plot_evolution_files, thresholds=[threshold])

    for (particle, backend, grid), values in sorted(_group(clumping, "particle", "backend", "grid").items()):
        simulations = {value.simulation for value in values if _MODEL_RE.match(value.simulation)}
        if len(simulations) < 2:
            continue
        for snapshot in sorted({value.snapshot for value in values}):
            snapshot_values = [value for value in values if value.snapshot == snapshot]
            by_box = defaultdict(list)
            for value in snapshot_values:
                match = _MODEL_RE.match(value.simulation)
                if match:
                    by_box[match.group("box")].append(value)
            for subject, box_values in sorted(by_box.items()):
                representatives: dict[str, AidaResult] = {}
                for value in sorted(box_values, key=lambda item: item.path.name):
                    representatives.setdefault(value.simulation, value)
                box_values = list(representatives.values())
                if len({value.simulation for value in box_values}) < 2:
                    continue
                input_paths = _unique_paths(box_values)
                output = canonical_plot_path(analysis_root, "clumping/model-comparison", subject, _snapshot_name(snapshot), particle, f"{backend}-{_grid_name(grid)}", "model_comparison.png")
                render("clumping/model-comparison", output, input_paths, plot_result_files, title=f"{subject} {particle} model comparison, snapshot {snapshot:03d}")
                cdm = next((candidate for candidate in box_values if candidate.simulation.lower().endswith("_cdm")), None)
                if cdm:
                    for value in box_values:
                        match = _MODEL_RE.match(value.simulation)
                        if match and match.group("model").upper() != "CDM":
                            output = canonical_plot_path(analysis_root, "clumping/model-comparison", subject, _snapshot_name(snapshot), particle, f"{backend}-{_grid_name(grid)}", f"{match.group('model').lower()}_relative_to_cdm.png")
                            render("clumping/model-comparison", output, [value.path], plot_result_files, relative_to_baseline=cdm.path)

    # Evolution relative to CDM: one plot per non-CDM model, with one curve
    # for every snapshot shared by all models in the same simulation box.
    for (particle, backend, grid), values in sorted(_group(clumping, "particle", "backend", "grid").items()):
        by_box: dict[str, list[AidaResult]] = defaultdict(list)
        for value in values:
            match = _MODEL_RE.match(value.simulation)
            if match:
                by_box[match.group("box")].append(value)
        for box, box_values in sorted(by_box.items()):
            representatives: dict[tuple[str, int | None], AidaResult] = {}
            for value in sorted(box_values, key=lambda item: item.path.name):
                representatives.setdefault((value.simulation, value.snapshot), value)
            selected = list(representatives.values())
            if len({value.simulation for value in selected}) < 2 or len({value.snapshot for value in selected}) < 2:
                continue
            inputs = _unique_paths(selected)
            output_dir = (
                Path(analysis_root)
                / "clumping"
                / "aida-tng"
                / "model-comparison"
                / box
                / "combined-snapshots"
                / particle
                / f"{backend}-grid{grid}"
                / "relative-to-cdm"
            )
            if dry_run:
                _record(rows, "clumping/model-comparison", output_dir, inputs, status="planned")
                outputs.append(output_dir)
                continue
            try:
                written = plot_model_evolution_files(
                    inputs,
                    output_dir=output_dir,
                    relative_to_cdm=True,
                    particle_type=particle,
                )
            except (ValueError, OSError) as exc:
                _record(rows, "clumping/model-comparison", output_dir, inputs, status="skipped", message=str(exc))
                continue
            for item in written:
                _record(rows, "clumping/model-comparison", Path(item), inputs)
                outputs.append(Path(item))

    ionized = [result for result in results if result.kind == "ionization"]
    for (simulation, snapshot, particle), values in sorted(_group(ionized, "simulation", "snapshot", "particle").items()):
        output = canonical_plot_path(analysis_root, "clumping/ionization", simulation, _snapshot_name(snapshot), particle, "ionized-sweep", "ionized_sweep.png")
        render("clumping/ionization", output, _unique_paths(values), plot_result_files, sweep_axis="ionized", ionized_density_thresholds=None)

    spectra = [result for result in results if result.kind == "power-spectrum"]
    for (simulation, particle, method), values in sorted(_group(spectra, "simulation", "particle", "method").items()):
        output = canonical_plot_path(analysis_root, "power-spectra", simulation, "combined", particle, method, "power_spectra.png")
        # The evolution overview is intentionally a single, reproducible
        # configuration: 256^3 spectra evaluated with the Pylians engine.
        # Keep the generic combined plot behavior for all other methods.
        if method == "smoothed-pylians_both":
            selected = [value for value in values if value.grid == 256]
            render(
                "power-spectra",
                output,
                _unique_paths(selected),
                plot_power_spectrum_files,
                engine="pylians",
                color_by_snapshot=True,
            )
        else:
            render("power-spectra", output, _unique_paths(values), plot_power_spectrum_files, engine="both")

    # Relative evolution: one all-snapshot plot per non-CDM model and box.
    # Match each snapshot to its CDM baseline and use the same 256^3 Pylians
    # configuration as the absolute evolution overview.
    selected_spectra = [
        value
        for value in spectra
        if value.method == "smoothed-pylians_both" and value.grid == 256 and value.particle in {"gas", "dm"}
    ]
    by_simulation = defaultdict(list)
    for value in selected_spectra:
        by_simulation[(value.simulation, value.particle)].append(value)
    for (simulation, particle), values in sorted(by_simulation.items()):
        match = _MODEL_RE.match(simulation)
        if not match or match.group("model").upper() == "CDM":
            continue
        box = match.group("box")
        cdm_simulation = f"{box}_CDM"
        baselines = {
            value.snapshot: value.path
            for value in by_simulation.get((cdm_simulation, particle), [])
            if value.snapshot is not None
        }
        if not baselines:
            continue
        output = canonical_plot_path(
            analysis_root,
            "power-spectra/model-comparison",
            box,
            "combined-snapshots",
            particle,
            "smoothed-pylians_both-grid256",
            f"{match.group('model').lower()}_relative_to_cdm.png",
        )
        render(
            "power-spectra/model-comparison",
            output,
            _unique_paths(values) + _unique_paths(baselines.values()),
            plot_relative_power_spectrum_evolution_files,
            baselines=baselines,
            engine="pylians",
            color_by_snapshot=True,
            title=f"{box} {match.group('model')} gas power relative to CDM (256^3, Pylians)",
        )

    for simulation in sorted({result.simulation for result in clumping}):
        paths = _unique_paths(result for result in clumping if result.simulation == simulation)
        if paths:
            output = canonical_plot_path(analysis_root, "performance", simulation, "combined", "combined", "campaign", "campaign_manifest.csv")
            if dry_run:
                _record(rows, "performance", output, paths, status="planned")
            else:
                try:
                    written = plot_campaign_files(paths, analysis_root=analysis_root)
                    for item in written:
                        _record(rows, "performance", Path(item), paths)
                        outputs.append(Path(item))
                except (ValueError, OSError) as exc:
                    _record(rows, "performance", output, paths, status="skipped", message=str(exc))

    manifest = Path(analysis_root) / "clumping" / "aida-tng" / "aida-tng-plots.csv"
    _write_manifest(rows, manifest)
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate the canonical AIDA-TNG plot catalog.")
    parser.add_argument("--results-root", default="results/aida-tng")
    parser.add_argument("--analysis-root", default="results/analysis")
    parser.add_argument("--threshold", type=float, default=20.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--archive-existing", action="store_true")
    return parser


def aida_tng_plot_main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    outputs = generate_aida_tng_plots(
        args.results_root,
        args.analysis_root,
        threshold=args.threshold,
        dry_run=args.dry_run,
        archive_existing=args.archive_existing,
    )
    print(f"Planned/wrote {len(outputs)} AIDA-TNG plot outputs.")
    print(f"Manifest: {Path(args.analysis_root) / 'clumping' / 'aida-tng' / 'aida-tng-plots.csv'}")


if __name__ == "__main__":
    aida_tng_plot_main()
