"""Stable identifiers for estimators, field builders, and workflows."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class MethodSpec:
    """The non-numerical contract for one supported method or workflow."""

    identifier: str
    domain: str
    description: str
    supported_particle_types: tuple[str, ...]
    field_representation: str
    weighting: str
    mask_semantics: str
    field_builder: str
    estimator: str
    selection: str
    producer: str
    command_kind: str | None = None
    command_variant: str | None = None
    grid_requirements: tuple[str, ...] = ()
    optional_dependencies: tuple[str, ...] = ()
    execution_modes: tuple[str, ...] = ("local",)
    presets: tuple[str, ...] = ()
    legacy_backends: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class MethodRegistry:
    """Registry with stable identifiers and backwards-compatible presets."""

    def __init__(self, specs: Iterable[MethodSpec] = ()) -> None:
        self._specs: dict[str, MethodSpec] = {}
        self._presets: dict[str, str] = {}
        for spec in specs:
            self.register(spec)

    def register(self, spec: MethodSpec) -> MethodSpec:
        if spec.identifier in self._specs:
            raise ValueError(f"Method identifier already registered: {spec.identifier}")
        self._specs[spec.identifier] = spec
        for preset in (spec.identifier, *spec.presets, *spec.legacy_backends):
            previous = self._presets.setdefault(preset, spec.identifier)
            if previous != spec.identifier:
                raise ValueError(f"Method preset already points to {previous}: {preset}")
        return spec

    def get(self, identifier_or_preset: str) -> MethodSpec:
        try:
            identifier = self._presets[identifier_or_preset]
        except KeyError as exc:
            known = ", ".join(sorted(self._presets))
            raise KeyError(f"Unknown method or preset {identifier_or_preset!r}; known values: {known}") from exc
        return self._specs[identifier]

    def expand_preset(self, preset: str) -> dict[str, object]:
        """Return a serializable method contract for a legacy preset."""

        return self.get(preset).to_dict()

    def catalog(self) -> list[dict[str, object]]:
        return [self._specs[key].to_dict() for key in sorted(self._specs)]

    @property
    def identifiers(self) -> tuple[str, ...]:
        return tuple(sorted(self._specs))

    @property
    def presets(self) -> tuple[str, ...]:
        return tuple(sorted(self._presets))


def _spec(
    identifier: str,
    domain: str,
    description: str,
    particles: tuple[str, ...],
    representation: str,
    weighting: str,
    mask: str,
    *,
    grid: tuple[str, ...] = (),
    optional: tuple[str, ...] = (),
    modes: tuple[str, ...] = ("local",),
    presets: tuple[str, ...] = (),
    legacy: tuple[str, ...] = (),
    command_kind: str | None = None,
    command_variant: str | None = None,
) -> MethodSpec:
    producer_by_domain = {
        "clumping": "clumping_factor.methods.clumping",
        "transmission": "clumping_factor.methods.clumping",
        "alternative": "clumping_factor.methods.clumping",
        "power-spectrum": "clumping_factor.methods.power_spectrum",
        "forest": "clumping_factor.methods.forest",
        "thermodynamics": "clumping_factor.methods.thermodynamics",
        "diagnostics": "clumping_factor.diagnostics",
        "campaign": "clumping_factor.visualization",
        "operations": "clumping_factor.infrastructure",
    }
    return MethodSpec(
        identifier=identifier,
        domain=domain,
        description=description,
        supported_particle_types=particles,
        field_representation=representation,
        weighting=weighting,
        mask_semantics=mask,
        field_builder=representation,
        estimator=identifier,
        selection=mask,
        producer=producer_by_domain[domain],
        command_kind=command_kind,
        command_variant=command_variant,
        grid_requirements=grid,
        optional_dependencies=optional,
        execution_modes=modes,
        presets=presets,
        legacy_backends=legacy,
    )


_SPECS = [
    _spec("clumping.sphere", "clumping", "Mass deposition with spherical top-hat smoothing", ("gas", "dm", "both"), "density-grid", "mass", "threshold-on-measured-field", grid=("grid-size", "mas", "radius-mode"), modes=("full", "chunked", "threaded", "pbs"), presets=("sphere",), legacy=("sphere",), command_kind="clumping-compute", command_variant="sphere"),
    _spec("clumping.cube", "clumping", "Mass deposition with cubic top-hat smoothing", ("gas", "dm", "both"), "density-grid", "mass", "threshold-on-measured-field", grid=("grid-size", "mas", "radius-mode"), modes=("full", "chunked", "threaded", "pbs"), presets=("cube",), legacy=("cube",), command_kind="clumping-compute", command_variant="cube"),
    _spec("clumping.pylians", "clumping", "Pylians mass-assignment and smoothing estimator", ("gas", "dm", "both"), "density-grid", "mass", "threshold-on-measured-field", grid=("grid-size", "mas", "filter-type"), optional=("Pylians",), modes=("full", "chunked", "threaded", "pbs"), presets=("pylians",), legacy=("pylians",), command_kind="clumping-compute", command_variant="pylians"),
    _spec("clumping.raw-cell-weighted", "clumping", "Native-cell density clumping with cell weighting", ("gas",), "native-cells", "cell", "threshold-on-native-density", modes=("full", "chunked", "pbs"), presets=("raw",), legacy=("raw",), command_kind="clumping-compute", command_variant="raw"),
    _spec("clumping.raw-volume-weighted", "clumping", "Native-cell density clumping with volume weighting", ("gas",), "native-cells", "volume", "threshold-on-native-density", modes=("full", "chunked", "pbs"), presets=("raw-volume",), legacy=("raw-volume",), command_kind="clumping-compute", command_variant="raw-volume"),
    _spec("clumping.mask-target", "clumping", "Separate field construction for mask and target", ("gas", "dm", "both"), "paired-density-grids", "mass", "mask-field-selects-target-field", grid=("grid-size",), modes=("full", "chunked", "threaded", "pbs"), presets=("mask-target",), legacy=("masked",), command_kind="clumping-compute", command_variant="sphere"),
    _spec("transmission.raw", "transmission", "Native-cell density moments with grid-derived optical-depth weights", ("gas",), "native-cells-plus-density-grid", "volume-and-transmission", "grid-derived-transmission-weight", grid=("grid-size", "mas"), modes=("chunked", "threaded", "pbs"), presets=("raw-transmission",), legacy=("raw-transmission",), command_kind="clumping-compute", command_variant="raw-transmission"),
    _spec("transmission.voronoi", "transmission", "Voronoi-neighbour transmission estimator", ("gas",), "voronoi-cells", "volume", "transmission-selection", optional=("scipy",), modes=("chunked", "threaded", "pbs"), presets=("voronoi-transmission",), legacy=("voronoi-transmission",), command_kind="clumping-compute", command_variant="voronoi-transmission"),
    _spec("alternative.estimators", "alternative", "Compatibility umbrella for alternative clumping estimators", ("gas", "dm"), "density-grid-or-native", "method-defined", "method-defined", grid=("optional-grid",), modes=("full", "chunked", "threaded", "pbs"), presets=("alternative",)),
    _spec("alternative.raw-volume", "alternative", "Davies Eq. 13 estimator on native volume-weighted cells", ("gas",), "native-cells", "recombination-and-volume", "overdensity-and-ionization", modes=("chunked", "pbs"), presets=("alternative-raw-volume",), command_kind="alternative-compute", command_variant="raw-volume"),
    _spec("alternative.grid-masked", "alternative", "Davies Eq. 13 estimator with a gridded IGM mask", ("gas", "dm"), "native-target-plus-density-grid-mask", "recombination-and-volume", "separate-grid-mask", grid=("grid-size",), modes=("full", "chunked", "threaded", "pbs"), presets=("alternative-grid",), command_kind="alternative-compute", command_variant="grid"),
    _spec("alternative.ionized-sweep", "alternative", "Ionized-fraction threshold sweep", ("gas",), "native-cells", "volume", "ionized-fraction-cut", modes=("chunked", "pbs"), presets=("ionized-sweep",), command_kind="ionized-sweep"),
    _spec("power-spectrum.numpy", "power-spectrum", "NumPy FFT density power spectrum", ("gas", "dm", "both"), "density-grid", "mass", "not-applicable", grid=("grid-size", "bin-count"), modes=("full", "chunked", "threaded", "pbs"), presets=("numpy",), legacy=("numpy-power-spectrum",), command_kind="power-spectrum-compute", command_variant="numpy"),
    _spec("power-spectrum.pylians", "power-spectrum", "Pylians density power spectrum", ("gas", "dm", "both"), "density-grid", "mass", "not-applicable", grid=("grid-size",), optional=("Pylians",), modes=("full", "chunked", "threaded", "pbs"), presets=("pylians-power-spectrum",), command_kind="power-spectrum-compute", command_variant="pylians"),
    _spec("power-spectrum.combined", "power-spectrum", "NumPy and Pylians spectra from the same density field", ("gas", "dm", "both"), "density-grid", "mass", "not-applicable", grid=("grid-size",), optional=("Pylians",), modes=("full", "chunked", "threaded", "pbs"), presets=("both-power-spectrum",), command_kind="power-spectrum-compute", command_variant="both"),
    _spec("power-spectrum.arepo-comparison", "power-spectrum", "AREPO text versus local estimator comparison", ("gas", "dm", "both"), "spectrum-series", "method-defined", "not-applicable", optional=("matplotlib",), modes=("local",), presets=("arepo-comparison",)),
    _spec("power-spectrum.relative-evolution", "power-spectrum", "Relative power-spectrum evolution", ("gas", "dm", "both"), "spectrum-series", "method-defined", "not-applicable", modes=("local",), presets=("relative-evolution",)),
    _spec("forest.lyman-alpha", "forest", "Lyman-alpha forest spectra", ("gas",), "line-of-sight", "column-density", "ray-selection", optional=("h5py",), presets=("lyman-alpha",), command_kind="forest-spectra"),
    _spec("forest.mfp", "forest", "Mean-free-path calculation", ("gas",), "line-of-sight", "optical-depth", "ray-selection", presets=("mfp",), command_kind="forest-ionizing", command_variant="mfp"),
    _spec("forest.gamma-hi", "forest", "Gamma_HI radiation model", ("gas",), "radiation-field", "volume", "ionizing-selection", presets=("gamma-hi", "gamma_hi"), command_kind="forest-ionizing", command_variant="gamma"),
    _spec("forest.caches", "forest", "Forest and radiation cache artifacts", ("gas",), "cache", "n/a", "cache-key", presets=("forest-cache",)),
    _spec("forest.snapshot", "forest", "Snapshot workflow orchestration", ("gas",), "workflow", "method-defined", "workflow-selection", modes=("local", "threaded", "pbs"), presets=("snapshot-workflow",), command_kind="forest-snapshot"),
    _spec("thermodynamics.particle-temperature", "thermodynamics", "Particle temperature diagnostic", ("gas",), "particle-values", "mass", "temperature-selection", presets=("particle-temperature",)),
    _spec("thermodynamics.snapshot-temperature", "thermodynamics", "Snapshot temperature diagnostic", ("gas",), "snapshot-values", "volume", "temperature-selection", modes=("full", "chunked", "pbs"), presets=("snapshot-temperature",), command_kind="temperature"),
    _spec("diagnostics.density-ratio", "diagnostics", "Density-ratio diagnostic", ("gas", "dm", "both"), "derived-series", "method-defined", "selection-spec", modes=("chunked", "pbs"), presets=("density-ratio",), command_kind="diagnostics", command_variant="density-ratio"),
    _spec("diagnostics.equations", "diagnostics", "Equation 5-13 diagnostic suite", ("gas",), "derived-series", "method-defined", "selection-spec", modes=("chunked", "pbs"), presets=("equations", "eq5-13"), command_kind="diagnostics", command_variant="equations"),
    _spec("diagnostics.igm-checks", "diagnostics", "IGM checks and story plots", ("gas",), "derived-series", "method-defined", "selection-spec", presets=("igm-checks",)),
    _spec("campaign.tng", "campaign", "TNG campaign analysis", ("gas", "dm", "both"), "result-series", "method-defined", "campaign-selection", presets=("tng-campaign",)),
    _spec("campaign.thesan", "campaign", "THESAN campaign analysis", ("gas", "dm", "both"), "result-series", "method-defined", "campaign-selection", presets=("thesan-campaign",)),
    _spec("campaign.aida-tng", "campaign", "AIDA-TNG campaign analysis", ("gas", "dm", "both"), "result-series", "method-defined", "campaign-selection", presets=("aida-tng",)),
    _spec("campaign.evolution", "campaign", "Evolution plots", ("gas", "dm", "both"), "result-series", "method-defined", "campaign-selection", presets=("evolution",)),
    _spec("campaign.model", "campaign", "Model comparison plots", ("gas", "dm", "both"), "result-series", "method-defined", "campaign-selection", presets=("model-comparison",)),
    _spec("campaign.benchmark", "campaign", "Benchmark and grid convergence plots", ("gas", "dm", "both"), "result-series", "method-defined", "campaign-selection", presets=("benchmark",)),
    _spec("operations.validation", "operations", "Result validation", ("gas", "dm", "both"), "artifact", "n/a", "validation-rules", presets=("validate",)),
    _spec("operations.provenance", "operations", "Provenance collection", ("gas", "dm", "both"), "artifact", "n/a", "provenance-rules", presets=("provenance",)),
    _spec("operations.campaigns", "operations", "Declarative campaign planning", ("gas", "dm", "both"), "task-manifest", "n/a", "task-selection", presets=("campaigns",)),
    _spec("operations.pbs", "operations", "Generic PBS planning and worker surface", ("gas", "dm", "both"), "task-manifest", "n/a", "task-selection", presets=("pbs",)),
]

METHOD_REGISTRY = MethodRegistry(_SPECS)


def expand_preset(preset: str) -> dict[str, object]:
    return METHOD_REGISTRY.expand_preset(preset)


def method_catalog(path: str | Path | None = None) -> list[dict[str, object]]:
    catalog = METHOD_REGISTRY.catalog()
    if path is not None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(catalog, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return catalog
