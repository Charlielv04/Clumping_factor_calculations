# Clumping Factor Calculations

Domain-organized scientific tools for clumping, transmission, power spectra,
Lyman-alpha forest and radiation workflows, thermodynamics, and diagnostics.

## Structure

```text
Clumping_factor_calculations/
  campaigns/                  Declarative simulation, method, and resource matrices
  docs/                       Architecture, guardrails, and decision records
  reports/
    consolidation/            Script-replacement audit
    migrations/               Schema-2 source/destination checksum manifest
  results/                    Canonical science, analysis, and preserved archives
  src/clumping_factor/
    command.py                The only public command router
    methods/
      registry.py             Sole owner of stable method identifiers and capabilities
      clumping/               Fields, estimators, transmission, alternatives, sweeps
      power_spectrum/          NumPy/Pylians computation services
      forest/                 Spectra, MFP, Gamma_HI, caches, snapshot workflow, line data
      thermodynamics/         Particle and snapshot temperature calculations
    diagnostics/              Equation and density-ratio calculations
    visualization/            Result, campaign, evolution, model, equation, benchmark, IGM plots
    infrastructure/           Loading, deposition, models, caching, results, paths, validation, campaigns
  tests/                      Numerical parity, contracts, commands, campaigns, architecture
```

The source root deliberately contains only `command.py`, `__init__.py`, and
the domain/infrastructure packages. There are no compatibility import shims,
historical console aliases, method-specific PBS scripts, or executable legacy
implementations.

## Install and verify

```bash
python -m pip install -e ".[dev]"
pytest
ruff check src tests
mypy src/clumping_factor
```

Pylians is optional. Install it separately when running its parity or production
methods.

## Public command tree

`clumping` is the only installed executable:

```text
clumping clumping compute|alternative|ionized-sweep
clumping power compute|plot|compare
clumping forest spectra|ionizing|snapshot
clumping temperature compute
clumping diagnostics equations|density-ratio
clumping plot result|campaign|evolution|model|equations|benchmark|igm
clumping results validate
clumping campaign plan|submit
clumping methods catalog
```

Use nested help for the authoritative options, for example:

```bash
clumping clumping compute --help
clumping forest snapshot --help
clumping plot benchmark --help
```

Python callers import configuration dataclasses, services, and results from
their canonical domain packages. `argparse.Namespace` is converted at the CLI
boundary and is not a service API.

## Implementation contract

All future implementations must preserve these boundaries:

- Put scientific ownership in exactly one domain package. Use `methods/clumping`
  for fields, clumping, transmission, and alternative estimators;
  `methods/power_spectrum` for spectra; `methods/forest` for forest and
  radiation; `methods/thermodynamics` for temperature; `diagnostics` for
  equations and density-ratio calculations; and `visualization` for plots.
- Put loading, deposition, units, preprocessing, caching, provenance, result
  serialization, validation, and path construction in `infrastructure`.
- Keep CLI adapters limited to parsing and typed service invocation. Scientific
  formulas, default paths, result naming, and scheduler logic must not be added
  to `command.py`, CLI adapters, PBS files, or shell scripts.
- Register every executable method once in `methods.registry`. A registry entry
  must state its stable identifier, supported particles, field representation,
  weighting, selection/mask semantics, grid requirements, optional dependencies,
  execution modes, and command capabilities. Do not introduce a second method
  name or a backend-specific path convention.
- Add a typed configuration, service, result model, command adapter, parity test,
  and (where applicable) campaign capability for every new method.

## Method registry

Every executable calculation has one stable identifier in
`clumping_factor.methods.registry`. The registry records supported particle
types, field representation, weighting, mask semantics, grid requirements,
optional dependencies, execution modes, and command capability. Short names
such as `sphere`, `raw-volume`, and `pylians` are registry presets, not hidden
bundles of behavior.

```bash
clumping methods catalog
clumping methods catalog --output reports/method-catalog.json
```

## Declarative campaigns

Campaign TOML is the only scheduler workflow source. It owns simulations,
snapshots, methods, grids, method-specific options, algorithmic execution
settings, and scheduler resources. The planner expands a deterministic task
manifest and the submit command renders generic PBS workers.

```bash
clumping campaign plan campaigns/tng-smoke.toml
clumping campaign submit campaigns/tng-smoke.toml
clumping campaign submit campaigns/tng-smoke.toml --execute
```

Submission is a dry run unless `--execute` is supplied. See
[`campaigns/README.md`](campaigns/README.md) and the checked-in AIDA-TNG,
THESAN, diagnostic, forest, and alternative-estimator campaigns.

## Strict schema-2 results

JSON result readers accept schema 2 only. Every result requires:

- `method_spec`, including actual settings in `method_spec.configuration`;
- `selection_spec`;
- `execution_spec`;
- provenance;
- normalized simulation identity.

Canonical paths are derived from the specifications:

```text
results/<family>/<simulation>/<domain>/<method>/<particle>/
  snapshotNNN/science-<12hex>/execution-<12hex>_runNNN.json
```

The science hash uses canonical JSON for `method_spec` plus `selection_spec`.
The execution hash uses algorithmic execution settings; queue, walltime,
campaign, task, and scheduler CPU metadata remain recorded but do not affect
identity.

The one-time results consolidation accounted for all 4,428 tracked source
artifacts. Science companions (HDF5, CSV, and similar run products) live beside
their owning JSON in `execution-<12hex>_runNNN.artifacts/` and are indexed with
checksums in the JSON result. Derived plots and tables live in the hashed
`results/analysis/` hierarchy, where each analysis directory contains a
`manifest.json` and its `artifacts/`. Unique historical material is preserved
under `results/archive/` with an archive inventory and is excluded from normal
result discovery.

The complete source-to-destination and checksum record is
[`reports/migrations/results-consolidation-final-manifest.csv`](reports/migrations/results-consolidation-final-manifest.csv).
The reference input formerly stored in `results/` is now
[`reference_data/Tigm_Davies_10000K.dat`](reference_data/Tigm_Davies_10000K.dat).
There is intentionally no migration or organizer command.

### Result and artifact rules

Producers must use the shared infrastructure path and manifest builders. They
must never construct a result path by concatenating a historical folder name,
and must never write an unregistered `backend`-only result. A schema-2 science
record contains normalized `method_spec`, `selection_spec`, `execution_spec`,
provenance, and simulation identity; actual field and estimator settings belong
inside `method_spec.configuration`.

Run products are written beside their owning JSON under the matching
`.artifacts/` directory. Each artifact entry records its relative path, role,
media type, byte size, and SHA-256 checksum. If a product cannot be associated
deterministically with a science owner, preserve it as an analysis or archive
manifest with explicit provenance rather than inventing an owner.

Analysis products use this shape:

```text
results/analysis/<domain>/<family>/<analysis-kind>/<subject>/
  <readable-method-label>/analysis-<12hex>/
    manifest.json
    artifacts/<descriptive-filename>
```

The analysis hash covers normalized plotting/analysis options and canonical
input identities. Its manifest records the generator, inputs, legacy source
paths, and checksums for every artifact. External output paths are allowed only
as explicit user-requested escape hatches and must receive a validating
`analysis-external` sidecar.

Unique historical material belongs under `results/archive/<import-id>/` with an
archive inventory and is excluded from normal discovery. Byte-identical files
may share one canonical owner; the migration manifest must retain every alias.
Non-identical collisions must stop the operation rather than receive an
arbitrary suffix.

Simulation identities are normalized at the infrastructure boundary. In
particular, `output_4_128_sl`, `output_4_128_rsl`, and the recorded in-house
THESAN base paths map to their `thesan-mini-*` identities, and
`Thesan-1-parallelized` is represented as additional deterministic runs of
`Thesan-1`. Embedded metadata takes precedence over misleading legacy folders.

Validate one or more artifacts with:

```bash
clumping results validate results
```

## Ownership rules

- Scientific formulas belong in their domain package, never in `command.py`,
  CLI adapters, campaign rendering, or shell.
- Shared loading, deposition, preprocessing, caching, serialization,
  provenance, validation, and path logic belong in `infrastructure`.
- Plot discovery reads schema metadata rather than interpreting historical
  directory names.
- Forest HDF5, equation tables, spectra, mean-free-path, Gamma-HI, benchmark,
  campaign, and visualization producers all use the same companion or analysis
  manifest contract.
- Campaign TOML is the only operational source. The planner owns task manifests
  and generic PBS rendering; scheduler scripts must not duplicate scientific
  defaults or result-path logic.
- `clumping results validate results` is the required pre-publication check. It
  validates schema paths, companion and analysis checksums, archive inventories,
  provenance, and forbidden legacy roots.
- New methods require a registry entry, typed configuration, service result,
  command coverage, numerical parity test, and campaign capability where
  applicable.
- Historical material is recoverable through Git history only.

See [`docs/architecture.md`](docs/architecture.md) and
[`docs/guardrails.md`](docs/guardrails.md) for the enforceable boundaries.
