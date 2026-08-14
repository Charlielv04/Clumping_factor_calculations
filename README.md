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
  results/                    Canonical scientific artifacts and non-JSON analyses
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

The one-time clean-break migration upgraded 1,549 historical JSON sources to
1,494 canonical destinations and deduplicated 55 payload-identical copies.
Source paths, destination paths, methods, payload checksums, and file checksums
are recorded in
[`reports/migrations/schema2-results.csv`](reports/migrations/schema2-results.csv).
There is intentionally no migration or organizer command.

Validate one or more artifacts with:

```bash
clumping results validate results/path/to/result.json
```

Non-JSON analysis products and human-authored reports were not moved.

## Ownership rules

- Scientific formulas belong in their domain package, never in `command.py`,
  CLI adapters, campaign rendering, or shell.
- Shared loading, deposition, preprocessing, caching, serialization,
  provenance, validation, and path logic belong in `infrastructure`.
- Plot discovery reads schema metadata rather than interpreting historical
  directory names.
- New methods require a registry entry, typed configuration, service result,
  command coverage, numerical parity test, and campaign capability where
  applicable.
- Historical material is recoverable through Git history only.

See [`docs/architecture.md`](docs/architecture.md) and
[`docs/guardrails.md`](docs/guardrails.md) for the enforceable boundaries.
