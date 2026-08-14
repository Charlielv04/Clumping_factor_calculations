# Architecture

The suite has four explicit boundaries:

1. **Field construction** loads particles/cells and builds native or gridded
   fields.
2. **Estimator** consumes fields and produces a numerical result.
3. **Selection** describes masks, target fields, thresholds, and cuts.
4. **Execution** describes local, chunked, parallel, or PBS scheduling.

`clumping_factor.methods.registry` is the single declarative catalog for these
contracts. Each entry has a stable identifier, supported particle types,
field representation, weighting, mask semantics, field-builder, estimator,
selection and producer ownership, grid requirements, optional dependencies,
method-specific execution modes, and documented legacy presets.

The packages under `clumping_factor.methods`, `diagnostics`, `visualization`,
and `infrastructure` expose typed configuration/service/result/CLI-adapter
boundaries. The existing top-level modules remain compatibility facades and
continue to own the established numerical implementations.

New result documents retain historical keys and add `method_spec`,
`selection_spec`, and `execution_spec`. Existing documents are read without
requiring those fields. New producers supply an explicit method identifier;
heuristic inference exists only for legacy reads, and an unrecognized legacy
document is labeled `legacy.unknown`. Canonical result paths are derived by
`clumping_factor.results.canonical_result_path`.

Typed campaign files describe simulation identity, snapshots, particles,
registered methods, grids, execution settings, and PBS resources. The planner
owns matrix expansion, validation, commands, and output paths. The original
explicit-command campaign format is compatibility-only.

The public command tree is:

```text
clumping clumping compute|alternative
clumping power compute|plot|compare
clumping forest spectra|ionizing|snapshot
clumping temperature compute
clumping diagnostics equations|density-ratio
clumping plot campaign|evolution|igm
clumping results validate|organize
clumping campaign plan|submit
```

## Decision records

ADR-001 keeps scientific kernels in Python services and makes CLI adapters
routing-only. ADR-002 makes campaign manifests and canonical paths
deterministic and declarative. ADR-003 preserves old imports and console
commands until a separately approved removal.
