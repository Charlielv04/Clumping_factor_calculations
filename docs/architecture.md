# Architecture

The suite is domain-first. Numerical ownership is under `methods`, diagnostics
under `diagnostics`, plots under `visualization`, and shared mechanics under
`infrastructure`. The package root is routing only.

Each executable domain exposes a configuration dataclass, a computation
service, a result boundary, a thin CLI adapter, and parity tests. CLI adapters
may parse arguments; services accept typed configuration objects only.

`methods.registry` is the sole owner of stable method identifiers, presets,
supported particle types, field and weighting semantics, selection behavior,
requirements, dependencies, execution modes, and campaign command capability.

Result identity is metadata-driven. Schema-2 readers require normalized method,
selection, execution, provenance, and simulation objects. Paths use hashes of
canonical scientific and algorithmic execution specifications. Scheduler-only
metadata never changes artifact identity.

Campaign TOML is the only source for cluster task matrices. The planner validates
registry capabilities, derives result paths, and renders generic workers. Shell
scripts do not own scientific or path defaults.

The clean-break cutover intentionally removed old imports, console aliases,
schema-1 reading, executable comparison copies, organizers, method-specific
submission scripts, and legacy source. Git history is the archive.
