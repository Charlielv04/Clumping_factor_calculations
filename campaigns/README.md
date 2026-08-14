# Declarative campaigns

Campaign TOML files are the source of truth for task matrices. New campaigns
declare one or more `[[simulations]]` tables plus `[matrix]`, `[execution]`, and
`[resources]` tables;
the planner validates method presets against the method registry, expands the
Cartesian product, and derives canonical output paths. Commands and PBS
scripts are generated artifacts, not campaign inputs. Generate a sorted,
reviewable manifest with:

```text
clumping campaign plan campaigns/tng-smoke.toml
```

Use `clumping campaign submit ...` to render generic PBS workers. Submission
is dry-run by default; `--execute` is the explicit opt-in for `qsub`.

Every executable method is capability-checked by the planner. Combined power
spectra use `power-spectrum.combined`; alternative raw/grid methods may provide their
method-specific inputs under `[method_options."alternative.raw-volume"]` or
`[method_options."alternative.grid-masked"]` (for example, `mfp_file`).

Explicit `[[tasks]]` command lists are intentionally rejected. All campaigns
use the typed matrix format demonstrated by `tng-smoke.toml`.
