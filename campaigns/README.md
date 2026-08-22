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
For a single scheduler submission that keeps the campaign tasks independent,
use `submit-array`: it renders one PBS/OpenPBS job array, indexed over the
deterministic task manifest.  Pass `--array-syntax torque` on Torque systems.

```text
clumping campaign submit-array campaigns/tng-smoke.toml
clumping campaign submit-array campaigns/tng-smoke.toml --execute
```

Every executable method is capability-checked by the planner. Combined power
spectra use `power-spectrum.combined`; alternative raw/grid methods may provide their
method-specific inputs under `[method_options."alternative.raw-volume"]` or
`[method_options."alternative.grid-masked"]` (for example, `mfp_file`).

Explicit `[[tasks]]` command lists are intentionally rejected. All campaigns
use the typed matrix format demonstrated by `tng-smoke.toml`.

`matrix.snapshots` may also be the string `"available"`. In that mode the
planner discovers `snapdir_###/snap_###.hdf5` directories and single-file
`snap_###.hdf5` snapshots independently under each simulation's `base_path`.
The folded AIDA-TNG power-spectrum campaign demonstrates this mode:

```text
clumping campaign plan campaigns/aida-tng-power-spectrum-folded.toml
clumping campaign submit campaigns/aida-tng-power-spectrum-folded.toml --execute
```

It runs the combined NumPy/Pylians estimator for DM on 256³, 512³, and 1024³
meshes with fold factors 1, 2, and 4. The first command is the reviewable
dry-run; use `--manifest` to choose its JSON manifest path.

Generated workers explicitly activate the `clumping-factor` conda environment,
with a fallback to `~/.conda/envs/clumping-factor/bin`, and include `#PBS -V` as
an additional environment safeguard. They also change to `$PBS_O_WORKDIR` so
relative canonical result paths resolve from the submission directory.
Scheduler stdout and stderr are written to `logs/pbs/` with the scheduler's
native job-name and job-ID suffixes (for example, `campaign-name.o589737`).
