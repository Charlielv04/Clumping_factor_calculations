# Clumping Factor Calculations

Modular clumping factor tools for TNG gas and dark matter snapshots.

The calculation command writes JSON summaries. Plotting is intentionally separate, so compute runs do not create figures unless requested.

## Repository structure

```text
Clumping_factor_calculations/
  campaigns/                    Declarative simulation and method matrices.
  docs/
    architecture.md             Architectural overview and public interfaces.
    decisions/                  Architecture decision records (ADRs).
    guardrails.md               Rules that keep scientific code out of adapters and shell scripts.
  src/clumping_factor/
    methods/
      registry.py                Stable method specifications and legacy presets.
      clumping/                  Clumping and alternative-estimator boundaries.
      power_spectrum/            NumPy, Pylians, comparison, and plotting boundaries.
      forest/                    Forest and radiation workflow boundaries.
      thermodynamics/            Particle and snapshot-temperature boundaries.
    diagnostics/                 Equation and density-ratio diagnostics.
    visualization/               Campaign, evolution, and IGM plotting adapters.
    infrastructure/              Campaign planning, paths, provenance, validation, and PBS rendering.
    forest/                      Established forest/radiation numerical implementations.
    command.py                   Unified public command router.
    results.py                   Normalized result contracts and canonical paths.
    cli.py, grid.py, ...         Established numerical kernels and compatibility entry points.
  tests/                         Numerical, contract, CLI, campaign, and architecture tests.
  scripts/                       Legacy operational helpers retained where still needed.
  results/                       Generated scientific and analysis artifacts.
  pyproject.toml                 Package metadata and compatibility console commands.
```

The organization is **domain-first**. Each domain exposes a configuration
model, service, result model, and thin CLI adapter. Scientific formulas remain
in the established Python implementations; adapters only parse, validate, and
delegate. Shared concerns such as loading, deposition, units, result paths,
provenance, caching, and cluster planning have one owner outside the domain
adapters.

### Where new code belongs

- Add or modify a scientific calculation in its domain service or established
  numerical module, never in `command.py`, a CLI adapter, or a shell script.
- Register every public computational method in
  `clumping_factor.methods.registry` with its stable identifier, supported
  particles, field representation, weighting, selection semantics, grid
  requirements, optional dependencies, execution modes, and command capability.
- Put reusable loading, serialization, provenance, path, cache, or scheduler
  behavior in the shared infrastructure layer.
- Put plots and campaign analysis in `visualization`; put scientific checks and
  equation tests in `diagnostics`.
- Add a compatibility alias only when an existing command or import must remain
  usable during the migration.

## Architecture and public commands

New integrations should use the domain facades and declarative method registry
described in [`docs/architecture.md`](docs/architecture.md). The long-term
interface is the single `clumping` command tree:

```text
clumping clumping compute|alternative
clumping power compute|plot|compare
clumping forest spectra|ionizing|snapshot
clumping temperature compute
clumping diagnostics equations|density-ratio
clumping plot campaign|evolution|igm
clumping results validate|organize
clumping campaign plan|submit
clumping methods catalog
```

Existing console commands and legacy imports remain supported as compatibility
aliases. Existing result documents are read unchanged; newly written result
documents add normalized method, selection, and execution specifications.

Inspect the complete generated method catalog with:

```bash
clumping methods catalog
```

Legacy backend names such as `sphere`, `raw-volume`, and `pylians` are presets.
They expand to stable identifiers such as `clumping.sphere`,
`clumping.raw-volume-weighted`, and `clumping.pylians`. A result producer must
write an explicit registered identifier; unrecognized historical results are
reported as `legacy.unknown` rather than guessed.

The primary workflow uses `clumping-compute` and `clumping-plot`. Additional
installed commands support evolution/campaign plots, equation diagnostics,
forest and ionizing calculations, and the alternative estimator. Run any
command with `--help` for its supported interface. The previous multi-node
partial/shard workflow has been removed; chunked gridded runs now parallelize
on one node through `clumping-compute --load-mode chunked --threads N`.

## Declarative campaigns

New cluster campaigns describe scientific and execution choices in TOML rather
than duplicating them across submission scripts. A campaign declares one or
more simulations and a snapshot × particle × method × grid matrix:

```toml
name = "tng-smoke"
output_root = "results"

[[simulations]]
family = "tng"
name = "tng100-3"
base_path = "./tng100-3/output"

[matrix]
snapshots = [98]
particle_types = ["gas", "dm"]
methods = ["clumping.sphere", "power-spectrum.numpy"]
grids = [256]

[execution]
threads = 4
load_mode = "chunked"

[resources]
cpus = 4
memory = "16gb"
walltime = "01:00:00"
queue = "mini"
```

Plan a deterministic manifest before submitting anything:

```bash
clumping campaign plan campaigns/tng-smoke.toml
clumping campaign submit campaigns/tng-smoke.toml
```

Submission is a dry run unless `--execute` is supplied. The planner validates
method compatibility, expands the matrix, derives canonical output paths, and
renders generic PBS workers. Scheduler resources do not silently change
scientific execution settings. See [`campaigns/README.md`](campaigns/README.md)
for method-specific options and compatibility behavior.

## Install

```bash
python -m pip install -e ".[test]"
```

For development checks, install `.[dev]`; this includes the test, lint, type,
and coverage tools used by CI.

## Reproducibility and result artifacts

New compute summaries use result schema version 2. They record the Git revision
and dirty state, Python and scientific-library versions, input snapshot file
signatures, execution parameters, unit conventions, and estimator definition.
Schema version 1 results remain readable.

Result files are written atomically, so an interrupted job cannot replace a
previous valid result with partial JSON. The generated `results/` tree is not
tracked by Git; retain production artifacts in project storage and keep only
small, curated regression fixtures under `tests/`.

On clusters with old system compilers, install the scientific stack from wheels first:

```bash
python -m pip install --only-binary=:all: -r requirements-core.txt
python -m pip install illustris-python
python -m pip install -e .
```

If Pylians tries to rebuild NumPy inside build isolation, install it after Cython with build isolation disabled:

```bash
python -m pip install "Cython<3"
python -m pip install --no-build-isolation Pylians
```

Pylians is intentionally not installed by `environment.yml`: its native extensions
assume Unix linker and OpenMP flags and do not build with MSVC on Windows. The
Windows environment can still analyze result JSON files and use the `sphere` and
`cube` backends. Install Pylians separately on the Linux compute cluster when the
`pylians` backend is needed.

## Compute

```bash
clumping-compute \
  --base-path ./tng100-3/output \
  --simulation-name tng100-3 \
  --snapshot 98 \
  --particle-type gas \
  --backend sphere \
  --radius-mode sphere \
  --grid-size 256 \
  --radius-bins 10 \
  --radius-bin-batch-size 2 \
  --load-mode auto
```

Backends:

- `sphere`: SciPy spherical tophat smoothing
- `cube`: SciPy cube tophat smoothing
- `pylians`: optional Pylians mass assignment and smoothing
- `raw`: raw gas-cell density calculation matching the first legacy gas script; only valid with `--particle-type gas`
- `raw-volume`: raw gas-cell density calculation weighted by each gas cell volume; only valid with `--particle-type gas`
- `raw-transmission`: native gas-cell, volume-weighted density clumping with a grid-derived `exp(-tau_eff)` weight; only valid with `--particle-type gas`
- `voronoi-transmission`: grid-free native gas-cell density clumping with a periodic Voronoi-neighbor gradient and `exp(-tau_eff)` weight; only valid with `--particle-type gas`

`raw-transmission` reads `HI_Fraction`, `HII_Fraction`, and hydrogen abundance from `GFM_Metals[:,0]`. It verifies `HI_Fraction + HII_Fraction ~= 1`, builds an auxiliary volume-weighted neutral-hydrogen grid, and returns one scalar rather than an overdensity-threshold sweep:

```bash
clumping-compute \
  --base-path ../Thesan-1/output \
  --simulation-name Thesan-1 \
  --snapshot 81 \
  --particle-type gas \
  --backend raw-transmission \
  --grid-size 512 \
  --mas CIC \
  --load-mode chunked \
  --sigma-bar-ion-cm2 <AREPO-RT-group-average> \
  --sigma-bar-ion-source "THESAN AREPO-RT first ionizing group"
```

The cross-section has no implicit default. For PBS submissions, provide the same values through `SIGMA_BAR_ION_CM2` and `SIGMA_BAR_ION_SOURCE`.

For THESAN snapshots containing `PhotonDensity`, use `--sigma-bar-ion-mode thesan-photon-groups` to derive the gray cross-section from all three groups. The code uses the existing `THESAN_SIGMA_C_CM3_S` coefficients from the Gamma_HI calculation, converts `c*sigma` to sigma, and volume-weights the three photon-number densities. The derived value and group weights are written to the result diagnostics.

The grid-free native-cell variant avoids CIC/TSC mass assignment:

```bash
clumping-compute \
  --base-path ../Thesan-1/output \
  --simulation-name Thesan-1 \
  --snapshot 81 \
  --particle-type gas \
  --backend voronoi-transmission \
  --load-mode chunked \
  --voronoi-neighbors 32 \
  --sigma-bar-ion-cm2 <AREPO-RT-group-average> \
  --sigma-bar-ion-source "THESAN AREPO-RT first ionizing group"
```

The current snapshots provide cell centers and native cell volumes, but not shared Voronoi-face geometry. The implementation therefore reconstructs a periodic nearest-cell stencil with `cKDTree` and obtains a weighted local least-squares gradient from neighboring native-cell values. All valid cells and the neighbor index table must be retained in memory for this search; `--memory-limit` will reject a run whose estimated working set is too large. `--voronoi-gradient-batch-size` controls gradient accumulation memory, while `--threads` controls the neighbor query workers.

For gridded gas calculations, `--radius-mode sphere` treats each gas cell volume as a sphere and `--radius-mode cube` uses the cube root of the cell volume. The default is `sphere`.

All gridded backends support `--mas CIC` (cloud-in-cell, the default) and `--mas TSC` (triangular-shaped cloud). CIC distributes particle mass over 8 neighboring cells; TSC uses a wider, smoothly weighted 27-cell stencil. The selected assignment is applied before the sphere, cube, or Pylians smoothing step.

```bash
clumping-compute \
  --base-path ./tng100-3/output \
  --simulation-name tng100-3 \
  --snapshot 98 \
  --particle-type gas \
  --backend sphere \
  --mas TSC \
  --grid-size 256
```

For PBS submissions, set `MAS=TSC`. Non-default mass assignment is included in job names and output filenames.

For Thesan, TNG, and AIDA-TNG production runs, PBS helpers write to the canonical `results/<family>/...` tree by default. Direct `clumping-compute` calls keep the legacy default path unless `--output` is supplied, so use an explicit canonical `--output` path for ad hoc production runs.

For large snapshots, `--load-mode auto` estimates whether a full particle load is safe and switches to chunked HDF5 reads when needed. Use `--load-mode chunked` to force streaming, `--chunk-size` to control particle/cell reads per chunk, and `--max-full-load-gb` to tune the automatic cutoff. Add `--verbose` for progress logs; `--progress-interval 10` reports every 10 chunks instead of the default 25.

For gridded chunked runs, `--threads` controls same-node parallel grid building. Snapshot files are weighted by their particle counts and assigned to workers to balance the expected load. The summary pass uses the same partitions. Each worker builds private grid accumulators, writes its completed grid under `$TMPDIR`, and the parent memory-maps and reduces each result as it arrives. Worker files and their temporary directory are removed after success or failure.

`--radius-bin-batch-size` controls how many radius-bin grids each worker fills during one particle-file pass. The default is `1` for memory safety. Larger values reduce repeated particle reads. Each worker uses approximately `batch size + 2` full grids at peak: one final grid, the active batch grids, and one smoothing output grid. Use `radius_bin_stream_passes`, `grids_per_worker`, and `estimated_total_worker_grid_bytes` from the result diagnostics when selecting a value.

Use `--memory-limit 24gb` to enforce the job's grid-memory budget and `--memory-safety-fraction` to reserve space for Python, HDF5, kernels, and other allocations. The build preserves the requested batch size while reducing workers, then reduces the batch size if one worker still does not fit. It fails before reading particle arrays when even one worker with batch size 1 is too large. PBS submissions pass the selected `MEM_<grid>` value automatically. `--temp-dir` overrides `$TMPDIR` when worker grids should use another node-local filesystem.

Snapshot summaries are cached under `results/.cache/summaries/` by default. The cache key includes the canonical snapshot path, snapshot number, particle type, gas-radius mode, and each snapshot file's size and modification time. Use `--summary-cache off` for a cold benchmark or `--summary-cache refresh` to force a rebuild. Cache writes are atomic, and concurrent jobs wait on a directory lock instead of rescanning the same snapshot.

`--work-partition auto` keeps whole-file scheduling when its predicted worker imbalance is at most 10%. Otherwise it splits each large snapshot file into at most `--max-file-readers` ranges, default `2`, and balances those ranges independently. Use `files` and `ranges` to force either behavior during comparisons.

Conservative starting points are:

- grid 128: batch size `5` or `10`;
- grid 256: batch size `2` or `5`;
- grid 512: start with batch size `1`; use `2` only with enough memory for all worker-private grids plus parent-process overhead.

Benchmark timings are written into the result JSON under `timings`. For chunked gridded runs, the most useful fields are:

- `chunk_summary`: initial pass used to find valid counts and radius bins.
- `metadata_inspection`: per-file particle-count inspection used for worker balancing.
- `parallel_chunk_summary`: wall time for the parallel summary pass.
- `parallel_grid_build`: wall time spent inside the local worker pool.
- `reduce_worker_grids`: time spent summing worker-private grids in the parent process.
- `density_conversion`: final mass-grid to density-grid conversion.
- `worker_stream_total`: summed worker time spent reading/streaming chunks.
- `worker_deposit_total` or `worker_assignment_total`: summed worker time spent depositing particles into grids.
- `worker_smooth_total`: summed worker smoothing time.
- `worker_grid_write_total`: summed time writing worker grids to temporary storage.
- `worker_io_total`: measured HDF5 dataset read time.
- `worker_preprocess_total`: validation and particle-radius calculation time.
- `temporary_cleanup`: temporary-directory setup and cleanup overhead.
- `worker_total_max`: slowest worker runtime, usually the best indicator of parallel wall-clock balance.
- `build_density_grid`: total density-grid construction time.

Thesan-1 snapshot 81 can be run with:

```bash
clumping-compute \
  --base-path ../Thesan-1 \
  --simulation-name Thesan-1 \
  --snapshot 81 \
  --particle-type gas \
  --backend sphere \
  --grid-size 256 \
  --load-mode chunked \
  --threads 8 \
  --memory-limit 32gb \
  --summary-cache auto \
  --work-partition auto \
  --verbose
```

Verify that an output was produced by the current pipeline:

```bash
python scripts/validate_chunked_result.py results/thesan/Thesan-2/gas/sphere/snapshot080_grid256/*.json
```

## AREPO/THESAN DM Power Spectrum Post-processing

The standard AREPO power-spectrum workflow reads the complete snapshot and
allocates the full particle array `P`. This is not practical for the full
THESAN snapshot. For snapshot 081, a one-rank test attempted to allocate
approximately 134 GB for hundreds of millions of particles on the rank, which
exceeded the available memory.

To make the calculation feasible, enable `POWERSPECTRUM_IN_POSTPROCESSING`.
This selects the streaming implementation in `pm_periodic.c`: snapshot files
are read in rounds, and the selected particle type is processed in chunks of
up to one million particles. For the dark-matter spectrum, use particle mask
`2`, which selects Type 1 dark matter. In the successful run, 30 MPI ranks
processed 1,200 snapshot files, with 40 rounds per rank. The density field was
accumulated directly on the PM mesh, keeping memory use modest on each rank.

Several changes were required for this post-processing route:

1. `calculate_power_spectra()` was changed to honor the requested particle
   mask on its first pass. It now enables only the particle types selected by
   the mask instead of constructing an all-particle spectrum before computing
   individual types. This makes the calculation genuinely DM-only.
2. The `MTNG` compile flag caused `long_range_init()` to return before
   `pm_init_periodic()` was called. Consequently, the FFT decomposition object
   `myplan` and `maxfftsize` were uninitialized when the streaming routine ran.
   The post-processing branch in `main.c` now calls `pm_init_periodic()`
   immediately before `calculate_power_spectra()`.
3. Cleanup was reordered to respect AREPO's stack-like `mymalloc()` allocator.
   The streaming routine allocates `Sndpm_count`, `Sndpm_offset`,
   `Rcvpm_count`, `Rcvpm_offset`, and `rhogrid`; therefore the post-processing
   cleanup must free them in reverse allocation order: `rhogrid`,
   `Rcvpm_offset`, `Rcvpm_count`, `Sndpm_offset`, and `Sndpm_count`. The
   original order caused `Wrong call of myfree(): not the last allocated block`.

With these changes, the full DM calculation completed successfully. It
processed all 1,200 files and 9,261,000,000 dark-matter particles. The power-
spectrum calculation itself took approximately 365 seconds. The output file
was:

```text
powerspec_081.txt
```

The file contains three appended spectra: the normal PM spectrum, a folded
spectrum, and a double-folded spectrum. These are produced because
`pmforce_do_powerspec()` calls `pmforce_periodic(1, typeflag)`,
`pmforce_periodic(2, typeflag)`, and `pmforce_periodic(3, typeflag)`.

Each spectrum block begins with four header lines containing, in order:
`a`, `BINS_PS`, total particle mass, and total particle number. The header is
followed by `BINS_PS = 2000` rows. The five columns are written explicitly as
`Kbin[i]`, `DeltaUncorrected[i]`, `PowerUncorrected[i]`, `CountModes[i]`, and
`ShotLimit[i]`. They represent, respectively, wavenumber `k`, dimensionless
power `Delta^2(k)`, uncorrected power, the number of Fourier modes in the bin,
and the shot-noise limit.

The wavenumbers are in inverse internal length units. Since the simulation
uses kpc/h for length, multiply the first column by 1,000 to obtain
`h Mpc^-1`. The first populated bin is approximately `9.73e-05` internal
units, or `0.0973 h/Mpc`, consistent with the fundamental mode of the
64.69 cMpc/h simulation box.

The local JSON result can be compared directly with an AREPO text output using
the dedicated plotting command:

```bash
clumping-power-spectrum-compare \
  --arepo results/thesan/Thesan-1/powerspec_081.txt \
  --local results/thesan/Thesan-1/dm/power-spectrum/mas-only_both/snapshot081_grid256/threads8_run001.json \
  --arepo-block 0 \
  --local-engine numpy \
  --output results/analysis/power-spectra/thesan/Thesan-1/snapshot081/dm/arepo_vs_local.png
```

The command produces an upper panel with the two spectra and a lower panel
showing the local-to-AREPO ratio. AREPO blocks `0`, `1`, and `2` select the
normal, folded, and double-folded spectra, respectively. The default
`--k-unit-factor 1000` converts the stored inverse kpc/h wavenumbers to
`h Mpc^-1`; set it to `1` for plots in internal units. Use `--field power` to
compare the uncorrected power instead of `Delta^2(k)`.

To compare AREPO against both local density-grid constructions and both local
estimators in one figure, pass both JSON files and select `--local-engine both`:

```bash
clumping-power-spectrum-compare \
  --arepo results/thesan/Thesan-1/powerspec_081.txt \
  --local results/thesan/Thesan-1/dm/power-spectrum/mas-only_both/snapshot081_grid256/threads8_run001.json \
         results/thesan/Thesan-1/dm/power-spectrum/smoothed-pylians_both/snapshot081_grid256/threads8_run001.json \
  --arepo-block all \
  --local-engine both \
  --output results/analysis/power-spectra/thesan/Thesan-1/snapshot081/dm/arepo_vs_all_local.png
```

This compares AREPO with NumPy and Pylians on the mass-assignment-only grid,
and with NumPy and Pylians on the Pylians-smoothed grid. Use repeated
`--local-label` options to provide custom legend labels.

One remaining robustness issue affects single-rank runs only: an `int`
particle counter overflows for 9.26 billion dark-matter particles and reports
the low 32-bit remainder. The 30-rank calculation avoids this because each
rank counts only a fraction of the files before the global reduction. The
local particle counters should nevertheless be changed to a 64-bit type
before treating the post-processing mode as fully robust for arbitrary MPI
layouts.

## Separate IGM Mask And Target Fields

By default, the same density field defines the threshold mask and the clumping factor. To define the IGM mask from one field but measure clumping on another, use the `--mask-*` and `--target-*` options.

Example: select IGM cells using the total matter field, then measure gas clumping there:

```bash
clumping-compute \
  --base-path ../tng100-3/output \
  --simulation-name tng100-3 \
  --snapshot 98 \
  --particle-type gas \
  --backend sphere \
  --target-particle-type gas \
  --target-backend sphere \
  --mask-particle-type both \
  --mask-backend sphere \
  --grid-size 256 \
  --output results/tng/tng100-3/gas/sphere-masked-total/snapshot098_grid256/threads1_batch1_run001.json
```

Example: select IGM cells from the DM field, then measure gas clumping:

```bash
clumping-compute \
  --base-path ../tng100-3/output \
  --simulation-name tng100-3 \
  --snapshot 98 \
  --particle-type gas \
  --backend sphere \
  --target-particle-type gas \
  --target-backend sphere \
  --mask-particle-type dm \
  --mask-backend sphere \
  --grid-size 256 \
  --output results/tng/tng100-3/gas/sphere-masked-dm/snapshot098_grid256/threads1_batch1_run001.json
```

## Plot

```bash
clumping-plot \
  results/tng/tng100-3/gas/sphere/snapshot098_grid256/threads1_batch1_run001.json \
  --output results/analysis/clumping/tng/tng100-3/snapshot098/gas/sphere/gas_sphere.png
```

Multiple JSON files can be plotted together:

```bash
clumping-plot \
  results/tng/tng100-3/gas/*/snapshot098_grid256/*.json \
  --output results/analysis/clumping/tng/tng100-3/snapshot098/gas/combined/comparison_grid256.png
```

Plot the number of cells included in the IGM mask as a function of overdensity threshold:

```bash
clumping-plot \
  results/tng/tng100-3/gas/sphere/snapshot098_grid256/threads1_batch1_run001.json \
  results/tng/tng100-3/gas/cube/snapshot098_grid256/threads1_batch1_run001.json \
  results/tng/tng100-3/gas/pylians/snapshot098_grid256/threads1_batch1_run001.json \
  --quantity cell-count \
  --output results/analysis/cell-count/tng/tng100-3/snapshot098/gas/combined/gas_backend_igm_cell_counts.png
```

## Redshift Evolution

New result files include the snapshot scale factor and redshift from the HDF5 header. Submit one bounded PBS array task per snapshot while retaining same-node parallelism inside each task:

```bash
SNAPSHOTS="40 50 60 70 80" \
BASE_PATH=../Thesan-2/output SIMULATION_NAME=Thesan-2 \
PARTICLE=gas BACKEND=sphere GRID=256 \
NCPUS=8 THREADS=8 MAX_CONCURRENT=8 MEM=32gb \
bash scripts/submit_evolution_jobs.sh
```

Each output filename includes its snapshot number. After the jobs finish, combine them at one or more fixed overdensity thresholds:

```bash
clumping-evolution-plot \
  results/thesan/Thesan-2/gas/sphere/snapshot*_grid256/threads8_batch2_run001.json \
  --threshold 10 --threshold 20 \
  --output results/analysis/clumping/thesan/Thesan-2/combined-snapshots/gas/sphere/gas_sphere_clumping_vs_redshift_grid256.png
```

The evolution plot command verifies that all inputs use the same particle, mask, backend, grid, and threshold configuration before interpolating the requested threshold values.

To plot the complete overdensity sweep for every dark-matter model, with one curve per snapshot, pass the result directory (or individual JSON files):

```bash
clumping-model-evolution-plot \
  results/aida-tng/L35n1080_CDM/dm/raw-volume \
  results/aida-tng/L35n1080_SIDM1/dm/raw-volume \
  results/aida-tng/L35n1080_vSIDM/dm/raw-volume \
  results/aida-tng/L35n1080_WDM3/dm/raw-volume
```

This writes one plot per complete model under `results/analysis/clumping/aida-tng/L35n1080_CDM/combined-snapshots/dm/raw-volume/`. Use `--relative-to-cdm` to write the corresponding same-snapshot proportional differences under the `relative-to-cdm` subdirectory. Models missing any snapshot in the supplied set are skipped.

### AIDA-TNG plot catalog

For a complete, question-first plot tree, use the AIDA-TNG catalog command:

```bash
clumping-aida-tng-plots --archive-existing
```

It discovers clumping, ionized-sweep, and power-spectrum JSON results under
`results/aida-tng`, delegates rendering to the existing plotters, and writes a
CSV catalog to `results/analysis/clumping/aida-tng/aida-tng-plots.csv`. Use
`--dry-run` to inspect the planned outputs without moving or creating plots.

Canonical AIDA-TNG analysis outputs use this layout:

```text
results/analysis/
  clumping/
    aida-tng/
      evolution/<simulation>/combined/<particle>/<method>/
      model-comparison/<box>/<snapshot>/<particle>/<method>/
      model-comparison/<box>/combined-snapshots/<particle>/<method>/relative-to-cdm/
      method-comparison/<simulation>/<snapshot>/<particle>/<grid>/
      grid-comparison/<simulation>/combined/<particle>/<method>/
      ionization/<simulation>/<snapshot>/<particle>/ionized-sweep/
  power-spectra/aida-tng/<simulation>/<snapshot>/<particle>/<method>/
  performance/aida-tng/<simulation>/combined/<particle>/<backend>/
  manifests/aida-tng-plots.csv
```

Existing AIDA-TNG PNGs are archived under `results/analysis/archive/aida-tng/`
when `--archive-existing` is supplied. JSON result files are never modified.

Relative-to-CDM snapshot-evolution plots are written one per non-CDM model
under `model-comparison/<box>/combined-snapshots/.../relative-to-cdm/`; each
plot contains all snapshots shared by the model set.

## Results Organization

The `results/` tree is organized by data product first, then by simulation family. Do not add campaign names to canonical paths; campaign/source folder names belong in manifests and metadata.

```text
results/
  thesan/
  tng/
  aida-tng/
  forest/
  analysis/
```

Legacy direct commands commonly write backend-named clumping paths:

```text
results/<family>/<simulation>/<particle>/<backend>/snapshot<SNAPSHOT>_grid<GRID>/threads<THREADS>_batch<BATCH>_run<RUN>.json
```

Declarative campaigns instead derive this component from the stable registered
method identifier. In both cases, path construction is owned by
`clumping_factor.results`; submission scripts must not construct result paths
independently.

Examples:

```text
results/thesan/Thesan-1/dm/pylians/snapshot081_grid512/threads16_batch10_run001.json
results/tng/tng100-3/gas/cube/snapshot098_grid256/threads4_batch10_run002.json
results/tng/tng100-3/gas/raw-volume/snapshot098_nogrid/threads1_batch1_run001.json
results/aida-tng/L35n1080_CDM/gas/sphere/snapshot017_grid256/threads8_batch2_run001.json
```

Forest spectra outputs use:

```text
results/forest/<family>/<simulation>/snapshot<SNAPSHOT>/<line>/<los-stem>_<line>.hdf5
```

Example:

```text
results/forest/thesan/Thesan-2/snapshot080/lya/rays_080_lya.hdf5
```

Analysis products use:

```text
results/analysis/<plot-type>/<family>/<simulation>/<snapshot>/<particle>/<backend>/<file>
```

`<plot-type>` is one of:

- `performance`
- `clumping`
- `cell-count`
- `misc`

Examples:

```text
results/analysis/performance/thesan/Thesan-1/snapshot081/dm/pylians/performance_grid512.png
results/analysis/clumping/tng/tng100-3/snapshot098/gas/raw/gas_raw_vs_grid_256.png
results/analysis/cell-count/thesan/Thesan-2/combined-snapshots/dm/pylians/Thesan-2_dm_pylians_grid512_cell_counts.png
```

Manifests and migration reports live under:

```text
results/analysis/manifests/
```

The current organizer scripts write:

```text
results/analysis/manifests/thesan_results_manifest.csv
results/analysis/manifests/thesan_duplicate_report.csv
results/analysis/manifests/thesan_move_plan.csv
results/analysis/manifests/tng_results_manifest.csv
results/analysis/manifests/tng_duplicate_report.csv
results/analysis/manifests/tng_move_plan.csv
```

PBS logs stay outside `results/`:

```text
logs/<source-campaign>/
```

Cache files are generated data and are not part of the canonical scientific result tree:

```text
results/.cache/summaries/
```

Future PBS clumping runs default to canonical layout for Thesan, TNG, and recognized AIDA-TNG simulations when `RESULTS_LAYOUT=auto`. Use `RESULTS_LAYOUT=legacy` only for compatibility checks. Direct command-line runs should either pass an explicit canonical `--output` path or be followed by the organizer.

Forest runs default to the canonical `results/forest/...` layout unless `--output` is supplied:

```bash
clumping-forest \
  --los-dir ../Thesan-2/los \
  --simulation-name Thesan-2 \
  --snapshots 54 80 \
  --output-dir results/forest
```

The forest command can produce Lyman-alpha spectra and MFP in one run from the
same COLT ray file:

```bash
clumping-forest \
  --los-file /path/to/rays_080.hdf5 \
  --simulation-name Thesan-1 \
  --compute-mfp --mfp-starts-per-ray 100 --mfp-cross-check
```

The outputs are written under the same snapshot directory as `lya/` and
`mfp912/`. This command consumes an existing COLT ray file; constructing that
file from a raw snapshot remains a separate COLT operation.

For a complete snapshot workflow, explicitly select the desired products:

```bash
clumping-snapshot \
  --base-path /lustre/work/carlos.lopez/Thesan-1/output \
  --snapshot 80 --simulation-name Thesan-1 \
  --los-file /lustre/work/carlos.lopez/Thesan-1/postprocessing/los/rays_080.hdf5 \
  --products lya mfp gamma equations \
  --temperature-file /path/to/Tigm_Thesan1.dat \
  --threshold-min -1 --threshold-max 25 --threshold-count 200 \
  --photon-group-tests 0 1 2 0+1 1+2 0+1+2 \
  --ionized-sweep --ionized-cut-min 0.9 --ionized-cut-max 0.9999 \
  --ionized-cut-count 200 --ionized-density-thresholds 1 5 10 15 20 25 \
  --mfp-cross-check --gamma-cross-check --verbose
```

The command writes `manifest.json`, `lya/`, `mfp912/`, `gamma_hi/`, and
`equations/` under the canonical snapshot directory. Successful products are
reused when their manifest fingerprint and outputs still match. Use
`--refresh-products` to recompute selected products. Independent products keep
running after a failure, but the command exits nonzero and records the failure
in the manifest. Gamma reads HDF5 data in chunks controlled by
`--gamma-chunk-size` (default 1,000,000 cells).

The equivalent Python API is:

```python
from clumping_factor.forest import SnapshotWorkflowConfig, run_snapshot_workflow

result = run_snapshot_workflow(SnapshotWorkflowConfig(
    base_path="/path/to/output", snapshot=80, simulation_name="Thesan-1",
    los_file="/path/to/rays_080.hdf5", products=["lya", "mfp", "gamma"],
))
```

Ionizing observables use the same THESAN/COLT ray format as the forest pipeline.
The MFP command samples periodic starting positions, measures the proper distance
to `tau_912 = 1`, continuing through periodic ray copies when one traversal is
too transparent, and can independently re-evaluate the supplied scalar equation:

```bash
clumping-ionizing mfp \
  --los-file ../Thesan-1/rays_080.hdf5 \
  --starts-per-ray 100 \
  --seed 0 \
  --cross-check \
  --output results/forest/thesan/Thesan-1/snapshot080/mfp_912.json
```

`--seed` makes the random origins reproducible. The result is reported in proper
Mpc/h, matching `get_mfp_from_sim.py`. Transparent rays continue through
periodic copies until unit optical depth is reached.

The Gamma command streams any explicitly listed snapshot pieces and applies the
volume-weighted, `HI_Fraction < 0.5` calculation from `get_gamma_from_sim.py`:

```bash
clumping-ionizing gamma \
  --base-path /path/to/output --snapshot 80 \
  --cross-check \
  --verbose --progress-interval 10 \
  --output results/forest/thesan/Thesan-1/snapshot080/gamma_hi.json
```

With `--cross-check`, both commands evaluate an independent scalar form of the
supplied scripts and store `cross_check.passed` plus the absolute numerical
difference in the output JSON. The regression suite also compares identical
rays, starting indices, snapshot cells, masks, unit conversions, and cached
table values; this mirrors the legacy-vs-new checks used for Lyman-alpha.

The repository-level `simloader.zip` is the upstream reader used by the supplied
MFP script. It remains available for exact legacy runs; the integrated command
uses `clumping_factor.forest.los_loader`, whose conversion behavior is regression
tested against that reader and is already shared with the Lyman-alpha pipeline.

Eq. 5--13 diagnostics can calculate both ionizing inputs when their tables are
missing. Gamma_HI is read from the snapshot itself; MFP additionally requires the
matching COLT ray file. Both generated tables are cached in `snapdir_NNN` and
reused on later runs:

```bash
clumping-equation-tests \
  --base-path /path/to/output --snapshot 80 \
  --compute-missing-ionizing --mfp-los-file /path/to/rays_080.hdf5 \
  --sigma-hi-cm2 6.3e-18 --output equations_080.json
```

For the Eq. 13-only command, use `--compute-missing-mfp` with the same
`--mfp-los-file` option.

Generated tables have adjacent `.meta.json` provenance sidecars containing file
signatures, snapshot/redshift, algorithm version, constants, and calculation
settings. A mismatch automatically regenerates the cache; use
`--refresh-ionizing-cache` to force this. Provenance-free MFP/Gamma_HI tables are
rejected by default. For a deliberate historical comparison only, pass
`--allow-legacy-ionizing-table` (or set `ALLOW_LEGACY_IONIZING_TABLE=1` in the PBS
submission scripts).

Existing legacy clumping folders can be audited without moving files:

```bash
python scripts/organize_thesan_results.py
python scripts/organize_tng_results.py
```

To copy files into the canonical layout after reviewing the move plan:

```bash
python scripts/organize_thesan_results.py --apply
python scripts/organize_tng_results.py --apply
```

Use `--move` with `--apply` only when you intentionally want to relocate originals. If a canonical destination exists and is byte-identical, the source is removed during move mode. If a JSON destination exists with different content, the organizer refuses to overwrite it; for plot collisions it preserves the extra file with a source/hash suffix.

Current repository organization notes:

- `Legacy files/` contains old standalone scripts kept for reference. A cleaner future step would be renaming it to `legacy/` and adding a short README that states these scripts are not production entry points.
- Historical comparison plots are stored under `results/analysis/misc/tng/tng100-3/snapshot098/<particle>/combined/`.
- `reports/` contains manuscript/report artifacts and `tools/` contains local binary tooling. They are separate from scientific run outputs and should not be mixed into `results/`.
