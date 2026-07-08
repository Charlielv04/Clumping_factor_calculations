# Clumping Factor Calculations

Modular clumping factor tools for TNG gas and dark matter snapshots.

The calculation command writes JSON summaries. Plotting is intentionally separate, so compute runs do not create figures unless requested.

## Repository Structure

```text
Clumping_factor_calculations/
  src/clumping_factor/
    cli.py          Command-line entry points for compute and plot.
    grid.py         Density-grid construction, smoothing, and same-node chunked parallelism.
    loaders.py      Snapshot metadata, full particle loading, and chunked HDF5 readers.
    clumping.py     Threshold sweeps and mask/target clumping-factor calculations.
    raw_gas.py      Raw gas-cell clumping paths that do not build particle grids.
    preprocess.py   Validation, gas-cell radius calculation, and particle indexing helpers.
    plotting.py     Plot generation from JSON result files.
    results.py      Output path handling and JSON serialization.
  tests/            Unit tests for CLI behavior, loading, grids, masks, plotting, and raw gas paths.
  scripts/          PBS helper scripts for same-node cluster runs.
  pyproject.toml    Package metadata and console scripts.
```

The primary workflow uses `clumping-compute` and `clumping-plot`. Additional
installed commands support evolution/campaign plots, equation diagnostics,
forest and ionizing calculations, and the alternative estimator. Run any
command with `--help` for its supported interface. The previous multi-node
partial/shard workflow has been removed; chunked gridded runs now parallelize
on one node through `clumping-compute --load-mode chunked --threads N`.

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

Clumping JSON outputs use:

```text
results/<family>/<simulation>/<particle>/<backend>/snapshot<SNAPSHOT>_grid<GRID>/threads<THREADS>_batch<BATCH>_run<RUN>.json
```

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
