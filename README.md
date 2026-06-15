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

Only `clumping-compute` and `clumping-plot` are user-facing console commands. The previous multi-node partial/shard workflow has been removed; chunked gridded runs now parallelize on one node through `clumping-compute --load-mode chunked --threads N`.

## Install

```bash
python -m pip install -e ".[test]"
```

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

Outputs are saved under `results/<simulation>/` unless `--output` is supplied. The simulation name is inferred from `--base-path` by default, or can be set explicitly with `--simulation-name`.

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
python scripts/validate_chunked_result.py results/Thesan-2-scaling-new/*.json
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
  --output results/tng100-3/gas_clumping_masked_by_total_sphere_256.json
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
  --output results/tng100-3/gas_clumping_masked_by_dm_sphere_256.json
```

## Plot

```bash
clumping-plot results/tng100-3/gas_sphere_snapshot098_grid256.json --output results/tng100-3/gas_sphere.png
```

Multiple JSON files can be plotted together:

```bash
clumping-plot results/*/*.json --output results/comparison.png
```

Plot the number of cells included in the IGM mask as a function of overdensity threshold:

```bash
clumping-plot \
  results/tng100-3/gas_sphere_256.json \
  results/tng100-3/gas_cube_256.json \
  results/tng100-3/gas_pylians_256.json \
  --quantity cell-count \
  --output results/tng100-3/gas_backend_igm_cell_counts.png
```
