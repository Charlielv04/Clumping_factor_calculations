# PBS Runs On idark

The IPMU idark documentation uses PBS resource requests of the form `select=1:ncpus=<n>:mem=<size>`. The submission helpers use `QUEUE=auto` by default:

- one-CPU jobs are submitted to `tiny`;
- jobs requesting more than one CPU are submitted to `mini`;
- explicitly setting `QUEUE=tiny` with `NCPUS>1` is rejected before submission;
- set `QUEUE=<name>` to request a specific larger queue when required locally.

Job names and result filenames include the grid size and parallel size, so benchmark runs with different resources do not overwrite each other.

Submit one smoke-test job:

```bash
GRIDS=256 PARTICLES=gas BACKENDS=cube \
WALLTIME_256=01:00:00 MEM_256=4gb NCPUS=1 THREADS=1 \
bash scripts/submit_clumping_jobs.sh
```

Submit all 256-grid jobs:

```bash
GRIDS=256 MEM_256=4gb WALLTIME_256=04:00:00 NCPUS=2 \
bash scripts/submit_clumping_jobs.sh
```

Submit a TNG100-3 comparison across methods, grid sizes, and parallel sizes:

```bash
for ncpus in 1 2 4 8; do
  BASE_PATH=../tng100-3/output \
  SIMULATION_NAME=tng100-3-benchmark \
  SNAPSHOT=98 \
  PARTICLES=gas \
  BACKENDS="sphere cube pylians" \
  GRIDS="128 256 512" \
  LOAD_MODE=chunked \
  NCPUS="${ncpus}" \
  THREADS="${ncpus}" \
  RADIUS_BIN_BATCH_SIZE=2 \
  MEM_128=8gb MEM_256=16gb MEM_512=32gb \
  WALLTIME_128=01:00:00 WALLTIME_256=02:00:00 WALLTIME_512=08:00:00 \
  bash scripts/submit_clumping_jobs.sh
done
```

With `QUEUE=auto`, the `ncpus=1` jobs use `tiny` and parallel jobs use `mini`. Output names contain `_grid<grid>_threads<threads>_batch<batch-size>`, while PBS job names also include `_b<batch-size>`.

The selected `MEM_<grid>` value is also passed to `clumping-compute --memory-limit`. Chunked grid builds automatically reduce the effective worker count, then the radius-bin batch size, to stay within that allocation with a 10% safety reserve. Set `MEMORY_SAFETY_FRACTION` in the job environment to change the reserve. Worker result grids use node-local `$TMPDIR` and are reduced and deleted incrementally.

`SUBMIT_MODE=throttled` is the default. Jobs submitted by one invocation are chained with PBS `afterok` dependencies so they do not simultaneously stream the same snapshot. Set `SUBMIT_MODE=parallel` only for intentional filesystem-contention tests. `REPETITIONS=3` submits three uniquely named runs; result filenames end in `_run1`, `_run2`, and `_run3`. Use `DEPEND_ON=<job-id>` to continue a throttled chain across separate invocations.

Summary caching and range scheduling are enabled by default through `SUMMARY_CACHE=auto` and `WORK_PARTITION=auto`. Set `SUMMARY_CACHE=off` for cold-cache measurements, `SUMMARY_CACHE=refresh` to rebuild it, `WORK_PARTITION=files` for the old whole-file scheduler, or `WORK_PARTITION=ranges MAX_FILE_READERS=2` for a forced range test.

Stable scaling benchmark:

```bash
dependency=""
for ncpus in 8 16; do
  submission="$(
    BASE_PATH=../Thesan-2/output \
    SIMULATION_NAME=Thesan-2-scaling-new \
    SNAPSHOT=80 PARTICLES=gas BACKENDS="cube pylians" GRIDS=256 \
    LOAD_MODE=chunked NCPUS="${ncpus}" THREADS="${ncpus}" \
    RADIUS_BIN_BATCH_SIZE=10 MEM_256=32gb WALLTIME_256=08:00:00 \
    SUMMARY_CACHE=auto WORK_PARTITION=auto SUBMIT_MODE=throttled REPETITIONS=3 \
    DEPEND_ON="${dependency}" bash scripts/submit_clumping_jobs.sh
  )"
  printf '%s\n' "${submission}"
  dependency="$(printf '%s\n' "${submission}" | tail -n 1)"
done
```

For Thesan-1 snapshot 81, set `BASE_PATH=../Thesan-1/output SNAPSHOT=81 SIMULATION_NAME=Thesan-1 LOAD_MODE=chunked THREADS=<n>`; results and logs will be written under simulation-specific subdirectories. Progress logging is enabled by default in PBS jobs; tune `CHUNK_SIZE`, `THREADS`, and `PROGRESS_INTERVAL` if needed. Chunked gridded runs use same-node local workers and cap the effective worker count by the number of snapshot files.

Submit larger grids only after checking queue limits with `qstat -Q` or `qstat -Qf`.

For redshift evolution, submit one bounded array task per snapshot. Each task keeps the existing same-node worker pool, while `MAX_CONCURRENT` limits simultaneous snapshot reads:

```bash
SNAPSHOTS="40 50 60 70 80" \
BASE_PATH=../Thesan-2/output SIMULATION_NAME=Thesan-2 \
PARTICLE=gas BACKEND=sphere GRID=256 \
NCPUS=8 THREADS=8 MAX_CONCURRENT=8 MEM=32gb \
bash scripts/submit_evolution_jobs.sh
```

Submit tiny one-thread Davies et al. Eq. 13 alternative-clumping jobs across the THESAN reionization snapshots. Each PBS array task reads one snapshot, streams `PhotonDensity`, and writes one JSON result:

```bash
BASE_PATH=/lustre/work/carlos.lopez/Thesan-2/output \
SIMULATION_NAME=Thesan-2 \
MFP_FILE=/lustre/work/carlos.lopez/Thesan-1/mfp_Thesan1.dat \
MAX_CONCURRENT=8 MEM=8gb WALLTIME=02:00:00 \
bash scripts/submit_alternative_clumping_evolution_jobs.sh
```

By default this submits snapshots `54 55 ... 80` on the `tiny` queue with `NCPUS=1 THREADS=1`, writes to `results/<simulation>/alternative_clumping/`, uses photon groups `0 1 2`, runs the `raw-volume` native gas-cell threshold sweep from overdensity `-1` to `25` with 200 samples, and enables verbose progress every 5 chunks. Override `SNAPSHOTS`, `OUTPUT_DIR`, `CHUNK_SIZE`, `PHOTON_GROUPS`, `BACKEND`, `THRESHOLD_MIN`, `THRESHOLD_MAX`, `THRESHOLD_COUNT`, `CHI_E_SOURCE`, `N_H_SOURCE`, or `FULLY_IONIZED=1` when needed. For `BACKEND=grid`, also set `MASK_PARTICLE_TYPE=gas`, `dm`, or `both`.

```bash
GRIDS=512 MEM_512=8gb WALLTIME_512=08:00:00 NCPUS=2 \
bash scripts/submit_clumping_jobs.sh
```

If PBS rejects a job with "violates queue and/or server resource limits", lower `MEM_*`, `WALLTIME_*`, or `NCPUS`, or choose an allowed larger queue with `QUEUE=<name>`.
