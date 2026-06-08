# PBS Runs On idark

The IPMU idark README shows PBS jobs using modest resource requests, for example `select=1:ncpus=2:mem=4gb`, short walltime, and optionally `-q tiny`. Start with small submissions and increase resources only after a job reaches the queue.

Submit one smoke-test job:

```bash
GRIDS=256 PARTICLES=gas BACKENDS=cube \
WALLTIME_256=01:00:00 MEM_256=4gb NCPUS=2 \
bash scripts/submit_clumping_jobs.sh
```

Submit all 256-grid jobs:

```bash
GRIDS=256 MEM_256=4gb WALLTIME_256=04:00:00 NCPUS=2 \
bash scripts/submit_clumping_jobs.sh
```

For Thesan-1 snapshot 81, set `BASE_PATH=../Thesan-1/output SNAPSHOT=81 SIMULATION_NAME=Thesan-1 LOAD_MODE=chunked`; results and logs will be written under simulation-specific subdirectories. Tune `CHUNK_SIZE` if needed.

Submit larger grids only after checking queue limits with `qstat -Q` or `qstat -Qf`.

```bash
GRIDS=512 MEM_512=8gb WALLTIME_512=08:00:00 NCPUS=2 \
bash scripts/submit_clumping_jobs.sh
```

If PBS rejects a job with "violates queue and/or server resource limits", lower `MEM_*`, `WALLTIME_*`, or `NCPUS`, or choose an allowed queue with `QUEUE=<name>`.
