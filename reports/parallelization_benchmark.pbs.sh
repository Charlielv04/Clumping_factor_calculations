#!/bin/sh
#PBS -N paper-thesan2-parallel-benchmark
#PBS -V
#PBS -o logs/pbs/
#PBS -e logs/pbs/
#PBS -J 1-6
#PBS -q mini
#PBS -l select=1:ncpus=16:mem=128gb
#PBS -l walltime=24:00:00
set -eu
if [ -f "$HOME/.conda/etc/profile.d/conda.sh" ]; then
    . "$HOME/.conda/etc/profile.d/conda.sh"
    conda activate clumping-factor
elif [ -x "$HOME/.conda/envs/clumping-factor/bin/clumping" ]; then
    export PATH="$HOME/.conda/envs/clumping-factor/bin:$PATH"
    export CONDA_DEFAULT_ENV=clumping-factor
    export CONDA_PREFIX="$HOME/.conda/envs/clumping-factor"
else
    echo 'clumping-factor conda environment was not found' >&2
    exit 127
fi
cd "${PBS_O_WORKDIR:-.}"
task_index="${PBS_ARRAY_INDEX:-${PBS_ARRAYID:-}}"
if [ -z "$task_index" ]; then
    echo 'PBS array index was not set' >&2
    exit 2
fi
case "$task_index" in
    1)
        exec clumping clumping compute --base-path /lustre/work/carlos.lopez/Thesan-2/output --simulation-name Thesan-2 --snapshot 80 --particle-type dm --backend pylians --grid-size 512 --filter-type Top-Hat --radius-bins 10 --threshold-count 200 --threshold-max 25.0 --threshold-min -1.0 --threads 1 --load-mode chunked --chunk-size 1000000 --mas CIC --radius-bin-batch-size 1 --summary-cache off --work-partition auto --max-file-readers 2 --memory-limit 128gb --memory-safety-fraction 0.1 --run-label w1-r1 --output reports/parallelization_benchmark_runs/thesan/Thesan-2/clumping/clumping.pylians/dm/snapshot080/science-093413ed5f82/execution-6c428357a776_run001.json
        ;;
    2)
        exec clumping clumping compute --base-path /lustre/work/carlos.lopez/Thesan-2/output --simulation-name Thesan-2 --snapshot 80 --particle-type dm --backend pylians --grid-size 512 --filter-type Top-Hat --radius-bins 10 --threshold-count 200 --threshold-max 25.0 --threshold-min -1.0 --threads 16 --load-mode chunked --chunk-size 1000000 --mas CIC --radius-bin-batch-size 1 --summary-cache off --work-partition auto --max-file-readers 2 --memory-limit 128gb --memory-safety-fraction 0.1 --run-label w16-r1 --output reports/parallelization_benchmark_runs/thesan/Thesan-2/clumping/clumping.pylians/dm/snapshot080/science-093413ed5f82/execution-9c4a6a3b174e_run002.json
        ;;
    3)
        exec clumping clumping compute --base-path /lustre/work/carlos.lopez/Thesan-2/output --simulation-name Thesan-2 --snapshot 80 --particle-type dm --backend pylians --grid-size 512 --filter-type Top-Hat --radius-bins 10 --threshold-count 200 --threshold-max 25.0 --threshold-min -1.0 --threads 1 --load-mode chunked --chunk-size 1000000 --mas CIC --radius-bin-batch-size 1 --summary-cache off --work-partition auto --max-file-readers 2 --memory-limit 128gb --memory-safety-fraction 0.1 --run-label w1-r2 --output reports/parallelization_benchmark_runs/thesan/Thesan-2/clumping/clumping.pylians/dm/snapshot080/science-093413ed5f82/execution-6c428357a776_run003.json
        ;;
    4)
        exec clumping clumping compute --base-path /lustre/work/carlos.lopez/Thesan-2/output --simulation-name Thesan-2 --snapshot 80 --particle-type dm --backend pylians --grid-size 512 --filter-type Top-Hat --radius-bins 10 --threshold-count 200 --threshold-max 25.0 --threshold-min -1.0 --threads 16 --load-mode chunked --chunk-size 1000000 --mas CIC --radius-bin-batch-size 1 --summary-cache off --work-partition auto --max-file-readers 2 --memory-limit 128gb --memory-safety-fraction 0.1 --run-label w16-r2 --output reports/parallelization_benchmark_runs/thesan/Thesan-2/clumping/clumping.pylians/dm/snapshot080/science-093413ed5f82/execution-9c4a6a3b174e_run004.json
        ;;
    5)
        exec clumping clumping compute --base-path /lustre/work/carlos.lopez/Thesan-2/output --simulation-name Thesan-2 --snapshot 80 --particle-type dm --backend pylians --grid-size 512 --filter-type Top-Hat --radius-bins 10 --threshold-count 200 --threshold-max 25.0 --threshold-min -1.0 --threads 1 --load-mode chunked --chunk-size 1000000 --mas CIC --radius-bin-batch-size 1 --summary-cache off --work-partition auto --max-file-readers 2 --memory-limit 128gb --memory-safety-fraction 0.1 --run-label w1-r3 --output reports/parallelization_benchmark_runs/thesan/Thesan-2/clumping/clumping.pylians/dm/snapshot080/science-093413ed5f82/execution-6c428357a776_run005.json
        ;;
    6)
        exec clumping clumping compute --base-path /lustre/work/carlos.lopez/Thesan-2/output --simulation-name Thesan-2 --snapshot 80 --particle-type dm --backend pylians --grid-size 512 --filter-type Top-Hat --radius-bins 10 --threshold-count 200 --threshold-max 25.0 --threshold-min -1.0 --threads 16 --load-mode chunked --chunk-size 1000000 --mas CIC --radius-bin-batch-size 1 --summary-cache off --work-partition auto --max-file-readers 2 --memory-limit 128gb --memory-safety-fraction 0.1 --run-label w16-r3 --output reports/parallelization_benchmark_runs/thesan/Thesan-2/clumping/clumping.pylians/dm/snapshot080/science-093413ed5f82/execution-9c4a6a3b174e_run006.json
        ;;
    *)
        echo "Invalid PBS array index: $task_index" >&2
        exit 2
        ;;
esac
