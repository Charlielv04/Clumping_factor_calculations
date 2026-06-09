#!/bin/bash

set -euo pipefail

: "${MANIFEST:?MANIFEST must be set, e.g. partials/Thesan-1/snapshot081/gas_sphere_grid256/manifest.json}"

PROJECT_DIR="$(pwd)"
CONDA_ENV="${CONDA_ENV:-clumping-factor}"
PARTIAL_DIR="${PARTIAL_DIR:-partials}"
SHARD_COUNT="${SHARD_COUNT:-32}"
MEM="${MEM:-8gb}"
WALLTIME="${WALLTIME:-12:00:00}"
NCPUS="${NCPUS:-1}"
QUEUE="${QUEUE:-}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-25}"
VERBOSE="${VERBOSE:-1}"

mkdir -p logs/partials

for shard_index in $(seq 0 $((SHARD_COUNT - 1))); do
  name="cf_partial_${shard_index}_of_${SHARD_COUNT}"
  args=(
    -N "${name}"
    -o "${PROJECT_DIR}/logs/partials/${name}.out"
    -e "${PROJECT_DIR}/logs/partials/${name}.err"
    -l "select=1:ncpus=${NCPUS}:mem=${MEM}"
    -l "walltime=${WALLTIME}"
    -v "PROJECT_DIR=${PROJECT_DIR},CONDA_ENV=${CONDA_ENV},MANIFEST=${MANIFEST},PARTIAL_DIR=${PARTIAL_DIR},SHARD_INDEX=${shard_index},SHARD_COUNT=${SHARD_COUNT},PROGRESS_INTERVAL=${PROGRESS_INTERVAL},VERBOSE=${VERBOSE}"
    scripts/partial_job.pbs
  )
  if [[ -n "${QUEUE}" ]]; then
    args=(-q "${QUEUE}" "${args[@]}")
  fi
  qsub "${args[@]}"
done
