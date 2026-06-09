#!/bin/bash

set -euo pipefail

: "${BASE_PATH:?BASE_PATH must be set}"
: "${SNAPSHOT:?SNAPSHOT must be set}"
: "${PARTICLE:?PARTICLE must be set}"
: "${BACKEND:?BACKEND must be set}"
: "${GRID:?GRID must be set}"

PROJECT_DIR="$(pwd)"
CONDA_ENV="${CONDA_ENV:-clumping-factor}"
SIMULATION_NAME="${SIMULATION_NAME:-}"
RADIUS_BINS="${RADIUS_BINS:-10}"
RADIUS_MODE="${RADIUS_MODE:-sphere}"
CHUNK_SIZE="${CHUNK_SIZE:-1000000}"
PARTIAL_DIR="${PARTIAL_DIR:-partials}"
THREADS="${THREADS:-1}"
SHARD_COUNT="${SHARD_COUNT:-32}"
MEM="${MEM:-4gb}"
WALLTIME="${WALLTIME:-02:00:00}"
NCPUS="${NCPUS:-1}"
QUEUE="${QUEUE:-}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-25}"
VERBOSE="${VERBOSE:-1}"

mkdir -p logs/partials

for shard_index in $(seq 0 $((SHARD_COUNT - 1))); do
  name="cf_summary_${shard_index}_of_${SHARD_COUNT}"
  args=(
    -N "${name}"
    -o "${PROJECT_DIR}/logs/partials/${name}.out"
    -e "${PROJECT_DIR}/logs/partials/${name}.err"
    -l "select=1:ncpus=${NCPUS}:mem=${MEM}"
    -l "walltime=${WALLTIME}"
    -v "PROJECT_DIR=${PROJECT_DIR},CONDA_ENV=${CONDA_ENV},BASE_PATH=${BASE_PATH},SIMULATION_NAME=${SIMULATION_NAME},SNAPSHOT=${SNAPSHOT},PARTICLE=${PARTICLE},BACKEND=${BACKEND},GRID=${GRID},RADIUS_BINS=${RADIUS_BINS},RADIUS_MODE=${RADIUS_MODE},CHUNK_SIZE=${CHUNK_SIZE},PARTIAL_DIR=${PARTIAL_DIR},THREADS=${THREADS},SHARD_INDEX=${shard_index},SHARD_COUNT=${SHARD_COUNT},PROGRESS_INTERVAL=${PROGRESS_INTERVAL},VERBOSE=${VERBOSE}"
    scripts/summary_job.pbs
  )
  if [[ -n "${QUEUE}" ]]; then
    args=(-q "${QUEUE}" "${args[@]}")
  fi
  qsub "${args[@]}"
done
