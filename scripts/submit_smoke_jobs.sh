#!/bin/bash

set -euo pipefail

GRIDS="${GRIDS:-64}"
PARTICLES="${PARTICLES:-gas dm}"
BACKENDS="${BACKENDS:-sphere cube pylians}"
MEM="${MEM:-4gb}"
WALLTIME="${WALLTIME:-01:00:00}"
NCPUS="${NCPUS:-1}"
THREADS="${THREADS:-${NCPUS}}"
RADIUS_BIN_BATCH_SIZE="${RADIUS_BIN_BATCH_SIZE:-1}"
RADIUS_MODE="${RADIUS_MODE:-sphere}"
LOAD_MODE="${LOAD_MODE:-auto}"
CHUNK_SIZE="${CHUNK_SIZE:-1000000}"
MAX_FULL_LOAD_GB="${MAX_FULL_LOAD_GB:-16}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-25}"
VERBOSE="${VERBOSE:-1}"
QUEUE="${QUEUE:-auto}"
BASE_PATH="${BASE_PATH:-../tng100-3/output}"
SIMULATION_NAME="${SIMULATION_NAME:-}"

if [[ -z "${SIMULATION_NAME}" ]]; then
  base_trimmed="${BASE_PATH%/}"
  SIMULATION_NAME="$(basename "${base_trimmed}")"
  if [[ "${SIMULATION_NAME}" == "output" ]]; then
    SIMULATION_NAME="$(basename "$(dirname "${base_trimmed}")")"
  fi
fi

mkdir -p "logs/${SIMULATION_NAME}" "results/${SIMULATION_NAME}"
JOB_SIMULATION_NAME="${SIMULATION_NAME//[^A-Za-z0-9_]/_}"

for grid in ${GRIDS}; do
  for particle in ${PARTICLES}; do
    for backend in ${BACKENDS}; do
      name="smoke_${JOB_SIMULATION_NAME}_${particle}_${backend}_${grid}_b${RADIUS_BIN_BATCH_SIZE}"
      selected_queue=""
      if [[ "${QUEUE}" == "auto" ]]; then
        if (( NCPUS == 1 )); then
          selected_queue="tiny"
        else
          selected_queue="mini"
        fi
      elif [[ "${QUEUE}" == "tiny" && NCPUS -gt 1 ]]; then
        echo "QUEUE=tiny cannot be used with NCPUS=${NCPUS}; use QUEUE=auto, QUEUE=mini, or another larger queue." >&2
        exit 1
      elif [[ "${QUEUE}" != "default" && "${QUEUE}" != "none" ]]; then
        selected_queue="${QUEUE}"
      fi
      args=(
        -N "${name}"
        -o "$(pwd)/logs/${SIMULATION_NAME}/${name}.out"
        -e "$(pwd)/logs/${SIMULATION_NAME}/${name}.err"
        -l "select=1:ncpus=${NCPUS}:mem=${MEM}"
        -l "walltime=${WALLTIME}"
        -v "PROJECT_DIR=$(pwd),BASE_PATH=${BASE_PATH},SIMULATION_NAME=${SIMULATION_NAME},GRID=${grid},PARTICLE=${particle},BACKEND=${backend},THREADS=${THREADS},RADIUS_BIN_BATCH_SIZE=${RADIUS_BIN_BATCH_SIZE},RADIUS_MODE=${RADIUS_MODE},LOAD_MODE=${LOAD_MODE},CHUNK_SIZE=${CHUNK_SIZE},MAX_FULL_LOAD_GB=${MAX_FULL_LOAD_GB},PROGRESS_INTERVAL=${PROGRESS_INTERVAL},VERBOSE=${VERBOSE}"
        scripts/smoke_job.pbs
      )
      if [[ -n "${selected_queue}" ]]; then
        args=(-q "${selected_queue}" "${args[@]}")
      fi
      qsub "${args[@]}"
    done
  done
done
