#!/bin/bash

set -euo pipefail

GRIDS="${GRIDS:-64}"
PARTICLES="${PARTICLES:-gas dm}"
BACKENDS="${BACKENDS:-sphere cube pylians}"
MEM="${MEM:-4gb}"
WALLTIME="${WALLTIME:-01:00:00}"
NCPUS="${NCPUS:-2}"
THREADS="${THREADS:-1}"
RADIUS_MODE="${RADIUS_MODE:-sphere}"
QUEUE="${QUEUE:-}"
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
      name="smoke_${JOB_SIMULATION_NAME}_${particle}_${backend}_${grid}"
      args=(
        -N "${name}"
        -o "$(pwd)/logs/${SIMULATION_NAME}/${name}.out"
        -e "$(pwd)/logs/${SIMULATION_NAME}/${name}.err"
        -l "select=1:ncpus=${NCPUS}:mem=${MEM}"
        -l "walltime=${WALLTIME}"
        -v "PROJECT_DIR=$(pwd),BASE_PATH=${BASE_PATH},SIMULATION_NAME=${SIMULATION_NAME},GRID=${grid},PARTICLE=${particle},BACKEND=${backend},THREADS=${THREADS},RADIUS_MODE=${RADIUS_MODE}"
        scripts/smoke_job.pbs
      )
      if [[ -n "${QUEUE}" ]]; then
        args=(-q "${QUEUE}" "${args[@]}")
      fi
      qsub "${args[@]}"
    done
  done
done
