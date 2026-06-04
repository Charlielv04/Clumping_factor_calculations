#!/bin/bash

set -euo pipefail

GRIDS="${GRIDS:-64}"
PARTICLES="${PARTICLES:-gas dm}"
BACKENDS="${BACKENDS:-sphere cube pylians}"
MEM="${MEM:-4gb}"
WALLTIME="${WALLTIME:-01:00:00}"
NCPUS="${NCPUS:-2}"
THREADS="${THREADS:-1}"
QUEUE="${QUEUE:-}"

mkdir -p logs results

for grid in ${GRIDS}; do
  for particle in ${PARTICLES}; do
    for backend in ${BACKENDS}; do
      name="smoke_${particle}_${backend}_${grid}"
      args=(
        -N "${name}"
        -o "$(pwd)/logs/${name}.out"
        -e "$(pwd)/logs/${name}.err"
        -l "select=1:ncpus=${NCPUS}:mem=${MEM}"
        -l "walltime=${WALLTIME}"
        -v "PROJECT_DIR=$(pwd),GRID=${grid},PARTICLE=${particle},BACKEND=${backend},THREADS=${THREADS}"
        scripts/smoke_job.pbs
      )
      if [[ -n "${QUEUE}" ]]; then
        args=(-q "${QUEUE}" "${args[@]}")
      fi
      qsub "${args[@]}"
    done
  done
done

