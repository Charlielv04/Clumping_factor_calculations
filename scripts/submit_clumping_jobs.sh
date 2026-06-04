#!/bin/bash

set -euo pipefail

mkdir -p logs results

PROJECT_DIR="$(pwd)"
BASE_PATH="${BASE_PATH:-../tng100-3/output}"
CONDA_ENV="${CONDA_ENV:-clumping-factor}"
SNAPSHOT="${SNAPSHOT:-98}"
RADIUS_BINS="${RADIUS_BINS:-10}"
THREADS="${THREADS:-1}"
WALLTIME_256="${WALLTIME_256:-01:00:00}"
WALLTIME_512="${WALLTIME_512:-04:00:00}"
WALLTIME_1024="${WALLTIME_1024:-12:00:00}"
MEM_256="${MEM_256:-4gb}"
MEM_512="${MEM_512:-4gb}"
MEM_1024="${MEM_1024:-4gb}"
NCPUS="${NCPUS:-2}"
QUEUE="${QUEUE:-}"
MAIL_USER="${MAIL_USER:-}"
GRIDS="${GRIDS:-256 512 1024}"
PARTICLES="${PARTICLES:-gas dm}"
BACKENDS="${BACKENDS:-sphere cube pylians}"

submit_one() {
  local particle="$1"
  local backend="$2"
  local grid="$3"
  local mem walltime name
  local -a qsub_args

  case "${grid}" in
    256)
      mem="${MEM_256}"
      walltime="${WALLTIME_256}"
      ;;
    512)
      mem="${MEM_512}"
      walltime="${WALLTIME_512}"
      ;;
    1024)
      mem="${MEM_1024}"
      walltime="${WALLTIME_1024}"
      ;;
    *)
      echo "Unsupported grid size: ${grid}" >&2
      exit 1
      ;;
  esac

  name="cf_${particle}_${backend}_${grid}"

  qsub_args=(
    -N "${name}" \
    -o "${PROJECT_DIR}/logs/${name}.out" \
    -e "${PROJECT_DIR}/logs/${name}.err" \
    -l "select=1:ncpus=${NCPUS}:mem=${mem}" \
    -l "walltime=${walltime}" \
    -v "PROJECT_DIR=${PROJECT_DIR},BASE_PATH=${BASE_PATH},CONDA_ENV=${CONDA_ENV},SNAPSHOT=${SNAPSHOT},RADIUS_BINS=${RADIUS_BINS},THREADS=${THREADS},PARTICLE=${particle},BACKEND=${backend},GRID=${grid}" \
    scripts/clumping_job.pbs
  )

  if [[ -n "${QUEUE}" ]]; then
    qsub_args=(-q "${QUEUE}" "${qsub_args[@]}")
  fi

  if [[ -n "${MAIL_USER}" ]]; then
    qsub_args=(-M "${MAIL_USER}" -m ae "${qsub_args[@]}")
  fi

  qsub "${qsub_args[@]}"
}

for grid in ${GRIDS}; do
  for particle in ${PARTICLES}; do
    for backend in ${BACKENDS}; do
      submit_one "${particle}" "${backend}" "${grid}"
    done
  done
done
