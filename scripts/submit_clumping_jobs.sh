#!/bin/bash

set -euo pipefail

mkdir -p logs results

PROJECT_DIR="$(pwd)"
BASE_PATH="${BASE_PATH:-../tng100-3/output}"
CONDA_ENV="${CONDA_ENV:-clumping-factor}"
SNAPSHOT="${SNAPSHOT:-98}"
RADIUS_BINS="${RADIUS_BINS:-10}"
THREADS="${THREADS:-1}"
WALLTIME_256="${WALLTIME_256:-24:00:00}"
WALLTIME_512="${WALLTIME_512:-48:00:00}"
WALLTIME_1024="${WALLTIME_1024:-72:00:00}"
MEM_256="${MEM_256:-128gb}"
MEM_512="${MEM_512:-256gb}"
MEM_1024="${MEM_1024:-512gb}"
NCPUS="${NCPUS:-4}"
QUEUE="${QUEUE:-}"
MAIL_USER="${MAIL_USER:-}"

submit_one() {
  local particle="$1"
  local backend="$2"
  local grid="$3"
  local mem walltime name queue_args mail_args

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
  queue_args=()
  mail_args=()

  if [[ -n "${QUEUE}" ]]; then
    queue_args=(-q "${QUEUE}")
  fi

  if [[ -n "${MAIL_USER}" ]]; then
    mail_args=(-M "${MAIL_USER}" -m ae)
  fi

  qsub \
    -N "${name}" \
    -o "${PROJECT_DIR}/logs/${name}.out" \
    -e "${PROJECT_DIR}/logs/${name}.err" \
    -l "select=1:ncpus=${NCPUS}:mem=${mem}" \
    -l "walltime=${walltime}" \
    "${queue_args[@]}" \
    "${mail_args[@]}" \
    -v "PROJECT_DIR=${PROJECT_DIR},BASE_PATH=${BASE_PATH},CONDA_ENV=${CONDA_ENV},SNAPSHOT=${SNAPSHOT},RADIUS_BINS=${RADIUS_BINS},THREADS=${THREADS},PARTICLE=${particle},BACKEND=${backend},GRID=${grid}" \
    scripts/clumping_job.pbs
}

for grid in 256 512 1024; do
  for particle in gas dm; do
    for backend in sphere cube pylians; do
      submit_one "${particle}" "${backend}" "${grid}"
    done
  done
done

