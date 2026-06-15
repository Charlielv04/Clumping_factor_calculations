#!/bin/bash

set -euo pipefail

mkdir -p logs results

PROJECT_DIR="$(pwd)"
BASE_PATH="${BASE_PATH:-../tng100-3/output}"
SIMULATION_NAME="${SIMULATION_NAME:-}"
CONDA_ENV="${CONDA_ENV:-clumping-factor}"
SNAPSHOT="${SNAPSHOT:-98}"
RADIUS_BINS="${RADIUS_BINS:-10}"
RADIUS_BIN_BATCH_SIZE="${RADIUS_BIN_BATCH_SIZE:-1}"
RADIUS_MODE="${RADIUS_MODE:-sphere}"
MAS="${MAS:-CIC}"
MAS="${MAS^^}"
LOAD_MODE="${LOAD_MODE:-auto}"
CHUNK_SIZE="${CHUNK_SIZE:-1000000}"
MAX_FULL_LOAD_GB="${MAX_FULL_LOAD_GB:-16}"
MEMORY_SAFETY_FRACTION="${MEMORY_SAFETY_FRACTION:-0.1}"
SUMMARY_CACHE="${SUMMARY_CACHE:-auto}"
SUMMARY_CACHE_DIR="${SUMMARY_CACHE_DIR:-results/.cache/summaries}"
WORK_PARTITION="${WORK_PARTITION:-auto}"
MAX_FILE_READERS="${MAX_FILE_READERS:-2}"
SUBMIT_MODE="${SUBMIT_MODE:-throttled}"
REPETITIONS="${REPETITIONS:-1}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-25}"
VERBOSE="${VERBOSE:-1}"
NCPUS="${NCPUS:-2}"
THREADS="${THREADS:-${NCPUS}}"
WALLTIME_256="${WALLTIME_256:-01:00:00}"
WALLTIME_512="${WALLTIME_512:-04:00:00}"
WALLTIME_1024="${WALLTIME_1024:-12:00:00}"
WALLTIME_128="${WALLTIME_128:-${WALLTIME_256}}"
WALLTIME_384="${WALLTIME_384:-${WALLTIME_512}}"
WALLTIME_640="${WALLTIME_640:-${WALLTIME_1024}}"
WALLTIME_768="${WALLTIME_768:-${WALLTIME_1024}}"
WALLTIME_896="${WALLTIME_896:-${WALLTIME_1024}}"
MEM_256="${MEM_256:-4gb}"
MEM_512="${MEM_512:-4gb}"
MEM_1024="${MEM_1024:-4gb}"
MEM_128="${MEM_128:-${MEM_256}}"
MEM_384="${MEM_384:-${MEM_512}}"
MEM_640="${MEM_640:-${MEM_1024}}"
MEM_768="${MEM_768:-${MEM_1024}}"
MEM_896="${MEM_896:-${MEM_1024}}"
QUEUE="${QUEUE:-auto}"
MAIL_USER="${MAIL_USER:-}"
GRIDS="${GRIDS:-128 384 640 768 896}"
PARTICLES="${PARTICLES:-gas dm}"
BACKENDS="${BACKENDS:-sphere cube pylians}"
TARGET_PARTICLE_TYPE="${TARGET_PARTICLE_TYPE:-}"
TARGET_BACKEND="${TARGET_BACKEND:-}"
MASK_PARTICLE_TYPE="${MASK_PARTICLE_TYPE:-}"
MASK_BACKEND="${MASK_BACKEND:-}"
TARGET_RADIUS_MODE="${TARGET_RADIUS_MODE:-}"
MASK_RADIUS_MODE="${MASK_RADIUS_MODE:-}"
LAST_JOB_ID=""

if [[ "${MAS}" != "CIC" && "${MAS}" != "TSC" ]]; then
  echo "MAS must be CIC or TSC, got: ${MAS}" >&2
  exit 1
fi

if [[ -z "${SIMULATION_NAME}" ]]; then
  base_trimmed="${BASE_PATH%/}"
  SIMULATION_NAME="$(basename "${base_trimmed}")"
  if [[ "${SIMULATION_NAME}" == "output" ]]; then
    SIMULATION_NAME="$(basename "$(dirname "${base_trimmed}")")"
  fi
fi
mkdir -p "logs/${SIMULATION_NAME}" "results/${SIMULATION_NAME}"
JOB_SIMULATION_NAME="${SIMULATION_NAME//[^A-Za-z0-9_]/_}"

submit_one() {
  local particle="$1"
  local backend="$2"
  local grid="$3"
  local run_label="$4"
  local dependency="$5"
  local mem walltime name selected_queue resource_size
  local submission_output
  local -a qsub_args

  case "${grid}" in
    128)
      mem="${MEM_128}"
      walltime="${WALLTIME_128}"
      ;;
    256)
      mem="${MEM_256}"
      walltime="${WALLTIME_256}"
      ;;
    384)
      mem="${MEM_384}"
      walltime="${WALLTIME_384}"
      ;;
    512)
      mem="${MEM_512}"
      walltime="${WALLTIME_512}"
      ;;
    640)
      mem="${MEM_640}"
      walltime="${WALLTIME_640}"
      ;;
    768)
      mem="${MEM_768}"
      walltime="${WALLTIME_768}"
      ;;
    896)
      mem="${MEM_896}"
      walltime="${WALLTIME_896}"
      ;;
    1024)
      mem="${MEM_1024}"
      walltime="${WALLTIME_1024}"
      ;;
    *)
      echo "Unsupported grid size: ${grid}. Supported defaults are 128-step grids from 128 to 1024." >&2
      exit 1
      ;;
  esac

  if (( NCPUS == 1 )); then
    resource_size="serial"
  else
    resource_size="parallel${NCPUS}"
  fi
  name="cf_${JOB_SIMULATION_NAME}_${particle}_${backend}_g${grid}_${resource_size}_b${RADIUS_BIN_BATCH_SIZE}"
  name="${name}_r${run_label}"
  if [[ "${MAS}" != "CIC" ]]; then
    name="${name}_mas${MAS,,}"
  fi

  case "${QUEUE}" in
    auto)
      if (( NCPUS == 1 )); then
        selected_queue="tiny"
      else
        selected_queue="mini"
      fi
      ;;
    tiny)
      if (( NCPUS > 1 )); then
        echo "QUEUE=tiny cannot be used with NCPUS=${NCPUS}; use QUEUE=auto, QUEUE=mini, or another larger queue." >&2
        exit 1
      fi
      selected_queue="tiny"
      ;;
    default|none)
      selected_queue=""
      ;;
    *)
      selected_queue="${QUEUE}"
      ;;
  esac

  echo "Submitting ${name}: grid=${grid}, ncpus=${NCPUS}, threads=${THREADS}, mem=${mem}, walltime=${walltime}, queue=${selected_queue:-default}, dependency=${dependency:-none}"

  qsub_args=(
    -N "${name}" \
    -o "${PROJECT_DIR}/logs/${SIMULATION_NAME}/${name}.out" \
    -e "${PROJECT_DIR}/logs/${SIMULATION_NAME}/${name}.err" \
    -l "select=1:ncpus=${NCPUS}:mem=${mem}" \
    -l "walltime=${walltime}" \
    -v "PROJECT_DIR=${PROJECT_DIR},BASE_PATH=${BASE_PATH},SIMULATION_NAME=${SIMULATION_NAME},CONDA_ENV=${CONDA_ENV},SNAPSHOT=${SNAPSHOT},RADIUS_BINS=${RADIUS_BINS},RADIUS_BIN_BATCH_SIZE=${RADIUS_BIN_BATCH_SIZE},RADIUS_MODE=${RADIUS_MODE},MAS=${MAS},THREADS=${THREADS},NCPUS=${NCPUS},RESOURCE_SIZE=${resource_size},MEMORY_LIMIT=${mem},MEMORY_SAFETY_FRACTION=${MEMORY_SAFETY_FRACTION},SUMMARY_CACHE=${SUMMARY_CACHE},SUMMARY_CACHE_DIR=${SUMMARY_CACHE_DIR},WORK_PARTITION=${WORK_PARTITION},MAX_FILE_READERS=${MAX_FILE_READERS},RUN_LABEL=${run_label},LOAD_MODE=${LOAD_MODE},CHUNK_SIZE=${CHUNK_SIZE},MAX_FULL_LOAD_GB=${MAX_FULL_LOAD_GB},PROGRESS_INTERVAL=${PROGRESS_INTERVAL},VERBOSE=${VERBOSE},PARTICLE=${particle},BACKEND=${backend},GRID=${grid},TARGET_PARTICLE_TYPE=${TARGET_PARTICLE_TYPE},TARGET_BACKEND=${TARGET_BACKEND},MASK_PARTICLE_TYPE=${MASK_PARTICLE_TYPE},MASK_BACKEND=${MASK_BACKEND},TARGET_RADIUS_MODE=${TARGET_RADIUS_MODE},MASK_RADIUS_MODE=${MASK_RADIUS_MODE}" \
    scripts/clumping_job.pbs
  )

  if [[ -n "${selected_queue}" ]]; then
    qsub_args=(-q "${selected_queue}" "${qsub_args[@]}")
  fi

  if [[ -n "${MAIL_USER}" ]]; then
    qsub_args=(-M "${MAIL_USER}" -m ae "${qsub_args[@]}")
  fi

  if [[ -n "${dependency}" ]]; then
    qsub_args=(-W "depend=afterok:${dependency}" "${qsub_args[@]}")
  fi

  submission_output="$(qsub "${qsub_args[@]}")"
  echo "${submission_output}"
  LAST_JOB_ID="${submission_output}"
}

previous_job="${DEPEND_ON:-}"
for repetition in $(seq 1 "${REPETITIONS}"); do
  for grid in ${GRIDS}; do
    for particle in ${PARTICLES}; do
      for backend in ${BACKENDS}; do
        dependency=""
        if [[ "${SUBMIT_MODE}" == "throttled" ]]; then
          dependency="${previous_job}"
        elif [[ "${SUBMIT_MODE}" != "parallel" ]]; then
          echo "SUBMIT_MODE must be throttled or parallel, got: ${SUBMIT_MODE}" >&2
          exit 1
        fi
        submit_one "${particle}" "${backend}" "${grid}" "${repetition}" "${dependency}"
        if [[ "${SUBMIT_MODE}" == "throttled" ]]; then
          previous_job="${LAST_JOB_ID}"
        fi
      done
    done
  done
done
