#!/bin/bash

set -euo pipefail

mkdir -p logs results

PROJECT_DIR="$(pwd)"
BASE_PATH="${BASE_PATH:-../tng100-3/output}"
SIMULATION_NAME="${SIMULATION_NAME:-}"
RESULTS_LAYOUT="${RESULTS_LAYOUT:-auto}"
RESULTS_FAMILY="${RESULTS_FAMILY:-}"
CONDA_ENV="${CONDA_ENV:-clumping-factor}"
SNAPSHOT="${SNAPSHOT:-98}"
GRIDS="${GRIDS:-256}"
PARTICLES="${PARTICLES:-dm}"
SMOOTHINGS="${SMOOTHINGS:-none}"
SPECTRUM_ENGINES="${SPECTRUM_ENGINES:-numpy}"
PYLIANS_AXIS="${PYLIANS_AXIS:-0}"
PYLIANS_MAS="${PYLIANS_MAS:-auto}"
RADIUS_BINS="${RADIUS_BINS:-10}"
RADIUS_BIN_BATCH_SIZE="${RADIUS_BIN_BATCH_SIZE:-1}"
RADIUS_MODE="${RADIUS_MODE:-sphere}"
FILTER_TYPE="${FILTER_TYPE:-Top-Hat}"
MAS="${MAS:-CIC}"
MAS="${MAS^^}"
LOAD_MODE="${LOAD_MODE:-auto}"
CHUNK_SIZE="${CHUNK_SIZE:-1000000}"
MAX_FULL_LOAD_GB="${MAX_FULL_LOAD_GB:-16}"
BIN_COUNT="${BIN_COUNT:-40}"
BINNING="${BINNING:-log}"
K_MIN="${K_MIN:-}"
K_MAX="${K_MAX:-}"
SUBMIT_MODE="${SUBMIT_MODE:-throttled}"
REPETITIONS="${REPETITIONS:-1}"
VERBOSE="${VERBOSE:-1}"
NCPUS="${NCPUS:-2}"
THREADS="${THREADS:-${NCPUS}}"
WALLTIME_128="${WALLTIME_128:-01:00:00}"
WALLTIME_256="${WALLTIME_256:-02:00:00}"
WALLTIME_384="${WALLTIME_384:-04:00:00}"
WALLTIME_512="${WALLTIME_512:-08:00:00}"
WALLTIME_640="${WALLTIME_640:-12:00:00}"
WALLTIME_768="${WALLTIME_768:-12:00:00}"
WALLTIME_896="${WALLTIME_896:-12:00:00}"
WALLTIME_1024="${WALLTIME_1024:-24:00:00}"
MEM_128="${MEM_128:-8gb}"
MEM_256="${MEM_256:-16gb}"
MEM_384="${MEM_384:-32gb}"
MEM_512="${MEM_512:-64gb}"
MEM_640="${MEM_640:-96gb}"
MEM_768="${MEM_768:-128gb}"
MEM_896="${MEM_896:-160gb}"
MEM_1024="${MEM_1024:-192gb}"
QUEUE="${QUEUE:-auto}"
MAIL_USER="${MAIL_USER:-}"
LAST_JOB_ID=""

job_script="${PROJECT_DIR}/scripts/power_spectrum_job.pbs"
if [[ ! -f "${job_script}" ]]; then
  echo "Run this script from the Clumping_factor_calculations directory." >&2
  exit 1
fi

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
  local smoothing="$2"
  local engine="$3"
  local grid="$4"
  local run_label="$5"
  local dependency="$6"
  local mem walltime selected_queue resource_size name submission_output
  local mem_var="MEM_${grid}"
  local walltime_var="WALLTIME_${grid}"
  local -a qsub_args

  case "${grid}" in
    128) mem="${MEM_128}"; walltime="${WALLTIME_128}" ;;
    256) mem="${MEM_256}"; walltime="${WALLTIME_256}" ;;
    384) mem="${MEM_384}"; walltime="${WALLTIME_384}" ;;
    512) mem="${MEM_512}"; walltime="${WALLTIME_512}" ;;
    640) mem="${MEM_640}"; walltime="${WALLTIME_640}" ;;
    768) mem="${MEM_768}"; walltime="${WALLTIME_768}" ;;
    896) mem="${MEM_896}"; walltime="${WALLTIME_896}" ;;
    1024) mem="${MEM_1024}"; walltime="${WALLTIME_1024}" ;;
    *)
      mem="${!mem_var:-}"
      walltime="${!walltime_var:-}"
      if [[ -z "${mem}" || -z "${walltime}" ]]; then
        echo "Unsupported grid size: ${grid}. Set MEM_${grid} and WALLTIME_${grid} to submit custom grids." >&2
        exit 1
      fi
      ;;
  esac

  if (( NCPUS > 56 )); then
    echo "NCPUS=${NCPUS} exceeds the largest same-node CPU count currently supported by this submit script. Use NCPUS<=56 or add a multi-node submission path." >&2
    exit 1
  fi
  if (( NCPUS == 1 )); then
    resource_size="serial"
  else
    resource_size="parallel${NCPUS}"
  fi
  case "${QUEUE}" in
    auto)
      if (( NCPUS == 1 )); then
        selected_queue="tiny"
      elif (( NCPUS >= 64 || grid >= 2048 )); then
        selected_queue="small"
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
    default|none) selected_queue="" ;;
    *) selected_queue="${QUEUE}" ;;
  esac

  name="pk_${JOB_SIMULATION_NAME}_${particle}_${smoothing}_${engine}_g${grid}_${resource_size}_r${run_label}"
  if [[ "${MAS}" != "CIC" ]]; then
    name="${name}_mas${MAS,,}"
  fi

  echo "Submitting ${name}: grid=${grid}, particle=${particle}, smoothing=${smoothing}, engine=${engine}, ncpus=${NCPUS}, threads=${THREADS}, mem=${mem}, walltime=${walltime}, queue=${selected_queue:-default}, dependency=${dependency:-none}"
  qsub_args=(
    -N "${name}"
    -o "${PROJECT_DIR}/logs/${SIMULATION_NAME}/${name}.out"
    -e "${PROJECT_DIR}/logs/${SIMULATION_NAME}/${name}.err"
    -l "select=1:ncpus=${NCPUS}:mem=${mem}"
    -l "walltime=${walltime}"
    -v "PROJECT_DIR=${PROJECT_DIR},BASE_PATH=${BASE_PATH},SIMULATION_NAME=${SIMULATION_NAME},RESULTS_LAYOUT=${RESULTS_LAYOUT},RESULTS_FAMILY=${RESULTS_FAMILY},CONDA_ENV=${CONDA_ENV},SNAPSHOT=${SNAPSHOT},PARTICLE=${particle},GRID=${grid},SMOOTHING=${smoothing},SPECTRUM_ENGINE=${engine},PYLIANS_AXIS=${PYLIANS_AXIS},PYLIANS_MAS=${PYLIANS_MAS},RADIUS_BINS=${RADIUS_BINS},RADIUS_BIN_BATCH_SIZE=${RADIUS_BIN_BATCH_SIZE},RADIUS_MODE=${RADIUS_MODE},FILTER_TYPE=${FILTER_TYPE},MAS=${MAS},THREADS=${THREADS},NCPUS=${NCPUS},RESOURCE_SIZE=${resource_size},LOAD_MODE=${LOAD_MODE},CHUNK_SIZE=${CHUNK_SIZE},MAX_FULL_LOAD_GB=${MAX_FULL_LOAD_GB},BIN_COUNT=${BIN_COUNT},BINNING=${BINNING},K_MIN=${K_MIN},K_MAX=${K_MAX},RUN_LABEL=${run_label},VERBOSE=${VERBOSE}"
    "${job_script}"
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
      for smoothing in ${SMOOTHINGS}; do
        for engine in ${SPECTRUM_ENGINES}; do
          dependency=""
          if [[ "${SUBMIT_MODE}" == "throttled" ]]; then
            dependency="${previous_job}"
          elif [[ "${SUBMIT_MODE}" != "parallel" ]]; then
            echo "SUBMIT_MODE must be throttled or parallel, got: ${SUBMIT_MODE}" >&2
            exit 1
          fi
          submit_one "${particle}" "${smoothing}" "${engine}" "${grid}" "${repetition}" "${dependency}"
          if [[ "${SUBMIT_MODE}" == "throttled" ]]; then
            previous_job="${LAST_JOB_ID}"
          fi
        done
      done
    done
  done
done
