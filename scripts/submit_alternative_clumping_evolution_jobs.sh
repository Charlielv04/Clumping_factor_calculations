#!/bin/bash

set -euo pipefail

SNAPSHOTS="${SNAPSHOTS:?Set SNAPSHOTS to a space-separated list, for example '80' or '54 58 61 64 70 75 80'.}"
BASE_PATH="${BASE_PATH:-../Thesan-2/output}"
MFP_FILE="${MFP_FILE:?Set MFP_FILE to the THESAN mean-free-path table, e.g. /lustre/work/carlos.lopez/Thesan-1/mfp_Thesan1.dat.}"
SIMULATION_NAME="${SIMULATION_NAME:-}"
OUTPUT_DIR="${OUTPUT_DIR:-results}"
NCPUS="${NCPUS:-1}"
THREADS="${THREADS:-${NCPUS}}"
MAX_CONCURRENT="${MAX_CONCURRENT:-8}"
MEM="${MEM:-8gb}"
WALLTIME="${WALLTIME:-02:00:00}"
QUEUE="${QUEUE:-auto}"
CONDA_ENV="${CONDA_ENV:-clumping-factor}"
ALLOW_LEGACY_IONIZING_TABLE="${ALLOW_LEGACY_IONIZING_TABLE:-0}"
CHUNK_SIZE="${CHUNK_SIZE:-1000000}"
LOAD_MODE="${LOAD_MODE:-chunked}"
MAX_FULL_LOAD_GB="${MAX_FULL_LOAD_GB:-16}"
MEMORY_SAFETY_FRACTION="${MEMORY_SAFETY_FRACTION:-0.1}"
SUMMARY_CACHE="${SUMMARY_CACHE:-auto}"
SUMMARY_CACHE_DIR="${SUMMARY_CACHE_DIR:-results/.cache/summaries}"
WORK_PARTITION="${WORK_PARTITION:-auto}"
MAX_FILE_READERS="${MAX_FILE_READERS:-2}"
CLUMPING_MAX_GRID_CELLS="${CLUMPING_MAX_GRID_CELLS:-}"
PHOTON_GROUPS="${PHOTON_GROUPS:-0 1 2}"
HYDROGEN_MASS_FRACTION="${HYDROGEN_MASS_FRACTION:-0.76}"
ALPHA_HII_CM3_S="${ALPHA_HII_CM3_S:-2.59e-13}"
CHI_E="${CHI_E:-1.08}"
CHI_E_SOURCE="${CHI_E_SOURCE:-constant}"
N_H_SOURCE="${N_H_SOURCE:-cosmic-mean}"
BACKEND="${BACKEND:-raw-volume}"
THRESHOLD_MIN="${THRESHOLD_MIN:--1}"
THRESHOLD_MAX="${THRESHOLD_MAX:-25}"
THRESHOLD_COUNT="${THRESHOLD_COUNT:-200}"
GRID_SIZE="${GRID_SIZE:-256}"
RADIUS_BINS="${RADIUS_BINS:-10}"
RADIUS_BIN_BATCH_SIZE="${RADIUS_BIN_BATCH_SIZE:-1}"
MAS="${MAS:-CIC}"
MAS="${MAS^^}"
MASK_PARTICLE_TYPE="${MASK_PARTICLE_TYPE:-}"
MASK_BACKEND="${MASK_BACKEND:-sphere}"
MASK_RADIUS_MODE="${MASK_RADIUS_MODE:-sphere}"
FULLY_IONIZED="${FULLY_IONIZED:-1}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-5}"
VERBOSE="${VERBOSE:-1}"

if (( NCPUS > 56 )); then
  echo "NCPUS=${NCPUS} exceeds the largest same-node CPU count currently supported by this submit script. Use NCPUS<=56." >&2
  exit 1
fi
if [[ "${MAS}" != "CIC" && "${MAS}" != "TSC" ]]; then
  echo "MAS must be CIC or TSC, got: ${MAS}" >&2
  exit 1
fi
if [[ "${BACKEND}" == "raw-volume" && ( "${NCPUS}" != "1" || "${THREADS}" != "1" ) ]]; then
  echo "BACKEND=raw-volume is streaming single-process; use NCPUS=1 THREADS=1 or BACKEND=grid for multithreading." >&2
  exit 1
fi
if [[ "${BACKEND}" == "grid" && -z "${MASK_PARTICLE_TYPE}" ]]; then
  echo "BACKEND=grid requires MASK_PARTICLE_TYPE=gas, dm, or both." >&2
  exit 1
fi

if [[ -z "${SIMULATION_NAME}" ]]; then
  base_trimmed="${BASE_PATH%/}"
  SIMULATION_NAME="$(basename "${base_trimmed}")"
  if [[ "${SIMULATION_NAME}" == "output" ]]; then
    SIMULATION_NAME="$(basename "$(dirname "${base_trimmed}")")"
  fi
fi
read -r -a snapshot_array <<< "${SNAPSHOTS}"
if (( ${#snapshot_array[@]} == 0 )); then
  echo "SNAPSHOTS must contain at least one snapshot number." >&2
  exit 1
fi
snapshots_csv="$(IFS=:; echo "${snapshot_array[*]}")"
read -r -a photon_group_array <<< "${PHOTON_GROUPS}"
photon_groups_csv="$(IFS=:; echo "${photon_group_array[*]}")"
last_index=$((${#snapshot_array[@]} - 1))
project_dir="$(pwd)"
job_simulation_name="${SIMULATION_NAME//[^A-Za-z0-9_]/_}"
if [[ "${BACKEND}" == "grid" ]]; then
  name="cfalt_${job_simulation_name}_grid_g${GRID_SIZE}_t${THREADS}"
else
  name="cfalt_${job_simulation_name}_rawvol"
fi

case "${QUEUE}" in
  auto)
    if (( NCPUS == 1 )); then
      selected_queue="tiny"
    else
      selected_queue="mini"
    fi
    ;;
  default|none)
    selected_queue=""
    ;;
  *)
    selected_queue="${QUEUE}"
    ;;
esac

mkdir -p "logs/${SIMULATION_NAME}" "${OUTPUT_DIR}"

if [[ "${MAX_CONCURRENT}" != "" ]]; then
  echo "Note: this PBS qsub accepts -J X-Y[:Z] but not -J X-Y%N; MAX_CONCURRENT=${MAX_CONCURRENT} is not applied." >&2
fi

env_vars="PROJECT_DIR=${project_dir},SNAPSHOTS_CSV=${snapshots_csv},BASE_PATH=${BASE_PATH},MFP_FILE=${MFP_FILE},SIMULATION_NAME=${SIMULATION_NAME},OUTPUT_DIR=${OUTPUT_DIR},CONDA_ENV=${CONDA_ENV},ALLOW_LEGACY_IONIZING_TABLE=${ALLOW_LEGACY_IONIZING_TABLE},CHUNK_SIZE=${CHUNK_SIZE},LOAD_MODE=${LOAD_MODE},MAX_FULL_LOAD_GB=${MAX_FULL_LOAD_GB},MEMORY_LIMIT=${MEM},MEMORY_SAFETY_FRACTION=${MEMORY_SAFETY_FRACTION},SUMMARY_CACHE=${SUMMARY_CACHE},SUMMARY_CACHE_DIR=${SUMMARY_CACHE_DIR},WORK_PARTITION=${WORK_PARTITION},MAX_FILE_READERS=${MAX_FILE_READERS},PHOTON_GROUPS_CSV=${photon_groups_csv},HYDROGEN_MASS_FRACTION=${HYDROGEN_MASS_FRACTION},ALPHA_HII_CM3_S=${ALPHA_HII_CM3_S},CHI_E=${CHI_E},CHI_E_SOURCE=${CHI_E_SOURCE},N_H_SOURCE=${N_H_SOURCE},BACKEND=${BACKEND},THRESHOLD_MIN=${THRESHOLD_MIN},THRESHOLD_MAX=${THRESHOLD_MAX},THRESHOLD_COUNT=${THRESHOLD_COUNT},GRID_SIZE=${GRID_SIZE},RADIUS_BINS=${RADIUS_BINS},RADIUS_BIN_BATCH_SIZE=${RADIUS_BIN_BATCH_SIZE},MAS=${MAS},MASK_PARTICLE_TYPE=${MASK_PARTICLE_TYPE},MASK_BACKEND=${MASK_BACKEND},MASK_RADIUS_MODE=${MASK_RADIUS_MODE},NCPUS=${NCPUS},THREADS=${THREADS},FULLY_IONIZED=${FULLY_IONIZED},PROGRESS_INTERVAL=${PROGRESS_INTERVAL},VERBOSE=${VERBOSE}"
if [[ -n "${CLUMPING_MAX_GRID_CELLS}" ]]; then
  env_vars="${env_vars},CLUMPING_MAX_GRID_CELLS=${CLUMPING_MAX_GRID_CELLS}"
fi

qsub_args=(
  -N "${name}" \
  -J "0-${last_index}" \
  -o "${project_dir}/logs/${SIMULATION_NAME}/${name}.out" \
  -e "${project_dir}/logs/${SIMULATION_NAME}/${name}.err" \
  -l "select=1:ncpus=${NCPUS}:mem=${MEM}" \
  -l "walltime=${WALLTIME}" \
  -v "${env_vars}" \
  scripts/alternative_clumping_evolution_job.pbs
)
if [[ -n "${selected_queue}" ]]; then
  qsub_args=(-q "${selected_queue}" "${qsub_args[@]}")
fi

echo "Submitting ${name}: backend=${BACKEND}, snapshots=${#snapshot_array[@]}, ncpus=${NCPUS}, threads=${THREADS}, mem=${MEM}, walltime=${WALLTIME}, queue=${selected_queue:-default}"
echo "Snapshot list: ${SNAPSHOTS}"
qsub "${qsub_args[@]}"
