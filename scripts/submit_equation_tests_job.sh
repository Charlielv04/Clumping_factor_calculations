#!/bin/bash

set -euo pipefail

SNAPSHOT="${SNAPSHOT:?Set SNAPSHOT to a single snapshot number, for example 80.}"
BASE_PATH="${BASE_PATH:-../Thesan-2/output}"
MFP_FILE="${MFP_FILE:?Set MFP_FILE to the THESAN mean-free-path table.}"
SIGMA_HI_CM2="${SIGMA_HI_CM2:-6.3e-18}"
SIMULATION_NAME="${SIMULATION_NAME:-}"
OUTPUT_DIR="${OUTPUT_DIR:-results}"
OUTPUT="${OUTPUT:-}"
CONDA_ENV="${CONDA_ENV:-clumping-factor}"
NCPUS="${NCPUS:-1}"
MEM="${MEM:-8gb}"
WALLTIME="${WALLTIME:-02:00:00}"
QUEUE="${QUEUE:-auto}"
GAMMA_HI_FILE="${GAMMA_HI_FILE:-}"
TEMPERATURE_FILE="${TEMPERATURE_FILE:-}"
REDUCED_SPEED_OF_LIGHT_FRACTION="${REDUCED_SPEED_OF_LIGHT_FRACTION:-0.2}"
C_TILDE_CM_S="${C_TILDE_CM_S:-}"
PHOTON_GROUPS="${PHOTON_GROUPS:-0}"
THRESHOLDS="${THRESHOLDS:-}"
THRESHOLD_MIN="${THRESHOLD_MIN:--1}"
THRESHOLD_MAX="${THRESHOLD_MAX:-25}"
THRESHOLD_COUNT="${THRESHOLD_COUNT:-200}"
IONIZED_CUTS="${IONIZED_CUTS:-}"
CHUNK_SIZE="${CHUNK_SIZE:-1000000}"
HYDROGEN_MASS_FRACTION="${HYDROGEN_MASS_FRACTION:-0.76}"
CHI_E="${CHI_E:-1.08}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-5}"
VERBOSE="${VERBOSE:-1}"

if (( NCPUS != 1 )); then
  echo "clumping-equation-tests is a one-pass streaming job; use NCPUS=1." >&2
  exit 1
fi

if [[ -z "${SIMULATION_NAME}" ]]; then
  base_trimmed="${BASE_PATH%/}"
  SIMULATION_NAME="$(basename "${base_trimmed}")"
  if [[ "${SIMULATION_NAME}" == "output" ]]; then
    SIMULATION_NAME="$(basename "$(dirname "${base_trimmed}")")"
  fi
fi

case "${QUEUE}" in
  auto)
    selected_queue="tiny"
    ;;
  default|none)
    selected_queue=""
    ;;
  *)
    selected_queue="${QUEUE}"
    ;;
esac

project_dir="$(pwd)"
job_simulation_name="${SIMULATION_NAME//[^A-Za-z0-9_]/_}"
snapshot_padded="$(printf "%03d" "${SNAPSHOT}")"
name="cfeq_${job_simulation_name}_s${snapshot_padded}"

mkdir -p "logs/${SIMULATION_NAME}" "${OUTPUT_DIR}"

read -r -a photon_group_array <<< "${PHOTON_GROUPS}"
photon_groups_csv="$(IFS=:; echo "${photon_group_array[*]}")"
thresholds_csv=""
if [[ -n "${THRESHOLDS}" ]]; then
  read -r -a threshold_array <<< "${THRESHOLDS}"
  thresholds_csv="$(IFS=:; echo "${threshold_array[*]}")"
fi
ionized_cuts_csv=""
if [[ -n "${IONIZED_CUTS}" ]]; then
  read -r -a ionized_cut_array <<< "${IONIZED_CUTS}"
  ionized_cuts_csv="$(IFS=:; echo "${ionized_cut_array[*]}")"
fi

env_vars="PROJECT_DIR=${project_dir},SNAPSHOT=${SNAPSHOT},BASE_PATH=${BASE_PATH},MFP_FILE=${MFP_FILE},SIGMA_HI_CM2=${SIGMA_HI_CM2},SIMULATION_NAME=${SIMULATION_NAME},OUTPUT_DIR=${OUTPUT_DIR},OUTPUT=${OUTPUT},CONDA_ENV=${CONDA_ENV},GAMMA_HI_FILE=${GAMMA_HI_FILE},TEMPERATURE_FILE=${TEMPERATURE_FILE},REDUCED_SPEED_OF_LIGHT_FRACTION=${REDUCED_SPEED_OF_LIGHT_FRACTION},C_TILDE_CM_S=${C_TILDE_CM_S},PHOTON_GROUPS_CSV=${photon_groups_csv},THRESHOLDS_CSV=${thresholds_csv},THRESHOLD_MIN=${THRESHOLD_MIN},THRESHOLD_MAX=${THRESHOLD_MAX},THRESHOLD_COUNT=${THRESHOLD_COUNT},IONIZED_CUTS_CSV=${ionized_cuts_csv},CHUNK_SIZE=${CHUNK_SIZE},HYDROGEN_MASS_FRACTION=${HYDROGEN_MASS_FRACTION},CHI_E=${CHI_E},PROGRESS_INTERVAL=${PROGRESS_INTERVAL},VERBOSE=${VERBOSE}"

qsub_args=(
  -N "${name}"
  -o "${project_dir}/logs/${SIMULATION_NAME}/${name}.out"
  -e "${project_dir}/logs/${SIMULATION_NAME}/${name}.err"
  -l "select=1:ncpus=${NCPUS}:mem=${MEM}"
  -l "walltime=${WALLTIME}"
  -v "${env_vars}"
  scripts/equation_tests_job.pbs
)
if [[ -n "${selected_queue}" ]]; then
  qsub_args=(-q "${selected_queue}" "${qsub_args[@]}")
fi

echo "Submitting ${name}: snapshot=${SNAPSHOT}, ncpus=${NCPUS}, mem=${MEM}, walltime=${WALLTIME}, queue=${selected_queue:-default}"
echo "Thresholds: ${THRESHOLDS:-${THRESHOLD_MIN} to ${THRESHOLD_MAX} (${THRESHOLD_COUNT})}"
echo "Ionized cuts: ${IONIZED_CUTS:-none}"
qsub "${qsub_args[@]}"
