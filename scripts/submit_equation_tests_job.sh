#!/bin/bash

set -euo pipefail

SNAPSHOT="${SNAPSHOT:?Set SNAPSHOT to a single snapshot number, for example 80.}"
BASE_PATH="${BASE_PATH:-../Thesan-2/output}"
MFP_FILE="${MFP_FILE:-}"
COMPUTE_MISSING_IONIZING="${COMPUTE_MISSING_IONIZING:-0}"
MFP_LOS_FILE="${MFP_LOS_FILE:-}"
ALLOW_MFP_LOS_REDSHIFT_MISMATCH="${ALLOW_MFP_LOS_REDSHIFT_MISMATCH:-0}"
MFP_STARTS_PER_RAY="${MFP_STARTS_PER_RAY:-100}"
MFP_SEED="${MFP_SEED:-0}"
GAMMA_HI_THRESHOLD="${GAMMA_HI_THRESHOLD:-0.5}"
REFRESH_IONIZING_CACHE="${REFRESH_IONIZING_CACHE:-0}"
ALLOW_LEGACY_IONIZING_TABLE="${ALLOW_LEGACY_IONIZING_TABLE:-0}"
SIGMA_HI_CM2="${SIGMA_HI_CM2:-6.3e-18}"
SIMULATION_NAME="${SIMULATION_NAME:-}"
OUTPUT_DIR="${OUTPUT_DIR:-results}"
OUTPUT="${OUTPUT:-}"
CONDA_ENV="${CONDA_ENV:-clumping-factor}"
NCPUS="${NCPUS:-1}"
THREADS="${THREADS:-${NCPUS}}"
MEM="${MEM:-8gb}"
WALLTIME="${WALLTIME:-02:00:00}"
QUEUE="${QUEUE:-auto}"
GAMMA_HI_FILE="${GAMMA_HI_FILE:-}"
TEMPERATURE_FILE="${TEMPERATURE_FILE:-}"
REDUCED_SPEED_OF_LIGHT_FRACTION="${REDUCED_SPEED_OF_LIGHT_FRACTION:-0.2}"
C_TILDE_CM_S="${C_TILDE_CM_S:-}"
PHOTON_GROUPS="${PHOTON_GROUPS:-0}"
PHOTON_GROUP_TESTS="${PHOTON_GROUP_TESTS:-0 1 2 0+1 1+2 0+1+2}"
THRESHOLDS="${THRESHOLDS:-}"
THRESHOLD_MIN="${THRESHOLD_MIN:--1}"
THRESHOLD_MAX="${THRESHOLD_MAX:-25}"
THRESHOLD_COUNT="${THRESHOLD_COUNT:-200}"
IONIZED_DENSITY_THRESHOLDS="${IONIZED_DENSITY_THRESHOLDS:-1 5 10 15 20 25}"
IONIZED_CUTS="${IONIZED_CUTS:-}"
IONIZED_SWEEP="${IONIZED_SWEEP:-1}"
IONIZED_CUT_MIN="${IONIZED_CUT_MIN:-0.9}"
IONIZED_CUT_MAX="${IONIZED_CUT_MAX:-0.9999}"
IONIZED_CUT_COUNT="${IONIZED_CUT_COUNT:-200}"
CHUNK_SIZE="${CHUNK_SIZE:-1000000}"
HYDROGEN_MASS_FRACTION="${HYDROGEN_MASS_FRACTION:-0.76}"
CHI_E="${CHI_E:-1.08}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-5}"
VERBOSE="${VERBOSE:-1}"

if [[ -z "${SIMULATION_NAME}" ]]; then
  base_trimmed="${BASE_PATH%/}"
  SIMULATION_NAME="$(basename "${base_trimmed}")"
  if [[ "${SIMULATION_NAME}" == "output" ]]; then
    SIMULATION_NAME="$(basename "$(dirname "${base_trimmed}")")"
  fi
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

project_dir="$(pwd)"
job_simulation_name="${SIMULATION_NAME//[^A-Za-z0-9_]/_}"
snapshot_padded="$(printf "%03d" "${SNAPSHOT}")"
name="cfeq_${job_simulation_name}_s${snapshot_padded}"

mkdir -p "logs/${SIMULATION_NAME}" "${OUTPUT_DIR}"

read -r -a photon_group_array <<< "${PHOTON_GROUPS}"
photon_groups_csv="$(IFS=:; echo "${photon_group_array[*]}")"
photon_group_tests_csv=""
if [[ -n "${PHOTON_GROUP_TESTS}" ]]; then
  read -r -a photon_group_test_array <<< "${PHOTON_GROUP_TESTS}"
  photon_group_tests_csv="$(IFS=:; echo "${photon_group_test_array[*]}")"
fi
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
ionized_density_thresholds_csv=""
if [[ -n "${IONIZED_DENSITY_THRESHOLDS}" ]]; then
  read -r -a ionized_density_threshold_array \
    <<< "${IONIZED_DENSITY_THRESHOLDS}"
  ionized_density_thresholds_csv="$(
    IFS=:
    echo "${ionized_density_threshold_array[*]}"
  )"
fi

env_vars="PROJECT_DIR=${project_dir},SNAPSHOT=${SNAPSHOT},BASE_PATH=${BASE_PATH},MFP_FILE=${MFP_FILE},COMPUTE_MISSING_IONIZING=${COMPUTE_MISSING_IONIZING},MFP_LOS_FILE=${MFP_LOS_FILE},ALLOW_MFP_LOS_REDSHIFT_MISMATCH=${ALLOW_MFP_LOS_REDSHIFT_MISMATCH},MFP_STARTS_PER_RAY=${MFP_STARTS_PER_RAY},MFP_SEED=${MFP_SEED},GAMMA_HI_THRESHOLD=${GAMMA_HI_THRESHOLD},REFRESH_IONIZING_CACHE=${REFRESH_IONIZING_CACHE},ALLOW_LEGACY_IONIZING_TABLE=${ALLOW_LEGACY_IONIZING_TABLE},SIGMA_HI_CM2=${SIGMA_HI_CM2},SIMULATION_NAME=${SIMULATION_NAME},OUTPUT_DIR=${OUTPUT_DIR},OUTPUT=${OUTPUT},CONDA_ENV=${CONDA_ENV},GAMMA_HI_FILE=${GAMMA_HI_FILE},TEMPERATURE_FILE=${TEMPERATURE_FILE},REDUCED_SPEED_OF_LIGHT_FRACTION=${REDUCED_SPEED_OF_LIGHT_FRACTION},C_TILDE_CM_S=${C_TILDE_CM_S},PHOTON_GROUPS_CSV=${photon_groups_csv},PHOTON_GROUP_TESTS_CSV=${photon_group_tests_csv},THRESHOLDS_CSV=${thresholds_csv},THRESHOLD_MIN=${THRESHOLD_MIN},THRESHOLD_MAX=${THRESHOLD_MAX},THRESHOLD_COUNT=${THRESHOLD_COUNT},IONIZED_DENSITY_THRESHOLDS_CSV=${ionized_density_thresholds_csv},IONIZED_CUTS_CSV=${ionized_cuts_csv},IONIZED_SWEEP=${IONIZED_SWEEP},IONIZED_CUT_MIN=${IONIZED_CUT_MIN},IONIZED_CUT_MAX=${IONIZED_CUT_MAX},IONIZED_CUT_COUNT=${IONIZED_CUT_COUNT},CHUNK_SIZE=${CHUNK_SIZE},THREADS=${THREADS},HYDROGEN_MASS_FRACTION=${HYDROGEN_MASS_FRACTION},CHI_E=${CHI_E},PROGRESS_INTERVAL=${PROGRESS_INTERVAL},VERBOSE=${VERBOSE}"

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

echo "Submitting ${name}: snapshot=${SNAPSHOT}, ncpus=${NCPUS}, threads=${THREADS}, mem=${MEM}, walltime=${WALLTIME}, queue=${selected_queue:-default}"
echo "Thresholds: ${THRESHOLDS:-${THRESHOLD_MIN} to ${THRESHOLD_MAX} (${THRESHOLD_COUNT})}"
echo "Ionized cuts: ${IONIZED_CUTS:-none}"
echo "Ionized density thresholds: ${IONIZED_DENSITY_THRESHOLDS}"
echo "Photon group tests: ${PHOTON_GROUP_TESTS:-${PHOTON_GROUPS}}"
if [[ -z "${IONIZED_CUTS}" && "${IONIZED_SWEEP}" == "1" ]]; then
  echo "Ionized sweep: ${IONIZED_CUT_MIN} to ${IONIZED_CUT_MAX} (${IONIZED_CUT_COUNT})"
fi
qsub "${qsub_args[@]}"
