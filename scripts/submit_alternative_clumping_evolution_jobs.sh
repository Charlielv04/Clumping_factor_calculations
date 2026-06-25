#!/bin/bash

set -euo pipefail

SNAPSHOTS="${SNAPSHOTS:-54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80}"
BASE_PATH="${BASE_PATH:-../Thesan-2/output}"
MFP_FILE="${MFP_FILE:?Set MFP_FILE to the THESAN mean-free-path table, e.g. /lustre/work/carlos.lopez/Thesan-1/mfp_Thesan1.dat.}"
SIMULATION_NAME="${SIMULATION_NAME:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
NCPUS="${NCPUS:-1}"
THREADS="${THREADS:-1}"
MAX_CONCURRENT="${MAX_CONCURRENT:-8}"
MEM="${MEM:-8gb}"
WALLTIME="${WALLTIME:-02:00:00}"
QUEUE="${QUEUE:-tiny}"
CONDA_ENV="${CONDA_ENV:-clumping-factor}"
CHUNK_SIZE="${CHUNK_SIZE:-1000000}"
PHOTON_GROUPS="${PHOTON_GROUPS:-0 1 2}"
HYDROGEN_MASS_FRACTION="${HYDROGEN_MASS_FRACTION:-0.76}"
ALPHA_HII_CM3_S="${ALPHA_HII_CM3_S:-2.59e-13}"
CHI_E="${CHI_E:-1.08}"
CHI_E_SOURCE="${CHI_E_SOURCE:-constant}"
N_H_SOURCE="${N_H_SOURCE:-simulation-volume-mean}"
FULLY_IONIZED="${FULLY_IONIZED:-0}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-5}"
VERBOSE="${VERBOSE:-1}"

if (( NCPUS != 1 || THREADS != 1 )); then
  echo "This helper is intended for tiny one-thread jobs; set NCPUS=1 THREADS=1." >&2
  exit 1
fi

if [[ -z "${SIMULATION_NAME}" ]]; then
  base_trimmed="${BASE_PATH%/}"
  SIMULATION_NAME="$(basename "${base_trimmed}")"
  if [[ "${SIMULATION_NAME}" == "output" ]]; then
    SIMULATION_NAME="$(basename "$(dirname "${base_trimmed}")")"
  fi
fi
if [[ -z "${OUTPUT_DIR}" ]]; then
  OUTPUT_DIR="results/${SIMULATION_NAME}/alternative_clumping"
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
name="cfalt_${job_simulation_name}"

mkdir -p "logs/${SIMULATION_NAME}" "${OUTPUT_DIR}"

qsub \
  -N "${name}" \
  -q "${QUEUE}" \
  -J "0-${last_index}%${MAX_CONCURRENT}" \
  -o "${project_dir}/logs/${SIMULATION_NAME}/${name}.out" \
  -e "${project_dir}/logs/${SIMULATION_NAME}/${name}.err" \
  -l "select=1:ncpus=${NCPUS}:mem=${MEM}" \
  -l "walltime=${WALLTIME}" \
  -v "PROJECT_DIR=${project_dir},SNAPSHOTS_CSV=${snapshots_csv},BASE_PATH=${BASE_PATH},MFP_FILE=${MFP_FILE},SIMULATION_NAME=${SIMULATION_NAME},OUTPUT_DIR=${OUTPUT_DIR},CONDA_ENV=${CONDA_ENV},CHUNK_SIZE=${CHUNK_SIZE},PHOTON_GROUPS_CSV=${photon_groups_csv},HYDROGEN_MASS_FRACTION=${HYDROGEN_MASS_FRACTION},ALPHA_HII_CM3_S=${ALPHA_HII_CM3_S},CHI_E=${CHI_E},CHI_E_SOURCE=${CHI_E_SOURCE},N_H_SOURCE=${N_H_SOURCE},FULLY_IONIZED=${FULLY_IONIZED},PROGRESS_INTERVAL=${PROGRESS_INTERVAL},VERBOSE=${VERBOSE}" \
  scripts/alternative_clumping_evolution_job.pbs
