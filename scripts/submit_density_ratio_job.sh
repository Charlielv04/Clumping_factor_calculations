#!/bin/bash

set -euo pipefail

BASE_PATH="${BASE_PATH:?Set BASE_PATH, for example /lustre/work/carlos.lopez/Thesan-2/output.}"
SNAPSHOT="${SNAPSHOT:?Set SNAPSHOT, for example 80.}"
SIMULATION_NAME="${SIMULATION_NAME:-Thesan-2}"
OUTPUT_DIR="${OUTPUT_DIR:-/lustre/work/carlos.lopez/Thesan-2/results/forest_parallelized}"
CONDA_ENV="${CONDA_ENV:-clumping-factor}"
NCPUS="${NCPUS:-1}"
MEM="${MEM:-16gb}"
WALLTIME="${WALLTIME:-08:00:00}"
QUEUE="${QUEUE:-tiny}"
THRESHOLD_MIN="${THRESHOLD_MIN:--1}"
THRESHOLD_MAX="${THRESHOLD_MAX:-25}"
THRESHOLD_COUNT="${THRESHOLD_COUNT:-200}"
CHUNK_SIZE="${CHUNK_SIZE:-1000000}"
HYDROGEN_MASS_FRACTION="${HYDROGEN_MASS_FRACTION:-0.76}"

project_dir="$(pwd)"
mkdir -p "${project_dir}/logs/${SIMULATION_NAME}"
snapshot_padded="$(printf '%03d' "${SNAPSHOT}")"
name="cfdr_${SIMULATION_NAME//[^A-Za-z0-9_]/_}_s${snapshot_padded}"

env_vars="PROJECT_DIR=${project_dir},BASE_PATH=${BASE_PATH},SNAPSHOT=${SNAPSHOT},SIMULATION_NAME=${SIMULATION_NAME},OUTPUT_DIR=${OUTPUT_DIR},CONDA_ENV=${CONDA_ENV},THRESHOLD_MIN=${THRESHOLD_MIN},THRESHOLD_MAX=${THRESHOLD_MAX},THRESHOLD_COUNT=${THRESHOLD_COUNT},CHUNK_SIZE=${CHUNK_SIZE},HYDROGEN_MASS_FRACTION=${HYDROGEN_MASS_FRACTION}"

qsub \
  -q "${QUEUE}" \
  -N "${name}" \
  -o "${project_dir}/logs/${SIMULATION_NAME}/${name}.out" \
  -e "${project_dir}/logs/${SIMULATION_NAME}/${name}.err" \
  -l "select=1:ncpus=${NCPUS}:mem=${MEM}" \
  -l "walltime=${WALLTIME}" \
  -v "${env_vars}" \
  scripts/density_ratio_job.pbs
