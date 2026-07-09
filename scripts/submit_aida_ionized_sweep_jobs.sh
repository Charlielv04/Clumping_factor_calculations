#!/bin/bash

set -euo pipefail

AIDA_ROOT="${AIDA_ROOT:-/lustre/work/egaraldi/AIDA-TNG}"
SIMULATIONS="${SIMULATIONS:-L35n1080_CDM L35n1080_SIDM1 L35n1080_vSIDM L35n1080_WDM3 L75n910_CDM L75n910_SIDM1 L75n910_vSIDM L75n910_WDM3}"
OUTPUT_DIR="${OUTPUT_DIR:-results}"
CONDA_ENV="${CONDA_ENV:-clumping-factor}"
NCPUS="${NCPUS:-1}"
MEM="${MEM:-8gb}"
WALLTIME="${WALLTIME:-02:00:00}"
QUEUE="${QUEUE:-tiny}"
IONIZED_DENSITY_THRESHOLDS="${IONIZED_DENSITY_THRESHOLDS:-1 5 10 15 20 25}"
IONIZED_CUT_MIN="${IONIZED_CUT_MIN:-0.9}"
IONIZED_CUT_MAX="${IONIZED_CUT_MAX:-0.9999}"
IONIZED_CUT_COUNT="${IONIZED_CUT_COUNT:-200}"
CHUNK_SIZE="${CHUNK_SIZE:-1000000}"
HII_SOURCE="${HII_SOURCE:-auto}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-5}"
VERBOSE="${VERBOSE:-1}"
DRY_RUN="${DRY_RUN:-0}"

project_dir="$(pwd)"
job_script="${project_dir}/scripts/ionized_sweep_job.pbs"
if [[ ! -f "${job_script}" ]]; then
  echo "Run this script from the Clumping_factor_calculations directory." >&2
  exit 1
fi

found=0
submitted=0
for simulation in ${SIMULATIONS}; do
  base_path="${AIDA_ROOT}/${simulation}/output"
  if [[ ! -d "${base_path}" ]]; then
    echo "Skipping missing simulation output: ${base_path}" >&2
    continue
  fi

  while IFS= read -r snapshot_dir; do
    snapshot_name="$(basename "${snapshot_dir}")"
    snapshot="${snapshot_name#snapdir_}"
    if [[ ! "${snapshot}" =~ ^[0-9]+$ ]]; then
      echo "Skipping unrecognized snapshot directory: ${snapshot_dir}" >&2
      continue
    fi
    snapshot_int="$((10#${snapshot}))"
    snapshot_padded="$(printf "%03d" "${snapshot_int}")"
    output="${OUTPUT_DIR}/aida-tng/${simulation}/gas/ionized-sweep/snapshot${snapshot_padded}_nogrid/threads1_batch1_run001.json"
    job_sim="${simulation//[^A-Za-z0-9_]/_}"
    name="cfis_${job_sim}_s${snapshot_padded}"
    ((found += 1))

    echo "AIDA-TNG ${simulation}: snapshot ${snapshot_padded} -> ${output}"
    if [[ "${DRY_RUN}" == "1" ]]; then
      continue
    fi

    mkdir -p "logs/${simulation}" "$(dirname "${output}")"
    env_vars="PROJECT_DIR=${project_dir},CONDA_ENV=${CONDA_ENV},SNAPSHOT=${snapshot_int},BASE_PATH=${base_path},SIMULATION_NAME=${simulation},OUTPUT=${output},IONIZED_DENSITY_THRESHOLDS=${IONIZED_DENSITY_THRESHOLDS},IONIZED_CUT_MIN=${IONIZED_CUT_MIN},IONIZED_CUT_MAX=${IONIZED_CUT_MAX},IONIZED_CUT_COUNT=${IONIZED_CUT_COUNT},CHUNK_SIZE=${CHUNK_SIZE},HII_SOURCE=${HII_SOURCE},PROGRESS_INTERVAL=${PROGRESS_INTERVAL},VERBOSE=${VERBOSE}"
    qsub_args=(
      -N "${name}"
      -o "${project_dir}/logs/${simulation}/${name}.out"
      -e "${project_dir}/logs/${simulation}/${name}.err"
      -l "select=1:ncpus=${NCPUS}:mem=${MEM}"
      -l "walltime=${WALLTIME}"
      -v "${env_vars}"
      "${job_script}"
    )
    if [[ -n "${QUEUE}" && "${QUEUE}" != "default" && "${QUEUE}" != "none" ]]; then
      qsub_args=(-q "${QUEUE}" "${qsub_args[@]}")
    fi
    qsub "${qsub_args[@]}"
    ((submitted += 1))
  done < <(find "${base_path}" -maxdepth 1 -mindepth 1 -type d -name 'snapdir_*' | sort)
done

if (( found == 0 )); then
  echo "No AIDA-TNG snapshots were found under ${AIDA_ROOT}." >&2
  exit 1
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "Dry run complete: found ${found} available simulation/snapshot combinations."
else
  echo "Submission complete: submitted ${submitted} ionized-sweep jobs."
fi
