#!/bin/bash

set -euo pipefail

AIDA_ROOT="${AIDA_ROOT:-/lustre/work/egaraldi/AIDA-TNG}"
SIMULATIONS="${SIMULATIONS:-L35n1080_CDM L35n1080_SIDM1 L35n1080_vSIDM L35n1080_WDM3 L75n910_CDM L75n910_SIDM1 L75n910_vSIDM L75n910_WDM3}"
RESULTS_FAMILY="${RESULTS_FAMILY:-aida-tng}"
PARTICLES="${PARTICLES:-gas}"
BACKENDS="${BACKENDS:-voronoi-transmission}"
# The native-cell backend does not use a Cartesian grid.  The shared PBS
# submitter still expects one grid label for resource and output naming.
GRIDS="${GRIDS:-256}"
SIGMA_BAR_ION_CM2="${SIGMA_BAR_ION_CM2:-}"
SIGMA_BAR_ION_SOURCE="${SIGMA_BAR_ION_SOURCE:-}"
SIGMA_BAR_ION_MODE="${SIGMA_BAR_ION_MODE:-explicit}"
VORONOI_NEIGHBORS="${VORONOI_NEIGHBORS:-32}"
VORONOI_GRADIENT_BATCH_SIZE="${VORONOI_GRADIENT_BATCH_SIZE:-100000}"
DRY_RUN="${DRY_RUN:-0}"

project_dir="$(pwd)"
submitter="${project_dir}/scripts/submit_clumping_jobs.sh"
if [[ ! -f "${submitter}" ]]; then
  echo "Run this script from the Clumping_factor_calculations directory." >&2
  exit 1
fi

if [[ " ${BACKENDS} " == *" voronoi-transmission "* ]]; then
  if [[ " ${PARTICLES} " != *" gas "* || " ${PARTICLES} " == *" dm "* ]]; then
    echo "voronoi-transmission is gas-only; set PARTICLES=gas." >&2
    exit 1
  fi
  if [[ "${SIGMA_BAR_ION_MODE}" == "explicit" ]]; then
    : "${SIGMA_BAR_ION_CM2:?Set SIGMA_BAR_ION_CM2 for voronoi-transmission.}"
  fi
  : "${SIGMA_BAR_ION_SOURCE:?Set SIGMA_BAR_ION_SOURCE for voronoi-transmission.}"
fi

found=0
submitted=0
for simulation in ${SIMULATIONS}; do
  output_dir="${AIDA_ROOT}/${simulation}/output"
  if [[ ! -d "${output_dir}" ]]; then
    echo "Skipping missing simulation output: ${output_dir}" >&2
    continue
  fi

  snapshot_dirs=()
  while IFS= read -r snapshot_dir; do
    snapshot_dirs+=("${snapshot_dir}")
  done < <(find "${output_dir}" -maxdepth 1 -mindepth 1 -type d -name 'snapdir_*' | sort)

  if (( ${#snapshot_dirs[@]} == 0 )); then
    echo "Skipping ${simulation}: no snapdir_* directories found in ${output_dir}" >&2
    continue
  fi

  for snapshot_dir in "${snapshot_dirs[@]}"; do
    snapshot_name="$(basename "${snapshot_dir}")"
    snapshot="${snapshot_name#snapdir_}"
    if [[ ! "${snapshot}" =~ ^[0-9]+$ ]]; then
      echo "Skipping unrecognized snapshot directory: ${snapshot_dir}" >&2
      continue
    fi
    snapshot="$((10#${snapshot}))"
    ((found += 1))
    echo "AIDA-TNG ${simulation}: snapshot ${snapshot} (${snapshot_dir})"

    if [[ "${DRY_RUN}" == "1" ]]; then
      continue
    fi

    BASE_PATH="${output_dir}" \
    SIMULATION_NAME="${simulation}" \
    PARTICLES="${PARTICLES}" \
    BACKENDS="${BACKENDS}" \
    GRIDS="${GRIDS}" \
    SOURCE_CAMPAIGN="AIDA-TNG" \
    RESULTS_LAYOUT="canonical" \
    RESULTS_FAMILY="${RESULTS_FAMILY}" \
    SNAPSHOT="${snapshot}" \
    SIGMA_BAR_ION_CM2="${SIGMA_BAR_ION_CM2}" \
    SIGMA_BAR_ION_SOURCE="${SIGMA_BAR_ION_SOURCE}" \
    SIGMA_BAR_ION_MODE="${SIGMA_BAR_ION_MODE}" \
    VORONOI_NEIGHBORS="${VORONOI_NEIGHBORS}" \
    VORONOI_GRADIENT_BATCH_SIZE="${VORONOI_GRADIENT_BATCH_SIZE}" \
      bash "${submitter}"
    ((submitted += 1))
  done
done

if (( found == 0 )); then
  echo "No AIDA-TNG snapshots were found under ${AIDA_ROOT}." >&2
  exit 1
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "Dry run complete: found ${found} available simulation/snapshot combinations."
else
  echo "Submission complete: processed ${submitted} available simulation/snapshot combinations."
fi
