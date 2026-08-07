#!/bin/bash

set -euo pipefail

THESAN_ROOT="${THESAN_ROOT:-/lustre/work/carlos.lopez}"
SIMULATIONS="${SIMULATIONS:-Thesan-1 Thesan-2}"
RESULTS_FAMILY="${RESULTS_FAMILY:-thesan}"
PARTICLES="${PARTICLES:-gas}"
BACKENDS="${BACKENDS:-voronoi-transmission}"
# The native-cell backend does not use a Cartesian grid. The shared PBS
# submitter still expects one grid label for resource and output naming.
GRIDS="${GRIDS:-256}"
SIGMA_BAR_ION_CM2="${SIGMA_BAR_ION_CM2:-}"
SIGMA_BAR_ION_MODE="${SIGMA_BAR_ION_MODE:-thesan-photon-groups}"
SIGMA_BAR_ION_SOURCE="${SIGMA_BAR_ION_SOURCE:-THESAN_SIGMA_C_CM3_S groups 0-2; volume-weighted PhotonDensity average}"
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
    : "${SIGMA_BAR_ION_CM2:?Set SIGMA_BAR_ION_CM2 when using explicit mode.}"
  fi
  : "${SIGMA_BAR_ION_SOURCE:?Set SIGMA_BAR_ION_SOURCE for voronoi-transmission.}"
fi

found=0
submitted=0
for simulation in ${SIMULATIONS}; do
  output_dir="${THESAN_ROOT}/${simulation}/output"
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
    echo "THESAN ${simulation}: snapshot ${snapshot} (${snapshot_dir})"

    if [[ "${DRY_RUN}" == "1" ]]; then
      continue
    fi

    BASE_PATH="${output_dir}" \
    SIMULATION_NAME="${simulation}" \
    PARTICLES="${PARTICLES}" \
    BACKENDS="${BACKENDS}" \
    GRIDS="${GRIDS}" \
    SOURCE_CAMPAIGN="THESAN" \
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
  echo "No Thesan snapshots were found under ${THESAN_ROOT}." >&2
  exit 1
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "Dry run complete: found ${found} available simulation/snapshot combinations."
else
  echo "Submission complete: processed ${submitted} available simulation/snapshot combinations."
fi
