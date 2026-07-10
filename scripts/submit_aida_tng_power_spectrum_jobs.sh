#!/bin/bash

set -euo pipefail

AIDA_ROOT="${AIDA_ROOT:-/lustre/work/egaraldi/AIDA-TNG}"
SIMULATIONS="${SIMULATIONS:-L35n1080_CDM L35n1080_SIDM1 L35n1080_vSIDM L35n1080_WDM3 L75n910_CDM L75n910_SIDM1 L75n910_vSIDM L75n910_WDM3}"
PARTICLES="${PARTICLES:-dm}"
GRIDS="${GRIDS:-256}"
SMOOTHINGS="${SMOOTHINGS:-none pylians}"
SPECTRUM_ENGINES="${SPECTRUM_ENGINES:-both}"
REPETITIONS="${REPETITIONS:-1}"
CONDA_ENV="${CONDA_ENV:-clumping-factor}"
NCPUS="${NCPUS:-2}"
THREADS="${THREADS:-${NCPUS}}"
MEM="${MEM:-16gb}"
WALLTIME="${WALLTIME:-02:00:00}"
QUEUE="${QUEUE:-auto}"
MAX_CONCURRENT="${MAX_CONCURRENT:-}"
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
PYLIANS_AXIS="${PYLIANS_AXIS:-0}"
RESULTS_LAYOUT="${RESULTS_LAYOUT:-canonical}"
RESULTS_FAMILY="${RESULTS_FAMILY:-aida-tng}"
VERBOSE="${VERBOSE:-1}"
DRY_RUN="${DRY_RUN:-0}"

project_dir="$(pwd)"
job_script="${project_dir}/scripts/power_spectrum_array_job.pbs"
if [[ ! -f "${job_script}" ]]; then
  echo "Run this script from the Clumping_factor_calculations directory." >&2
  exit 1
fi

found=0
task_count=0
manifest_dir="${project_dir}/logs/aida-tng-power-spectrum"
mkdir -p "${manifest_dir}"
timestamp="$(date +%Y%m%d_%H%M%S)"
manifest="${manifest_dir}/tasks_${timestamp}.tsv"
: > "${manifest}"

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
    found=$((found + 1))
    echo "AIDA-TNG power spectrum ${simulation}: snapshot ${snapshot} (${snapshot_dir})"

    for run_label in $(seq 1 "${REPETITIONS}"); do
      for grid in ${GRIDS}; do
        for particle in ${PARTICLES}; do
          for smoothing in ${SMOOTHINGS}; do
            for engine in ${SPECTRUM_ENGINES}; do
              printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                "${output_dir}" "${simulation}" "${snapshot}" "${particle}" "${grid}" "${smoothing}" "${engine}" "${run_label}" \
                >> "${manifest}"
              task_count=$((task_count + 1))
            done
          done
        done
      done
    done
  done
done

if (( found == 0 )); then
  echo "No AIDA-TNG snapshots were found under ${AIDA_ROOT}." >&2
  exit 1
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "Dry run complete: found ${found} available simulation/snapshot combinations and would submit ${task_count} array tasks."
  echo "Task manifest: ${manifest}"
  exit 0
fi

if (( task_count == 0 )); then
  echo "No power-spectrum tasks were generated." >&2
  exit 1
fi

last_index=$((task_count - 1))
job_name="pk_aida_tng"
selected_queue=""
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

array_range="0-${last_index}"
if [[ -n "${MAX_CONCURRENT}" ]]; then
  echo "Note: this submitter uses PBS -J ${array_range}. If this cluster supports -J X-Y%N, set it manually or adapt array_range; MAX_CONCURRENT=${MAX_CONCURRENT} is recorded but not applied." >&2
fi

env_vars="PROJECT_DIR=${project_dir},TASK_MANIFEST=${manifest},CONDA_ENV=${CONDA_ENV},RESULTS_LAYOUT=${RESULTS_LAYOUT},RESULTS_FAMILY=${RESULTS_FAMILY},RADIUS_BINS=${RADIUS_BINS},RADIUS_BIN_BATCH_SIZE=${RADIUS_BIN_BATCH_SIZE},RADIUS_MODE=${RADIUS_MODE},FILTER_TYPE=${FILTER_TYPE},MAS=${MAS},THREADS=${THREADS},NCPUS=${NCPUS},RESOURCE_SIZE=aida-array,LOAD_MODE=${LOAD_MODE},CHUNK_SIZE=${CHUNK_SIZE},MAX_FULL_LOAD_GB=${MAX_FULL_LOAD_GB},BIN_COUNT=${BIN_COUNT},BINNING=${BINNING},K_MIN=${K_MIN},K_MAX=${K_MAX},PYLIANS_AXIS=${PYLIANS_AXIS},VERBOSE=${VERBOSE}"

qsub_args=(
  -N "${job_name}"
  -J "${array_range}"
  -o "${manifest_dir}/${job_name}.out"
  -e "${manifest_dir}/${job_name}.err"
  -l "select=1:ncpus=${NCPUS}:mem=${MEM}"
  -l "walltime=${WALLTIME}"
  -v "${env_vars}"
  "${job_script}"
)
if [[ -n "${selected_queue}" ]]; then
  qsub_args=(-q "${selected_queue}" "${qsub_args[@]}")
fi

echo "Submitting ${job_name} as one PBS array job with ${task_count} tasks."
echo "Manifest: ${manifest}"
echo "Queue: ${selected_queue:-default}; resources: ncpus=${NCPUS}, mem=${MEM}, walltime=${WALLTIME}"
qsub "${qsub_args[@]}"
