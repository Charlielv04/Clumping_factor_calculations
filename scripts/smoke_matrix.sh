#!/bin/bash

set -u

BASE_PATH="${BASE_PATH:-../tng100-3/output}"
SIMULATION_NAME="${SIMULATION_NAME:-}"
SNAPSHOT="${SNAPSHOT:-98}"
GRID="${GRID:-64}"
RADIUS_BINS="${RADIUS_BINS:-4}"
RADIUS_MODE="${RADIUS_MODE:-sphere}"
THRESHOLD_COUNT="${THRESHOLD_COUNT:-16}"
THREADS="${THREADS:-1}"
LOAD_MODE="${LOAD_MODE:-auto}"
CHUNK_SIZE="${CHUNK_SIZE:-1000000}"
MAX_FULL_LOAD_GB="${MAX_FULL_LOAD_GB:-16}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-25}"
VERBOSE="${VERBOSE:-1}"

mkdir -p results logs
if [[ -z "${SIMULATION_NAME}" ]]; then
  base_trimmed="${BASE_PATH%/}"
  SIMULATION_NAME="$(basename "${base_trimmed}")"
  if [[ "${SIMULATION_NAME}" == "output" ]]; then
    SIMULATION_NAME="$(basename "$(dirname "${base_trimmed}")")"
  fi
fi
mkdir -p "results/${SIMULATION_NAME}" "logs/${SIMULATION_NAME}"

status=0

for particle in gas dm; do
  for backend in sphere cube pylians; do
    name="smoke_${particle}_${backend}_${GRID}"
    output="results/${SIMULATION_NAME}/${name}.json"
    log="logs/${SIMULATION_NAME}/${name}.log"
    echo "=== ${name} ==="
    if clumping-compute \
      --base-path "${BASE_PATH}" \
      --simulation-name "${SIMULATION_NAME}" \
      --snapshot "${SNAPSHOT}" \
      --particle-type "${particle}" \
      --backend "${backend}" \
      --radius-mode "${RADIUS_MODE}" \
      --load-mode "${LOAD_MODE}" \
      --chunk-size "${CHUNK_SIZE}" \
      --max-full-load-gb "${MAX_FULL_LOAD_GB}" \
      --progress-interval "${PROGRESS_INTERVAL}" \
      --grid-size "${GRID}" \
      --radius-bins "${RADIUS_BINS}" \
      --threshold-count "${THRESHOLD_COUNT}" \
      --threads "${THREADS}" \
      $([[ "${VERBOSE}" == "1" ]] && printf '%s' "--verbose") \
      --output "${output}" >"${log}" 2>&1; then
      echo "PASS ${name}"
    else
      echo "FAIL ${name}  see ${log}"
      status=1
    fi
  done
done

exit "${status}"
