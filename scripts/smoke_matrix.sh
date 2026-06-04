#!/bin/bash

set -u

BASE_PATH="${BASE_PATH:-../tng100-3/output}"
SNAPSHOT="${SNAPSHOT:-98}"
GRID="${GRID:-64}"
RADIUS_BINS="${RADIUS_BINS:-4}"
THRESHOLD_COUNT="${THRESHOLD_COUNT:-16}"
THREADS="${THREADS:-1}"

mkdir -p results logs

status=0

for particle in gas dm; do
  for backend in sphere cube pylians; do
    name="smoke_${particle}_${backend}_${GRID}"
    output="results/${name}.json"
    log="logs/${name}.log"
    echo "=== ${name} ==="
    if clumping-compute \
      --base-path "${BASE_PATH}" \
      --snapshot "${SNAPSHOT}" \
      --particle-type "${particle}" \
      --backend "${backend}" \
      --grid-size "${GRID}" \
      --radius-bins "${RADIUS_BINS}" \
      --threshold-count "${THRESHOLD_COUNT}" \
      --threads "${THREADS}" \
      --output "${output}" >"${log}" 2>&1; then
      echo "PASS ${name}"
    else
      echo "FAIL ${name}  see ${log}"
      status=1
    fi
  done
done

exit "${status}"

