#!/usr/bin/env bash
# Plan or submit the adaptive-SubfindHsml DM-clumping campaign.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "${script_dir}/.." && pwd)"
campaign="${project_dir}/campaigns/dm-clumping-subfind-hsml.toml"
manifest="${project_dir}/campaigns/dm-clumping-subfind-hsml.manifest.json"

case "${1:---plan}" in
  --plan)
    cd "${project_dir}"
    clumping campaign plan "${campaign}" --manifest "${manifest}"
    ;;
  --submit)
    cd "${project_dir}"
    clumping campaign plan "${campaign}" --manifest "${manifest}"
    clumping campaign submit "${campaign}" --execute
    ;;
  *)
    echo "Usage: $0 [--plan|--submit]" >&2
    exit 2
    ;;
esac
