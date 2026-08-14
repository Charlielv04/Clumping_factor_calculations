# Results

Science results use strict schema 2 paths. Companion run products live beside
their owning JSON, derived analyses live under `analysis/` with a checksum
manifest, and preserved historical material lives under `archive/` and is
excluded from normal discovery.

Validate this tree with `clumping results validate results`. The canonical path
and artifact rules are described in the [repository README](../README.md#strict-schema-2-results);
the completed source-to-destination inventory is in
[`reports/migrations/results-consolidation-final-manifest.csv`](../reports/migrations/results-consolidation-final-manifest.csv).
