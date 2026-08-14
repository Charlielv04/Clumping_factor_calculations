# Results Consolidation Preflight

This checksum inventory covers all 4,428 on-disk tracked result files
(1,348,644,705 bytes), including eight long-path analysis artifacts. Every
source has a SHA-256 checksum, explicit action, and destination in the CSV.

- 1,493 canonical science moves and one `Thesan-1-parallelized` run moved to
  the same science identity as deterministic `run002`.
- 32 active forest spectra become strict owner companions; two archived
  pre-pull spectra remain archive material. Six ionizing products become
  strict forest runs, with conflicting MFP/Gamma products deterministically
  assigned additional runs.
- 1,794 active derived artifacts receive hashed analysis destinations. Their
  companion specification table records the exact hash inputs and semantic
  domain/family/kind/subject/method labels: `aida-tng`, `thesan`, or
  `combined` families; named evolution, comparison, ionization, equations,
  IGM, performance, benchmark, and campaign categories; and only three
  genuinely shallow AIDA summary tables remain `legacy-analysis`. 582
  historical items receive archive destinations; 113 caches are deleted; the
  reference temperature table relocates; `results/README.md` is retained.
- 284 duplicate groups yield 405 alias removals and 266,392,279 reclaimable
  bytes. There are zero non-identical destination collisions and zero
  unresolved classifications.

No artifact has moved, been deleted, or been rewritten during preflight.
