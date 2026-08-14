# Architecture guardrails

- `scripts/` may launch jobs or call the public command tree, but must not
  implement scientific formulas or define canonical result paths.
- `*_cli.py` modules may parse and route arguments, but numerical algorithms
  belong in services and method implementations.
- Result writers must preserve legacy keys and add normalized method,
  selection, and execution contracts.
- Campaign manifests are deterministic inputs; generated results are never
  rewritten by planning or validation.
