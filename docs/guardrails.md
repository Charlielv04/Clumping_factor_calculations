# Architecture guardrails

- The source root contains only `command.py`, `__init__.py`, and packages.
- `command.py` routes commands and does not import scientific libraries.
- CLI adapters parse arguments and call typed services; service APIs do not
  accept `argparse.Namespace`.
- Scientific implementations live in exactly one domain package.
- Method identifiers and executable capabilities are declared only by the
  method registry.
- Result readers accept strict schema 2 only; method identity is never guessed.
- Canonical output paths are derived only by infrastructure result-path code.
- Campaigns are typed matrices; explicit command tasks are rejected.
- No `scripts/`, compatibility modules, legacy source, or executable comparative
  implementations may be added.
- Generated logs, caches, bytecode, and backups remain ignored.
