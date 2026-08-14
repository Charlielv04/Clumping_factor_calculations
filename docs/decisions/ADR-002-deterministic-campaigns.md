# ADR-002: Deterministic campaign planning

Campaign tasks are sorted by stable task id and serialized with sorted JSON
keys. Canonical paths are pure functions of declared task metadata and never
depend on directory enumeration order.
