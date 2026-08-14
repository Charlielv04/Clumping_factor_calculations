# ADR-001: Keep numerical kernels out of adapters and shell

CLI adapters translate command names to services. PBS workers execute a
manifest command. Neither layer chooses thresholds, estimators, masks, or
output paths. Those values come from typed configuration, the method registry,
or an explicit campaign file.
