# ADR-003: Compatibility facades

Existing import paths, console scripts, result keys, and generated artifacts
remain valid. New domain packages delegate to those implementations; they do
not copy numerical algorithms. Removing a facade requires a separate
migration decision and parity evidence.
