"""Independent scalar reference calculations used for scientific cross-checks."""

from .ionizing import calculate_mean_free_paths_reference, gamma_hi_from_snapshot_files_reference

__all__ = ["calculate_mean_free_paths_reference", "gamma_hi_from_snapshot_files_reference"]
