"""Streaming spatial folding helpers for power-spectrum calculations."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from typing import Any

import numpy as np

from clumping_factor.infrastructure.models import ParticleData


def validate_fold_factor(fold_factor: int | float, box_size: float) -> int:
    """Validate one integer spatial-fold factor and return it as an ``int``."""
    value = int(fold_factor)
    if value != fold_factor or value < 1:
        raise ValueError("fold_factor must be a positive integer.")
    if not np.isfinite(box_size) or box_size <= 0:
        raise ValueError("The periodic box size must be positive and finite.")
    effective_box = float(box_size) / value
    if not np.isfinite(effective_box) or effective_box <= 0:
        raise ValueError("fold_factor is incompatible with the periodic box size.")
    return value


def validate_fold_factors(fold_factors: Iterable[int | float], box_size: float) -> tuple[int, ...]:
    values = tuple(validate_fold_factor(value, box_size) for value in fold_factors)
    if not values:
        raise ValueError("At least one fold factor is required.")
    if len(set(values)) != len(values):
        raise ValueError("fold_factors must not contain duplicates.")
    return values


def folded_box_size(box_size: float, fold_factor: int) -> float:
    factor = validate_fold_factor(fold_factor, box_size)
    return float(box_size) / factor


def fold_coordinates(coords: np.ndarray, box_size: float, fold_factor: int) -> np.ndarray:
    """Remap every particle once into the periodic effective folded box."""
    factor = validate_fold_factor(fold_factor, box_size)
    if factor == 1:
        return np.asarray(coords)
    effective_box = float(box_size) / factor
    folded = np.mod(np.asarray(coords, dtype=np.float64), effective_box)
    if folded.size and (np.any(folded < 0) or np.any(folded >= effective_box)):
        raise ValueError("Folded particle coordinates escaped the effective periodic box.")
    return np.ascontiguousarray(folded, dtype=np.float64)


def fold_particle_data(particles: ParticleData, fold_factor: int) -> ParticleData:
    effective_box = folded_box_size(particles.lbox, fold_factor)
    return ParticleData(
        coords=fold_coordinates(particles.coords, particles.lbox, fold_factor),
        radii=particles.radii,
        masses=particles.masses,
        lbox=effective_box,
        particle_type=particles.particle_type,
        metadata={**particles.metadata, "fold_factor": int(fold_factor), "effective_box_size": effective_box},
    )


def fold_chunk_factory(chunk_factory: Callable[[], Iterable[dict[str, Any]]], fold_factor: int) -> Callable[[], Iterator[dict[str, Any]]]:
    """Wrap a chunk stream, remapping only the current chunk."""
    def factory() -> Iterator[dict[str, Any]]:
        for source in chunk_factory():
            chunk = dict(source)
            source_box = float(chunk["lbox"])
            effective_box = folded_box_size(source_box, fold_factor)
            chunk["coords"] = fold_coordinates(chunk["coords"], source_box, fold_factor)
            chunk["lbox"] = effective_box
            yield chunk
    return factory
