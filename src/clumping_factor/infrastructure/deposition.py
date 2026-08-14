from __future__ import annotations

from typing import Iterable

import numpy as np


def assignment_indices_and_weights(
    coords: np.ndarray, lbox: float, grid_size: int, mas: str,
) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    """Yield periodic flat grid indices and weights for a mass-assignment scheme."""
    scaled = np.mod(np.asarray(coords, dtype=np.float64), float(lbox)) * (int(grid_size) / float(lbox))
    if mas == "CIC":
        base = np.floor(scaled).astype(np.int64)
        fraction = scaled - base
        axis_indices: list[tuple[np.ndarray, ...]] = [(base[:, axis] % grid_size, (base[:, axis] + 1) % grid_size) for axis in range(3)]
        axis_weights: list[tuple[np.ndarray, ...]] = [(1.0 - fraction[:, axis], fraction[:, axis]) for axis in range(3)]
        offsets = range(2)
    elif mas == "TSC":
        center = np.floor(scaled + 0.5).astype(np.int64)
        distance = scaled - center
        axis_indices = [
            ((center[:, axis] - 1) % grid_size, center[:, axis] % grid_size, (center[:, axis] + 1) % grid_size)
            for axis in range(3)
        ]
        axis_weights = []
        for axis in range(3):
            d = distance[:, axis]
            axis_weights.append((0.5 * (0.5 - d) ** 2, 0.75 - d**2, 0.5 * (0.5 + d) ** 2))
        offsets = range(3)
    else:
        raise ValueError("mas must be 'CIC' or 'TSC'.")

    for ox in offsets:
        for oy in offsets:
            for oz in offsets:
                flat_indices = axis_indices[0][ox] * grid_size**2 + axis_indices[1][oy] * grid_size + axis_indices[2][oz]
                weights = axis_weights[0][ox] * axis_weights[1][oy] * axis_weights[2][oz]
                yield flat_indices, weights


def add_deposited_mass(
    target_grid: np.ndarray,
    coords: np.ndarray,
    masses: np.ndarray,
    lbox: float,
    grid_size: int,
    mas: str = "CIC",
) -> None:
    """Deposit masses into a periodic grid in-place using CIC or TSC."""
    if coords.size == 0:
        return
    masses = np.asarray(masses, dtype=target_grid.dtype)
    flat_grid = target_grid.ravel()
    for flat_indices, assignment_weights in assignment_indices_and_weights(coords, lbox, grid_size, mas):
        np.add.at(flat_grid, flat_indices, masses * assignment_weights)
