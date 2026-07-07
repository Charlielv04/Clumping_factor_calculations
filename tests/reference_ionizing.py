"""Independent regression copies of the two supplied notebook calculations.

Keep these scalar and deliberately unoptimized: they are the oracle used to
detect accidental changes in the production implementation.
"""

import numpy as np


def supplied_mfp_from_starts(data, starting_indices, *, X, mH, sigma_912, Mpc):
    samples_per_ray = len(starting_indices) // len(data.rays)
    output = []
    cursor = 0
    for ray in data.rays:
        all_idx = np.arange(len(ray.density_cgs), dtype=int)
        for _ in range(samples_per_ray):
            start = int(starting_indices[cursor])
            cursor += 1
            density = ray.density_cgs.take(all_idx + start, mode="wrap")
            xhi = ray.xHI.take(all_idx + start, mode="wrap")
            segments = ray.segments_cgs.take(all_idx + start, mode="wrap")
            integral = np.cumsum(density * xhi * X / mH * segments * sigma_912)
            index = np.argmin(np.abs(integral - 1.0))
            free_path = segments[:index].sum() + segments[index] * (
                1.0 - integral[index - 1]
            ) / (integral[index] - integral[index - 1])
            output.append(free_path / Mpc * data.hubble_param)
    return np.asarray(output)


def supplied_gamma_from_arrays(photon_density, masses, density, hi_fraction, *, a,
                               unit_length_cm, hubble_param, sigma_c, threshold=0.5):
    numerator = 0.0
    denominator = 0.0
    mask = (np.asarray(hi_fraction) < threshold).astype(int)
    volume = np.asarray(masses) / np.asarray(density)
    conversion = 1e63 / (unit_length_cm * a / hubble_param) ** 3 * a
    for band in range(3):
        numerator += np.sum(
            np.asarray(photon_density)[:, band]
            * conversion
            * sigma_c[band]
            * mask
            * volume
        )
    denominator += np.sum(mask * volume)
    return numerator / denominator
