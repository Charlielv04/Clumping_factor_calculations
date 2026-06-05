import numpy as np
import json
from scipy.ndimage import convolve
import h5py
import illustris_python as il
import matplotlib.pyplot as plt
from time import perf_counter

def load_data(N_grid, N_radius, verbose=False):
    t0 = perf_counter()
    dm_data = il.snapshot.loadSubset(
        './tng100-3/output',
        98,
        1,
        fields=['Coordinates']
    )

    with h5py.File('tng100-3/output/snapdir_098/snap_098.0.hdf5', 'r') as snapfile:
        Lbox = snapfile['Header'].attrs['BoxSize']
        Mass = snapfile['Header'].attrs['MassTable'][1]

    coords = dm_data
    n_particles = dm_data.shape[0]
    mean_spacing = Lbox / n_particles**(1/3)
    radius = np.full(n_particles, mean_spacing)
    mass = np.full(n_particles, Mass)

    cell_size = Lbox /N_grid
    cell_volume = cell_size**3

    if verbose:
        print("Number of particles:", n_particles)
        print("Lbox:", Lbox)
        print("Grid size:", N_grid)
        print("Cell size:", cell_size)
        print("Mean particle spacing:", mean_spacing)
        print("Mean radius in grid cells:", mean_spacing / cell_size)

    particles = (coords, radius, mass)
    params = {
        "Lbox": Lbox,
        "n_particles": n_particles,
        "cell_size": cell_size,
        "cell_volume": cell_volume,
        "mean_spacing": mean_spacing,
        "N_grid": N_grid,
        "N_radius": N_radius,
    }
    print(f"[timing] load_data: {perf_counter() - t0:.3f} s")
    return particles, params


def spherical_tophat_kernel(radius_physical, cell_size):
    radius_cells = radius_physical / cell_size
    kernel_radius = int(np.ceil(radius_cells))

    gx, gy, gz = np.meshgrid(
        np.arange(-kernel_radius, kernel_radius + 1),
        np.arange(-kernel_radius, kernel_radius + 1),
        np.arange(-kernel_radius, kernel_radius + 1),
        indexing="ij"
    )

    r = np.sqrt(gx**2 + gy**2 + gz**2)

    kernel = (r <= radius_cells).astype(np.float32)

    if kernel.sum() == 0:
        kernel[kernel_radius, kernel_radius, kernel_radius] = 1.0

    kernel /= kernel.sum()

    return kernel

def radius_binning(particles, params):
    total_t0 = perf_counter()
    coords, radius, masses = particles

    t0 = perf_counter()
    scale = params['N_grid'] / params['Lbox']
    ix = (coords[:, 0] * scale).astype(np.int64)
    iy = (coords[:, 1] * scale).astype(np.int64)
    iz = (coords[:, 2] * scale).astype(np.int64)

    ix = np.clip(ix, 0, params['N_grid'] - 1)
    iy = np.clip(iy, 0, params['N_grid'] - 1)
    iz = np.clip(iz, 0, params['N_grid'] - 1)

    flat_index = ix * params['N_grid'] ** 2 + iy * params['N_grid'] + iz
    print(f"[timing] particle cell indexing: {perf_counter() - t0:.3f} s")

    t0 = perf_counter()
    r_min = radius.min()
    r_max = radius.max()

    if np.isclose(r_min, r_max):
        radius_group_id = np.zeros(params['n_particles'], dtype=np.int64)
        group_radii = np.array([r_min], dtype=np.float32)
    else:
        # Use logarithmic bins if radii vary significantly
        if r_min > 0 and r_max / r_min > 3:
            radius_edges = np.geomspace(r_min, r_max, params['N_radius'] + 1)
        else:
            radius_edges = np.linspace(r_min, r_max, params['N_radius'] + 1)

        radius_group_id = np.digitize(radius, radius_edges) - 1
        radius_group_id = np.clip(radius_group_id, 0, params['N_radius'] - 1)

        group_radii = np.full(params['N_radius'], np.nan, dtype=np.float32)

        for i in range(params['N_radius']):
            mask = radius_group_id == i
            if np.any(mask):
                group_radii[i] = np.median(radius[mask])
    print(f"[timing] radius grouping: {perf_counter() - t0:.3f} s")

    # Final smoothed mass grid
    t0 = perf_counter()
    smoothed_mass_grid = np.zeros((params['N_grid'], params['N_grid'], params['N_grid']), dtype=np.float32)
    print(f"[timing] allocate smoothed grid: {perf_counter() - t0:.3f} s")

    group_timings = []

    for group_id, group_radius in enumerate(group_radii):
        if not np.isfinite(group_radius):
            continue

        group_t0 = perf_counter()
        t0 = perf_counter()
        group_mask = radius_group_id == group_id
        n_group = np.sum(group_mask)
        mask_time = perf_counter() - t0

        if n_group == 0:
            continue

        print()
        print("Radius group:", group_id)
        print("Particles:", n_group)
        print("Radius:", group_radius)
        print("Radius in grid cells:", group_radius / params['cell_size'])

        t0 = perf_counter()
        group_mass_grid = np.bincount(
            flat_index[group_mask],
            weights=masses[group_mask],
            minlength=params['N_grid'] ** 3
        ).reshape((params['N_grid'], params['N_grid'], params['N_grid'])).astype(np.float32)
        deposit_time = perf_counter() - t0

        t0 = perf_counter()
        kernel = spherical_tophat_kernel(group_radius, params['cell_size'])
        kernel_time = perf_counter() - t0

        print("Kernel shape:", kernel.shape)
        print("Kernel cells:", np.sum(kernel > 0))
        print("Kernel sum:", kernel.sum())

        t0 = perf_counter()
        smoothed_group_mass_grid = convolve(
            group_mass_grid,
            kernel,
            mode="wrap"
        )
        convolve_time = perf_counter() - t0

        t0 = perf_counter()
        smoothed_mass_grid += smoothed_group_mass_grid
        accumulate_time = perf_counter() - t0

        group_total_time = perf_counter() - group_t0
        group_timings.append(
            (group_id, n_group, mask_time, deposit_time, kernel_time, convolve_time, accumulate_time, group_total_time)
        )

        print(f"[timing] group mask: {mask_time:.3f} s")
        print(f"[timing] group deposit: {deposit_time:.3f} s")
        print(f"[timing] group kernel: {kernel_time:.3f} s")
        print(f"[timing] group convolve: {convolve_time:.3f} s")
        print(f"[timing] group accumulate: {accumulate_time:.3f} s")
        print(f"[timing] group total: {group_total_time:.3f} s")

    # Convert mass per grid cell to density
    t0 = perf_counter()
    density_grid = smoothed_mass_grid / params['cell_volume']
    print(f"[timing] density conversion: {perf_counter() - t0:.3f} s")

    # Sanity check: total mass should be conserved approximately
    print()
    print("Input total mass:", np.sum(masses))
    print("Grid total mass:", np.sum(smoothed_mass_grid))
    print("Relative mass error:", (np.sum(smoothed_mass_grid) - np.sum(masses)) / np.sum(masses))

    if group_timings:
        timing_array = np.array(group_timings, dtype=float)
        print()
        print("Timing summary over radius groups")
        print("Total mask time:", timing_array[:, 2].sum())
        print("Total deposit time:", timing_array[:, 3].sum())
        print("Total kernel time:", timing_array[:, 4].sum())
        print("Total convolve time:", timing_array[:, 5].sum())
        print("Total accumulate time:", timing_array[:, 6].sum())
        slowest_group = timing_array[np.argmax(timing_array[:, 7])]
        print(
            "Slowest group:",
            int(slowest_group[0]),
            "particles:",
            int(slowest_group[1]),
            "time:",
            slowest_group[7],
        )

    print(f"[timing] radius_binning total: {perf_counter() - total_t0:.3f} s")

    return density_grid

def clumping_factor(threshold, density_grid):
    overdensity = (density_grid / density_grid.mean() - 1)

    IGM_mask = overdensity < threshold

    if np.sum(IGM_mask) == 0:
        return np.nan

    IGM_densities = density_grid[IGM_mask]

    Clumping_factor = np.mean(IGM_densities ** 2) / np.mean(IGM_densities) ** 2

    return Clumping_factor


def clumping_factor_sweep(thresholds, density_grid):
    rho = density_grid.ravel()
    mean_density = rho.mean()

    if mean_density == 0:
        return np.full_like(thresholds, np.nan, dtype=np.float64)

    t0 = perf_counter()
    overdensity = rho / mean_density - 1
    print(f"[timing] clumping sweep overdensity flatten: {perf_counter() - t0:.3f} s")

    t0 = perf_counter()
    order = np.argsort(overdensity)
    overdensity_sorted = overdensity[order]
    rho_sorted = rho[order].astype(np.float64)
    print(f"[timing] clumping sweep sort: {perf_counter() - t0:.3f} s")

    t0 = perf_counter()
    cumulative_rho = np.cumsum(rho_sorted)
    cumulative_rho2 = np.cumsum(rho_sorted ** 2)
    print(f"[timing] clumping sweep cumulative sums: {perf_counter() - t0:.3f} s")

    t0 = perf_counter()
    thresholds = np.asarray(thresholds)
    indices = np.searchsorted(overdensity_sorted, thresholds, side="left")

    clumping_factors = np.full(thresholds.shape, np.nan, dtype=np.float64)
    valid = indices > 0

    selected_counts = indices[valid]
    mean_rho = cumulative_rho[selected_counts - 1] / selected_counts
    mean_rho2 = cumulative_rho2[selected_counts - 1] / selected_counts

    nonzero = mean_rho > 0
    valid_positions = np.flatnonzero(valid)
    clumping_factors[valid_positions[nonzero]] = mean_rho2[nonzero] / mean_rho[nonzero] ** 2
    print(f"[timing] clumping sweep threshold lookup: {perf_counter() - t0:.3f} s")

    return clumping_factors


def main():
    total_t0 = perf_counter()
    N_grid = 256
    N_radius = 10
    particles, params = load_data(N_grid, N_radius)

    t0 = perf_counter()
    density_grid = radius_binning(particles, params)
    print(f"[timing] density grid build total: {perf_counter() - t0:.3f} s")

    thresholds = np.linspace(-1, 25, 200)

    t0 = perf_counter()
    clumping_factors = clumping_factor_sweep(thresholds, density_grid)
    print(f"[timing] clumping threshold sweep: {perf_counter() - t0:.3f} s")

    with open(f"Clumping_factor_dm_{N_grid}.json", "w", encoding="utf-8") as f:
        json.dump({
            "source_script": "DM_clumping_factor_radius.py",
            "N_grid": N_grid,
            "N_radius": N_radius,
            "thresholds": thresholds.tolist(),
            "clumping_factors": [None if not np.isfinite(v) else float(v) for v in clumping_factors],
        }, f, indent=2)

    t0 = perf_counter()
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(thresholds, clumping_factors, label="Clumping factor")

    ax.grid(True)

    fig.savefig(f"Clumping_factor_dm_{N_grid}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[timing] plot save: {perf_counter() - t0:.3f} s")
    print(f"[timing] main total: {perf_counter() - total_t0:.3f} s")


if __name__ == "__main__":
    main()
