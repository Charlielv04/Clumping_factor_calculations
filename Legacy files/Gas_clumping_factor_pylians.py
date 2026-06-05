import numpy as np
import json
import h5py
import illustris_python as il
import matplotlib.pyplot as plt
from time import perf_counter

try:
    import MAS_library as MASL
    import smoothing_library as SL
except ImportError as exc:
    MASL = None
    SL = None
    PYLIANS_IMPORT_ERROR = exc
else:
    PYLIANS_IMPORT_ERROR = None


def require_pylians():
    if MASL is None or SL is None:
        raise ImportError(
            "Pylians is required for this script. Install it with "
            "`python -m pip install Pylians`, then rerun this file."
        ) from PYLIANS_IMPORT_ERROR


def load_data(N_grid, N_radius, verbose=False):
    t0 = perf_counter()
    gas_data = il.snapshot.loadSubset(
        './tng100-3/output',
        98,
        0,
        fields=['Coordinates', 'Density', 'Masses']
    )

    with h5py.File('tng100-3/output/snapdir_098/snap_098.0.hdf5', 'r') as snapfile:
        Lbox = snapfile['Header'].attrs['BoxSize']

    coords = np.asarray(gas_data['Coordinates'], dtype=np.float32)
    density = np.asarray(gas_data['Density'], dtype=np.float32)
    masses = np.asarray(gas_data['Masses'], dtype=np.float32)

    valid = np.isfinite(density) & np.isfinite(masses) & (density > 0) & (masses > 0)
    coords = np.ascontiguousarray(coords[valid], dtype=np.float32)
    masses = np.ascontiguousarray(masses[valid], dtype=np.float32)
    density = density[valid]

    cell_volume_gas = masses / density
    radius = np.asarray((3.0 * cell_volume_gas / (4.0 * np.pi)) ** (1.0 / 3.0), dtype=np.float32)

    n_particles = coords.shape[0]
    cell_size = Lbox / N_grid
    cell_volume = cell_size**3

    if verbose:
        print("Number of gas cells:", n_particles)
        print("Lbox:", Lbox)
        print("Grid size:", N_grid)
        print("Grid cell size:", cell_size)
        print("Median gas radius:", np.median(radius))
        print("Median gas radius in grid cells:", np.median(radius) / cell_size)

    particles = (coords, radius, masses)
    params = {
        "Lbox": np.float32(Lbox),
        "n_particles": n_particles,
        "cell_size": cell_size,
        "cell_volume": cell_volume,
        "N_grid": N_grid,
        "N_radius": N_radius,
    }
    print(f"[timing] load_data: {perf_counter() - t0:.3f} s")
    return particles, params


def make_radius_groups(radius, params):
    t0 = perf_counter()
    r_min = radius.min()
    r_max = radius.max()

    if np.isclose(r_min, r_max):
        radius_group_id = np.zeros(params['n_particles'], dtype=np.int64)
        group_radii = np.array([r_min], dtype=np.float32)
    else:
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
    return radius_group_id, group_radii


def assign_mass_to_grid(group_pos, group_mass_grid, group_masses, lbox, mas):
    try:
        MASL.MA(group_pos, group_mass_grid, lbox, mas, W=group_masses, verbose=False)
    except TypeError:
        MASL.MA(group_pos, group_mass_grid, lbox, mas, verbose=False)
        if not np.allclose(group_masses, group_masses[0]):
            raise TypeError(
                "This Pylians MAS_library.MA build does not accept W= weights, "
                "but gas masses are not constant."
            )
        group_mass_grid *= group_masses[0]


def build_density_grid_pylians(particles, params, mas='CIC', filter_type='Top-Hat', threads=1):
    require_pylians()

    total_t0 = perf_counter()
    coords, radius, masses = particles
    radius_group_id, group_radii = make_radius_groups(radius, params)

    t0 = perf_counter()
    smoothed_mass_grid = np.zeros(
        (params['N_grid'], params['N_grid'], params['N_grid']),
        dtype=np.float32,
    )
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
        print("Gas cells:", n_group)
        print("Radius:", group_radius)
        print("Radius in grid cells:", group_radius / params['cell_size'])

        t0 = perf_counter()
        if n_group == params['n_particles']:
            group_pos = coords
            group_masses = masses
        else:
            group_pos = np.ascontiguousarray(coords[group_mask], dtype=np.float32)
            group_masses = np.ascontiguousarray(masses[group_mask], dtype=np.float32)
        slice_time = perf_counter() - t0

        t0 = perf_counter()
        group_mass_grid = np.zeros_like(smoothed_mass_grid)
        assign_mass_to_grid(group_pos, group_mass_grid, group_masses, params['Lbox'], mas)
        assignment_time = perf_counter() - t0

        t0 = perf_counter()
        W_k = SL.FT_filter(params['Lbox'], float(group_radius), params['N_grid'], filter_type, threads)
        filter_time = perf_counter() - t0

        t0 = perf_counter()
        smoothed_group_mass_grid = SL.field_smoothing(group_mass_grid, W_k, threads).astype(np.float32, copy=False)
        smooth_time = perf_counter() - t0

        t0 = perf_counter()
        smoothed_mass_grid += smoothed_group_mass_grid
        accumulate_time = perf_counter() - t0

        group_total_time = perf_counter() - group_t0
        group_timings.append(
            (
                group_id,
                n_group,
                mask_time,
                slice_time,
                assignment_time,
                filter_time,
                smooth_time,
                accumulate_time,
                group_total_time,
            )
        )

        print(f"[timing] group mask: {mask_time:.3f} s")
        print(f"[timing] group position slice: {slice_time:.3f} s")
        print(f"[timing] group Pylians {mas} assignment: {assignment_time:.3f} s")
        print(f"[timing] group Pylians filter build: {filter_time:.3f} s")
        print(f"[timing] group Pylians smoothing: {smooth_time:.3f} s")
        print(f"[timing] group accumulate: {accumulate_time:.3f} s")
        print(f"[timing] group total: {group_total_time:.3f} s")

    t0 = perf_counter()
    density_grid = smoothed_mass_grid / params['cell_volume']
    print(f"[timing] density conversion: {perf_counter() - t0:.3f} s")

    print()
    input_total_mass = np.sum(masses, dtype=np.float64)
    print("Input total gas mass:", input_total_mass)
    print("Grid total gas mass:", np.sum(smoothed_mass_grid, dtype=np.float64))
    print(
        "Relative mass error:",
        (np.sum(smoothed_mass_grid, dtype=np.float64) - input_total_mass) / input_total_mass,
    )

    if group_timings:
        timing_array = np.array(group_timings, dtype=float)
        print()
        print("Timing summary over radius groups")
        print("Total mask time:", timing_array[:, 2].sum())
        print("Total position slice time:", timing_array[:, 3].sum())
        print("Total Pylians assignment time:", timing_array[:, 4].sum())
        print("Total Pylians filter build time:", timing_array[:, 5].sum())
        print("Total Pylians smoothing time:", timing_array[:, 6].sum())
        print("Total accumulate time:", timing_array[:, 7].sum())

    print(f"[timing] build_density_grid_pylians total: {perf_counter() - total_t0:.3f} s")
    return density_grid


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
    MAS = 'CIC'
    Filter = 'Top-Hat'
    threads = 1

    particles, params = load_data(N_grid, N_radius)

    t0 = perf_counter()
    density_grid = build_density_grid_pylians(
        particles,
        params,
        mas=MAS,
        filter_type=Filter,
        threads=threads,
    )
    print(f"[timing] density grid build total: {perf_counter() - t0:.3f} s")

    thresholds = np.linspace(-1, 25, 200)

    t0 = perf_counter()
    clumping_factors = clumping_factor_sweep(thresholds, density_grid)
    print(f"[timing] clumping threshold sweep: {perf_counter() - t0:.3f} s")

    with open(f"Clumping_factor_gas_pylians_{MAS}_{Filter}_{N_grid}.json", "w", encoding="utf-8") as f:
        json.dump({
            "source_script": "Gas_clumping_factor_pylians.py",
            "N_grid": N_grid,
            "N_radius": N_radius,
            "MAS": MAS,
            "Filter": Filter,
            "threads": threads,
            "thresholds": thresholds.tolist(),
            "clumping_factors": [None if not np.isfinite(v) else float(v) for v in clumping_factors],
        }, f, indent=2)

    t0 = perf_counter()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, clumping_factors, label=f"Gas Pylians {MAS} + {Filter}")
    ax.set_xlabel("Overdensity threshold")
    ax.set_ylabel("Gas clumping factor")
    ax.legend()
    ax.grid(True)

    fig.savefig(f"Clumping_factor_gas_pylians_{MAS}_{Filter}_{N_grid}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[timing] plot save: {perf_counter() - t0:.3f} s")
    print(f"[timing] main total: {perf_counter() - total_t0:.3f} s")


if __name__ == "__main__":
    main()
