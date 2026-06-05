import numpy as np
import json
import h5py
import illustris_python as il
import matplotlib.pyplot as plt

# Load gas data
gas_data = il.snapshot.loadSubset(
    './tng100-3/output',
    98,
    0,
    fields=['Coordinates', 'Density', 'Masses']
)

# Read box size
with h5py.File('tng100-3/output/snapdir_098/snap_098.0.hdf5', 'r') as snapfile:
    Lbox = snapfile['Header'].attrs['BoxSize']

coords = gas_data['Coordinates']
rho = gas_data['Density']
mass = gas_data['Masses']

# Cell volume in code units
cell_volume = mass / rho

# Mean gas density in the box
rho_mean = np.sum(mass) / (Lbox ** 3)

# Overdensity in the usual sense: rho / <rho> - 1
overdensity = rho / rho_mean - 1

# IGM threshold
threshold = 10.0

IGM_mask = overdensity < threshold

# Slice definition
z_min = 0.0
z_max = 3000.0    # much better than 300 for a first try

# Select cells inside the slice
slice_mask = (
    (coords[:, 2] > z_min) &
    (coords[:, 2] < z_max)
)

x = coords[slice_mask, 0]
y = coords[slice_mask, 1]
od = overdensity[slice_mask]
vol = cell_volume[slice_mask]

# IGM cells
igm_mask = od < threshold

# 2D binning
bins = 200
range_xy = [[0, Lbox], [0, Lbox]]

# Background map: gas counts
count_map, xedges, yedges = np.histogram2d(
    x,
    y,
    bins=bins,
    range=range_xy
)

# Total gas volume per pixel
total_volume_map, _, _ = np.histogram2d(
    x,
    y,
    bins=[xedges, yedges],
    weights=vol
)

# IGM gas volume per pixel
igm_volume_map, _, _ = np.histogram2d(
    x[igm_mask],
    y[igm_mask],
    bins=[xedges, yedges],
    weights=vol[igm_mask]
)

# IGM volume fraction in each pixel
igm_volume_fraction = np.divide(
    igm_volume_map,
    total_volume_map,
    out=np.full_like(total_volume_map, np.nan, dtype=float),
    where=total_volume_map > 0
)

# Optional: require at least a few cells per pixel to avoid noisy pixels
min_cells = 2
valid_pixels = count_map >= min_cells

# Define IGM-dominated projected zones
IGM_zone = (igm_volume_fraction == 1) & valid_pixels

# Plot
fig, ax = plt.subplots(figsize=(8, 8))

# Background
ax.imshow(
    np.log10(count_map.T + 1),
    origin='lower',
    cmap='viridis',
    extent=(0, Lbox, 0, Lbox)
)

# Red overlay only where IGM dominates by volume
overlay = np.where(IGM_zone, igm_volume_fraction, np.nan)

ax.imshow(
    overlay.T,
    origin='lower',
    cmap='Reds',
    extent=(0, Lbox, 0, Lbox),
    alpha=0.35,
    vmin=0.5,
    vmax=1.0
)

# White contour around IGM-dominated regions
ax.contour(
    IGM_zone.T.astype(float),
    levels=[0.5],
    origin='lower',
    extent=(0, Lbox, 0, Lbox),
    colors='white',
    linewidths=0.6
)

ax.set_xlim(0, Lbox)
ax.set_ylim(0, Lbox)
ax.set_axis_off()

fig.savefig("gas_slice_IGM_overlay.png", dpi=300, bbox_inches="tight", pad_inches=0)
plt.close(fig)

# Diagnostics
print("Box size =", Lbox)
print("Slice thickness =", z_max - z_min)
print("Slice thickness / box size =", (z_max - z_min) / Lbox)
print("Global IGM fraction by cell count =", np.mean(overdensity < threshold))
print("Global IGM fraction by volume =", np.sum(cell_volume[overdensity < threshold]) / np.sum(cell_volume))

IGM_densities = gas_data['Density'][IGM_mask]

Clumping_factor = np.mean(IGM_densities**2)/np.mean(IGM_densities)**2

print("Clumping factor =", Clumping_factor)

def clumping_factor(threshold):
    IGM_mask = overdensity < threshold

    if np.sum(IGM_mask) == 0:
        return np.nan

    IGM_densities = gas_data['Density'][IGM_mask]

    Clumping_factor = np.mean(IGM_densities ** 2) / np.mean(IGM_densities) ** 2

    return Clumping_factor

x = np.linspace(-1, 25, 200)
# Compute y
y = np.array([clumping_factor(t) for t in x])

with open("Clumping_factor.json", "w", encoding="utf-8") as f:
    json.dump({
        "source_script": "Gas_clumping_factor.py",
        "threshold": threshold,
        "thresholds": x.tolist(),
        "clumping_factors": [None if not np.isfinite(v) else float(v) for v in y],
        "single_threshold_clumping_factor": None if not np.isfinite(Clumping_factor) else float(Clumping_factor),
        "rho_mean": float(rho_mean),
        "overdensity_definition": "Density / rho_mean - 1",
        "global_IGM_fraction_by_cell_count": float(np.mean(overdensity < threshold)),
        "global_IGM_fraction_by_volume": float(np.sum(cell_volume[overdensity < threshold]) / np.sum(cell_volume)),
    }, f, indent=2)

# Plot
fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(x, y)

ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.set_title("Plot of y = f(x)")

ax.grid(True)

fig.savefig("Clumping_factor.png", dpi=300, bbox_inches="tight")
plt.close(fig)
