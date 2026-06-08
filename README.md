# Clumping Factor Calculations

Modular clumping factor tools for TNG gas and dark matter snapshots.

The calculation command writes JSON summaries. Plotting is intentionally separate, so compute runs do not create figures unless requested.

## Install

```bash
python -m pip install -e ".[test]"
```

On clusters with old system compilers, install the scientific stack from wheels first:

```bash
python -m pip install --only-binary=:all: -r requirements-core.txt
python -m pip install illustris-python
python -m pip install -e .
```

If Pylians tries to rebuild NumPy inside build isolation, install it after Cython with build isolation disabled:

```bash
python -m pip install "Cython<3"
python -m pip install --no-build-isolation Pylians
```

## Compute

```bash
clumping-compute \
  --base-path ./tng100-3/output \
  --simulation-name tng100-3 \
  --snapshot 98 \
  --particle-type gas \
  --backend sphere \
  --radius-mode sphere \
  --grid-size 256 \
  --radius-bins 10
```

Backends:

- `sphere`: SciPy spherical tophat smoothing
- `cube`: SciPy cube tophat smoothing
- `pylians`: optional Pylians mass assignment and smoothing
- `raw`: raw gas-cell density calculation matching the first legacy gas script; only valid with `--particle-type gas`
- `raw-volume`: raw gas-cell density calculation weighted by each gas cell volume; only valid with `--particle-type gas`

For gridded gas calculations, `--radius-mode sphere` treats each gas cell volume as a sphere and `--radius-mode cube` uses the cube root of the cell volume. The default is `sphere`.

Outputs are saved under `results/<simulation>/` unless `--output` is supplied. The simulation name is inferred from `--base-path` by default, or can be set explicitly with `--simulation-name`.

Thesan-1 snapshot 81 can be run with:

```bash
clumping-compute \
  --base-path ../Thesan-1 \
  --simulation-name Thesan-1 \
  --snapshot 81 \
  --particle-type gas \
  --backend sphere \
  --grid-size 256
```

## Separate IGM Mask And Target Fields

By default, the same density field defines the threshold mask and the clumping factor. To define the IGM mask from one field but measure clumping on another, use the `--mask-*` and `--target-*` options.

Example: select IGM cells using the total matter field, then measure gas clumping there:

```bash
clumping-compute \
  --base-path ../tng100-3/output \
  --simulation-name tng100-3 \
  --snapshot 98 \
  --particle-type gas \
  --backend sphere \
  --target-particle-type gas \
  --target-backend sphere \
  --mask-particle-type both \
  --mask-backend sphere \
  --grid-size 256 \
  --output results/tng100-3/gas_clumping_masked_by_total_sphere_256.json
```

Example: select IGM cells from the DM field, then measure gas clumping:

```bash
clumping-compute \
  --base-path ../tng100-3/output \
  --simulation-name tng100-3 \
  --snapshot 98 \
  --particle-type gas \
  --backend sphere \
  --target-particle-type gas \
  --target-backend sphere \
  --mask-particle-type dm \
  --mask-backend sphere \
  --grid-size 256 \
  --output results/tng100-3/gas_clumping_masked_by_dm_sphere_256.json
```

## Plot

```bash
clumping-plot results/tng100-3/gas_sphere_snapshot098_grid256.json --output results/tng100-3/gas_sphere.png
```

Multiple JSON files can be plotted together:

```bash
clumping-plot results/*/*.json --output results/comparison.png
```
