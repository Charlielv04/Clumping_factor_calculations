# Clumping Factor Calculations

Modular clumping factor tools for TNG gas and dark matter snapshots.

The calculation command writes JSON summaries. Plotting is intentionally separate, so compute runs do not create figures unless requested.

## Install

```bash
python -m pip install -e ".[test]"
```

## Compute

```bash
clumping-compute \
  --base-path ./tng100-3/output \
  --snapshot 98 \
  --particle-type gas \
  --backend sphere \
  --grid-size 256 \
  --radius-bins 10
```

Backends:

- `sphere`: SciPy spherical tophat smoothing
- `cube`: SciPy cube tophat smoothing
- `pylians`: optional Pylians mass assignment and smoothing

Outputs are saved under `results/` unless `--output` is supplied.

## Plot

```bash
clumping-plot results/gas_sphere_snapshot098_grid256.json --output results/gas_sphere.png
```

Multiple JSON files can be plotted together:

```bash
clumping-plot results/*.json --output results/comparison.png
```

