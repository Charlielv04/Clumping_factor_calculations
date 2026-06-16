# Comparison with Oku & Cen (2025), arXiv:2511.09364v1

## Scope

This note compares the repository's current clumping-factor calculations with the transmission-weighted recombination clumping factor in *Recombination Clumping Factor of Physically Defined Intergalactic Medium at the Epoch of Reionization* (Oku & Cen 2025). The proposed changes are post-processing operations on existing snapshots. They do not require running a new hydrodynamic or radiative-transfer simulation.

## What the paper calculates

The paper defines a soft, physically motivated IGM selector

```text
F = exp(-tau_eff)
tau_eff = 0.5 * sigma_bar_ion * n_HI**2 / |grad(n_HI)|
```

where `F` is close to one in gas traversed by external ionizing photons and close to zero in self-shielded gas. The gray hydrogen cross-section `sigma_bar_ion` is averaged over the assumed UV-background spectrum between the HI and HeII ionization edges.

Its recombination clumping factor is

```text
C_rec,F = <alpha(T) n_HII n_e F>_V
          / (<alpha(T)>_V <n_HII>_V <n_e>_V <F>_V)
```

and the normalized quantity used for most comparisons is

```text
C_norm,F = <alpha(T) n_HII n_e F>_V
           / (alpha_B(10^4 K) <n_HII>_V <n_e>_V).
```

This is important: `F` is not simply used to compute an ordinary weighted density variance. In `C_norm,F`, it multiplies the local recombination rate in the numerator, while the denominator contains unweighted full-volume means of `n_HII` and `n_e`. Implementing `<rho^2 F>/<rho F>^2` would therefore not reproduce the paper.

The paper contrasts this with a hard overdensity selector, `theta(Delta_th - Delta)`. It finds the transmission-based result is lower during partial reionization because dense self-shielded cells can pass a density threshold but receive very small `F`.

## What this repository currently calculates

### Gridded backends

The `sphere`, `cube`, and `pylians` paths construct a mass-density grid and calculate

```text
C_rho(Delta_th) = <rho_target^2>_selected / <rho_target>_selected^2,
selected where rho_mask / <rho_mask> - 1 < Delta_th.
```

The mask and target density fields may differ, which is useful infrastructure, but selection remains a binary overdensity cut and the target remains a density field.

### Raw gas backends

The `raw` backend averages equally over gas cells. Because moving-mesh gas cells have unequal volumes, this is not a volume average.

The `raw-volume` backend correctly forms volume-weighted moments of gas mass density:

```text
C_rho,V = [sum(rho^2 V) / sum(V)] / [sum(rho V) / sum(V)]^2.
```

This is the closest current numerical foundation to the paper, but it still lacks ionization state, electron abundance, temperature-dependent recombination, neutral-density gradients, and transmission weighting.

## Main differences

| Aspect | Current repository | Oku & Cen |
|---|---|---|
| Physical quantity | Gas, DM, or total mass-density clumping | Hydrogen recombination-rate clumping |
| IGM definition | Hard overdensity threshold | Soft transmission field `F = exp(-tau_eff)` |
| Local numerator | `rho^2` | `alpha(T) n_HII n_e F` |
| Averaging | Equal grid-cell volume; raw path is cell- or volume-weighted | Volume-weighted |
| Temperature | Not used | Case-B `alpha(T)` |
| Ionization state | Not used | `n_HI`, `n_HII`, and `n_e` |
| Spatial derivative | Not used | `|grad(n_HI)|` |
| Output | Sweep versus overdensity threshold | Evolution versus redshift or neutral fraction |

Consequently, the existing density-only result should not be interpreted as the paper's recombination clumping factor, even when both use a volume average. Agreement can occur only in the restrictive limit of nearly uniform temperature and ionization, `n_HII proportional to rho`, `n_e proportional to rho`, and `F` approximated by a hard density mask.

## Snapshot fields needed

For TNG-like gas snapshots, the minimum useful fields are:

- `Density` and `Masses`, to obtain physical gas density and native cell volume.
- `NeutralHydrogenAbundance`, to obtain `n_HI` and `n_HII`.
- `ElectronAbundance`, to obtain `n_e`.
- `InternalEnergy`, plus composition/electron abundance, to derive temperature.
- `Coordinates`, to construct or sample the spatial `n_HI` field used by the gradient.
- Snapshot scale factor, Hubble parameter, and unit metadata, because `tau_eff` requires physical cgs number densities and a physical spatial gradient.

TNG full snapshots document these gas fields, but `NeutralHydrogenAbundance` is not available in every reduced snapshot and both neutral fraction and temperature require caution for star-forming effective-equation-of-state gas. Other simulations, including THESAN variants, should be detected by inspecting the actual HDF5 field names rather than assuming the TNG schema.

The gray cross-section is not generally a snapshot field. It should be an explicit input or be derived from a documented UV-background spectrum. The chosen value and spectrum must be written to the result metadata.

## Exact mapping for THESAN chemistry fields

THESAN is a particularly suitable input because it evolves radiation and non-equilibrium thermochemistry. A field named `NeutralHydrogenAbundance` is not required if the snapshot provides `HI_Fraction` and `HII_Fraction`.

Before processing a production snapshot, inspect one `PartType0` group and record all dataset names, shapes, units, and header attributes. The implementation should support aliases rather than hard-code one release spelling. The expected mapping is:

```text
x_HI   <- HI_Fraction
x_HII  <- HII_Fraction
x_HeI  <- HeI_Fraction
x_HeII <- HeII_Fraction
x_HeIII<- HeIII_Fraction
x_e    <- ElectronAbundance = n_e / n_H
u      <- InternalEnergy
```

The hydrogen and helium fields are expected to be ion-stage fractions within their respective elements. This convention must be verified numerically on the actual files:

```text
x_HI + x_HII                         approximately 1
x_HeI + x_HeII + x_HeIII             approximately 1
x_e                                  >= 0
```

Small floating-point departures are harmless. Large or systematic departures indicate a different convention, a missing ion stage, or a field that stores a mass fraction rather than an ion-stage fraction.

For each gas cell, first convert the comoving gas density to physical cgs density, `rho_phys`. With hydrogen and helium mass fractions `X` and `Y`, calculate

```text
n_H   = X * rho_phys / m_p
n_He  = Y * rho_phys / (4 * m_p)
n_HI  = x_HI  * n_H
n_HII = x_HII * n_H
n_e   = ElectronAbundance * n_H
```

Use `HII_Fraction` directly. Only use `x_HII = 1 - x_HI` as a fallback after confirming hydrogen closure and documenting that fallback.

The helium fields provide a valuable independent electron-density check:

```text
n_e,species = n_HII + n_He * (x_HeII + 2*x_HeIII)
x_e,species = n_e,species / n_H
```

Compare `x_e,species` with `ElectronAbundance`. If they agree, retain `ElectronAbundance` as the primary field because it follows the simulation's own convention. If they do not agree, stop and diagnose the field definitions rather than choosing one silently. If THESAN tracks only singly ionized helium in the relevant output, use the available stages and state that limitation.

### Temperature from internal energy

For a monatomic gas with `gamma = 5/3`, convert `InternalEnergy` from `(km/s)^2` to `erg/g` and calculate

```text
mu = 4 / (1 + 3*X + 4*X*x_e)
T  = (gamma - 1) * u_cgs * mu * m_p / k_B.
```

Here `mu` is dimensionless in proton-mass units and `x_e = n_e/n_H`. This is the standard TNG/AREPO conversion. The calculation should read `gamma` and abundance assumptions from simulation metadata when available, and otherwise store the adopted values, for example `X = 0.76`, in the result.

### Physical unit conversion

The current repository can use code units for a density-only dimensionless ratio. The optical-depth calculation cannot: `sigma_bar_ion * n_HI**2 / |grad n_HI|` is dimensionless only when number density and length are in consistent physical units.

For the usual TNG/AREPO snapshot convention,

```text
rho_phys = Density
           * (1e10 M_sun / h)
           / (ckpc / h)^3
           * a^-3
```

converted finally to `g cm^-3`. Coordinates and box size become physical lengths through

```text
r_phys = a * r_comoving / h
```

with kpc converted to cm. Gas-cell physical volume can be computed either from physical mass divided by `rho_phys`, or from the comoving `Masses/Density` volume multiplied by `(a/h)^3` with the appropriate kpc units. Verify that the sum of cell volumes agrees with the physical box volume to numerical precision.

## THESAN computation in the current chunked setup

The closest implementation to the paper is a three-pass hybrid calculation. It reuses the repository's HDF5 chunk streaming and periodic grid assignment, while keeping the final averages on native Voronoi-cell volumes.

### Pass 0: inspect and validate the snapshot

Read the header and the first non-empty `PartType0` group. Resolve field aliases and validate the fraction identities above. Record redshift, scale factor, Hubble parameter, box size, abundance assumptions, and the exact snapshot field names. Fail early if `HI_Fraction`, `InternalEnergy`, or a usable electron-density source is absent.

### Pass 1: construct the neutral-hydrogen field

For every gas chunk:

1. Read `Coordinates`, `Density`, `Masses`, and `HI_Fraction`.
2. Convert density and cell volume to physical units.
3. Calculate native-cell `n_HI`.
4. Deposit `n_HI * V_cell` to a periodic uniform-grid numerator.
5. Deposit `V_cell` using the identical assignment stencil to a grid-volume denominator.

After reduction across workers, form

```text
n_HI_grid = deposited(n_HI * V_cell) / deposited(V_cell).
```

Depositing numerator and denominator separately is preferable to treating each cell as a mass particle: it produces a volume-weighted neutral number density and does not let the numerous small dense cells dominate merely by cell count. The same CIC or TSC stencil must be used for both arrays. Grid cells with negligible deposited volume require an explicit fill or smoothing policy, although a well-sampled full cosmological box should have few at practical resolution.

### Grid operation: derive transmission

Convert the grid spacing to physical cm and use periodic centered differences:

```text
dn_dx = (roll(n_HI, -1, x) - roll(n_HI, +1, x)) / (2*dx_phys)
dn_dy = ...
dn_dz = ...
grad_n_HI = sqrt(dn_dx**2 + dn_dy**2 + dn_dz**2)
```

Then calculate

```text
tau_eff = 0.5 * sigma_bar_ion * n_HI_grid**2 / grad_n_HI
F_grid  = exp(-tau_eff).
```

Handle limiting cases physically:

- `n_HI = 0` and zero gradient: set `tau_eff = 0`, hence `F = 1`.
- `n_HI > 0` and zero gradient: set `tau_eff = infinity`, hence `F = 0`.
- Clip finite `tau_eff` to a range such as `[0, 700]` before exponentiation.

Do not add a tunable gradient floor without reporting it: a floor can artificially make uniform neutral regions transmissive. A clearly defined zero-gradient branch is easier to interpret. As a numerical-systematics test, also calculate gradients after one-cell smoothing or with a fourth-order periodic stencil.

The paper derives `sigma_bar_ion` from its assumed UV-background spectrum. THESAN follows three photon groups, so the most faithful THESAN adaptation is to calculate a photon-number- or photoionization-rate-weighted effective HI cross-section from the first ionizing group, if its group-averaged cross-section is available in the simulation parameters. Otherwise expose `sigma_bar_ion` as a required input and report results for a documented fiducial value and plausible alternatives. This cross-section choice is a genuine physical systematic, not only a numerical option.

### Pass 2: integrate on native gas-cell volumes

Stream the gas cells again and read `Density`, `Masses`, `InternalEnergy`, `HI_Fraction`, `HII_Fraction`, `ElectronAbundance`, and coordinates. For each valid cell:

1. Calculate `V_cell`, `n_HII`, `n_e`, and `T` in physical units.
2. Calculate case-B `alpha_B(T)` using the same Hui-Gnedin fit as the paper.
3. Periodically interpolate `F_grid` to the cell center with the same CIC/TSC convention used in Pass 1.
4. Accumulate double-precision scalar sums:

```text
S_V       += V_cell
S_alpha   += alpha_B(T) * V_cell
S_HII     += n_HII * V_cell
S_e       += n_e * V_cell
S_F       += F * V_cell
S_HI      += n_HI * V_cell
S_rec_F   += alpha_B(T) * n_HII * n_e * F * V_cell
S_HII_F   += n_HII * F * V_cell
```

Reduce these scalars across workers. Define `<q>_V = S_q/S_V`, then calculate exactly

```text
C_rec,F  = <alpha n_HII n_e F>_V
           / (<alpha>_V <n_HII>_V <n_e>_V <F>_V)

C_norm,F = <alpha n_HII n_e F>_V
           / (alpha_B(1e4 K) <n_HII>_V <n_e>_V)

t_rec     = <n_HII F>_V / <alpha n_HII n_e F>_V.
```

Also report the THESAN ionization diagnostics

```text
<x_HI>_V = sum(x_HI * V_cell) / sum(V_cell)
<x_HI>_M = sum(x_HI * M_H_cell) / sum(M_H_cell)
```

where `M_H_cell = X * M_cell`. These allow direct comparison with the paper's plots versus volume- and mass-weighted neutral fraction.

### Comparison products from the same pass

Accumulate three additional numerator variants at almost no extra I/O cost:

```text
F = 1
F = theta(Delta_th - Delta_b) for Delta_th = 50, 100, 200
F = exp(-tau_eff)
```

For the hard-mask comparison, use baryon overdensity `Delta_b = rho_b/<rho_b>` and follow the paper's Eq. 11 denominator, which remains based on full-volume `<n_HII>` and `<n_e>`. This is not the same normalization as the repository's current selected-region density variance.

## Resolution and interpretation for THESAN

The transmission estimator introduces an analysis scale through the `n_HI` grid and its derivative. Run at least three grid sizes, for example 256, 512, and 1024 where memory permits, and compare CIC with TSC. The appropriate result is a convergence band rather than a single unqualified number.

THESAN's self-consistent radiation field means this post-processing is physically better founded than applying the estimator to TNG. However, Oku & Cen assume external uniform radiation and an outside-in topology, while THESAN generally produces source-driven, spatially inhomogeneous reionization. A difference from the paper can therefore be a real topology effect rather than an implementation discrepancy.

## Recommended post-processing implementation

### Stage 1: recombination clumping without transmission

Add a native-cell, volume-weighted calculation of

```text
C_norm = <alpha(T) n_HII n_e>_V
         / (alpha_B(10^4 K) <n_HII>_V <n_e>_V).
```

This isolates field loading, unit conversion, temperature, ionization, and recombination-coefficient logic before introducing a noisy gradient. It is also scientifically more meaningful than density-only clumping.

### Stage 2: grid-derived transmission with native-cell integration

This is the best first approximation to the paper for existing moving-mesh snapshots:

1. Deposit neutral-hydrogen number density, not neutral mass alone, onto a uniform grid using a volume-conservative scheme.
2. Compute a periodic finite-difference gradient of the physical `n_HI` grid.
3. Evaluate `tau_eff` and `F` on the grid, with explicit handling for `|grad(n_HI)| -> 0` and clipping before `exp`.
4. Interpolate `F` back to gas-cell centers.
5. Accumulate native-cell volume sums of `alpha(T) n_HII n_e F`, `n_HII`, `n_e`, and optionally `F`.
6. Report both `C_norm,F` (Eq. 7) and `C_rec,F` (Eq. 5), plus the component means needed to audit them.

Using native cell volumes for the final integral avoids replacing the moving-mesh volume measure with an arbitrary analysis grid. The grid is then used only for the spatial derivative and transmission estimate.

### Stage 3: fully gridded cross-check

Deposit volume integrals of `n_HI`, `n_HII`, `n_e`, and `alpha n_HII n_e` onto the same grid, divide by deposited volume, compute `F`, and evaluate all averages on grid cells. This is easier to validate and parallelize with the current grid pipeline, but the result is more directly resolution- and assignment-scheme-dependent.

Run Stage 2 and Stage 3 at several grid sizes. Their convergence difference is a useful systematic-error estimate.

## Suggested code structure

- Extend snapshot loading with an optional gas-chemistry field specification and clear missing-field errors.
- Add a `recombination.py` module for unit conversion, temperature, the Hui-Gnedin case-B coefficient, `tau_eff`, and scalar accumulation.
- Generalize grid deposition so it can deposit arbitrary extensive scalar numerators and volume, rather than only mass.
- Add a backend or statistic selector such as `--statistic density`, `recombination`, and `recombination-transmission`; do not overload the geometric `sphere/cube/pylians` backend names.
- Add `--sigma-bar-ion`, gradient method, gradient floor, and transmission-grid size to the command line and JSON metadata.
- Preserve chunked loading by accumulating denominator sums in one pass and by building the neutral-density grid in a streaming pass. A later pass can sample `F` and accumulate the numerator without retaining every gas cell in memory.

## Required validation

- Uniform ionized gas: `C_norm,F = alpha(T)/alpha(10^4 K)` and equals one at `10^4 K` when `F = 1`.
- Constant fields with fixed `F`: verify Eq. 5 and Eq. 7 analytically; Eq. 7 may be below one when `F < 1`.
- Two-zone self-shielding test: dense neutral cells should contribute little as `F -> 0`.
- Full-load and chunked paths must agree.
- Physical/comoving unit conversion must give identical dimensionless results under a controlled scale-factor transformation.
- Convergence tests over transmission-grid size, CIC/TSC assignment, and gradient stencil.
- Compare `F = 1`, hard overdensity masks, and derived `F` on the same snapshot to separate recombination physics from IGM selection.

## Scientific limitations for precomputed simulations

This method can be applied after the fact, but it cannot recreate missing radiation physics. The inferred `F` is a diagnostic derived from the snapshot's neutral-hydrogen structure, not a new radiative-transfer solution. Results inherit the original simulation's UV background, self-shielding prescription, thermal history, and reionization topology.

In particular, applying the method to ordinary TNG snapshots can reproduce the paper's estimator but not its reionization scenario. A radiation-hydrodynamic snapshot with resolved ionization fields is a closer physical comparison. The analysis should therefore label results as an "Oku-Cen-inspired post-processed transmission estimator" unless the simulation provides equivalent radiation/chemistry modeling and the same gray cross-section.

## Recommendation

Implement Stage 1 first, then the Stage 2 hybrid method. Keep the existing density clumping factors as separate diagnostics rather than replacing them. The scientifically useful comparison for every snapshot is a four-column set:

1. Existing volume-weighted density clumping.
2. Recombination clumping with `F = 1`.
3. Recombination clumping with the traditional overdensity mask.
4. Recombination clumping with derived `F = exp(-tau_eff)`.

That decomposition will show whether differences come from temperature/ionization physics or from the paper's physical IGM definition.

## Sources

- Oku, Y. & Cen, R. (2025), arXiv:2511.09364v1, especially Eqs. 3-7 and 9-11: https://arxiv.org/abs/2511.09364
- IllustrisTNG snapshot field specifications: https://www.tng-project.org/data/docs/specifications/
- Hui, L. & Gnedin, N. Y. (1997), recombination-coefficient fitting functions: https://arxiv.org/abs/astro-ph/9612232
- Rahmati, A. et al. (2013), self-shielding treatment cited by the paper: https://arxiv.org/abs/1210.7808
