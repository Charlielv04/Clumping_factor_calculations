# Plot showcase: clumping factors, IGM definitions, and current results

This document is a plot-first record of the project so far. It follows the
scientific progression from the first density-clumping calculations to the more
physical ionization, recombination, forest, and dark-matter-model comparisons.

The central idea running through all figures is that a "clumping factor" is not
one unique number. It depends on the matter component, the averaging measure, the
IGM selection, and the physical quantity placed in the numerator.

```text
Baseline density clumping:

C = <rho^2> / <rho>^2
```

The early project measured this density statistic. The later project asks how
that baseline changes when we move toward gas volumes, ionization state,
recombination rates, mean free paths, photoionization rates, and observationally
motivated IGM definitions.

## 1. The main Thesan-2 density-clumping story

The first complete scientific picture is the redshift evolution of density
clumping in Thesan-2. These plots show that the density field becomes more
inhomogeneous as structure grows, and that the measured amplitude depends on
whether the gas is analyzed on a grid or on native gas-cell volumes.

![Combined methods at Delta_th = 20](../results/analysis/clumping/thesan/Thesan-2/combined-snapshots/combined/methods_clumping_vs_redshift_delta20.png)

At fixed overdensity threshold `Delta_th = 20`, the clumping factor rises
steadily from high redshift to lower redshift. This is the cleanest summary of
the basic physical trend: structure formation drives the matter distribution
away from uniformity.

![Combined redshift-threshold panels](../results/analysis/clumping/thesan/Thesan-2/combined-snapshots/combined/methods_clumping_vs_redshift_threshold_panels.png)

The threshold panels show that this conclusion is not tied to only one density
cut. Higher thresholds include denser gas and therefore allow larger clumping
values. This is why the project moved from a single number to threshold sweeps.

![Methods across overdensity at selected redshifts](../results/analysis/clumping/thesan/Thesan-2/combined-snapshots/combined/methods_clumping_vs_overdensity_selected_redshifts.png)

This plot is the best visual summary of the method dependence. At the same
redshift, the inferred clumping depends on the estimator. That is a physics
lesson, not just a software detail: the definition controls which structures
count as part of the IGM and how strongly dense regions are weighted.

Representative `C20` values:

| Snapshot | Redshift | Gas sphere | Gas raw-volume | DM Pylians |
|---:|---:|---:|---:|---:|
| 000 | 20.042 | 1.067 | 1.147 | 1.066 |
| 020 | 11.436 | 1.223 | 1.590 | 1.223 |
| 040 | 8.268 | 1.475 | 2.144 | 1.484 |
| 060 | 6.597 | 1.789 | 2.600 | 1.812 |
| 080 | 5.491 | 2.123 | 2.967 | 2.163 |

## 2. Gas-only views: grid versus native gas volumes

The gas figures isolate one of the most important choices made in the project:
whether to reconstruct gas on a regular grid or to use the native moving-mesh
cell volumes.

![Gas sphere clumping vs redshift](../results/analysis/clumping/thesan/Thesan-2/combined-snapshots/gas/sphere/gas_sphere_clumping_vs_redshift_grid256.png)

The spherical smoothing path gives a stable gridded gas-density statistic. This
was one of the first mature versions of the calculation: density is deposited to
a mesh, smoothed, and then thresholded by overdensity.

![Gas Pylians clumping vs redshift](../results/analysis/clumping/thesan/Thesan-2/combined-snapshots/gas/pylians/gas_pylians_clumping_vs_redshift_grid256.png)

The Pylians result closely tracks the custom gridded calculation for the same
statistic. This agreement was a key validation step: the density trend is not an
artifact of only one implementation.

![Gas raw-volume clumping vs redshift](../results/analysis/clumping/thesan/Thesan-2/combined-snapshots/gas/raw-volume/gas_raw_volume_clumping_vs_redshift.png)

The raw-volume estimator is larger. This is physically meaningful because it
uses native gas-cell volumes rather than replacing the moving-mesh volume
measure with a uniform analysis grid. The difference is one of the clearest
reasons to keep multiple definitions visible.

![Gas sphere clumping vs overdensity](../results/analysis/clumping/thesan/Thesan-2/combined-snapshots/gas/sphere/gas_sphere_clumping_vs_overdensity_selected_redshifts.png)

![Gas Pylians clumping vs overdensity](../results/analysis/clumping/thesan/Thesan-2/combined-snapshots/gas/pylians/gas_pylians_clumping_vs_overdensity_selected_redshifts.png)

![Gas raw-volume clumping vs overdensity](../results/analysis/clumping/thesan/Thesan-2/combined-snapshots/gas/raw-volume/gas_raw_volume_clumping_vs_overdensity_selected_redshifts.png)

Together, these three threshold-sweep plots show the same core lesson from
three angles: the redshift trend is robust, but the amplitude depends on the
gas-density definition and averaging measure.

## 3. Dark-matter clumping as the baseline structure tracer

Dark matter is the clean collisionless structure tracer. Its clumping evolution
is therefore the baseline against which gas physics can be compared.

![DM Pylians clumping vs redshift](../results/analysis/clumping/thesan/Thesan-2/combined-snapshots/dm/pylians/dm_pylians_clumping_vs_redshift_grid256.png)

The DM clumping factor grows in the same broad direction as gas clumping, but it
does not encode gas pressure, cooling, ionization, or recombination physics.
This is why gas and dark matter were kept separate from the beginning.

![DM Pylians clumping vs overdensity](../results/analysis/clumping/thesan/Thesan-2/combined-snapshots/dm/pylians/dm_pylians_clumping_vs_overdensity_selected_redshifts.png)

The overdensity sweep shows how much of the inferred clumping comes from the
choice of IGM cut. The threshold is part of the physical definition, not a
cosmetic plotting parameter.

![DM Pylians clumping by grid size](../results/analysis/clumping/thesan/Thesan-2/combined-snapshots/dm/pylians/dm_pylians_clumping_delta20_vs_redshift_by_grid.png)

This grid-size comparison is one of the checks that the density statistic is
not dominated by one arbitrary mesh resolution. Resolution dependence remains a
systematic to report, but this plot is the reason the grid-based campaign became
usable for scientific comparison.

## 4. Snapshot 080 as the main method-comparison anchor

Snapshot 080 at `z = 5.491` became the main comparison point because many
definitions were run there. It is late enough for clumping to be visible and
still relevant to reionization-era IGM physics.

| Method | `C5` | `C10` | `C20` |
|---|---:|---:|---:|
| Gas sphere, grid 256 | 1.665 | 1.903 | 2.123 |
| Gas Pylians, grid 256 | 1.665 | 1.903 | 2.123 |
| Gas raw-volume | 1.872 | 2.322 | 2.967 |
| DM Pylians, grid 256 | 1.664 | 1.912 | 2.163 |

The plot story is simple: when two methods implement the same gridded statistic,
they agree; when the estimator changes to native gas-cell volume weighting, the
answer changes. That difference is part of the science.

## 5. From density clumping to ionization-aware clumping

The next group of figures marks the transition from density-only clumping to
IGM physics. Density clumping measures `rho`. Reionization observables depend on
hydrogen ionization, electron density, recombination rates, radiation fields,
and mean free paths.

The more physical recombination-style quantity is closer to

```text
C_rec,F =
    <alpha(T) n_HII n_e F>_V
    / (<alpha(T)>_V <n_HII>_V <n_e>_V <F>_V)
```

where `F` can represent a soft transmission or IGM-selection field. This is not
the same quantity as `<rho^2>/<rho>^2`.

![Ionized sweep, C13 and c-tilde panels](../results/analysis/clumping/thesan/Thesan-2/snapshot080_computed_ionizing/gas/equation-tests/ionized_sweep_C13_ctilde_panels.png)

This plot group is one of the clearest signs that the project moved beyond a
mass-density statistic. The ionized fraction cut changes the inferred
clumping-like quantities because it changes which gas participates in the
ionized IGM calculation.

![Ionized sweep, raw-volume panels](../results/analysis/clumping/thesan/Thesan-2/snapshot080_computed_ionizing/gas/equation-tests/ionized_sweep_raw_volume_panels.png)

The raw-volume ionized sweep keeps the native gas-volume logic while adding an
ionization selection. This is closer in spirit to a physical IGM statistic than
the original gridded density threshold.

## 6. Full equation diagnostics

The full diagnostic set decomposes the ionization-aware calculation into the
quantities that drive it. These figures are useful because they show where the
final clumping-like number comes from physically.

![C5 raw-volume overdensity diagnostic](../results/analysis/clumping/thesan/Thesan-2/snapshot080_computed_ionizing/gas/equation-tests/full_diagnostics/01_c5_raw_volume_overdensity.png)

This is the bridge from the old density-clumping language to the newer
equation-based diagnostics. It keeps the familiar overdensity axis but begins
to connect it to the paper-inspired quantities.

![Clumping and ionization panels](../results/analysis/clumping/thesan/Thesan-2/snapshot080_computed_ionizing/gas/equation-tests/full_diagnostics/02_clumping_ionization_panels.png)

These panels show that clumping and ionization selection are intertwined. Dense
gas is not automatically the gas that should dominate a reionization-relevant
statistic.

![Photon-group C13 panels](../results/analysis/clumping/thesan/Thesan-2/snapshot080_computed_ionizing/gas/equation-tests/full_diagnostics/03_c13_photon_group_panels.png)

The photon-group comparison matters because THESAN radiation is not a single
monochromatic field. The physical interpretation of transmission or ionization
weights depends on which radiation group is being used.

![nH volume diagnostic](../results/analysis/clumping/thesan/Thesan-2/snapshot080_computed_ionizing/gas/equation-tests/full_diagnostics/04_parameter_nH_V.png)

![nHI volume diagnostic](../results/analysis/clumping/thesan/Thesan-2/snapshot080_computed_ionizing/gas/equation-tests/full_diagnostics/05_parameter_nHI_V.png)

![nHII volume diagnostic](../results/analysis/clumping/thesan/Thesan-2/snapshot080_computed_ionizing/gas/equation-tests/full_diagnostics/06_parameter_nHII_V.png)

![Electron-density volume diagnostic](../results/analysis/clumping/thesan/Thesan-2/snapshot080_computed_ionizing/gas/equation-tests/full_diagnostics/07_parameter_ne_V.png)

These four plots separate the hydrogen and electron-density ingredients. This
is the key physical decomposition missing from density-only clumping.

![Photon-density diagnostic](../results/analysis/clumping/thesan/Thesan-2/snapshot080_computed_ionizing/gas/equation-tests/full_diagnostics/08_parameter_nGamma_V.png)

![Recombination-rate diagnostic](../results/analysis/clumping/thesan/Thesan-2/snapshot080_computed_ionizing/gas/equation-tests/full_diagnostics/09_parameter_R_rec.png)

![Ionization-rate diagnostic](../results/analysis/clumping/thesan/Thesan-2/snapshot080_computed_ionizing/gas/equation-tests/full_diagnostics/10_parameter_R_ion.png)

These panels move the analysis from density fields into rate fields. That is
the conceptual step needed before comparing to reionization observables.

![Gamma and c-tilde diagnostic](../results/analysis/clumping/thesan/Thesan-2/snapshot080_computed_ionizing/gas/equation-tests/full_diagnostics/11_parameter_R_gamma_ctilde.png)

![Q6 diagnostic](../results/analysis/clumping/thesan/Thesan-2/snapshot080_computed_ionizing/gas/equation-tests/full_diagnostics/12_parameter_Q6.png)

![MFP and neutral-hydrogen diagnostic](../results/analysis/clumping/thesan/Thesan-2/snapshot080_computed_ionizing/gas/equation-tests/full_diagnostics/13_parameter_nHI_mfp_over_nHI_V.png)

![Q12 and c-tilde diagnostic](../results/analysis/clumping/thesan/Thesan-2/snapshot080_computed_ionizing/gas/equation-tests/full_diagnostics/14_parameter_Q12_ctilde.png)

![nGamma c-tilde sigma over Gamma diagnostic](../results/analysis/clumping/thesan/Thesan-2/snapshot080_computed_ionizing/gas/equation-tests/full_diagnostics/15_parameter_nGamma_ctilde_sigma_over_Gamma.png)

The later diagnostics connect the clumping calculation to Gamma, mean free path,
and effective radiation factors. This is the beginning of the observational
bridge: the calculation is no longer only asking how lumpy the gas is, but how
that lumpiness enters reionization equations.

For snapshot 080, the equation-test output labelled `C5_paper_actual` gives:

| Threshold | `C5_paper_actual` |
|---:|---:|
| 1 | 1.364 |
| 5 | 1.894 |
| 10 | 2.350 |
| 20 | 3.003 |
| 25 | 3.264 |

## 7. AIDA-TNG: dark-matter-model plot gallery

The AIDA-TNG figures move the project toward its original objective: comparing
clumping predictions across dark matter models. These plots use the same
analysis language across CDM, WDM, SIDM, and velocity-dependent SIDM variants.

![L35n1080 snapshot 017 Pylians model comparison](../results/analysis/clumping/aida-tng/L35n1080/snapshot017/gas/pylians/model_comparison_grid256.png)

![L35n1080 snapshot 017 Pylians relative to CDM](../results/analysis/clumping/aida-tng/L35n1080/snapshot017/gas/pylians/model_relative_to_cdm_grid256.png)

At `z ~ 5`, the gas-density clumping differences between the L35n1080
models are modest. That is already scientifically useful: the simplest density
statistic may not be the most sensitive discriminator.

![L35n1080 snapshot 099 Pylians model comparison](../results/analysis/clumping/aida-tng/L35n1080/snapshot099/gas/pylians/model_comparison_grid256.png)

![L35n1080 snapshot 099 Pylians relative to CDM](../results/analysis/clumping/aida-tng/L35n1080/snapshot099/gas/pylians/model_relative_to_cdm_grid256.png)

At `z = 0`, clumping is much larger overall. The relative plot is the more
important physical view because it asks whether model differences survive after
removing the common structure-growth trend.

![L75n910 snapshot 099 Pylians model comparison](../results/analysis/clumping/aida-tng/L75n910/snapshot099/gas/pylians/model_comparison_grid256.png)

![L75n910 snapshot 099 Pylians model comparison, grid 512](../results/analysis/clumping/aida-tng/L75n910/snapshot099/gas/pylians/model_comparison_grid512.png)

The L75n910 plots add a second volume/resolution family. This matters because a
dark-matter-model claim should not depend on one box alone.

Representative Pylians `C20` values:

| Simulation | Snapshot | Redshift | `C20` |
|---|---:|---:|---:|
| L35n1080_CDM | 017 | 4.999 | 2.729 |
| L35n1080_SIDM1 | 017 | 4.999 | 2.726 |
| L35n1080_WDM3 | 017 | 4.999 | 2.721 |
| L35n1080_vSIDM | 017 | 4.999 | 2.729 |
| L35n1080_CDM | 099 | 0.000 | 8.511 |
| L35n1080_WDM3 | 099 | 0.000 | 8.529 |
| L35n1080_vSIDM | 099 | 0.000 | 8.555 |
| L75n910_CDM | 099 | 0.000 | 8.135 |
| L75n910_SIDM1 | 099 | 0.000 | 8.200 |
| L75n910_WDM3 | 099 | 0.000 | 8.103 |
| L75n910_vSIDM | 099 | 0.000 | 8.192 |

## 8. AIDA-TNG ionization and raw-volume views

The density-only model comparisons are only one layer. The ionization and
raw-volume plots ask whether model differences become clearer when the gas is
filtered by ionization or measured with native cell volumes.

![L35n1080 snapshot 017 ionized-sweep model comparison](../results/analysis/clumping/aida-tng/L35n1080/snapshot017/gas/ionized-sweep/model_comparison.png)

![L35n1080 snapshot 017 ionized-sweep relative to CDM](../results/analysis/clumping/aida-tng/L35n1080/snapshot017/gas/ionized-sweep/model_relative_to_cdm.png)

![L35n1080 snapshot 017 ionized-sweep model comparison panels](../results/analysis/clumping/aida-tng/L35n1080/snapshot017/gas/ionized-sweep/model_comparison_panels.png)

![L35n1080 snapshot 017 ionized-sweep relative panels](../results/analysis/clumping/aida-tng/L35n1080/snapshot017/gas/ionized-sweep/model_relative_to_cdm_panels.png)

These are the more physically promising AIDA-TNG plots because they begin to
separate total gas density from the ionized IGM gas relevant to reionization and
absorption.

![L35n1080 snapshot 017 raw-volume ionization comparison](../results/analysis/clumping/aida-tng/L35n1080/snapshot017/gas/raw-volume/ionization_model_comparison.png)

![L35n1080 snapshot 017 raw-volume relative to CDM](../results/analysis/clumping/aida-tng/L35n1080/snapshot017/gas/raw-volume/ionization_relative_to_cdm.png)

![L35n1080 snapshot 017 raw-volume ionization sweep](../results/analysis/clumping/aida-tng/L35n1080/snapshot017/gas/raw-volume/ionization_sweep.png)

The raw-volume view keeps the moving-mesh volume measure. If the goal is an IGM
volume average, these plots are closer to the desired physical averaging than
equal-cell or purely gridded density moments.

![L75n910 snapshot 099 raw-volume ionization comparison](../results/analysis/clumping/aida-tng/L75n910/snapshot099/gas/raw-volume/ionization_model_comparison.png)

![L75n910 snapshot 099 raw-volume relative to CDM](../results/analysis/clumping/aida-tng/L75n910/snapshot099/gas/raw-volume/ionization_relative_to_cdm.png)

![L75n910 snapshot 099 raw-volume ionization sweep](../results/analysis/clumping/aida-tng/L75n910/snapshot099/gas/raw-volume/ionization_sweep.png)

This second model family is useful as a check on whether the ionization-aware
patterns are robust across simulation volume/resolution choices.

## 9. Alternative clumping and observationally motivated quantities

The alternative Eq. 13 campaign was introduced to connect the simulation
analysis to observationally motivated clumping estimates. This is less about the
raw mass-density field and more about the form in which clumping enters
reionization and absorption equations.

![Eq. 13 clumping vs redshift](../results/Thesan-2/alternative_clumping/eq13_clumping_vs_redshift.png)

This figure is part of the transition from "clumping as density variance" to
"clumping as an ingredient in an observational or reionization equation."

JSON products:

- [Snapshot 055 Eq. 13](../results/Thesan-2/alternative_clumping/snapshot055_eq13.json)
- [Snapshot 060 Eq. 13](../results/Thesan-2/alternative_clumping/snapshot060_eq13.json)
- [Snapshot 065 Eq. 13](../results/Thesan-2/alternative_clumping/snapshot065_eq13.json)
- [Snapshot 070 Eq. 13](../results/Thesan-2/alternative_clumping/snapshot070_eq13.json)
- [Snapshot 075 Eq. 13](../results/Thesan-2/alternative_clumping/snapshot075_eq13.json)
- [Snapshot 080 Eq. 13](../results/Thesan-2/alternative_clumping/snapshot080_eq13.json)

## 10. Forest, Gamma, and MFP products

The Lyman-alpha forest direction is the observational bridge. Observations do
not measure the full 3D density field directly; they measure absorption along
lines of sight. The forest, Gamma, and MFP products therefore connect the
clumping project to quantities closer to reionization data.

Current Thesan-1 snapshot 080 scalar products:

| Quantity | Value | Notes |
|---|---:|---|
| `Gamma_HI` | `4.1228e-13 s^-1` | Volume-weighted `HI_Fraction < 0.5` calculation. |
| MFP at 912 Angstrom | `4.6868 proper Mpc / h` | Mean over 15000 ray samples. |
| MFP median | `4.4674 proper Mpc / h` | Same ray sample. |
| MFP 16th-84th interval | `1.3812` to `7.8003 proper Mpc / h` | Broad line-of-sight variation. |

Products:

- [Gamma HI JSON](../results/forest/thesan/Thesan-1/snapshot080/gamma_hi.json)
- [MFP 912 JSON](../results/forest/thesan/Thesan-1/snapshot080/mfp912/rays_080_mfp912.json)
- [Lyman-alpha spectra HDF5](../results/forest/thesan/Thesan-1/snapshot080/lya/rays_080_lya.hdf5)

## 11. Performance plots that matter for the physics campaign

These plots are not the scientific endpoint, but they explain why the project
could move from toy calculations to multi-snapshot physical comparisons.

![Grid scaling](../results/analysis/dm-pylians-all-grid-thread-scaling/grid_scaling.png)

![Performance dashboard](../results/analysis/dm-pylians-all-grid-thread-scaling/performance_dashboard.png)

The practical outcome is that the same estimator can be run across redshifts,
grids, and models. That consistency is what makes the physical comparisons
above interpretable.

![Clumping consistency across grids](../results/analysis/dm-pylians-all-grid-thread-scaling/clumping_consistency.png)

This is the most physics-relevant performance-era plot: it checks whether
changes in grid resolution alter the inferred clumping trend.

## 12. What the plot sequence says

The plots tell a staged story:

1. Density clumping grows with cosmic structure formation.
2. Gas and dark matter must be treated as different physical tracers.
3. Gridded gas and Pylians agree when they implement the same density statistic.
4. Native gas-volume weighting gives larger gas clumping, so the averaging
   measure is physically important.
5. Ionization-aware diagnostics move the project beyond density variance toward
   recombination and reionization physics.
6. AIDA-TNG model differences are modest in the simplest density statistic, so
   the stronger test is likely in ionization-aware or observationally connected
   quantities.
7. Forest, Gamma, and MFP products are the bridge from simulation fields to
   absorption-era observables.

The current best interpretation is that the original density clumping factor is
a necessary baseline, but not the final physical observable. The project is now
positioned to compare dark matter models using statistics that know about gas
volume, ionization state, recombination physics, and observational absorption
probes.
