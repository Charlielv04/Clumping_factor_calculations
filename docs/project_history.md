# Project history: clumping factors, IGM definitions, and current results

This document summarizes the scientific path of the project so far. The emphasis is
physics first: what quantity was being measured at each stage, why the definition
changed, and what the current results say about the matter distribution in the
simulations. Implementation details are included only when they changed the
physical interpretation or made a new scientific comparison possible.

## 1. Starting point: density clumping as a first observable

The first working target was the standard density clumping factor,

```text
C = <rho^2> / <rho>^2.
```

For a uniform field, `C = 1`. Values above one indicate that the density field
has developed inhomogeneous structure. This made `C` a natural first diagnostic:
it is simple, dimensionless, and sensitive to the small-scale structure that can
change between dark matter models.

The first standalone scripts computed gas and dark matter clumping separately.
That separation was physically important from the beginning. Dark matter is a
collisionless particle distribution, while gas in moving-mesh simulations is
stored in finite-volume cells with density, volume, temperature, and chemistry.
Even if both components are processed into a density field, they are not the same
physical tracer.

The earliest decision was therefore not only computational. It was the choice to
treat "the clumping factor" as a family of related statistics rather than a
single universal number. The relevant questions became:

- Which component is being measured: gas, dark matter, or total matter?
- Is the density field reconstructed on a grid or measured on native gas cells?
- Which gas is considered part of the IGM?
- Is the selection based on a hard overdensity threshold or on ionization physics?

## 2. Naive implementations and the first lessons

The initial scripts explored several density-construction choices:

- direct gas-cell calculations;
- dark-matter particle calculations;
- spherical smoothing;
- cubic smoothing;
- Pylians-based mass assignment and smoothing;
- radius-based and cube-based approximations to gas-cell size.

These scripts were valuable because they produced the first results quickly, but
they also exposed the main physical ambiguity. A density clumping factor is not
defined until the averaging measure and the selected region are defined. A gas
calculation that averages equally over gas cells is not the same as a volume
average, because AREPO gas cells have unequal volumes. A gridded calculation
measures a smoothed field on an analysis mesh, not the native moving-mesh field.

The first major decision was to standardize the density-threshold form

```text
C_rho(Delta_th) =
    <rho_target^2>_selected / <rho_target>_selected^2,

selected where rho_mask / <rho_mask> - 1 < Delta_th.
```

This made the IGM selection explicit. It also opened the door to separate mask
and target fields: for example, selecting low-density regions using one matter
field while measuring gas clumping inside that region.

## 3. From scripts to a reproducible scientific pipeline

The next stage converted the standalone calculations into the `clumping_factor`
package and command-line tools. The important scientific decision here was to
make result JSON files the stable interface between expensive simulation runs
and later analysis.

That mattered because each output now carries enough provenance to interpret the
number:

- simulation name, snapshot, redshift, and scale factor;
- particle type and backend;
- grid size and mass-assignment method;
- thresholds and clumping factors;
- run parameters and timing diagnostics;
- schema version and input metadata.

The package retained multiple backends because they answer slightly different
physical and numerical questions:

| Backend or path | Physical meaning |
|---|---|
| `sphere` | Density field smoothed with a spherical top-hat kernel. |
| `cube` | Density field smoothed with a cubic kernel. |
| `pylians` | Independent grid construction and smoothing path used as a comparison standard. |
| `raw` | Native gas-cell density moments with equal cell weighting; useful historically but not a volume average. |
| `raw-volume` | Native gas-cell density moments weighted by gas-cell volume. This is the closest density-only gas statistic to a physical volume average. |

The key physics decision was to keep these as distinct definitions rather than
forcing them to agree. Agreement between methods is useful evidence of numerical
stability. Disagreement is also useful, because it tells us which part of the
definition controls the inferred clumping.

## 4. Scaling up made the physics program feasible

Large snapshots made full in-memory processing impractical. The pipeline was
therefore extended to chunked loading and same-node parallelism. Scientifically,
this was not just a performance improvement. It made it possible to run the same
definition across multiple redshifts, grid sizes, methods, and simulations.

The scaling work allowed the project to move from "does the formula run?" to
"does the physical trend survive changes in method and resolution?" That is the
necessary condition for comparing dark matter models.

The benchmark campaign showed that the runtime depends strongly on grid size,
thread count, cache state, and worker balance. The practical outcome is that
grid sizes up to 512 and 1024 became usable for selected campaigns, while grid
256 became the main production-scale comparison point.

Relevant performance products:

- [Pylians grid/thread benchmark summary](../results/analysis/dm-pylians-all-grid-thread-scaling/benchmark_summary.csv)
- [Grid-scaling dashboard](../results/analysis/dm-pylians-all-grid-thread-scaling/grid_scaling.png)
- [Performance dashboard](../results/analysis/dm-pylians-all-grid-thread-scaling/performance_dashboard.png)

## 5. Thesan-2 density clumping: main redshift trend

The clearest current density-clumping sequence is the Thesan-2 redshift
evolution. At fixed overdensity threshold `Delta_th = 20`, the standard gridded
gas and dark matter clumping factors rise steadily from high redshift to lower
redshift as structure forms.

Representative values:

| Snapshot | Redshift | Gas sphere `C20` | Gas raw-volume `C20` | DM Pylians `C20` |
|---:|---:|---:|---:|---:|
| 000 | 20.042 | 1.067 | 1.147 | 1.066 |
| 010 | 14.360 | 1.134 | 1.334 | 1.134 |
| 020 | 11.436 | 1.223 | 1.590 | 1.223 |
| 030 | 9.581 | 1.336 | 1.869 | 1.339 |
| 040 | 8.268 | 1.475 | 2.144 | 1.484 |
| 050 | 7.322 | 1.629 | 2.388 | 1.645 |
| 060 | 6.597 | 1.789 | 2.600 | 1.812 |
| 070 | 5.994 | 1.955 | 2.793 | 1.987 |
| 080 | 5.491 | 2.123 | 2.967 | 2.163 |

The main physical reading is straightforward: the density field becomes more
inhomogeneous with time. The more subtle result is that the native gas-cell
volume-weighted statistic is systematically larger than the gridded gas
statistic. That means the averaging measure is not a harmless technicality. It
changes how strongly dense gas structures contribute to the final number.

Important plots:

- [Gas sphere clumping vs redshift](../results/analysis/clumping/thesan/Thesan-2/combined-snapshots/gas/sphere/gas_sphere_clumping_vs_redshift_grid256.png)
- [Gas raw-volume clumping vs redshift](../results/analysis/clumping/thesan/Thesan-2/combined-snapshots/gas/raw-volume/gas_raw_volume_clumping_vs_redshift.png)
- [DM Pylians clumping vs redshift](../results/analysis/clumping/thesan/Thesan-2/combined-snapshots/dm/pylians/dm_pylians_clumping_vs_redshift_grid256.png)
- [Combined methods at `Delta_th = 20`](../results/analysis/clumping/thesan/Thesan-2/combined-snapshots/combined/methods_clumping_vs_redshift_delta20.png)
- [Combined threshold panels](../results/analysis/clumping/thesan/Thesan-2/combined-snapshots/combined/methods_clumping_vs_redshift_threshold_panels.png)

## 6. Snapshot 080 method comparison

Snapshot 080, at `z = 5.491`, is a useful reference point because many methods
were run there. At this redshift the Universe is late enough in the simulation
for inhomogeneity to be visible, but still directly relevant to reionization-era
IGM questions.

Representative values:

| Method | `C5` | `C10` | `C20` |
|---|---:|---:|---:|
| Gas sphere, grid 256 | 1.665 | 1.903 | 2.123 |
| Gas Pylians, grid 256 | 1.665 | 1.903 | 2.123 |
| Gas raw-volume | 1.872 | 2.322 | 2.967 |
| DM Pylians, grid 256 | 1.664 | 1.912 | 2.163 |

Two decisions are supported by this comparison.

First, the custom gridded gas path and the Pylians path agree extremely well for
the same density statistic. That gives confidence that the gridded density
calculation is not dominated by an implementation artifact.

Second, the native volume-weighted gas statistic is larger. This is not a bug in
the gridded method; it is a different physical estimator. It preserves the
native gas-cell volume measure rather than first replacing the moving-mesh gas
field with a uniform analysis grid.

Important plots:

- [Method comparison across overdensity thresholds](../results/analysis/clumping/thesan/Thesan-2/combined-snapshots/combined/methods_clumping_vs_overdensity_selected_redshifts.png)
- [Gas Pylians clumping vs overdensity](../results/analysis/clumping/thesan/Thesan-2/combined-snapshots/gas/pylians/gas_pylians_clumping_vs_overdensity_selected_redshifts.png)
- [Gas raw-volume clumping vs overdensity](../results/analysis/clumping/thesan/Thesan-2/combined-snapshots/gas/raw-volume/gas_raw_volume_clumping_vs_overdensity_selected_redshifts.png)
- [DM Pylians clumping vs overdensity](../results/analysis/clumping/thesan/Thesan-2/combined-snapshots/dm/pylians/dm_pylians_clumping_vs_overdensity_selected_redshifts.png)

## 7. Moving beyond density-only clumping

The next scientific step was motivated by the difference between a density
clumping factor and a recombination or transmission-weighted IGM clumping
factor. Density clumping only knows about `rho`. Reionization observables depend
on ionization state, electron density, temperature, neutral hydrogen, and
radiation transport.

The comparison note in [paper_2511.09364_comparison.md](paper_2511.09364_comparison.md)
records the key distinction. The Oku and Cen style recombination clumping factor
is closer to

```text
C_rec,F =
    <alpha(T) n_HII n_e F>_V
    / (<alpha(T)>_V <n_HII>_V <n_e>_V <F>_V),
```

with a soft transmission field

```text
F = exp(-tau_eff).
```

This is not the same as replacing `rho` by gas density in `<rho^2>/<rho>^2`.
The numerator is a recombination-rate density, and the IGM selector can be a
continuous radiation/opacity weight rather than a hard density threshold.

The project therefore added equation tests and ionization-aware diagnostics. In
snapshot 080, the equation-test output labelled `C5_paper_actual` gives:

| Threshold | `C5_paper_actual` |
|---:|---:|
| 1 | 1.364 |
| 5 | 1.894 |
| 10 | 2.350 |
| 20 | 3.003 |
| 25 | 3.264 |

At `Delta_th = 20`, this is slightly above the raw-volume density clumping
value and clearly above the gridded density value. The physical interpretation
is that once the estimator includes hydrogen and ionization-related quantities,
it is no longer measuring only density contrast. It is measuring the clustering
of the gas that actually participates in recombination or neutral absorption.

Important diagnostic plots:

- [Ionized sweep, C13 and c-tilde panels](../results/analysis/clumping/thesan/Thesan-2/snapshot080_computed_ionizing/gas/equation-tests/ionized_sweep_C13_ctilde_panels.png)
- [Ionized sweep, raw-volume panels](../results/analysis/clumping/thesan/Thesan-2/snapshot080_computed_ionizing/gas/equation-tests/ionized_sweep_raw_volume_panels.png)
- [C5 raw-volume overdensity diagnostic](../results/analysis/clumping/thesan/Thesan-2/snapshot080_computed_ionizing/gas/equation-tests/full_diagnostics/01_c5_raw_volume_overdensity.png)
- [Clumping and ionization panels](../results/analysis/clumping/thesan/Thesan-2/snapshot080_computed_ionizing/gas/equation-tests/full_diagnostics/02_clumping_ionization_panels.png)
- [Photon-group C13 panels](../results/analysis/clumping/thesan/Thesan-2/snapshot080_computed_ionizing/gas/equation-tests/full_diagnostics/03_c13_photon_group_panels.png)
- [Gamma and c-tilde diagnostic](../results/analysis/clumping/thesan/Thesan-2/snapshot080_computed_ionizing/gas/equation-tests/full_diagnostics/11_parameter_R_gamma_ctilde.png)
- [MFP and neutral-hydrogen diagnostic](../results/analysis/clumping/thesan/Thesan-2/snapshot080_computed_ionizing/gas/equation-tests/full_diagnostics/13_parameter_nHI_mfp_over_nHI_V.png)

## 8. Forest, Gamma, and mean-free-path products

The Lyman-alpha forest direction was added because observations do not directly
measure the full three-dimensional density field. They measure absorption along
lines of sight. This makes forest and ionizing quantities the natural bridge
between the simulation clumping factors and observationally inferred IGM
properties.

Current Thesan-1 snapshot 080 products include:

| Quantity | Value | Notes |
|---|---:|---|
| `Gamma_HI` | `4.1228e-13 s^-1` | Volume-weighted `HI_Fraction < 0.5` calculation. |
| MFP at 912 A | `4.6868 proper Mpc / h` | Mean over 15000 ray samples. |
| MFP median | `4.4674 proper Mpc / h` | Same ray sample. |
| MFP 16th-84th interval | `1.3812` to `7.8003 proper Mpc / h` | Indicates broad line-of-sight variation. |

The `Gamma_HI` cross-check records a very small absolute numerical difference,
`4.35e-21 s^-1`, while the boolean `passed` flag is false. That should be
treated as a tolerance/reporting issue to inspect, not as evidence for a large
physical discrepancy.

Relevant products:

- [Gamma HI JSON](../results/forest/thesan/Thesan-1/snapshot080/gamma_hi.json)
- [MFP 912 JSON](../results/forest/thesan/Thesan-1/snapshot080/mfp912/rays_080_mfp912.json)
- [Lyman-alpha spectra HDF5](../results/forest/thesan/Thesan-1/snapshot080/lya/rays_080_lya.hdf5)

## 9. AIDA-TNG model comparisons

The AIDA-TNG campaign extends the project from one simulation history to model
comparisons. The same clumping definitions are run across CDM, WDM, SIDM, and
velocity-dependent SIDM variants. This is the direction needed for the original
scientific objective: test whether different dark matter models lead to
measurably different clumping factors.

For L35n1080 gas Pylians at grid 256:

| Model | Snapshot | Redshift | `C20` |
|---|---:|---:|---:|
| CDM | 017 | 4.999 | 2.729 |
| SIDM1 | 017 | 4.999 | 2.726 |
| WDM3 | 017 | 4.999 | 2.721 |
| vSIDM | 017 | 4.999 | 2.729 |
| CDM | 099 | 0.000 | 8.511 |
| WDM3 | 099 | 0.000 | 8.529 |
| vSIDM | 099 | 0.000 | 8.555 |

For L75n910 gas Pylians at grid 256 and snapshot 099:

| Model | Redshift | `C20` |
|---|---:|---:|
| CDM | 0.000 | 8.135 |
| SIDM1 | 0.000 | 8.200 |
| WDM3 | 0.000 | 8.103 |
| vSIDM | 0.000 | 8.192 |

At the sampled thresholds and snapshots, the model-to-model differences are
modest compared with the redshift evolution. This does not mean the model
comparison is uninteresting. It means the final constraint will likely depend on
choosing the right physical statistic, redshift range, and observable mapping,
not only on the simplest density clumping number.

Important AIDA-TNG comparison plots:

- [L35n1080 snapshot 017 Pylians model comparison](../results/analysis/clumping/aida-tng/L35n1080/snapshot017/gas/pylians/model_comparison_grid256.png)
- [L35n1080 snapshot 017 relative to CDM](../results/analysis/clumping/aida-tng/L35n1080/snapshot017/gas/pylians/model_relative_to_cdm_grid256.png)
- [L35n1080 snapshot 099 Pylians model comparison](../results/analysis/clumping/aida-tng/L35n1080/snapshot099/gas/pylians/model_comparison_grid256.png)
- [L75n910 snapshot 099 Pylians model comparison](../results/analysis/clumping/aida-tng/L75n910/snapshot099/gas/pylians/model_comparison_grid256.png)
- [L35n1080 snapshot 017 ionized-sweep model comparison](../results/analysis/clumping/aida-tng/L35n1080/snapshot017/gas/ionized-sweep/model_comparison.png)
- [L75n910 snapshot 099 raw-volume ionization comparison](../results/analysis/clumping/aida-tng/L75n910/snapshot099/gas/raw-volume/ionization_model_comparison.png)

## 10. Alternative clumping and observational bridge

The alternative Eq. 13 calculation was introduced to connect the internal
simulation measurements to observationally motivated clumping estimates,
especially the Davies-style absorption/reionization framing. These outputs live
under the older `results/Thesan-2/alternative_clumping/` path and cover
snapshots 055 through 080.

Relevant products:

- [Eq. 13 clumping vs redshift](../results/Thesan-2/alternative_clumping/eq13_clumping_vs_redshift.png)
- [Snapshot 055 Eq. 13 JSON](../results/Thesan-2/alternative_clumping/snapshot055_eq13.json)
- [Snapshot 060 Eq. 13 JSON](../results/Thesan-2/alternative_clumping/snapshot060_eq13.json)
- [Snapshot 065 Eq. 13 JSON](../results/Thesan-2/alternative_clumping/snapshot065_eq13.json)
- [Snapshot 070 Eq. 13 JSON](../results/Thesan-2/alternative_clumping/snapshot070_eq13.json)
- [Snapshot 075 Eq. 13 JSON](../results/Thesan-2/alternative_clumping/snapshot075_eq13.json)
- [Snapshot 080 Eq. 13 JSON](../results/Thesan-2/alternative_clumping/snapshot080_eq13.json)

The physical role of this branch is different from the density-clumping branch.
It asks how a clumping-like quantity enters reionization and absorption
equations, rather than only how inhomogeneous the mass density field is.

## 11. Current scientific picture

The project has moved through four scientific layers.

First, it established the density clumping factor as a baseline statistic for
inhomogeneity. This was necessary because it is the simplest controlled way to
compare gas, dark matter, redshift, and simulation model.

Second, it showed that numerical definitions matter physically. Gridded gas and
Pylians agree well when they implement the same statistic, while native
volume-weighted gas gives larger values. This means method comparison is not
only debugging; it is part of defining the observable.

Third, it extended the analysis toward IGM physics. Ionization state,
temperature, recombination coefficients, MFP, Gamma, and photon-group choices
are now part of the analysis tree. This is the path from density clumping to a
quantity that can be compared with reionization and Lyman-alpha constraints.

Fourth, it began the dark-matter-model comparison on AIDA-TNG. The current
model differences in simple density clumping are relatively small at the sampled
points, so the most promising next comparisons should use the more physical
IGM-aware diagnostics and observational bridge quantities.

## 12. Decisions made so far

The main decisions and their scientific motivations are:

| Decision | Motivation |
|---|---|
| Keep gas and dark matter separate. | They trace different physics and are represented differently in the simulations. |
| Use threshold sweeps instead of one fixed cut. | The IGM boundary is not unique, so trends with `Delta_th` are more informative than a single number. |
| Preserve several density-construction methods. | Method agreement tests numerical stability; method disagreement reveals definition dependence. |
| Add native volume-weighted gas. | Gas cells have physical volumes, and equal-cell averaging is not a volume average. |
| Store JSON provenance. | A clumping number is interpretable only with redshift, component, backend, grid, threshold, and units. |
| Add chunked and parallel execution. | Full scientific campaigns require many snapshots and large grids. |
| Add ionization/equation diagnostics. | Reionization observables depend on hydrogen, electrons, temperature, and radiation, not density alone. |
| Add forest, Gamma, and MFP products. | Observations probe absorption along lines of sight, so the project needs an observational bridge. |
| Run AIDA-TNG model comparisons. | The long-term goal is to test whether dark matter models produce distinguishable clumping signatures. |

## 13. Most important limitations

The density-only clumping factor should not be overinterpreted as a
recombination clumping factor. It is a useful baseline, but it does not include
temperature, ionization state, electron density, or transmission.

The overdensity threshold is a hard IGM selector. It is easy to compare and plot,
but it is not the same as a physically motivated radiation or self-shielding
criterion.

The native gas-cell and gridded estimators measure related but different
quantities. The difference between them should be reported as a systematic
definition dependence, not hidden.

The AIDA-TNG model differences seen in the simple gas density statistic are
small in the sampled cases. Stronger physical conclusions need the full
ionization-aware and observationally connected analysis.

Some result products are still transitional. In particular, older paths and
placeholder workflow manifests should be treated as project history unless their
JSON outputs contain complete successful provenance.

## 14. Recommended next steps

The next scientific step is to make the recombination/IGM-aware comparison as
systematic as the density-clumping comparison:

1. Compute `F = 1`, hard-threshold, and transmission-weighted recombination
   clumping on the same snapshots.
2. Run the same definitions across Thesan and AIDA-TNG where the required fields
   exist.
3. Quantify resolution dependence using grid 256, 512, and 1024 when feasible.
4. Compare model differences relative to CDM using the same redshift, threshold,
   and estimator.
5. Connect the preferred estimator to Lyman-alpha, MFP, and Gamma diagnostics so
   the result can be discussed in observational terms.

The project is now well positioned for that step. The early scripts established
the baseline observable; the package made it reproducible; the scaling work made
campaigns feasible; and the ionization/forest branches opened the path from a
density statistic to a physically interpretable IGM constraint.
