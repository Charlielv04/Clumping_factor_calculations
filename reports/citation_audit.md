# Citation verification audit — `apssamp_draft.tex`

Prepared 2026-08-23. This audit covers the 12 citation commands in the draft, which yield **15 source-use records** because three commands cite two works. Source page numbers below are PDF pages (the first page of each arXiv PDF is page 1). The source files were not changed.

## Result at a glance

| Check | Result |
|---|---:|
| Citation commands found | 12 |
| Individual cited-source uses | 15 |
| Fully supported uses | 10 |
| Partly supported uses | 5 |
| Unsupported substantive uses | 0 |
| Bibliography metadata issues | 3 major, 1 minor |

The five partly supported uses are not failures of the scientific propositions: three use a malformed Davies BibTeX record, one overstates how AIDA-TNG initial conditions are matched, and one uses a THESAN record with a mismatched arXiv identifier. The underlying papers support the intended claims.

## Verified source index

### S01 — `feng2019reionization`

J. H. Wise, *An Introductory Review on Cosmic Reionization*, **Contemporary Physics 60**, 145–167 (2019), doi:10.1080/00107514.2019.1631548. ArXiv: [1907.06653](https://arxiv.org/abs/1907.06653).

### S02 — intended replacement for `davies2024clumping`; also `Davies_2026`

F. B. Davies, S. E. I. Bosman, and S. R. Furlanetto, *The Predicament of Absorption-dominated Reionization. II. Observational Estimate of the Clumping Factor at the End of Reionization*, **The Astrophysical Journal 1005**, 36 (2026), doi:10.3847/1538-4357/ae75b3. ArXiv: [2406.18186](https://arxiv.org/abs/2406.18186).

`Davies_2026` is substantively correct but is missing its arXiv field. `davies2024clumping` is not a valid record for this paper: its title, authors, DOI, and year do not identify arXiv:2406.18186.

### S03 — `vogelsberger2019simulations`

M. Vogelsberger, F. Marinacci, P. Torrey, and E. Puchwein, *Cosmological Simulations of Galaxy Formation*, **Nature Reviews Physics 2**, 42–66 (2020), doi:10.1038/s42254-019-0127-2. ArXiv: [1909.07976](https://arxiv.org/abs/1909.07976).

### S04 — `garaldi2022thesan`

R. Kannan, E. Garaldi, A. Smith, R. Pakmor, V. Springel, M. Vogelsberger, and L. Hernquist, *Introducing the THESAN project: radiation-magnetohydrodynamic simulations of the epoch of reionization*, **Monthly Notices of the Royal Astronomical Society 511**, 4005–4037 (2022), doi:10.1093/mnras/stac238. ArXiv: [2110.00584](https://arxiv.org/abs/2110.00584).

The current key combines the title/DOI of this Kannan et al. paper with the arXiv identifier of a different Garaldi et al. paper.

### S05 — `despali2025aidatng`

G. Despali, L. Moscardini, D. Nelson, A. Pillepich, V. Springel, and M. Vogelsberger, *Introducing the AIDA-TNG project: galaxy formation in alternative dark matter models*, **Astronomy & Astrophysics 697**, A213 (2025), doi:10.1051/0004-6361/202553836. ArXiv: [2501.12439](https://arxiv.org/abs/2501.12439).

### S06 — `irsic2024wdm`

V. Iršič, M. Viel, M. G. Haehnelt, J. S. Bolton, M. Molaro, E. Puchwein, E. Boera, G. D. Becker, P. Gaikwad, L. Keating, and G. Kulkarni, *Unveiling Dark Matter Free-streaming at the Smallest Scales with High-redshift Lyman-alpha Forest*, **Physical Review D 109**, 043511 (2024), doi:10.1103/PhysRevD.109.043511. ArXiv: [2309.04533](https://arxiv.org/abs/2309.04533).

### S07 — `villasenor2022wdm`

B. Villasenor, B. Robertson, P. Madau, and E. Schneider, *New Constraints on Warm Dark Matter from the Lyman-alpha Forest Power Spectrum*, **Physical Review D 108**, 023502 (2023), doi:10.1103/PhysRevD.108.023502. ArXiv: [2209.14220](https://arxiv.org/abs/2209.14220).

### S08 — `tulin2018sidm`

S. Tulin and H.-B. Yu, *Dark Matter Self-interactions and Small Scale Structure*, **Physics Reports 730**, 1–57 (2018), doi:10.1016/j.physrep.2017.11.004. ArXiv: [1705.02358](https://arxiv.org/abs/1705.02358).

## In-text citation audit

### C01 — line 170 — `feng2019reionization` — Supported

**Draft sentence:** “The end of reionization occurred within approximately the first billion years after the Big Bang, although the duration and topology of the transition depend on the abundance and radiative properties of the early sources.”

**Evidence:** “reionization ended approximately one billion years after the Big Bang” (S01, p. 1, Abstract); the review also identifies its “main sources” and the “topology of the ionized regions” as core reionization questions (p. 8, Sec. 3).

**Assessment:** Supports the timing and source/topology dependence. No wording change required.

### C02 — line 178 — `feng2019reionization` — Supported

**Draft sentence:** “This dependence is the physical motivation for introducing a clumping factor in reionization calculations.”

**Evidence:** “the rate will be proportional to the product of the number density of protons and electrons” (S01, p. 7, Sec. 2.3); the review later states that a “factor C ≈ 3–5 during EoR accounts for enhanced recombinations in a clumpy ionized IGM” (p. 21, Sec. 4).

**Assessment:** Directly supports the density-squared recombination argument and the use of a clumping factor.

### C03 — line 178 — `davies2024clumping` — Partly supported

**Draft sentence:** “This dependence is the physical motivation for introducing a clumping factor in reionization calculations.”

**Evidence:** The intended Davies et al. paper defines the recombination clumping factor in Eq. (5) and writes the photoionization-equilibrium balance with the clumping factor in Eq. (6) (S02, p. 3, Sec. 2).

**Assessment:** The intended source supports the claim, but the cited BibTeX key does not identify that paper. Replace the key with a corrected Davies record for S02.

### C04 — line 195 — `davies2024clumping` — Partly supported

**Draft sentence:** “Davies et al. show that this distinction is also important when relating the clumping factor to the ionizing mean free path and photoionization rate.”

**Evidence:** “the exact definition varies considerably between different works” and the paper then relates the clumping factor to photoionization rate and mean free path (S02, p. 3, Sec. 2, Eqs. 6–8).

**Assessment:** The substance is supported. Replace malformed key `davies2024clumping` with the corrected S02 record.

### C05 — line 197 — `feng2019reionization` — Supported

**Draft sentence:** “The reionization review literature often quotes values of order a few for simulation-based clumping factors during the epoch of reionization, but these values depend on redshift, resolution, the IGM definition, and whether the calculation is restricted to ionized gas.”

**Evidence:** “factor C ≈ 3–5 during EoR” and “There are various definitions for the clumping factor” (S01, p. 21, Sec. 4); the definition shown there is explicitly “restricted to ionized regions.”

**Assessment:** Supports “order a few,” differing definitions, and ionized-region selection. The review is less explicit on numerical resolution; retain the sentence but consider adding a simulation-method citation if resolution dependence is central.

### C06 — line 197 — `davies2024clumping` — Partly supported

**Draft sentence:** “More recent analyses of the observed mean free path and photoionization rate infer larger effective global values under ionization-equilibrium assumptions.”

**Evidence:** “C ∼ 10–15” with “an average value of C ≈ 12 at z = 5–6” (S02, p. 5, Sec. 3.2); the derivation assumes photoionization equilibrium (p. 3, Sec. 2, Eq. 6).

**Assessment:** Strongly supported by S02, but not by the malformed `davies2024clumping` metadata. Correct the key.

### C07 — line 202 — `vogelsberger2019simulations` — Supported

**Draft sentence:** “The large-scale initial density field is usually established by a linear matter power spectrum, while the late-time distribution is obtained by solving the gravitational dynamics and, in hydrodynamical simulations, the equations governing gas flows and thermochemistry.”

**Evidence:** “Initial conditions for cosmological simulations specify the perturbations” and are specified by their matter power spectrum (S03, p. 2, Sec. 2.2). Figure 2 states that dark matter follows collisionless gravitational dynamics while gas is described by hydrodynamic equations (p. 9).

**Assessment:** Direct support. No change required.

### C08 — line 206 — `garaldi2022thesan` — Partly supported

**Draft sentence:** “The THESAN suite is a radiation-hydrodynamical set of simulations designed to model the sources, gas, and radiation field responsible for hydrogen reionization.”

**Evidence:** The title identifies “radiation-magnetohydrodynamic simulations of the epoch of reionization” (S04, p. 1); the abstract describes a suite modelling IGM properties and resolved galaxies during reionization (p. 1).

**Assessment:** The claim is supported, but the key’s eprint points to the different Garaldi et al. IGM-properties paper. Replace `2110.01628` with `2110.00584`, and list Kannan as first author.

### C09 — line 208 — `despali2025aidatng` — Partly supported

**Draft sentence:** “Because the simulations use the same baryonic model and matched initial conditions across the different dark-matter scenarios, differences in the resulting density fields can be related more directly to the underlying dark-matter physics.”

**Evidence:** AIDA-TNG combines dark-matter variations with the “fiducial IllustrisTNG galaxy formation model” (S05, p. 1, Abstract) and runs the same cosmological volumes in multiple dark-matter scenarios (p. 2, Sec. 2). However, for WDM, a suppressed power spectrum is used “to re-create the initial conditions” (p. 3, Sec. 2.1).

**Assessment:** Same baryonic model is supported; “matched initial conditions across” is too broad because WDM initial transfer functions differ. Suggested wording: “Because the simulations use the same baryonic model and comparable volumes/initial phases while varying the dark-matter prescription, …”.

### C10 — line 232 — `irsic2024wdm` — Supported

**Draft sentence:** “Current analyses generally place the lower bound on a thermal-relic WDM mass at several keV, with representative results ranging from roughly 3 to 6 keV depending on the data and assumptions about the thermal history.”

**Evidence:** “lower limits … of 5.7 keV (at 95% C.L.)” (S06, p. 1, Abstract); using comparable scale and thermal-history choices lowers the bound to 4.1 keV (p. 20, Conclusions).

**Assessment:** Supports the upper end of the stated representative range and its dependence on data/thermal assumptions.

### C11 — line 232 — `villasenor2022wdm` — Supported

**Draft sentence:** “Current analyses generally place the lower bound on a thermal-relic WDM mass at several keV, with representative results ranging from roughly 3 to 6 keV depending on the data and assumptions about the thermal history.”

**Evidence:** The abstract reports “a lower limit m_WDM > 3.1 keV (95 percent CL)” from Lyman-alpha forest data at z≈4.0–5.2 (S07, p. 1, Abstract).

**Assessment:** Supports the lower end of the cited range. Correct the bibliography page/DOI: the paper is PRD **108**, 023502, doi:10.1103/PhysRevD.108.023502, not 083529.

### C12 — line 237 — `tulin2018sidm` — Supported

**Draft sentence:** “Repeated scattering can redistribute energy within a halo, isotropize the velocity distribution, alter halo shapes, and produce central density cores or other changes in the inner density profile.”

**Evidence:** The review describes SIDM simulations as “rich dynamics for elastic scattering” (S08, p. 35, Sec. IV) and notes the standard isotropic, velocity-independent interaction specified by a fixed σ/m (p. 25, Sec. III). Its halo-profile discussion treats core sizes explicitly (p. 50, Sec. V).

**Assessment:** The review supports the physical summary. No change required.

### C13 — line 251 — `despali2025aidatng` — Supported

**Draft sentence:** “The AIDA-TNG suite includes both a constant-cross-section SIDM model and a velocity-dependent model, alongside CDM and several thermal-relic WDM models.”

**Evidence:** “three WDM models … and two self-interacting dark matter scenarios” with either a constant or velocity-dependent cross-section (S05, p. 17, Conclusions); Sec. 2 lists CDM, WDM, SIDM1, and vSIDM runs (p. 2, Table 1).

**Assessment:** Directly supported. No change required.

### C14 — line 447 — `Davies_2026` — Supported

**Draft sentence:** “Davies et al. provide one such approach.”

**Evidence:** The paper’s purpose is to estimate the clumping factor from ionizing-background information and mean free path (S02, p. 2, Sec. 1); Sec. 2 gives the estimator equations.

**Assessment:** Supported. Add `eprint = {2406.18186}` and `archivePrefix = {arXiv}` to make the record complete.

### C15 — line 709 — `garaldi2022thesan` — Partly supported

**Draft sentence:** “THESAN therefore adopts this approximation to make the simulations computationally feasible, after demonstrating convergence of the neutral-fraction evolution at this value.”

**Evidence:** “Convergence on the evolution of the neutral fraction is achieved with a value of c-tilde = 0.2 c” (S04, p. 26, Appendix A, Fig. A1); the appendix explains that the approximation avoids the short time steps imposed by light speed (p. 26).

**Assessment:** The statement is accurately supported by S04. The only problem is bibliographic: correct the S04 arXiv identifier and lead author as noted in C08.

## Bibliography corrections required

| Key | Problem | Required correction |
|---|---|---|
| `davies2024clumping` | Metadata does not resolve to the intended Davies clumping-estimator paper; its eprint belongs to S02 while title/authors/DOI/year do not. | Replace with the verified S02 record, or cite the existing corrected `Davies_2026` key after adding `eprint = {2406.18186}`. |
| `garaldi2022thesan` | Mixed records: title/DOI are Kannan et al. but eprint `2110.01628` is Garaldi et al. | Use Kannan et al., eprint `2110.00584`, DOI `10.1093/mnras/stac238`. |
| `villasenor2022wdm` | Wrong article number and DOI. | Use PRD **108**, 023502; doi:10.1103/PhysRevD.108.023502. |
| `Davies_2026` | Valid publisher record, but missing arXiv metadata. | Add `eprint = {2406.18186}` and `archivePrefix = {arXiv}`. |

## Bibliography-only appendix

These entries occur in `apssamp.bib` but are not cited by `apssamp_draft.tex`; consequently they have no in-text proposition to verify.

| Key | Reference / source | Status |
|---|---|---|
| `Garaldi_2022` | E. Garaldi et al., *The THESAN Project: Properties of the Intergalactic Medium and Its Connection to Reionization-era Galaxies*, MNRAS **512**, 4909–4933 (2022), doi:10.1093/mnras/stac257; [arXiv:2110.01628](https://arxiv.org/abs/2110.01628). | Uncited; metadata corresponds to the arXiv paper. |
| `marsh2021ultralight` | E. G. M. Ferreira, *Ultra-light Dark Matter*, A&A Review **29**, 7 (2021), doi:10.1007/s00159-021-00135-6; [arXiv:1912.03148](https://arxiv.org/abs/1912.03148). | Uncited. |
| `irsic2017fdm` | V. Iršič et al., *First Constraints on Fuzzy Dark Matter from Lyman-alpha Forest Data and Hydrodynamical Simulations*, PRL **119**, 031302 (2017), doi:10.1103/PhysRevLett.119.031302; [arXiv:1703.04683](https://arxiv.org/abs/1703.04683). | Uncited. |

## Completeness check

All citation commands in `apssamp_draft.tex` are represented: lines 170, 178, 195, 197, 202, 206, 208, 232, 237, 251, 447, and 709. The duplicated source records at lines 178, 197, and 232 were separately verified. All arXiv links in the source index resolve to the stated papers; S02 is the correct arXiv source for the Davies estimator, including the use currently keyed as `davies2024clumping`.

## Missing-citation review

This review identifies claims that are externally grounded but currently have no nearby source. It does **not** recommend citations for the report's own calculations, figure interpretation, validation outcomes, or performance measurements. “Required” means the statement introduces borrowed physics, a published method, a code, a simulation design, or a numerical constant; “recommended” strengthens a broad synthesis; “optional” is standard material whose source would help a non-specialist reader.

| ID | Line(s) | Priority | Uncited claim / recommended insertion point | Best source and action |
|---|---:|---|---|---|
| MC01 | 158 | Required | Dark-matter evidence; large-scale success of Lambda-CDM; and small-scale challenges. Add after the first paragraph of the Introduction. | **N01** — new source. This is the appropriate balanced review; avoid saying small-scale phenomena “cannot be explained” without qualification. |
| MC02 | 160 | Recommended | “Most comparisons … focus on the power spectrum” and the power-spectrum definition. Add after the first sentence. | **S03** — already in bibliography. Cite the cosmological-simulation review. |
| MC03 | 183–192 | Required | Density and recombination-weighted clumping-factor definitions, and the required specification of gas/volume. Add after each displayed definition block, or once after the paragraph introducing both. | **S01 + S02** — already in bibliography after correcting S02’s key. |
| MC04 | 204 | Required | Particle deposition, direct Voronoi use, and sensitivity to mass assignment, grid resolution, smoothing, and empty cells. Add at the end of the paragraph. | **N02 + N03** — new sources for AREPO/Voronoi representation and FFT-grid systematics. |
| MC05 | 212–226 | Required | CDM as cold/collisionless, hierarchical small-halo formation, and scale-dependent changes in alternative dark-matter models. Add after the CDM paragraph and after the small-scale comparison paragraph. | **N01 + S05** — cite the CDM review and the AIDA-TNG paper. |
| MC06 | 230 | Required | WDM velocity dispersion/free streaming, suppressed high-k power, delayed low-mass halos, and reduced dense structure. Add at the end of the WDM-physics paragraph. | **S06 + S05** — already in bibliography. |
| MC07 | 239 | Required | The contrast between WDM/FDM initial-power effects and SIDM halo-structure effects; baryonic dependence. Add at the end of the paragraph. | **S08 + N04** — cite the SIDM review plus the existing FDM review. |
| MC08 | 245–249 | Required | Velocity-dependent SIDM is stronger in dwarfs and weaker in clusters. Add after the second sentence in the vSIDM paragraph. | **S05** — already in bibliography; its Sec. 2.2 states this behavior and specifies vSIDM. |
| MC09 | 255–257 | Recommended | The synthesis that WDM, SIDM, and vSIDM affect density fields through distinct mechanisms. Add at the end of the first “Connection to this work” paragraph. | **S05 + S08** — already in bibliography. |
| MC10 | 266–274 | Required | CIC mass assignment, grid-scale artifacts, and smoothing of gridded particle fields. Add after the CIC/smoothing discussion. | **N03** — new source; it directly treats grid assignment, aliasing, and the CIC window. |
| MC11 | 276–277 | Required | Pylians as the implementation used for spherical smoothing. Add directly after “Pylians”. | **N05** — new software citation required by Pylians itself. |
| MC12 | 279–286 | Required | Physical IGM selection, overdensity masking, and ionization thresholds. Add after the mask-definition paragraph. | **S02 + N06** — S02 for clumping-phase definitions and N06 for self-shielding/IGM density selection. |
| MC13 | 289–295 | Required | Fourier-transform power-spectrum estimator, spherical k-shell averaging, and grid/assignment effects. Add after the introductory power-spectrum paragraph. | **N03** — new source. |
| MC14 | 300–304 | Required | Spatial folding as a technique to extend measurable high-k range; its normalization and overlap checks. Add after the folding paragraph. | **N07** — new source. |
| MC15 | 309–319 | Required | Mean-free-path interpretation; optical-depth expression; tau=1 convention; hydrogen fraction and 912-A cross-section. Add after the mean-free-path derivation. | **S02 + N08** — S02 supports the estimator context; N08 supports the atomic cross section. State explicitly if X=0.76 is a THESAN input assumption. |
| MC16 | 322–353 | Required | Photoionization-rate integral, three-group approximation, and listed group-averaged cross sections. Add after the group approximation and after the numerical coefficient list. | **S04 + N08** — cite the THESAN methodology/source for radiation groups and N08 for atomic cross sections. If coefficients are pipeline constants, also cite the project configuration/data documentation. |
| MC17 | 368–397 | Required | THESAN particle counts, the statement that Pylians is an established package, and method claims about grid/smoothing behavior. Add S04 after the N values and N05 after “Pylians”; do not cite the figure-specific findings. | **S04 + N05** — existing THESAN source plus new Pylians citation. |
| MC18 | 415–422 | Required | CIC-window suppression and the claim that deconvolution amplifies high-k noise/aliasing. Add after each explanatory paragraph. | **N03** — new source. |
| MC19 | 489–535 | Required | The ionization-equilibrium estimator, lambda_mfp approximation, and first estimator equations attributed to Davies et al. Add the citation in the subsection opening and after the derived estimator. | **S02** — existing corrected Davies source. The current citation at line 447 is too distant for displayed borrowed equations. |
| MC20 | 542–573 | Required | Photon-density closure, Gamma_gamma=c/lambda_mfp, and second estimator equations attributed to Davies et al. Add in the subsection opening and after the final estimator. | **S02** — existing corrected Davies source. |
| MC21 | 615–630 | Required | AIDA-TNG’s model comparison and any statements about the suite’s CDM/WDM/SIDM/vSIDM predictions. Add in the Application II introduction. | **S05** — already in bibliography. Replace the current placeholder “Reference the AIDA-TNG paper …” with a substantive citation. |
| MC22 | 702–727 | Required | AIDA-TNG particle/Voronoi representation and the use/meaning of `SubfindHsml`. Add after the opening appendix paragraph. | **S05 + N02** — cite AIDA-TNG for the run/data context and AREPO for moving-mesh/Voronoi representation. |
| MC23 | 730–739 | Optional | General statement that reduced speed of light relaxes radiation-transport time-step costs. Move or repeat the existing THESAN citation immediately after this claim. | **S04** — already in bibliography. The present citation comes later and is sufficient only if its scope is made visually unambiguous. |

### New-source index for missing citations

**N01.** J. S. Bullock and M. Boylan-Kolchin, *Small-Scale Challenges to the Lambda-CDM Paradigm*, **Annual Review of Astronomy and Astrophysics 55**, 343–387 (2017), doi:10.1146/annurev-astro-091916-055313. ArXiv: [1707.04256](https://arxiv.org/abs/1707.04256).

**N02.** V. Springel, *E pur si muove: Galilean-invariant cosmological hydrodynamical simulations on a moving mesh*, **Monthly Notices of the Royal Astronomical Society 401**, 791–851 (2010), doi:10.1111/j.1365-2966.2009.15715.x. ArXiv: [0901.4107](https://arxiv.org/abs/0901.4107).

**N03.** Y. P. Jing, *Correcting for the Alias Effect when Measuring the Power Spectrum Using a Fast Fourier Transform*, **The Astrophysical Journal 620**, 559–563 (2005), doi:10.1086/427087. ArXiv: [astro-ph/0409240](https://arxiv.org/abs/astro-ph/0409240).

**N04.** E. G. M. Ferreira, *Ultra-light Dark Matter*, **The Astronomy and Astrophysics Review 29**, 7 (2021), doi:10.1007/s00159-021-00135-6. ArXiv: [1912.03148](https://arxiv.org/abs/1912.03148). This is already present as `marsh2021ultralight` but currently unused.

**N05.** F. Villaescusa-Navarro, *Pylians: Python Libraries for the Analysis of Numerical Simulations*, Astrophysics Source Code Library, ascl:1811.008 (2018), [software citation](https://pylians3.readthedocs.io/en/master/citation.html). This is the project’s requested citation.

**N06.** A. Rahmati, A. Pawlik, M. Raičević, and J. Schaye, *On the Evolution of the H I Column Density Distribution in Cosmological Simulations*, **Monthly Notices of the Royal Astronomical Society 430**, 2427–2445 (2013), doi:10.1093/mnras/stt066. ArXiv: [1210.7808](https://arxiv.org/abs/1210.7808).

**N07.** S. Colombi, A. H. Jaffe, D. Novikov, and C. Pichon, *Accurate Estimators of Power Spectra in N-body Simulations*, **Monthly Notices of the Royal Astronomical Society 393**, 511–526 (2009), doi:10.1111/j.1365-2966.2008.14207.x. ArXiv: [0811.0313](https://arxiv.org/abs/0811.0313).

**N08.** D. A. Verner, G. J. Ferland, K. T. Korista, and D. G. Yakovlev, *Atomic Data for Astrophysics. II. New Analytic Fits for Photoionization Cross Sections of Atoms and Ions*, **Atomic Data and Nuclear Data Tables 64**, 1–103 (1996), doi:10.1006/adnd.1996.0001. Publisher link: [ScienceDirect](https://doi.org/10.1006/adnd.1996.0001); no arXiv version located.

### Explicit non-findings

No new literature citation is recommended for the project’s numerical experiments, figure-specific conclusions, direct comparisons between the author’s own calculation backends, AIDA/THESAN measurement values computed in this work, the reduced-speed pilot-run results, or the parallelization timing/results. These should instead be traceable to the figures, tables, code, and data-provenance material in the report.
