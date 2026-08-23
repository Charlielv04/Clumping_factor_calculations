# Compiling the paper locally

The paper can be compiled locally with the Tectonic executable and build helper in the project’s `.local-latex` folder. These instructions assume that PowerShell is being run from the project root:

```powershell
Set-Location "C:\Users\carlo\Documents\ClumpingFactorSuite"
```

## Preview version

To compile the shorter preview document:

```powershell
& ".\.local-latex\build_latex.ps1"
```

The PDF is written to:

```text
Clumping_factor_calculations\reports\build\apssamp_preview.pdf
```

## Full paper

To compile the full draft:

```powershell
& ".\.local-latex\build_latex.ps1" -Full
```

The PDF is written to:

```text
Clumping_factor_calculations\reports\build\apssamp_draft.pdf
```

The source file is `apssamp_draft.tex`, and the bibliography is `apssamp.bib`. Figures are stored in the `figures` subfolder.

The helper uses the local Tectonic cache and therefore does not require a separate LaTeX installation. If compilation fails because the local `.local-latex` tools or cache are missing, restore that folder before running the commands above.
