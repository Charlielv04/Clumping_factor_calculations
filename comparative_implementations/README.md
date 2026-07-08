# Comparative implementations

This directory preserves external or legacy implementations used only for
scientific regression comparisons. Files here are not production entry points
and must not be imported or executed as application code.

Regression tests may extract individual functions from these sources and
compare their numerical results with the maintained package implementation.
Keeping the references inside the repository makes those comparisons
reproducible on clean machines and CI runners.

## Forest optical-depth reference

`forest/compute_tau.py` and `forest/line_list.txt` are the legacy optical-depth
implementation and its atomic-line table. The test suite parses the function
definitions from the script without executing its production loop.
