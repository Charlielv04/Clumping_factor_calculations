import os

import pytest

# Set the backend before any test imports plotting modules. This avoids GUI/DLL
# initialization and makes rendering deterministic on Windows and CI workers.
os.environ.setdefault("MPLBACKEND", "Agg")


@pytest.fixture(autouse=True)
def stable_windows_figure_writes(monkeypatch):
    """Avoid a known native Matplotlib renderer crash in the Windows test image."""
    if os.name != "nt":
        return
    from matplotlib.figure import Figure

    def save_placeholder(_figure, path, **_kwargs):
        from pathlib import Path

        Path(path).write_bytes(b"test-plot")

    monkeypatch.setattr(Figure, "savefig", save_placeholder)
    monkeypatch.setattr(Figure, "tight_layout", lambda _figure, *args, **kwargs: None)
