from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ClumpingMethodResult:
    """Stable result envelope; numerical arrays remain legacy-compatible."""

    document: dict[str, Any]

