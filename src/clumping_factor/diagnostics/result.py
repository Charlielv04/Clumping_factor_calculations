from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DiagnosticResult:
    document: dict[str, Any]

