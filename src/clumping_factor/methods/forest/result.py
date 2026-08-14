from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ForestMethodResult:
    document: dict[str, Any]

