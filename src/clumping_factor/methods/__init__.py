"""Declarative method metadata and domain facades.

The registry describes how a calculation is assembled.  It deliberately does
not contain numerical code; implementations remain in the compatibility
modules under :mod:`clumping_factor`.
"""

from .registry import METHOD_REGISTRY, MethodRegistry, MethodSpec, expand_preset, method_catalog

__all__ = ["METHOD_REGISTRY", "MethodRegistry", "MethodSpec", "expand_preset", "method_catalog"]
