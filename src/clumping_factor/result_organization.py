"""Compatibility facade for the domain-neutral result organizer."""

from .infrastructure.result_organization import FamilyOrganizer, build_manifest, main

__all__ = ["FamilyOrganizer", "build_manifest", "main"]
