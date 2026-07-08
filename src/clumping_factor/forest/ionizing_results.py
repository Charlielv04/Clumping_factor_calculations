"""Typed ionizing results and shared document builders."""

from .ionizing import gamma_result_document, mfp_result_document
from .ionizing_models import GammaHIResult, MeanFreePathResult

__all__ = ["GammaHIResult", "MeanFreePathResult", "gamma_result_document", "mfp_result_document"]
