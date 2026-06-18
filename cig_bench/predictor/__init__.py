"""Inference pipelines for CIG_Bench tasks."""

from . import utils  # noqa: F401
from ._download import (  # noqa: F401
    ensure_weight,
    MODELSCOPE_REGISTRY,
    WEIGHT_REGISTRY,  # alias of MODELSCOPE_REGISTRY (backward compat)
)

__all__ = ["utils", "channel", "fault", "karst", "property", "rgt",
           "ensure_weight",
           "MODELSCOPE_REGISTRY", "WEIGHT_REGISTRY"]
