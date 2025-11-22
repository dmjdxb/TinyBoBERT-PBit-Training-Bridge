"""
Importance sampling for thermodynamic computing.

This module provides mathematically correct sampling methods that replace
the naive averaging in the original code.
"""

from .importance_sampling import (
    ImportanceSampler,
    ImportanceSampledAttention,
    SelfNormalizedEstimator,
    EffectiveSampleSize,
)

__all__ = [
    'ImportanceSampler',
    'ImportanceSampledAttention',
    'SelfNormalizedEstimator',
    'EffectiveSampleSize',
]