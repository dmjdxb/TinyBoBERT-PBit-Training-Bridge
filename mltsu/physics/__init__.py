"""
Physics-realistic modeling for thermodynamic computing.

This module provides physically accurate noise models, thermalization dynamics,
and device imperfections that real TSU hardware would exhibit.
"""

from .realistic_noise import (
    ThermodynamicNoiseModel,
    DeviceImperfections,
    OrnsteinUhlenbeckProcess,
    JohnsonNyquistNoise,
)
from .thermalization import (
    ThermalizationModel,
    DetailedBalanceChecker,
    MixingTimeEstimator,
    EffectiveSampleSize,
)
from .device_imperfections import (
    ManufacturingVariations,
    ReadoutErrors,
    CrosstalkModel,
    TemperatureDrift,
)

__all__ = [
    # Noise models
    'ThermodynamicNoiseModel',
    'DeviceImperfections',
    'OrnsteinUhlenbeckProcess',
    'JohnsonNyquistNoise',
    # Thermalization
    'ThermalizationModel',
    'DetailedBalanceChecker',
    'MixingTimeEstimator',
    'EffectiveSampleSize',
    # Device imperfections
    'ManufacturingVariations',
    'ReadoutErrors',
    'CrosstalkModel',
    'TemperatureDrift',
]