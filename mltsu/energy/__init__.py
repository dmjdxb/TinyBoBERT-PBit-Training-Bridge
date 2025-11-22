"""
Realistic energy accounting for thermodynamic computing.

This module provides accurate energy models that include ALL costs:
- Core switching energy
- Control logic overhead
- Data movement
- Thermal management
- Power delivery losses

These realistic estimates replace the oversimplified claims in the original code.
"""

from .realistic_accounting import (
    RealisticEnergyModel,
    EnergyBreakdown,
    SystemEnergyCalculator,
    EnergyComparison,
    PowerDeliveryModel,
)
from .overhead_models import (
    ControlOverhead,
    DataMovementEnergy,
    ThermalManagement,
    ErrorCorrectionOverhead,
    SystemIntegrationCosts,
)

__all__ = [
    # Energy accounting
    'RealisticEnergyModel',
    'EnergyBreakdown',
    'SystemEnergyCalculator',
    'EnergyComparison',
    'PowerDeliveryModel',
    # Overhead models
    'ControlOverhead',
    'DataMovementEnergy',
    'ThermalManagement',
    'ErrorCorrectionOverhead',
    'SystemIntegrationCosts',
]