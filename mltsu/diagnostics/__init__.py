"""
Convergence diagnostics for MCMC and thermodynamic sampling.

This module provides rigorous diagnostics to verify that sampling
has converged and results are reliable.
"""

from .convergence import (
    ConvergenceDiagnostics,
    GelmanRubinDiagnostic,
    EffectiveSampleSize,
    MCMCError,
    GewekeDiagnostic,
    HeidelbergerWelchTest,
)

__all__ = [
    'ConvergenceDiagnostics',
    'GelmanRubinDiagnostic',
    'EffectiveSampleSize',
    'MCMCError',
    'GewekeDiagnostic',
    'HeidelbergerWelchTest',
]