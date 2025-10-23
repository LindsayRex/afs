"""
Computable Flows Shim - High-level runtime adapter for physics-based flows.

This package compiles declarative energy specifications into fast, composable flows
using four core primitives: F_Dis, F_Proj, F_Multi, F_Con.
"""

__version__ = "0.1.0"

# Logging infrastructure
from .logging import configure_logging, get_logger, log_performance

__all__ = [
    'configure_logging',
    'get_logger', 
    'log_performance'
]