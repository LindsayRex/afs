"""
Computable Flows Shim - High-level runtime adapter for physics-based flows.

This package compiles declarative energy specifications into fast, composable flows
using four core primitives: F_Dis, F_Proj, F_Multi, F_Con.
"""

__version__ = "0.1.0"

# Core configuration and type system
from . import config
from .config import (
    configure_jax_environment,
    create_array,
    enforce_dtype,
    get_dtype,
    get_xla_flags_for_platform,
    ones,
    random_normal,
    validate_dtype_consistency,
    validate_xla_flags,
    zeros,
)

# Logging infrastructure
from .logging import configure_logging, get_logger, log_performance

__all__ = [
    # Configuration and types
    "config",
    "configure_jax_environment",
    "configure_logging",
    "create_array",
    "enforce_dtype",
    "get_dtype",
    "get_logger",
    "get_xla_flags_for_platform",
    "log_performance",
    "ones",
    "random_normal",
    "validate_dtype_consistency",
    "validate_xla_flags",
    "zeros",
]
