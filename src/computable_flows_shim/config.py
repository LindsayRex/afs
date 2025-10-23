# JAX Configuration and Type Enforcement
"""
Centralized configuration for JAX settings, dtype policies, and XLA flags.
This module ensures consistent dtype usage and JAX configuration across the entire AFS system.
"""

import os
import jax
import jax.numpy as jnp
from typing import Dict, Any, Optional

# =============================================================================
# DTYPE POLICY
# =============================================================================

# Default dtype for all AFS computations
# Using float64 for numerical stability in differential geometry (JAX x64 enabled by default)
DEFAULT_DTYPE = jnp.float64
DEFAULT_COMPLEX_DTYPE = jnp.complex128

# Explicit dtype mappings for different use cases
DTYPE_POLICY = {
    # Core computational dtypes (64-bit by default for numerical stability)
    'default': DEFAULT_DTYPE,
    'complex': DEFAULT_COMPLEX_DTYPE,

    # Specialized dtypes for specific operations
    'high_precision': jnp.float64,      # Default (for when precision is critical)
    'low_precision': jnp.float32,       # For memory-constrained operations
    'lowest_precision': jnp.float16,    # For extreme memory constraints
    'integer': jnp.int64,               # Default integer type (64-bit)
    'boolean': jnp.bool_,               # Boolean operations

    # Legacy compatibility (explicit precision specification)
    'float16': jnp.float16,
    'float32': jnp.float32,
    'float64': jnp.float64,
    'complex64': jnp.complex64,
    'complex128': jnp.complex128,
    'int32': jnp.int32,
    'int64': jnp.int64,
}

# =============================================================================
# JAX/XLA CONFIGURATION
# =============================================================================

# XLA Flags for different environments
# These should be set via XLA_FLAGS environment variable before importing JAX

# XLA Flags for different environments
# Simplified to essential optimizations only

XLA_FLAGS_CPU = [
    '--xla_cpu_multi_thread_eigen=true',           # Use multiple threads for Eigen operations
    '--xla_cpu_enable_fast_math=true',             # Enable fast math approximations
]

XLA_FLAGS_GPU = [
    '--xla_gpu_enable_fast_min_max=true',          # Optimize min/max operations
    '--xla_gpu_enable_async_all_reduce=true',      # Async collective operations
]

XLA_FLAGS_TPU = [
    '--xla_tpu_enable_async_collective_fusion=true',  # Fuse collective operations
]

# Performance flags (apply to all platforms)
XLA_FLAGS_PERFORMANCE = [
    '--xla_enable_fast_math=true',                  # General fast math
    '--xla_optimization_level=3',                   # Highest optimization level
]

# Debug flags (for development/troubleshooting)
XLA_FLAGS_DEBUG = [
    '--xla_dump_hlo_as_text=true',                  # Dump HLO as text
    '--xla_enable_dumping=true',                    # Enable general dumping
]

# =============================================================================
# CONFIGURATION FUNCTIONS
# =============================================================================

def get_dtype(name: str = 'default') -> jnp.dtype:
    """Get dtype by name from the policy."""
    if name not in DTYPE_POLICY:
        raise ValueError(f"Unknown dtype '{name}'. Available: {list(DTYPE_POLICY.keys())}")
    return DTYPE_POLICY[name]

def enforce_dtype(array: jax.Array, dtype: Optional[str] = None) -> jax.Array:
    """Enforce dtype on an array, converting if necessary."""
    target_dtype = get_dtype(dtype) if dtype else DEFAULT_DTYPE
    return jnp.asarray(array, dtype=target_dtype)

def get_xla_flags_for_platform(platform: str = 'auto') -> str:
    """
    Get XLA flags string for the specified platform.

    Args:
        platform: 'cpu', 'gpu', 'tpu', or 'auto' (detect automatically)

    Returns:
        Space-separated XLA flags string
    """
    if platform == 'auto':
        # Auto-detect platform
        try:
            devices = jax.devices()
            if any('TPU' in str(d) for d in devices):
                platform = 'tpu'
            elif any('GPU' in str(d) for d in devices):
                platform = 'gpu'
            else:
                platform = 'cpu'
        except:
            platform = 'cpu'  # Fallback

    # Get platform-specific flags
    if platform == 'cpu':
        flags = XLA_FLAGS_CPU.copy()
    elif platform == 'gpu':
        flags = XLA_FLAGS_GPU.copy()
    elif platform == 'tpu':
        flags = XLA_FLAGS_TPU.copy()
    else:
        raise ValueError(f"Unknown platform '{platform}'")

    # Add performance flags
    flags.extend(XLA_FLAGS_PERFORMANCE)

    # Add debug flags if in debug mode
    if os.getenv('AFS_DEBUG', '').lower() == 'true':
        flags.extend(XLA_FLAGS_DEBUG)

    flags_str = ' '.join(flags)

    # Validate flags for conflicts
    validate_xla_flags(flags_str)

    return flags_str

def configure_jax_environment():
    """
    Configure JAX environment with appropriate XLA flags and settings.
    Call this before importing other JAX-dependent modules.

    IMPORTANT: This enables 64-bit precision by default for numerical stability
    in differential geometry applications. Set AFS_DISABLE_64BIT=true to disable.
    """
    # Set XLA flags if not already set
    if 'XLA_FLAGS' not in os.environ:
        platform = os.getenv('AFS_JAX_PLATFORM', 'auto')
        os.environ['XLA_FLAGS'] = get_xla_flags_for_platform(platform)

    # Set JAX platform if specified
    jax_platform = os.getenv('JAX_PLATFORM_NAME')
    if jax_platform:
        jax.config.update('jax_platform_name', jax_platform)

    # Enable 64-bit precision by default for numerical stability
    # Only disable if explicitly requested (for performance/memory reasons)
    disable_64bit = os.getenv('AFS_DISABLE_64BIT', '').lower() == 'true'
    if not disable_64bit:
        jax.config.update('jax_enable_x64', True)

def validate_xla_flags(flags_str: str) -> bool:
    """
    Validate XLA flags string for conflicts and invalid combinations.

    Args:
        flags_str: Space-separated XLA flags string

    Returns:
        True if flags are valid, raises ValueError if invalid
    """
    flags = flags_str.split()

    # Check for conflicting optimization levels
    opt_levels = [f for f in flags if f.startswith('--xla_optimization_level=')]
    if len(opt_levels) > 1:
        raise ValueError(f"Multiple optimization levels specified: {opt_levels}")

    # Check for valid optimization levels
    for opt_flag in opt_levels:
        level = int(opt_flag.split('=')[1])
        if level not in [0, 1, 2, 3]:
            raise ValueError(f"Invalid optimization level: {level}")

    # Check for conflicting fast math settings
    fast_math_flags = [f for f in flags if 'fast_math' in f]
    conflicting_fast_math = []
    for flag in fast_math_flags:
        if flag.endswith('=false'):
            conflicting_fast_math.append(flag)

    if conflicting_fast_math:
        raise ValueError(f"Conflicting fast math settings: {conflicting_fast_math}")

    return True

def validate_dtype_consistency():
    """
    Validate that arrays use consistent dtypes according to policy.
    This is a development-time check that can be called in tests.
    """
    # This would be expanded to check actual arrays in the system
    # For now, just validate the policy is internally consistent
    assert DEFAULT_DTYPE == jnp.float64, "Default dtype should be float64 for numerical stability"
    assert DEFAULT_COMPLEX_DTYPE == jnp.complex128, "Default complex dtype should be complex128"
    return True

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_array(*args, dtype: Optional[str] = None, **kwargs) -> jax.Array:
    """Create array with enforced dtype."""
    target_dtype = get_dtype(dtype) if dtype else DEFAULT_DTYPE
    return jnp.array(*args, dtype=target_dtype, **kwargs)

def zeros(*args, dtype: Optional[str] = None, **kwargs) -> jax.Array:
    """Create zeros array with enforced dtype."""
    target_dtype = get_dtype(dtype) if dtype else DEFAULT_DTYPE
    return jnp.zeros(*args, dtype=target_dtype, **kwargs)

def ones(*args, dtype: Optional[str] = None, **kwargs) -> jax.Array:
    """Create ones array with enforced dtype."""
    target_dtype = get_dtype(dtype) if dtype else DEFAULT_DTYPE
    return jnp.ones(*args, dtype=target_dtype, **kwargs)

def random_normal(key, shape, dtype: Optional[str] = None, **kwargs) -> jax.Array:
    """Create random normal array with enforced dtype."""
    target_dtype = get_dtype(dtype) if dtype else DEFAULT_DTYPE
    return jax.random.normal(key, shape, dtype=target_dtype, **kwargs)

# =============================================================================
# AUTO-CONFIGURATION
# =============================================================================

# Auto-configure on import if this is the main AFS system
if __name__ != '__main__' and 'afs' in str(__file__).lower():
    configure_jax_environment()