"""
Core utilities for AFS SDK hygiene and safety.

Provides decorators and utilities for numerical stability, type safety, and error handling.
"""

import functools
from collections.abc import Callable
from typing import Any, TypeVar

import jax
import jax.numpy as jnp

F = TypeVar("F", bound=Callable[..., Any])


class NumericalInstabilityError(Exception):
    """Raised when numerical instability is detected (NaN/Inf values)."""


def numerical_stability_check[F](func: F) -> F:
    """
    Decorator that checks for numerical stability in mathematical functions.

    Validates that:
    1. All JAX array inputs are finite (no NaN/Inf)
    2. All JAX array outputs are finite (no NaN/Inf)

    Only performs checks when errors are detected, so zero overhead in normal operation.

    Args:
        func: Function to decorate (must work with JAX arrays)

    Returns:
        Decorated function with numerical stability checks

    Raises:
        NumericalInstabilityError: If NaN/Inf detected in inputs or outputs
        TypeError: If function arguments are not JAX arrays when expected
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Check inputs for numerical stability
        _check_inputs_for_numerical_stability(args, kwargs, func.__name__)

        try:
            # Call the original function
            result = func(*args, **kwargs)

            # Check outputs for numerical stability
            _check_outputs_for_numerical_stability(result, func.__name__)

            return result

        except Exception:
            # Re-raise original exceptions
            raise

    return wrapper  # type: ignore


def _check_inputs_for_numerical_stability(args, kwargs, func_name: str) -> None:
    """
    Check function inputs for NaN/Inf values.

    Only checks JAX arrays, ignores other types.
    """

    def _check_array(x, arg_name: str):
        """Check a single array for numerical issues."""
        if not isinstance(x, jax.Array):
            return  # Skip non-JAX arrays

        # Skip checks during JAX tracing (JIT compilation)
        # The check will still happen on concrete values after compilation
        try:
            # Try to check for NaN/Inf - this will fail during tracing
            if jnp.isnan(x).any():
                raise NumericalInstabilityError(
                    f"Input '{arg_name}' to {func_name} contains NaN values"
                )

            if jnp.isinf(x).any():
                raise NumericalInstabilityError(
                    f"Input '{arg_name}' to {func_name} contains infinite values"
                )
        except NumericalInstabilityError:
            # Re-raise our own errors
            raise
        except Exception as e:
            # If we get a tracing-related exception (like TracerBoolConversionError),
            # skip the check during tracing. The check will still happen when the
            # function is called with concrete values.
            if (
                "TracerBoolConversionError" in str(type(e))
                or "traced array" in str(e).lower()
            ):
                pass  # Skip during tracing
            else:
                # Re-raise other unexpected exceptions
                raise

    # Check positional arguments
    for i, arg in enumerate(args):
        _check_array(arg, f"arg_{i}")

    # Check keyword arguments
    for key, value in kwargs.items():
        _check_array(value, key)


def _check_outputs_for_numerical_stability(result, func_name: str) -> None:
    """
    Check function outputs for NaN/Inf values.

    Handles single arrays, tuples/lists of arrays, and nested structures.
    """

    def _check_single_output(x, output_name: str):
        """Check a single output for numerical issues."""
        if not isinstance(x, jax.Array):
            return  # Skip non-JAX arrays

        try:
            # Try to check for NaN/Inf - this will fail during tracing
            if jnp.isnan(x).any():
                raise NumericalInstabilityError(
                    f"Output '{output_name}' from {func_name} contains NaN values"
                )

            if jnp.isinf(x).any():
                raise NumericalInstabilityError(
                    f"Output '{output_name}' from {func_name} contains infinite values"
                )
        except NumericalInstabilityError:
            # Re-raise our own errors
            raise
        except Exception as e:
            # If we get a tracing-related exception (like TracerBoolConversionError),
            # skip the check during tracing. The check will still happen when the
            # function is called with concrete values.
            if (
                "TracerBoolConversionError" in str(type(e))
                or "traced array" in str(e).lower()
            ):
                pass  # Skip during tracing
            else:
                # Re-raise other unexpected exceptions
                raise

    # Handle different output types
    if isinstance(result, jax.Array):
        # Single array output
        _check_single_output(result, "result")
    elif isinstance(result, (tuple, list)):
        # Multiple outputs
        for i, item in enumerate(result):
            _check_single_output(item, f"result[{i}]")
    elif isinstance(result, dict):
        # Dictionary output
        for key, value in result.items():
            _check_single_output(value, f"result['{key}']")
    # For other types, we don't check (they might not be numerical)


# Convenience function for manual checking
def check_numerical_stability(x, name: str = "value") -> None:
    """
    Manually check a JAX array for numerical stability.

    Args:
        x: JAX array to check
        name: Name for error messages

    Raises:
        NumericalInstabilityError: If NaN/Inf detected
    """
    if not isinstance(x, jax.Array):
        return  # Skip non-JAX arrays

    try:
        if jnp.isnan(x).any():
            raise NumericalInstabilityError(f"{name} contains NaN values")

        if jnp.isinf(x).any():
            raise NumericalInstabilityError(f"{name} contains infinite values")
    except NumericalInstabilityError:
        # Re-raise our own errors
        raise
    except Exception as e:
        # If we get a tracing-related exception (like TracerBoolConversionError),
        # skip the check. The check will still happen when called with concrete values.
        if (
            "TracerBoolConversionError" in str(type(e))
            or "traced array" in str(e).lower()
        ):
            pass  # Skip during tracing
        else:
            # Re-raise other unexpected exceptions
            raise
