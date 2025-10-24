"""
Test-Driven Development for Numerical Stability Checks.

Tests the @numerical_stability_check decorator and related utilities.
Ensures NaN/Inf detection works correctly with minimal performance overhead.
"""

import pytest
import jax.numpy as jnp
from computable_flows_shim.core import numerical_stability_check, NumericalInstabilityError, check_numerical_stability


class TestNumericalStabilityDecorator:
    """Test the @numerical_stability_check decorator."""

    def test_decorator_preserves_function_behavior(self):
        """Test that decorator doesn't change normal function behavior."""
        @numerical_stability_check
        def add_arrays(x, y):
            return x + y

        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.array([4.0, 5.0, 6.0])
        result = add_arrays(x, y)

        expected = jnp.array([5.0, 7.0, 9.0])
        assert jnp.allclose(result, expected)

    def test_decorator_detects_nan_input(self):
        """Test that decorator detects NaN in inputs."""
        @numerical_stability_check
        def identity(x):
            return x

        x = jnp.array([1.0, float('nan'), 3.0])

        with pytest.raises(NumericalInstabilityError) as exc_info:
            identity(x)

        assert "contains NaN values" in str(exc_info.value)
        assert "arg_0" in str(exc_info.value)

    def test_decorator_detects_inf_input(self):
        """Test that decorator detects infinity in inputs."""
        @numerical_stability_check
        def identity(x):
            return x

        x = jnp.array([1.0, float('inf'), 3.0])

        with pytest.raises(NumericalInstabilityError) as exc_info:
            identity(x)

        assert "contains infinite values" in str(exc_info.value)

    def test_decorator_detects_nan_output(self):
        """Test that decorator detects NaN in outputs."""
        @numerical_stability_check
        def divide_by_zero(x):
            return x / 0.0  # Creates inf, but let's use a function that creates NaN

        @numerical_stability_check
        def create_nan(x):
            return jnp.sqrt(-1.0)  # NaN result

        x = jnp.array([1.0])

        with pytest.raises(NumericalInstabilityError) as exc_info:
            create_nan(x)

        assert "contains NaN values" in str(exc_info.value)
        assert "result" in str(exc_info.value)

    def test_decorator_detects_inf_output(self):
        """Test that decorator detects infinity in outputs."""
        @numerical_stability_check
        def divide_by_zero(x):
            return x / 0.0

        x = jnp.array([1.0])

        with pytest.raises(NumericalInstabilityError) as exc_info:
            divide_by_zero(x)

        assert "contains infinite values" in str(exc_info.value)

    def test_decorator_ignores_non_array_inputs(self):
        """Test that decorator ignores non-JAX array inputs."""
        @numerical_stability_check
        def func_with_scalars(x, y, scalar_param=1.0):
            return x + y + scalar_param

        x = jnp.array([1.0, 2.0])
        y = jnp.array([3.0, 4.0])

        # Should work fine with scalar parameters
        result = func_with_scalars(x, y, scalar_param=5.0)
        expected = jnp.array([9.0, 11.0])
        assert jnp.allclose(result, expected)

    def test_decorator_with_keyword_arguments(self):
        """Test decorator with keyword arguments."""
        @numerical_stability_check
        def func_with_kwargs(x, y=None):
            if y is None:
                return x * 2
            return x + y

        x = jnp.array([1.0, 2.0])
        y = jnp.array([3.0, 4.0])

        # Test with positional args
        result1 = func_with_kwargs(x)
        assert jnp.allclose(result1, jnp.array([2.0, 4.0]))

        # Test with keyword args
        result2 = func_with_kwargs(x, y=y)
        assert jnp.allclose(result2, jnp.array([4.0, 6.0]))

    def test_decorator_with_nan_in_kwargs(self):
        """Test decorator detects NaN in keyword arguments."""
        @numerical_stability_check
        def add(x, y):
            return x + y

        x = jnp.array([1.0, 2.0])
        y = jnp.array([float('nan'), 4.0])

        with pytest.raises(NumericalInstabilityError) as exc_info:
            add(x, y=y)

        assert "contains NaN values" in str(exc_info.value)
        assert "y" in str(exc_info.value)

    def test_decorator_with_multiple_outputs(self):
        """Test decorator with functions returning multiple values."""
        @numerical_stability_check
        def multiple_outputs(x):
            return x * 2, x + 1, x / 2

        x = jnp.array([2.0, 4.0])
        result1, result2, result3 = multiple_outputs(x)

        assert jnp.allclose(result1, jnp.array([4.0, 8.0]))
        assert jnp.allclose(result2, jnp.array([3.0, 5.0]))
        assert jnp.allclose(result3, jnp.array([1.0, 2.0]))

    def test_decorator_with_dict_output(self):
        """Test decorator with functions returning dictionaries."""
        @numerical_stability_check
        def dict_output(x):
            return {
                'doubled': x * 2,
                'halved': x / 2,
                'scalar': 42  # Non-array, should be ignored
            }

        x = jnp.array([2.0, 4.0])
        result = dict_output(x)

        assert jnp.allclose(result['doubled'], jnp.array([4.0, 8.0]))
        assert jnp.allclose(result['halved'], jnp.array([1.0, 2.0]))
        assert result['scalar'] == 42

    def test_decorator_with_nan_in_dict_output(self):
        """Test decorator detects NaN in dictionary outputs."""
        @numerical_stability_check
        def dict_output_with_nan(x):
            return {
                'good': x * 2,
                'bad': jnp.sqrt(-x)  # Creates NaN
            }

        x = jnp.array([1.0, 4.0])

        with pytest.raises(NumericalInstabilityError) as exc_info:
            dict_output_with_nan(x)

        assert "contains NaN values" in str(exc_info.value)
        assert "result['bad']" in str(exc_info.value)


class TestManualNumericalStabilityCheck:
    """Test the manual check_numerical_stability function."""

    def test_manual_check_passes_valid_array(self):
        """Test manual check passes for valid arrays."""
        x = jnp.array([1.0, 2.0, 3.0])
        # Should not raise
        check_numerical_stability(x, "test_array")

    def test_manual_check_detects_nan(self):
        """Test manual check detects NaN."""
        x = jnp.array([1.0, float('nan'), 3.0])

        with pytest.raises(NumericalInstabilityError) as exc_info:
            check_numerical_stability(x, "test_array")

        assert "test_array contains NaN values" in str(exc_info.value)

    def test_manual_check_detects_inf(self):
        """Test manual check detects infinity."""
        x = jnp.array([1.0, float('inf'), 3.0])

        with pytest.raises(NumericalInstabilityError) as exc_info:
            check_numerical_stability(x, "test_array")

        assert "test_array contains infinite values" in str(exc_info.value)

    def test_manual_check_ignores_non_arrays(self):
        """Test manual check ignores non-JAX arrays."""
        # Should not raise for non-arrays
        check_numerical_stability(42, "scalar")
        check_numerical_stability("string", "string")
        check_numerical_stability([1, 2, 3], "list")


class TestPerformanceCharacteristics:
    """Test that the decorator has minimal performance overhead."""

    def test_decorator_overhead_is_minimal(self):
        """Test that decorator overhead is negligible for normal operations."""
        import time

        @numerical_stability_check
        def simple_math(x):
            return x * 2 + 1

        x = jnp.array([1.0, 2.0, 3.0])

        # Time multiple calls
        start_time = time.time()
        for _ in range(1000):
            result = simple_math(x)
        end_time = time.time()

        # Should complete quickly (less than 1 second for 1000 calls)
        duration = end_time - start_time
        assert duration < 1.0, f"Decorator overhead too high: {duration}s for 1000 calls"

        # Verify results are still correct
        assert jnp.allclose(result, jnp.array([3.0, 5.0, 7.0]))


class TestErrorPropagation:
    """Test that original exceptions are preserved."""

    def test_original_exceptions_preserved(self):
        """Test that non-numerical errors are re-raised unchanged."""
        @numerical_stability_check
        def function_that_raises(x):
            raise ValueError("Original error message")

        x = jnp.array([1.0, 2.0])

        with pytest.raises(ValueError) as exc_info:
            function_that_raises(x)

        assert "Original error message" in str(exc_info.value)

    def test_numerical_errors_take_precedence(self):
        """Test that numerical errors are caught before other errors."""
        @numerical_stability_check
        def function_with_nan_and_error(x):
            if jnp.isnan(x).any():
                raise ValueError("This should not be reached")
            return x

        x = jnp.array([float('nan')])

        # Should catch NaN first, not the ValueError
        with pytest.raises(NumericalInstabilityError):
            function_with_nan_and_error(x)