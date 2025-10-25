"""
Cross-cutting tests for device x precision combinations.

Tests dtype enforcement and numerical behavior across CPU/GPU and float32/float64.
"""

import jax
import jax.numpy as jnp
import pytest
from jax import random

from computable_flows_shim import configure_jax_environment


class TestDevicePrecisionCrossCutting:
    """Test dtype enforcement across device and precision combinations."""

    @pytest.mark.parametrize(
        "device_type,precision",
        [
            ("cpu", jnp.float32),
            ("cpu", jnp.float64),
            ("gpu_mock", jnp.float32),
            ("gpu_mock", jnp.float64),
        ],
    )
    def test_array_creation_dtype_enforcement(self, device_type, precision):
        """Test that arrays are created with correct dtypes across devices."""
        # Configure JAX for this test
        configure_jax_environment()

        # Create test array
        test_array = jnp.array([1.0, 2.0, 3.0], dtype=precision)

        # Verify dtype
        assert test_array.dtype == precision, (
            f"Expected {precision}, got {test_array.dtype}"
        )

        # Verify device context (mock for GPU)
        if device_type == "gpu_mock":
            # In mock GPU mode, we still use CPU but simulate GPU behavior
            assert jax.default_backend() == "cpu"  # JAX limitation on Windows
        else:
            assert jax.default_backend() == "cpu"

    @pytest.mark.parametrize(
        "device_type,precision",
        [
            ("cpu", jnp.float32),
            ("cpu", jnp.float64),
            ("gpu_mock", jnp.float32),
            ("gpu_mock", jnp.float64),
        ],
    )
    def test_random_generation_dtype_consistency(self, device_type, precision):
        """Test random number generation maintains dtype across devices."""
        configure_jax_environment()

        key = random.PRNGKey(42)
        tolerance = 1e-5 if precision == jnp.float32 else 1e-12

        # Generate random arrays
        rand_array = random.normal(key, (10,), dtype=precision)

        # Verify dtype
        assert rand_array.dtype == precision

        # Verify numerical properties
        assert jnp.isfinite(rand_array).all()
        assert rand_array.std() > tolerance  # Should have non-zero variance

    @pytest.mark.parametrize(
        "device_type,precision",
        [
            ("cpu", jnp.float32),
            ("cpu", jnp.float64),
            ("gpu_mock", jnp.float32),
            ("gpu_mock", jnp.float64),
        ],
    )
    def test_matrix_operations_precision(self, device_type, precision):
        """Test matrix operations maintain precision across devices."""
        configure_jax_environment()

        tolerance = 1e-5 if precision == jnp.float32 else 1e-12

        # Create test matrices
        A = jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=precision)
        B = jnp.array([[2.0, 0.0], [0.0, 2.0]], dtype=precision)

        # Matrix multiplication
        C = jnp.dot(A, B)

        # Verify dtypes preserved
        assert A.dtype == precision
        assert B.dtype == precision
        assert C.dtype == precision

        # Verify numerical accuracy
        expected = jnp.array([[2.0, 4.0], [6.0, 8.0]], dtype=precision)
        assert jnp.allclose(C, expected, atol=tolerance)

    @pytest.mark.parametrize(
        "device_type,precision",
        [
            ("cpu", jnp.float32),
            ("cpu", jnp.float64),
            ("gpu_mock", jnp.float32),
            ("gpu_mock", jnp.float64),
        ],
    )
    def test_complex_operations_dtype_preservation(self, device_type, precision):
        """Test complex operations maintain dtype across devices."""
        configure_jax_environment()

        complex_dtype = jnp.complex64 if precision == jnp.float32 else jnp.complex128
        tolerance = 1e-5 if precision == jnp.float32 else 1e-12

        # Create complex arrays
        real_part = jnp.array([1.0, 2.0], dtype=precision)
        imag_part = jnp.array([0.5, 1.5], dtype=precision)
        complex_array = real_part + 1j * imag_part

        # Verify complex dtype
        assert complex_array.dtype == complex_dtype

        # Test complex operations
        magnitude = jnp.abs(complex_array)
        phase = jnp.angle(complex_array)

        assert magnitude.dtype == precision
        assert phase.dtype == precision

        # Verify numerical accuracy
        expected_mag = jnp.sqrt(real_part**2 + imag_part**2)
        assert jnp.allclose(magnitude, expected_mag, atol=tolerance)

    @pytest.mark.parametrize(
        "device_type,precision",
        [
            ("cpu", jnp.float32),
            ("cpu", jnp.float64),
            ("gpu_mock", jnp.float32),
            ("gpu_mock", jnp.float64),
        ],
    )
    def test_gradient_computation_precision(self, device_type, precision):
        """Test automatic differentiation maintains precision across devices."""
        configure_jax_environment()

        tolerance = 1e-5 if precision == jnp.float32 else 1e-12

        def test_function(x):
            return x**2 + 2 * x + 1

        # Compute gradient
        grad_fn = jax.grad(test_function)
        x = jnp.array(2.0, dtype=precision)
        gradient = grad_fn(x)

        # Verify dtype preservation
        assert gradient.dtype == precision

        # Verify numerical accuracy (derivative of x^2 + 2x + 1 = 2x + 2, at x=2: 6)
        expected = jnp.array(6.0, dtype=precision)
        assert jnp.allclose(gradient, expected, atol=tolerance)

    @pytest.mark.parametrize(
        "device_type,precision",
        [
            ("cpu", jnp.float32),
            ("cpu", jnp.float64),
            ("gpu_mock", jnp.float32),
            ("gpu_mock", jnp.float64),
        ],
    )
    def test_memory_layout_consistency(self, device_type, precision):
        """Test memory layout and array properties across devices."""
        configure_jax_environment()

        # Create arrays with different layouts
        contiguous_array = jnp.arange(12, dtype=precision).reshape(3, 4)
        transposed = contiguous_array.T

        # Verify dtypes
        assert contiguous_array.dtype == precision
        assert transposed.dtype == precision

        # Verify shapes
        assert contiguous_array.shape == (3, 4)
        assert transposed.shape == (4, 3)

        # Test memory operations
        flattened = contiguous_array.flatten()
        assert flattened.dtype == precision
        assert flattened.shape == (12,)


class TestPrecisionDependentBehavior:
    """Test behavior that changes based on precision requirements."""

    def test_float32_optimization_flags(self, device_precision_config):
        """Test that float32 configurations enable appropriate optimizations."""
        config = device_precision_config

        if config["float_dtype"] == jnp.float32:
            # Float32 should use lower tolerance
            assert config["tolerance"] == 1e-5
            assert config["complex_dtype"] == jnp.complex64
        else:
            # Float64 should use higher precision
            assert config["tolerance"] == 1e-12
            assert config["complex_dtype"] == jnp.complex128

    def test_numerical_stability_thresholds(self, device_precision_config):
        """Test numerical stability thresholds adapt to precision."""
        config = device_precision_config

        # Create test values near numerical limits
        if config["float_dtype"] == jnp.float32:
            small_value = jnp.array(
                1e-4, dtype=config["float_dtype"]
            )  # Larger than float32 tolerance
            # Float32 should handle this with appropriate tolerance
            assert jnp.abs(small_value) > config["tolerance"]
        else:
            small_value = jnp.array(
                1e-11, dtype=config["float_dtype"]
            )  # Larger than float64 tolerance
            # Float64 should handle much smaller values
            assert jnp.abs(small_value) > config["tolerance"]

    def test_device_precision_interaction(self, device_precision_config):
        """Test that device and precision settings work together."""
        config = device_precision_config

        # Create test computation
        x = jnp.linspace(
            0, 2 * jnp.pi, 1000, dtype=config["float_dtype"]
        )  # More points for accuracy
        y = jnp.sin(x) + jnp.cos(x)

        # Verify all operations preserve device and precision
        assert y.dtype == config["float_dtype"]
        assert jax.default_backend() == "cpu"  # Windows limitation

        # Verify numerical accuracy
        expected_max = jnp.sqrt(2.0)  # sin(x) + cos(x) max is sqrt(2)
        assert (
            jnp.abs(y.max() - expected_max) < 1e-6
        )  # Relaxed tolerance for discretization error
