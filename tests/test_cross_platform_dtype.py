"""
Cross-cutting tests for JAX dtype enforcement on available platforms.

Tests dtype enforcement across precision levels on the current JAX platform.
Note: JAX 0.6.2 does not support runtime platform switching via jax.platform().
Platform selection must be done via environment variables before JAX import.
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture(scope="session")
def current_platform():
    """Get the current JAX platform (set at import time)."""
    return jax.default_backend()


@pytest.fixture(scope="session")
def available_devices():
    """Get available JAX devices."""
    return jax.devices()


@pytest.fixture(scope="session")
def available_platforms():
    """Get list of available JAX platforms in current environment."""
    platforms = []
    try:
        # Test CPU (always available in JAX 0.6.2)
        devices = jax.devices("cpu")
        if devices:
            platforms.append("cpu")
    except (RuntimeError, ValueError):
        pass

    try:
        # Test GPU
        devices = jax.devices("gpu")
        if devices:
            platforms.append("gpu")
    except (RuntimeError, ValueError):
        pass

    try:
        # Test TPU
        devices = jax.devices("tpu")
        if devices:
            platforms.append("tpu")
    except (RuntimeError, ValueError):
        pass

    return platforms


@pytest.mark.dtype_parametrized
class TestDtypeEnforcement:
    """Tests for dtype enforcement on the current JAX platform."""

    @pytest.fixture(autouse=True)
    def setup_method(self, float_dtype):
        """Set up test method with dtype fixture."""
        self.float_dtype = float_dtype
        # Set tolerance based on precision
        self.tolerance = 1e-5 if float_dtype == jnp.float32 else 1e-12

    def test_dtype_consistency(self):
        """Test that dtype enforcement works consistently."""
        # Create arrays with different dtypes
        arr_f32 = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
        arr_f64 = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64)

        # Verify dtypes are preserved
        assert arr_f32.dtype == jnp.float32
        assert arr_f64.dtype == jnp.float64

        # Test basic operations maintain dtype
        result_f32 = arr_f32 + arr_f32
        result_f64 = arr_f64 + arr_f64

        assert result_f32.dtype == jnp.float32
        assert result_f64.dtype == jnp.float64

    def test_precision_operations(self):
        """Test precision-sensitive operations."""
        # Create test data
        x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=self.float_dtype)
        y = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=self.float_dtype)

        # Test matrix operations
        a = jnp.outer(x, y)
        assert a.dtype == self.float_dtype

        # Test eigenvalue computation (precision sensitive)
        if a.shape[0] <= 10:  # Only for small matrices to avoid timeout
            eigenvals = jnp.linalg.eigvals(a)
            assert eigenvals.dtype in [jnp.complex64, jnp.complex128]
            # Check that eigenvalues are finite
            assert jnp.all(jnp.isfinite(eigenvals))

    def test_memory_management(self):
        """Test memory operations."""
        # Test large array creation and operations
        size = 1000  # Reasonable size for CPU testing

        # Create large arrays
        x = jnp.ones(size, dtype=self.float_dtype)
        y = jnp.ones(size, dtype=self.float_dtype) * 2.0

        # Test operations
        z = x + y
        w = z * 3.0
        result = jnp.sum(w)

        assert result.dtype == self.float_dtype
        assert jnp.isfinite(result)

    def test_jit_compilation(self):
        """Test JIT compilation."""

        @jax.jit
        def test_function(x, y):
            return x * y + jnp.sin(x)

        x = jnp.array([1.0, 2.0, 3.0], dtype=self.float_dtype)
        y = jnp.array([0.5, 1.5, 2.5], dtype=self.float_dtype)

        result = test_function(x, y)

        assert result.dtype == self.float_dtype
        assert jnp.all(jnp.isfinite(result))

    def test_gradient_computation(self):
        """Test automatic differentiation."""

        def test_function(x):
            return jnp.sum(x**2 + jnp.sin(x))

        x = jnp.array([1.0, 2.0, 3.0], dtype=self.float_dtype)

        # Compute gradient
        grad_fn = jax.grad(test_function)
        gradient = grad_fn(x)

        assert gradient.dtype == self.float_dtype
        assert jnp.all(jnp.isfinite(gradient))

        # Verify gradient is approximately correct (2*x + cos(x))
        expected = 2 * x + jnp.cos(x)
        assert jnp.allclose(gradient, expected, atol=self.tolerance)

    def test_random_generation(self):
        """Test random number generation."""
        key = jax.random.PRNGKey(42)

        # Generate random arrays
        x = jax.random.normal(key, (100,), dtype=self.float_dtype)
        y = jax.random.uniform(key, (100,), dtype=self.float_dtype)

        assert x.dtype == self.float_dtype
        assert y.dtype == self.float_dtype
        assert jnp.all(jnp.isfinite(x))
        assert jnp.all(jnp.isfinite(y))

        # Check statistical properties (basic sanity check)
        assert jnp.abs(jnp.mean(x)) < 0.5  # Should be close to 0
        assert jnp.std(x) > 0.5  # Should have some variance

    def test_device_placement(self, available_devices):
        """Test that operations are placed on available devices."""
        x = jnp.array([1.0, 2.0, 3.0], dtype=self.float_dtype)

        # Check device placement
        if available_devices:
            # Array should be on one of the available devices
            assert x.device in available_devices

    def test_dtype_conversion(self):
        """Test dtype conversion works correctly."""
        # Start with different precisions
        x_f32 = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
        x_f64 = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64)

        # Convert between precisions
        converted_to_f64 = x_f32.astype(jnp.float64)
        converted_to_f32 = x_f64.astype(jnp.float32)

        assert converted_to_f64.dtype == jnp.float64
        assert converted_to_f32.dtype == jnp.float32

        # Values should be preserved (within precision limits)
        assert jnp.allclose(converted_to_f64, x_f64, atol=1e-6)
        assert jnp.allclose(converted_to_f32, x_f32, atol=1e-5)


class TestPlatformDetection:
    """Test platform detection and configuration."""

    def test_current_platform_detection(self, current_platform):
        """Test that current platform detection works correctly."""
        # Should be a string representing the platform
        assert isinstance(current_platform, str)
        assert current_platform in ["cpu", "gpu", "tpu"]

    def test_available_devices(self, available_devices):
        """Test that device detection works correctly."""
        # Should have at least one device
        assert len(available_devices) > 0
        # All devices should be JAX Device objects
        for device in available_devices:
            assert hasattr(device, "id")
            assert hasattr(device, "platform")

    def test_platform_backend_consistency(self, current_platform, available_devices):
        """Test that platform and device information is consistent."""
        # All devices should be on the current platform
        for device in available_devices:
            assert device.platform == current_platform

    def test_xla_flags_current_platform(self, current_platform):
        """Test that XLA flags are set appropriately for the current platform."""
        from computable_flows_shim.config import get_xla_flags_for_platform

        flags = get_xla_flags_for_platform(current_platform)
        assert isinstance(flags, str)
        assert len(flags) > 0

        # Check platform-specific flags are present
        if current_platform == "cpu":
            assert "--xla_cpu_multi_thread_eigen=true" in flags
        elif current_platform == "gpu":
            # GPU-specific flags would be checked here
            pass
        elif current_platform == "tpu":
            # TPU-specific flags would be checked here
            pass


class TestPrecisionPlatformMatrix:
    """Test precision levels on the current platform."""

    @pytest.mark.parametrize("precision", [jnp.float32, jnp.float64])
    def test_precision_compatibility(self, precision):
        """Test that each precision works on the current platform."""
        # Test basic operations with this precision
        x = jnp.array([1.0, 2.0, 3.0], dtype=precision)
        y = jnp.array([0.5, 1.5, 2.5], dtype=precision)

        result = x + y * 2.0 - jnp.sin(x)
        assert result.dtype == precision
        assert jnp.all(jnp.isfinite(result))

    def test_precision_limits_accuracy(self):
        """Test that precision limits are respected."""
        test_values = [1e-8, 1e-12, 1e-15]  # Test very small numbers

        # Test float32 precision limits
        x_f32 = jnp.array(test_values, dtype=jnp.float32)
        # Very small values should be representable or underflow to zero
        assert jnp.isfinite(x_f32).all() or jnp.all(x_f32 == 0.0)

        # Test float64 precision limits
        x_f64 = jnp.array(test_values, dtype=jnp.float64)
        # Should preserve more precision
        finite_count_f64 = jnp.sum(jnp.isfinite(x_f64))
        finite_count_f32 = jnp.sum(jnp.isfinite(x_f32))
        # float64 should handle more small values than float32
        assert finite_count_f64 >= finite_count_f32
