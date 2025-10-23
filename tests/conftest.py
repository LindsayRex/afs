"""
Pytest configuration and fixtures for AFS testing.

Provides JAX configuration, dtype fixtures, and testing utilities.
"""
import pytest
import jax
import jax.numpy as jnp

# Configure JAX environment for all tests
from computable_flows_shim import configure_jax_environment

# Configure JAX at session start
def pytest_sessionstart(session):
    """Configure JAX environment at the start of the test session."""
    configure_jax_environment()


@pytest.fixture(scope="session")
def jax_configured():
    """Fixture to ensure JAX is configured. Used by tests that need JAX setup."""
    # JAX is already configured in pytest_sessionstart
    return True


@pytest.fixture(params=[jnp.float32, jnp.float64])
def float_dtype(request):
    """Parametrize tests with different floating point precisions."""
    return request.param


@pytest.fixture(params=[jnp.complex64, jnp.complex128])
def complex_dtype(request):
    """Parametrize tests with different complex precisions."""
    return request.param


@pytest.fixture(params=[jnp.float32, jnp.float64])
def real_precision(request):
    """Parametrize tests with real number precisions for general use."""
    return request.param


@pytest.fixture(params=[jnp.complex64, jnp.complex128])
def complex_precision(request):
    """Parametrize tests with complex precisions for signal processing."""
    return request.param


@pytest.fixture
def precision_config(request):
    """Fixture providing precision configuration for tests."""
    # Default to high precision for numerical stability
    return {
        'float_dtype': jnp.float64,
        'complex_dtype': jnp.complex128,
        'tolerance': 1e-12
    }


@pytest.fixture
def low_precision_config(request):
    """Fixture providing low precision configuration for memory-constrained tests."""
    return {
        'float_dtype': jnp.float32,
        'complex_dtype': jnp.complex64,
        'tolerance': 1e-5
    }


def create_test_array(*args, dtype=jnp.float64, **kwargs):
    """Create test arrays with specified dtype."""
    return jnp.array(*args, dtype=dtype, **kwargs)


def create_test_zeros(*args, dtype=jnp.float64, **kwargs):
    """Create zero arrays with specified dtype."""
    return jnp.zeros(*args, dtype=dtype, **kwargs)


def create_test_ones(*args, dtype=jnp.float64, **kwargs):
    """Create ones arrays with specified dtype."""
    return jnp.ones(*args, dtype=dtype, **kwargs)


@pytest.fixture
def array_factory():
    """Factory fixture for creating test arrays with proper dtypes."""
    return {
        'array': create_test_array,
        'zeros': create_test_zeros,
        'ones': create_test_ones
    }


def pytest_configure(config):
    """Add custom markers for dtype testing."""
    config.addinivalue_line(
        "markers", "dtype_parametrized: marks tests that are parametrized with different dtypes"
    )
    config.addinivalue_line(
        "markers", "precision_sensitive: marks tests that require specific numerical precision"
    )
    config.addinivalue_line(
        "markers", "complex_operations: marks tests that use complex number operations"
    )