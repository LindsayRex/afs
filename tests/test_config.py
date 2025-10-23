"""
Tests for JAX configuration and dtype enforcement system.
"""

import os
import pytest
import jax
import jax.numpy as jnp
from unittest.mock import patch, MagicMock

from computable_flows_shim.config import (
    DEFAULT_DTYPE,
    DEFAULT_COMPLEX_DTYPE,
    DTYPE_POLICY,
    get_dtype,
    enforce_dtype,
    get_xla_flags_for_platform,
    configure_jax_environment,
    validate_dtype_consistency,
    validate_xla_flags,
    create_array,
    zeros,
    ones,
    random_normal
)


class TestDtypePolicy:
    """Test dtype policy and enforcement."""

    def test_default_dtypes(self):
        """Test default dtype values."""
        assert DEFAULT_DTYPE == jnp.float64
        assert DEFAULT_COMPLEX_DTYPE == jnp.complex128

    def test_dtype_policy_complete(self):
        """Test that all expected dtypes are in the policy."""
        expected_dtypes = [
            'default', 'complex', 'high_precision', 'low_precision',
            'integer', 'boolean', 'float32', 'float64', 'complex64', 'complex128'
        ]
        for dtype_name in expected_dtypes:
            assert dtype_name in DTYPE_POLICY

    def test_get_dtype(self):
        """Test get_dtype function."""
        assert get_dtype('default') == jnp.float64
        assert get_dtype('high_precision') == jnp.float64
        assert get_dtype('complex') == jnp.complex128

        with pytest.raises(ValueError, match="Unknown dtype"):
            get_dtype('invalid_dtype')

    @pytest.mark.dtype_parametrized
    def test_enforce_dtype_parametrized(self, float_dtype):
        """Test dtype enforcement on arrays with different input dtypes."""
        # Create array with specified dtype
        input_array = jnp.array([1.0, 2.0, 3.0], dtype=float_dtype)
        enforced = enforce_dtype(input_array, 'default')

        # Should always convert to default dtype (float64)
        assert enforced.dtype == jnp.float64
        assert jnp.allclose(enforced, input_array)

    def test_create_array_functions(self):
        """Test array creation functions with dtype enforcement."""
        # Test create_array
        arr = create_array([1.0, 2.0, 3.0])
        assert arr.dtype == jnp.float64

        # Test with explicit dtype
        arr_hp = create_array([1.0, 2.0, 3.0], dtype='high_precision')
        assert arr_hp.dtype == jnp.float64

        # Test zeros
        z = zeros((2, 3))
        assert z.dtype == jnp.float64
        assert z.shape == (2, 3)

        # Test ones
        o = ones((2, 3), dtype='low_precision')
        assert o.dtype == jnp.float32
        assert o.shape == (2, 3)


class TestXLAFlags:
    """Test XLA flag generation for different platforms."""

    def test_get_xla_flags_cpu(self):
        """Test CPU XLA flags."""
        flags = get_xla_flags_for_platform('cpu')
        assert '--xla_cpu_multi_thread_eigen=true' in flags
        assert '--xla_cpu_enable_fast_math=true' in flags
        assert '--xla_enable_fast_math=true' in flags
        assert '--xla_optimization_level=3' in flags

    def test_get_xla_flags_gpu(self):
        """Test GPU XLA flags."""
        flags = get_xla_flags_for_platform('gpu')
        assert '--xla_gpu_enable_fast_min_max=true' in flags
        assert '--xla_gpu_enable_async_all_reduce=true' in flags
        assert '--xla_enable_fast_math=true' in flags

    def test_get_xla_flags_tpu(self):
        """Test TPU XLA flags."""
        flags = get_xla_flags_for_platform('tpu')
        assert '--xla_tpu_enable_async_collective_fusion=true' in flags
        assert '--xla_enable_fast_math=true' in flags

    @patch('jax.devices')
    def test_get_xla_flags_auto_detection(self, mock_devices):
        """Test auto platform detection."""
        # Mock CPU-only environment
        mock_devices.return_value = [MagicMock()]
        mock_devices.return_value[0].platform = 'cpu'

        flags = get_xla_flags_for_platform('auto')
        assert '--xla_cpu_multi_thread_eigen=true' in flags

    def test_get_xla_flags_invalid_platform(self):
        """Test invalid platform raises error."""
        with pytest.raises(ValueError, match="Unknown platform"):
            get_xla_flags_for_platform('invalid')

    @patch.dict(os.environ, {'AFS_DEBUG': 'true'})
    def test_debug_flags_included(self):
        """Test debug flags are included when AFS_DEBUG=true."""
        flags = get_xla_flags_for_platform('cpu')
        assert '--xla_dump_hlo_as_text=true' in flags
        assert '--xla_enable_dumping=true' in flags


class TestConfiguration:
    """Test JAX environment configuration."""

    @patch.dict(os.environ, {}, clear=True)
    def test_configure_jax_environment_no_flags(self):
        """Test configuration when no XLA_FLAGS are set."""
        configure_jax_environment()
        assert 'XLA_FLAGS' in os.environ

    @patch.dict(os.environ, {'XLA_FLAGS': 'existing_flags'})
    def test_configure_jax_environment_preserves_existing(self):
        """Test configuration preserves existing XLA_FLAGS."""
        original_flags = os.environ['XLA_FLAGS']
        configure_jax_environment()
        assert os.environ['XLA_FLAGS'] == original_flags

    @patch.dict(os.environ, {'JAX_PLATFORM_NAME': 'cpu'})
    @patch('jax.config.update')
    def test_configure_jax_environment_platform(self, mock_config_update):
        """Test platform configuration."""
        configure_jax_environment()
        # Should set both platform name and enable x64
        mock_config_update.assert_any_call('jax_platform_name', 'cpu')
        mock_config_update.assert_any_call('jax_enable_x64', True)

    @patch.dict(os.environ, {'AFS_DISABLE_64BIT': 'true'})
    @patch('jax.config.update')
    def test_configure_jax_environment_64bit_disabled(self, mock_config_update):
        """Test 64-bit precision can be disabled."""
        configure_jax_environment()
        # Should not call jax_enable_x64 when disabled
        assert not any(call[0][0] == 'jax_enable_x64' for call in mock_config_update.call_args_list)


class TestValidation:
    """Test configuration validation."""

    def test_validate_dtype_consistency(self):
        """Test dtype consistency validation."""
        assert validate_dtype_consistency() is True

    def test_validate_xla_flags_valid(self):
        """Test XLA flags validation with valid flags."""
        valid_flags = "--xla_enable_fast_math=true --xla_optimization_level=3"
        assert validate_xla_flags(valid_flags) is True

    def test_validate_xla_flags_conflicting_opt_levels(self):
        """Test XLA flags validation catches conflicting optimization levels."""
        conflicting_flags = "--xla_optimization_level=3 --xla_optimization_level=2"
        with pytest.raises(ValueError, match="Multiple optimization levels"):
            validate_xla_flags(conflicting_flags)

    def test_validate_xla_flags_invalid_opt_level(self):
        """Test XLA flags validation catches invalid optimization levels."""
        invalid_flags = "--xla_optimization_level=5"
        with pytest.raises(ValueError, match="Invalid optimization level"):
            validate_xla_flags(invalid_flags)

    def test_validate_xla_flags_conflicting_fast_math(self):
        """Test XLA flags validation catches conflicting fast math settings."""
        conflicting_flags = "--xla_enable_fast_math=true --xla_cpu_enable_fast_math=false"
        with pytest.raises(ValueError, match="Conflicting fast math settings"):
            validate_xla_flags(conflicting_flags)

    def test_random_normal_dtype_enforcement(self):
        """Test random_normal respects dtype enforcement."""
        key = jax.random.key(42)
        arr = random_normal(key, (3, 3))
        assert arr.dtype == jnp.float64

        arr_hp = random_normal(key, (3, 3), dtype='high_precision')
        assert arr_hp.dtype == jnp.float64


class TestIntegration:
    """Integration tests for the configuration system."""

    @patch.dict(os.environ, {'AFS_JAX_PLATFORM': 'cpu'}, clear=True)
    def test_full_configuration_workflow(self):
        """Test complete configuration workflow."""
        # This should not raise any exceptions
        configure_jax_environment()

        # Check that XLA flags were set
        assert 'XLA_FLAGS' in os.environ
        flags = os.environ['XLA_FLAGS']
        assert '--xla_cpu_multi_thread_eigen=true' in flags

        # Test dtype functions work
        arr = create_array([1.0, 2.0, 3.0])
        assert arr.dtype == jnp.float64

        enforced = enforce_dtype(jnp.array([1.0, 2.0], dtype=jnp.float32))
        assert enforced.dtype == jnp.float64

    def test_dtype_policy_coverage(self):
        """Test that all dtype policy entries are valid JAX dtypes."""
        # Enable 64-bit precision for this test to properly test all dtypes
        original_x64 = jax.config.jax_enable_x64
        jax.config.update('jax_enable_x64', True)

        try:
            for name, dtype in DTYPE_POLICY.items():
                # Should be a valid JAX dtype - check by trying to create an array
                try:
                    test_array = jnp.array([1.0], dtype=dtype)
                    # If we get here, the dtype is valid
                    # The resulting dtype should match what we requested (or be compatible)
                    if dtype == jnp.float64:
                        assert test_array.dtype == jnp.float64
                    elif dtype == jnp.complex128:
                        assert test_array.dtype == jnp.complex128
                    elif dtype == jnp.float32:
                        assert test_array.dtype == jnp.float32
                    elif dtype == jnp.complex64:
                        assert test_array.dtype == jnp.complex64
                    elif dtype == jnp.float16:
                        assert test_array.dtype == jnp.float16
                    elif dtype == jnp.int32:
                        assert test_array.dtype == jnp.int32
                    elif dtype == jnp.bool_:
                        assert test_array.dtype == jnp.bool_
                except (TypeError, ValueError) as e:
                    pytest.fail(f"Invalid dtype in policy: {name} -> {dtype}, error: {e}")
        finally:
            # Restore original setting
            jax.config.update('jax_enable_x64', original_x64)

    def test_xla_flags_are_strings(self):
        """Test that XLA flag functions return strings."""
        for platform in ['cpu', 'gpu', 'tpu']:
            flags = get_xla_flags_for_platform(platform)
            assert isinstance(flags, str)
            assert len(flags) > 0