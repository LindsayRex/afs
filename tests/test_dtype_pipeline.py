"""
Pipeline-level dtype enforcement tests for AFS.

Tests end-to-end dtype consistency across the entire AFS processing pipeline,
validating that all components maintain proper precision and type enforcement.
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from computable_flows_shim.fda.certificates import estimate_gamma_lanczos
from computable_flows_shim.multi.transform_op import make_transform

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.mark.dtype_parametrized
class TestPipelineDtypeEnforcement:
    """End-to-end dtype enforcement tests for AFS processing pipeline."""

    @pytest.fixture(autouse=True)
    def setup_method(self, float_dtype):
        """Set up test method with dtype fixture."""
        self.float_dtype = float_dtype  # pylint: disable=attribute-defined-outside-init
        # Set tolerance based on precision
        self.tolerance = 1e-5 if float_dtype == jnp.float32 else 1e-12  # pylint: disable=attribute-defined-outside-init

    @pytest.fixture
    def sample_signal_1d(self):
        """Generate 1D test signal for pipeline testing."""
        key = jax.random.PRNGKey(42)
        x = jnp.linspace(0, 4 * jnp.pi, 128, dtype=self.float_dtype)
        # Create a signal with multiple frequency components
        signal = (
            jnp.sin(x)
            + 0.5 * jnp.sin(3 * x)
            + 0.3 * jnp.sin(5 * x)
            + 0.1 * jax.random.normal(key, x.shape, dtype=self.float_dtype) * 0.1
        )
        return signal

    @pytest.fixture
    def sample_signal_2d(self):
        """Generate 2D test signal for pipeline testing."""
        key = jax.random.PRNGKey(42)
        x = jnp.linspace(-2, 2, 64, dtype=self.float_dtype)
        y = jnp.linspace(-2, 2, 64, dtype=self.float_dtype)
        X, Y = jnp.meshgrid(x, y)  # pylint: disable=invalid-name
        # Create 2D pattern with some noise
        signal = jnp.exp(-(X**2 + Y**2)) * jnp.sin(3 * X) * jnp.cos(
            2 * Y
        ) + 0.05 * jax.random.normal(key, X.shape, dtype=self.float_dtype)
        return signal

    def test_transform_pipeline_dtype_consistency(self, sample_signal_1d):
        """Test that transform operations maintain dtype consistency throughout pipeline."""
        # Create wavelet transform
        transform = make_transform("haar", levels=3, ndim=1)

        # Forward transform
        coeffs = transform.forward(sample_signal_1d)

        # Verify coefficients maintain dtype (coeffs is a list of arrays)
        for i, coeff_array in enumerate(coeffs):
            assert coeff_array.dtype == self.float_dtype, (
                f"Coefficient array {i} dtype {coeff_array.dtype} != expected {self.float_dtype}"
            )

        # Inverse transform
        reconstruction = transform.inverse(coeffs)

        # Verify reconstruction maintains dtype
        assert reconstruction.dtype == self.float_dtype, (
            f"Reconstruction dtype {reconstruction.dtype} != expected {self.float_dtype}"
        )

        # Verify reconstruction accuracy
        relative_error = jnp.linalg.norm(
            reconstruction - sample_signal_1d
        ) / jnp.linalg.norm(sample_signal_1d)
        assert relative_error < self.tolerance, (
            f"Reconstruction error {relative_error} exceeds tolerance {self.tolerance}"
        )

    def test_lanczos_pipeline_dtype_consistency(self, sample_signal_1d):
        """Test that Lanczos operations maintain dtype in spectral analysis pipeline."""

        # Create a simple linear operator for testing
        def L_apply(v):  # pylint: disable=invalid-name
            # Simple diagonal operator for testing
            return 2.0 * v

        key = jax.random.PRNGKey(123)

        # Run Lanczos estimation
        gamma = estimate_gamma_lanczos(L_apply, key, sample_signal_1d.shape, k=8)

        # Verify gamma maintains appropriate dtype (should be real)
        assert gamma.dtype in [
            jnp.float32,
            jnp.float64,
        ], f"Gamma dtype {gamma.dtype} not real-valued"
        assert jnp.isfinite(gamma), f"Lanczos result {gamma} should be finite"

    def test_wavelet_lanczos_pipeline_integration(self, sample_signal_1d):
        """Test integrated wavelet-Lanczos pipeline maintains dtype consistency."""

        # Create wavelet transform
        transform = make_transform("haar", levels=2, ndim=1)

        # Define operator in wavelet space
        def L_wavelet(coeffs_flat):  # pylint: disable=invalid-name
            # coeffs_flat is a flattened array of all coefficients
            # Apply simple scaling to all coefficients
            return 1.5 * coeffs_flat

        key = jax.random.PRNGKey(456)

        # Test Lanczos in wavelet space
        gamma_wavelet = estimate_gamma_lanczos(
            L_wavelet, key, sample_signal_1d.shape, k=6, transform_op=transform
        )

        # Test Lanczos in physical space for comparison
        def L_physical(signal):  # pylint: disable=invalid-name
            # Apply same scaling operation in physical space
            return 1.5 * signal

        gamma_physical = estimate_gamma_lanczos(
            L_physical, key, sample_signal_1d.shape, k=6, transform_op=None
        )

        # Both should be finite and maintain dtype consistency
        assert jnp.isfinite(gamma_wavelet), (
            f"Wavelet space gamma {gamma_wavelet} should be finite"
        )
        assert jnp.isfinite(gamma_physical), (
            f"Physical space gamma {gamma_physical} should be finite"
        )
        assert gamma_wavelet.dtype in [jnp.float32, jnp.float64]
        assert gamma_physical.dtype in [jnp.float32, jnp.float64]

    def test_2d_transform_pipeline_dtype_consistency(self, sample_signal_2d):
        """Test 2D transform pipeline maintains dtype consistency."""
        # Create 2D wavelet transform
        transform = make_transform("haar", levels=2, ndim=2)

        # Forward transform
        coeffs = transform.forward(sample_signal_2d)

        # Verify coefficients maintain dtype (coeffs is a complex structure for 2D)
        # For 2D transforms, coeffs is typically a tuple/list structure
        def check_coeffs_dtype(coeffs):
            if isinstance(coeffs, (list, tuple)):
                for item in coeffs:
                    check_coeffs_dtype(item)
            else:
                # It's an array
                assert coeffs.dtype == self.float_dtype, (
                    f"2D coefficient dtype {coeffs.dtype} != expected {self.float_dtype}"
                )

        check_coeffs_dtype(coeffs)

        # Inverse transform
        reconstruction = transform.inverse(coeffs)

        # Verify reconstruction maintains dtype
        assert reconstruction.dtype == self.float_dtype, (
            f"2D reconstruction dtype {reconstruction.dtype} != expected {self.float_dtype}"
        )

        # Verify reconstruction accuracy
        relative_error = jnp.linalg.norm(
            reconstruction - sample_signal_2d
        ) / jnp.linalg.norm(sample_signal_2d)
        assert relative_error < self.tolerance, (
            f"2D reconstruction error {relative_error} exceeds tolerance {self.tolerance}"
        )

    def test_dtype_enforcement_across_jit_compilation(self):
        """Test that dtype consistency is maintained across JIT compilation boundaries."""

        @jax.jit
        def jit_transform_operation(x):
            # Simple operation that should maintain dtype
            return 2.0 * x + 1.0

        # Test with different input dtypes
        x_float = jnp.array([1.0, 2.0, 3.0], dtype=self.float_dtype)
        result = jit_transform_operation(x_float)

        # Verify output dtype matches input
        assert result.dtype == self.float_dtype, (
            f"JIT output dtype {result.dtype} != input dtype {self.float_dtype}"
        )

        # Verify numerical correctness
        expected = 2.0 * x_float + 1.0
        assert jnp.allclose(result, expected, atol=self.tolerance)

    def test_pipeline_memory_dtype_consistency(self, sample_signal_1d):
        """Test that intermediate computations maintain dtype consistency in memory."""
        # Create a multi-step pipeline
        transform1 = make_transform("haar", levels=2, ndim=1)
        transform2 = make_transform("db4", levels=2, ndim=1)

        # Step 1: First transform
        coeffs1 = transform1.forward(sample_signal_1d)
        # Check dtype of all coefficient arrays
        for _, coeff_array in enumerate(coeffs1):
            assert coeff_array.dtype == self.float_dtype

        # Step 2: Apply operation in transform space
        modified_coeffs1 = [coeff_array * 0.8 for coeff_array in coeffs1]  # Attenuation
        for coeff_array in modified_coeffs1:
            assert coeff_array.dtype == self.float_dtype

        # Step 3: Inverse first transform
        intermediate_signal = transform1.inverse(modified_coeffs1)
        assert intermediate_signal.dtype == self.float_dtype

        # Step 4: Second transform
        coeffs2 = transform2.forward(intermediate_signal)
        for coeff_array in coeffs2:
            assert coeff_array.dtype == self.float_dtype

        # Step 5: Apply different operation
        modified_coeffs2 = [
            coeff_array * 1.2 for coeff_array in coeffs2
        ]  # Amplification
        for coeff_array in modified_coeffs2:
            assert coeff_array.dtype == self.float_dtype

        # Step 6: Inverse second transform
        final_signal = transform2.inverse(modified_coeffs2)
        assert final_signal.dtype == self.float_dtype

        # Verify final result is reasonable (should be attenuated then amplified)
        # The net effect should be some modification of the original signal
        assert jnp.isfinite(final_signal).all(), (
            "Final pipeline result should be finite"
        )
        assert not jnp.allclose(final_signal, sample_signal_1d, atol=self.tolerance), (
            "Pipeline should modify signal"
        )

    def test_error_propagation_dtype_consistency(self):
        """Test that error calculations maintain dtype consistency."""
        # Create test data with known errors
        x_true = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=self.float_dtype)
        x_approx = x_true + jnp.array(
            [0.01, -0.005, 0.002, 0.008, -0.003], dtype=self.float_dtype
        )

        # Calculate various error metrics
        abs_error = jnp.abs(x_approx - x_true)
        rel_error = abs_error / jnp.abs(x_true)
        mse = jnp.mean(abs_error**2)
        rmse = jnp.sqrt(mse)
        max_error = jnp.max(abs_error)

        # All error metrics should maintain dtype
        assert abs_error.dtype == self.float_dtype
        assert rel_error.dtype == self.float_dtype
        assert mse.dtype == self.float_dtype
        assert rmse.dtype == self.float_dtype
        assert max_error.dtype == self.float_dtype

        # All should be non-negative
        assert jnp.all(abs_error >= 0)
        assert jnp.all(rel_error >= 0)
        assert mse >= 0
        assert rmse >= 0
        assert max_error >= 0
