"""
Test-Driven Development for TransformOp (Multiscale Transforms).

Following the RED-GREEN-REFACTOR cycle with Design by Contract.
Tests enforce mathematical properties of wavelet transforms and frame-aware operations.
"""

import pytest
import jax
import jax.numpy as jnp
from computable_flows_shim.multi.transform_op import TransformOp, make_jaxwt_transform, make_transform


class TestTransformOpContract:
    """
    Design by Contract tests for TransformOp.

    Contract: TransformOp provides mathematically correct wavelet forward/inverse transforms
    with proper frame metadata, JAX compatibility, and round-trip accuracy.
    """

    @pytest.fixture(params=[jnp.float32, jnp.float64])
    def float_dtype(self, request):
        """Parametrize tests with different floating point precisions."""
        return request.param

    @pytest.fixture
    def sample_1d_signal(self, float_dtype):
        """1D test signal that should be compressible with wavelets."""
        # Create a signal with some structure: smooth part + high-frequency component
        x = jnp.linspace(0, 4*jnp.pi, 64, dtype=float_dtype)
        signal = jnp.sin(x) + 0.1 * jnp.sin(20*x)  # Low + high frequency
        return signal

    @pytest.fixture
    def sample_2d_image(self, float_dtype):
        """2D test image for wavelet transforms."""
        # Simple 2D pattern
        x = jnp.linspace(-1, 1, 32, dtype=float_dtype)
        y = jnp.linspace(-1, 1, 32, dtype=float_dtype)
        X, Y = jnp.meshgrid(x, y)
        image = jnp.exp(-(X**2 + Y**2)) + 0.1 * jnp.sin(10*X) * jnp.cos(10*Y)
        return image

    def test_transform_op_creation(self):
        """RED: TransformOp should be created with proper structure."""
        # This will fail until TransformOp is implemented
        transform = make_jaxwt_transform('haar', levels=2)

        assert hasattr(transform, 'name')
        assert hasattr(transform, 'forward')
        assert hasattr(transform, 'inverse')
        assert hasattr(transform, 'frame')
        assert hasattr(transform, 'c')
        assert hasattr(transform, 'levels')

    def test_1d_round_trip_accuracy(self, sample_1d_signal, float_dtype):
        """RED: 1D forward/inverse should achieve perfect reconstruction."""
        transform = make_jaxwt_transform('haar', levels=2)

        # Forward transform
        coeffs = transform.forward(sample_1d_signal)

        # Inverse transform
        reconstructed = transform.inverse(coeffs)

        # Should achieve near-perfect reconstruction
        tolerance = 1e-5 if float_dtype == jnp.float32 else 1e-12
        error = jnp.max(jnp.abs(reconstructed - sample_1d_signal))
        assert error < tolerance, f"Round-trip error too large: {error} (dtype: {float_dtype})"

    def test_2d_round_trip_accuracy(self, sample_2d_image, float_dtype):
        """RED: 2D forward/inverse should achieve perfect reconstruction."""
        transform = make_jaxwt_transform('haar', levels=2, ndim=2)

        # Forward transform
        coeffs = transform.forward(sample_2d_image)

        # Inverse transform
        reconstructed = transform.inverse(coeffs)

        # Should achieve near-perfect reconstruction
        tolerance = 1e-5 if float_dtype == jnp.float32 else 1e-12
        error = jnp.max(jnp.abs(reconstructed - sample_2d_image))
        assert error < tolerance, f"Round-trip error too large: {error} (dtype: {float_dtype})"

    def test_frame_metadata_haar(self):
        """RED: Haar wavelet should be unitary frame."""
        transform = make_jaxwt_transform('haar', levels=2)

        assert transform.frame == 'unitary'
        assert transform.c == 1.0  # Unitary frames have c=1

    def test_frame_metadata_db4(self):
        """RED: Daubechies-4 should be tight frame."""
        transform = make_jaxwt_transform('db4', levels=2)

        assert transform.frame in ['tight', 'unitary']
        assert isinstance(transform.c, (int, float))
        assert transform.c > 0

    def test_jax_jit_compatibility(self, sample_1d_signal):
        """RED: Transform operations should be JIT-compatible."""
        transform = make_jaxwt_transform('haar', levels=2)

        # JIT compile forward and inverse
        jit_forward = jax.jit(transform.forward)
        jit_inverse = jax.jit(transform.inverse)

        # Should execute without errors
        coeffs = jit_forward(sample_1d_signal)
        reconstructed = jit_inverse(coeffs)

        assert reconstructed.shape == sample_1d_signal.shape

    def test_jax_grad_compatibility(self, sample_1d_signal):
        """RED: Forward transform should be differentiable."""
        transform = make_jaxwt_transform('haar', levels=2)

        def loss_fn(x):
            coeffs = transform.forward(x)
            # Simple loss on coefficients (sum of absolute values of all coeff arrays)
            total_loss = 0.0
            for coeff in coeffs:
                total_loss += jnp.sum(jnp.abs(coeff))
            return total_loss

        # Should be able to compute gradients
        grad_fn = jax.grad(loss_fn)
        grad = grad_fn(sample_1d_signal)

        assert grad.shape == sample_1d_signal.shape
        assert jnp.isfinite(grad).all()

    def test_coefficient_structure_preservation(self, sample_1d_signal):
        """RED: Coefficient structure should be preserved through forward/inverse."""
        transform = make_jaxwt_transform('haar', levels=2)

        coeffs1 = transform.forward(sample_1d_signal)
        coeffs2 = transform.forward(sample_1d_signal)

        # Forward transform should be deterministic (compare each coefficient array)
        assert len(coeffs1) == len(coeffs2)
        for c1, c2 in zip(coeffs1, coeffs2):
            assert jnp.allclose(c1, c2)

        # Inverse of forward should preserve structure
        reconstructed = transform.inverse(coeffs1)
        assert reconstructed.shape == sample_1d_signal.shape

    def test_transform_registry_integration(self):
        """RED: make_transform should create correct TransformOp instances."""
        # Test different wavelet types
        haar_transform = make_transform('haar', levels=2)
        db4_transform = make_transform('db4', levels=2)

        assert haar_transform.name == 'haar'
        assert db4_transform.name == 'db4'
        assert haar_transform.levels == 2
        assert db4_transform.levels == 2

    def test_error_handling_invalid_wavelet(self):
        """RED: Should handle invalid wavelet names gracefully during execution."""
        transform = make_jaxwt_transform('invalid_wavelet', levels=2)
        
        # Object creation should succeed
        assert transform is not None
        
        # But execution should fail
        x = jnp.ones(64)
        with pytest.raises(ValueError, match="Unknown wavelet"):
            transform.forward(x)

    def test_multilevel_consistency(self, sample_1d_signal):
        """RED: Different levels should maintain reconstruction quality."""
        transform_l1 = make_jaxwt_transform('haar', levels=1)
        transform_l2 = make_jaxwt_transform('haar', levels=2)

        # Both should achieve perfect reconstruction
        recon_l1 = transform_l1.inverse(transform_l1.forward(sample_1d_signal))
        recon_l2 = transform_l2.inverse(transform_l2.forward(sample_1d_signal))

        assert jnp.allclose(recon_l1, sample_1d_signal, atol=1e-6)
        assert jnp.allclose(recon_l2, sample_1d_signal, atol=1e-6)

    def test_frame_constant_mathematical_property(self, sample_1d_signal):
        """RED: Frame constant should satisfy Parseval/tight frame property."""
        transform = make_jaxwt_transform('haar', levels=2)

        if transform.frame == 'tight':
            # For tight frames: ||Wx||² = c ||x||²
            coeffs = transform.forward(sample_1d_signal)
            coeff_energy = jnp.sum(coeffs**2)
            signal_energy = jnp.sum(sample_1d_signal**2)

            # Should satisfy Parseval's theorem (up to numerical precision)
            ratio = coeff_energy / signal_energy
            assert abs(ratio - transform.c) < 1e-6, f"Frame constant violation: {ratio} vs {transform.c}"