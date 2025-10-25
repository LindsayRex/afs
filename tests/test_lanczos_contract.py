"""
Contract tests for FDA Lanczos hardening implementation.
"""

import sys
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from computable_flows_shim.fda.certificates import estimate_gamma_lanczos


@pytest.mark.dtype_parametrized
class TestLanczosContract:
    """Mathematical contracts for Lanczos spectral gap estimation."""

    @pytest.fixture(autouse=True)
    def setup_method(self, float_dtype):
        """Set up test method with dtype fixture."""
        self.float_dtype = float_dtype
        # Set tolerance based on precision
        self.tolerance = 1e-5 if float_dtype == jnp.float32 else 1e-12

    def test_lanczos_convergence_for_diagonal_matrix(self):
        """
        Given a diagonal matrix with known eigenvalues,
        When Lanczos is run with sufficient iterations,
        Then it should converge to the smallest eigenvalue.
        """

        # Diagonal matrix [[2, 0], [0, 5]] has eigenvalues 2, 5
        def L_apply(v):
            return jnp.array([2.0 * v[0], 5.0 * v[1]], dtype=self.float_dtype)

        key = jax.random.PRNGKey(42)
        gamma = estimate_gamma_lanczos(L_apply, key, (2,), k=10)

        # Should converge to min eigenvalue = 2.0
        assert jnp.isclose(
            gamma, 2.0, atol=self.tolerance
        ), f"Expected ~2.0, got {gamma}"

    def test_lanczos_matrix_free_consistency(self):
        """
        Given a linear operator defined matrix-free,
        When Lanczos is compared to direct eigenvalue computation,
        Then results should be consistent for small matrices.
        """
        # Create a small symmetric matrix
        A = jnp.array([[3.0, 1.0], [1.0, 4.0]], dtype=self.float_dtype)
        true_eigenvals = jnp.linalg.eigh(A)[0]  # [2.268, 4.732]

        def L_apply(v):
            return A @ v

        key = jax.random.PRNGKey(123)
        gamma = estimate_gamma_lanczos(L_apply, key, (2,), k=8)

        # Should be close to smallest eigenvalue
        assert jnp.isclose(
            gamma, true_eigenvals[0], atol=self.tolerance
        ), f"Expected ~{true_eigenvals[0]}, got {gamma}"

    def test_lanczos_handles_negative_eigenvalues(self):
        """
        Given a matrix with negative eigenvalues,
        When Lanczos estimates the spectral gap,
        Then it should correctly identify negative values.
        """

        # Matrix [[-1, 0], [0, 2]] has eigenvalues -1, 2
        def L_apply(v):
            return jnp.array([-1.0 * v[0], 2.0 * v[1]], dtype=self.float_dtype)

        key = jax.random.PRNGKey(456)
        gamma = estimate_gamma_lanczos(L_apply, key, (2,), k=10)

        # Should return -1.0 (most negative eigenvalue)
        assert jnp.isclose(
            gamma, -1.0, atol=self.tolerance
        ), f"Expected ~-1.0, got {gamma}"

    def test_lanczos_convergence_with_different_k(self):
        """
        Given the same operator with different iteration counts,
        When Lanczos is run with increasing k,
        Then estimates should improve (get closer to true value).
        """

        def L_apply(v):
            # Matrix with eigenvalues approximately [1.0, 6.0]
            return jnp.array(
                [2.0 * v[0] + v[1], v[0] + 5.0 * v[1]], dtype=self.float_dtype
            )

        key = jax.random.PRNGKey(789)

        gamma_k5 = estimate_gamma_lanczos(L_apply, key, (2,), k=5)
        gamma_k10 = estimate_gamma_lanczos(L_apply, key, (2,), k=10)

        # Higher k should give better estimate (closer to true min eigenvalue ~1.0)
        true_min = 1.0  # Approximate
        error_k5 = abs(gamma_k5 - true_min)
        error_k10 = abs(gamma_k10 - true_min)

        assert (
            error_k10 <= error_k5
        ), f"k=10 error {error_k10} should be <= k=5 error {error_k5}"

    def test_lanczos_jit_compatibility(self):
        """
        Given a Lanczos implementation,
        When JIT compiled,
        Then it should execute without errors.
        """

        def L_apply(v):
            return jnp.array([3.0 * v[0], 4.0 * v[1]], dtype=self.float_dtype)

        # Test that the function can be JIT compiled (without passing function as argument)
        @partial(jax.jit, static_argnums=(1, 2))  # Make shape and k static
        def jit_lanczos_fixed(key, shape, k):
            return estimate_gamma_lanczos(L_apply, key, shape, k)

        key = jax.random.PRNGKey(999)
        gamma = jit_lanczos_fixed(key, (2,), 8)

        assert jnp.isfinite(gamma), f"JIT result should be finite, got {gamma}"

    def test_lanczos_w_space_aware_integration(self):
        """
        Given a W-space aware L_apply function with real wavelet transform,
        When Lanczos is applied,
        Then it should work correctly in transformed space.
        """
        from computable_flows_shim.multi.transform_op import make_transform

        # Create a Haar wavelet transform for 4-element vectors
        transform = make_transform("haar", levels=1, ndim=1)

        # Simple operator that works on coefficient vectors (diagonal scaling)
        def L_coeffs(coeffs_flat):
            # Assume coeffs_flat represents flattened wavelet coefficients
            # Apply simple diagonal operator
            return 2.0 * coeffs_flat  # Scale all coefficients by 2

        key = jax.random.PRNGKey(111)

        # Test with W-space transform
        gamma_w_space = estimate_gamma_lanczos(
            L_coeffs, key, (4,), k=6, transform_op=transform
        )

        # Test without transform for comparison
        gamma_physical = estimate_gamma_lanczos(
            L_coeffs, key, (4,), k=6, transform_op=None
        )

        # Both should work and return finite values
        assert jnp.isfinite(
            gamma_w_space
        ), f"W-space Lanczos should return finite value, got {gamma_w_space}"
        assert jnp.isfinite(
            gamma_physical
        ), f"Physical space Lanczos should return finite value, got {gamma_physical}"
