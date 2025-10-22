"""
Test-Driven Development for Manifold Adapters.

Following the RED-GREEN-REFACTOR cycle with Design by Contract.
Tests ensure manifold operations satisfy geometric properties.
"""

import pytest
import jax
import jax.numpy as jnp
from computable_flows_shim.runtime.manifolds import (
    EuclideanManifold, SphereManifold, StiefelManifold, PositiveDefiniteManifold,
    create_manifold, MANIFOLD_REGISTRY
)


class TestManifoldAdaptersContract:
    """
    Design by Contract tests for manifold adapters.
    
    Contract: Manifold adapters provide correct geometric operations
    for Riemannian optimization with proper tangent space projections and retractions.
    """

    @pytest.fixture
    def euclidean_manifold(self):
        """Euclidean manifold for testing."""
        return EuclideanManifold()

    @pytest.fixture
    def sphere_manifold(self):
        """Unit sphere manifold for testing."""
        return SphereManifold(radius=1.0)

    @pytest.fixture
    def stiefel_manifold(self):
        """Stiefel manifold (3x2 orthogonal matrices) for testing."""
        return StiefelManifold(n=3, k=2)

    def test_euclidean_manifold_operations(self, euclidean_manifold):
        """RED: Euclidean manifold should have identity operations."""
        x = jnp.array([1.0, 2.0, 3.0])
        v = jnp.array([0.1, 0.2, 0.3])

        # Tangent projection should be identity
        proj_v = euclidean_manifold.proj_tangent(x, v)
        assert jnp.allclose(proj_v, v)

        # Retraction should be addition
        retracted = euclidean_manifold.retract(x, v)
        assert jnp.allclose(retracted, x + v)

        # Riemannian gradient should equal Euclidean
        grad = euclidean_manifold.riemannian_gradient(x, v)
        assert jnp.allclose(grad, v)

    def test_sphere_manifold_tangent_projection(self, sphere_manifold):
        """RED: Sphere tangent projection should be orthogonal to position vector."""
        # Point on unit sphere
        x = jnp.array([1.0, 0.0, 0.0])  # Already normalized
        v = jnp.array([0.5, 0.5, 0.5])  # Arbitrary vector

        proj_v = sphere_manifold.proj_tangent(x, v)

        # Projected vector should be orthogonal to x
        dot_product = jnp.sum(x * proj_v)
        assert abs(dot_product) < 1e-10

        # Original component along x should be removed
        expected_proj = v - (jnp.sum(x * v) / jnp.sum(x**2)) * x
        assert jnp.allclose(proj_v, expected_proj)

    def test_sphere_manifold_retraction(self, sphere_manifold):
        """RED: Sphere retraction should stay on sphere."""
        x = jnp.array([1.0, 0.0, 0.0])
        v = jnp.array([0.0, 0.1, 0.0])  # Tangent vector

        retracted = sphere_manifold.retract(x, v)

        # Result should be on unit sphere
        norm = jnp.sqrt(jnp.sum(retracted**2))
        assert abs(norm - 1.0) < 1e-6

    def test_sphere_manifold_riemannian_gradient(self, sphere_manifold):
        """RED: Riemannian gradient should be tangent to sphere."""
        x = jnp.array([0.0, 1.0, 0.0])  # Point on sphere
        euclidean_grad = jnp.array([1.0, 1.0, 1.0])

        riemannian_grad = sphere_manifold.riemannian_gradient(x, euclidean_grad)

        # Riemannian gradient should be tangent (orthogonal to x)
        dot_product = jnp.sum(x * riemannian_grad)
        assert abs(dot_product) < 1e-10

    def test_stiefel_manifold_tangent_projection(self, stiefel_manifold):
        """RED: Stiefel tangent projection should preserve structure."""
        # Create a 3x2 orthogonal matrix
        x = jnp.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])  # First 2 rows of identity
        v = jnp.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])  # Arbitrary matrix

        proj_v = stiefel_manifold.proj_tangent(x, v)

        # Check that x^T * proj_v is skew-symmetric (property of tangent space)
        xt_proj = x.T @ proj_v
        skew_check = xt_proj + xt_proj.T
        assert jnp.max(jnp.abs(skew_check)) < 1e-6

    def test_stiefel_manifold_retraction(self, stiefel_manifold):
        """RED: Stiefel retraction should produce orthogonal matrix."""
        x = jnp.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
        v = jnp.array([[0.01, 0.02], [0.03, 0.04], [0.05, 0.06]])

        retracted = stiefel_manifold.retract(x, v)

        # Result should be 3x2 with orthogonal columns
        # Check that columns are orthonormal
        col1 = retracted[:, 0]
        col2 = retracted[:, 1]

        norm1 = jnp.sqrt(jnp.sum(col1**2))
        norm2 = jnp.sqrt(jnp.sum(col2**2))
        dot_product = jnp.sum(col1 * col2)

        assert abs(norm1 - 1.0) < 1e-6
        assert abs(norm2 - 1.0) < 1e-6
        assert abs(dot_product) < 1e-6

    def test_positive_definite_manifold_operations(self):
        """RED: SPD manifold should handle symmetric matrices."""
        spd_manifold = PositiveDefiniteManifold()

        # Positive definite matrix
        x = jnp.array([[2.0, 0.5], [0.5, 1.0]])
        v = jnp.array([[0.1, 0.2], [0.3, 0.4]])  # Not necessarily symmetric

        # Tangent projection should symmetrize
        proj_v = spd_manifold.proj_tangent(x, v)
        assert jnp.allclose(proj_v, (v + v.T) / 2)

        # Retraction should preserve positive definiteness
        retracted = spd_manifold.retract(x, proj_v)
        # Check eigenvalues are positive (basic check)
        eigenvals = jnp.linalg.eigvals(retracted)
        assert jnp.all(eigenvals.real > 0)

    def test_manifold_factory(self):
        """RED: Factory function should create correct manifold instances."""
        euclidean = create_manifold('euclidean')
        assert isinstance(euclidean, EuclideanManifold)

        sphere = create_manifold('sphere', radius=2.0)
        assert isinstance(sphere, SphereManifold)
        assert sphere.radius == 2.0

        stiefel = create_manifold('stiefel', n=4, k=3)
        assert isinstance(stiefel, StiefelManifold)
        assert stiefel.n == 4 and stiefel.k == 3

    def test_unknown_manifold_error(self):
        """RED: Factory should raise error for unknown manifold types."""
        with pytest.raises(ValueError, match="Unknown manifold type"):
            create_manifold('unknown_manifold')

    def test_manifold_registry_integrity(self):
        """RED: Registry should contain expected manifold types."""
        expected_types = {'euclidean', 'sphere', 'stiefel', 'spd'}
        assert set(MANIFOLD_REGISTRY.keys()) == expected_types

    def test_f_dis_with_manifold_support(self):
        """RED: F_Dis should support Riemannian manifolds."""
        from computable_flows_shim.runtime.primitives import F_Dis
        from computable_flows_shim.runtime.manifolds import SphereManifold
        
        # Function with non-zero Riemannian gradient on sphere: f(x) = x[1] (y-coordinate)
        def f_value(state):
            x = state['x']
            return x[1]  # y-coordinate
        
        grad_f = jax.grad(f_value)
        
        # Test on sphere manifold
        manifolds = {'x': SphereManifold(radius=1.0)}
        
        # Start at a point on the sphere
        state = {'x': jnp.array([1.0, 0.0, 0.0])}
        
        # Apply one gradient step
        new_state = F_Dis(state, grad_f, step_alpha=0.1, manifolds=manifolds)
        
        # Result should still be on the sphere (approximately)
        x_new = new_state['x']
        norm = jnp.sqrt(jnp.sum(x_new**2))
        assert abs(norm - 1.0) < 1e-6
        
        # Should have moved (gradient descent should change the point)
        assert not jnp.allclose(state['x'], x_new)