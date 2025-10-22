"""
Manifold adapters for geometric optimization on Riemannian manifolds.

Provides the geometric primitives needed for manifold-aware optimization flows.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Protocol, Union
import jax
import jax.numpy as jnp

# Type aliases
Array = jnp.ndarray
State = Dict[str, Array]


class ManifoldAdapter(ABC):
    """
    Abstract base class for manifold adapters.
    
    Provides geometric operations needed for Riemannian optimization:
    - Project tangent vectors to tangent space
    - Retract from tangent space to manifold
    - Compute Riemannian gradient from Euclidean gradient
    """
    
    @abstractmethod
    def proj_tangent(self, x: Array, v: Array) -> Array:
        """
        Project a vector v onto the tangent space at point x.
        
        Args:
            x: Point on the manifold
            v: Vector to project
            
        Returns:
            Tangent vector at x
        """
        pass
    
    @abstractmethod
    def retract(self, x: Array, v: Array) -> Array:
        """
        Retract a tangent vector v at point x to the manifold.
        
        Args:
            x: Point on the manifold
            v: Tangent vector at x
            
        Returns:
            New point on the manifold
        """
        pass
    
    @abstractmethod
    def riemannian_gradient(self, x: Array, euclidean_grad: Array) -> Array:
        """
        Convert Euclidean gradient to Riemannian gradient.
        
        For most manifolds, this is just the projection to tangent space.
        
        Args:
            x: Point on the manifold
            euclidean_grad: Euclidean gradient
            
        Returns:
            Riemannian gradient (tangent vector)
        """
        return self.proj_tangent(x, euclidean_grad)


class EuclideanManifold(ManifoldAdapter):
    """
    Euclidean manifold (flat space).
    
    This is the default manifold for standard optimization.
    All operations are identity operations.
    """
    
    def proj_tangent(self, x: Array, v: Array) -> Array:
        """Tangent space is the entire space."""
        return v
    
    def retract(self, x: Array, v: Array) -> Array:
        """Retraction is just vector addition."""
        return x + v
    
    def riemannian_gradient(self, x: Array, euclidean_grad: Array) -> Array:
        """Riemannian gradient equals Euclidean gradient."""
        return euclidean_grad


class SphereManifold(ManifoldAdapter):
    """
    Sphere manifold: {x ∈ ℝⁿ | ‖x‖ = r}
    
    Useful for optimization over unit vectors or probability distributions.
    """
    
    def __init__(self, radius: float = 1.0):
        self.radius = radius
    
    def proj_tangent(self, x: Array, v: Array) -> Array:
        """Project v onto tangent space: v - (⟨x,v⟩/‖x‖²)x"""
        x_norm_sq = jnp.sum(x**2)
        return v - (jnp.sum(x * v) / x_norm_sq) * x
    
    def retract(self, x: Array, v: Array) -> Array:
        """Exponential map retraction: normalize(x + v) * radius"""
        x_new = x + v
        x_norm = jnp.sqrt(jnp.sum(x_new**2))
        return (x_new / x_norm) * self.radius
    
    def riemannian_gradient(self, x: Array, euclidean_grad: Array) -> Array:
        """Riemannian gradient on sphere."""
        return self.proj_tangent(x, euclidean_grad)


class StiefelManifold(ManifoldAdapter):
    """
    Stiefel manifold: {X ∈ ℝⁿᵏ | X^T X = Iₖ}
    
    Orthogonal matrices. Useful for optimization over rotations or orthonormal bases.
    """
    
    def __init__(self, n: int, k: int):
        self.n = n
        self.k = k
    
    def proj_tangent(self, x: Array, v: Array) -> Array:
        """Project onto tangent space: v - x(x^T v + v^T x)/2"""
        xtv = x.T @ v
        return v - x @ (xtv + xtv.T) / 2
    
    def retract(self, x: Array, v: Array) -> Array:
        """QR retraction: orthogonalize(x + v)"""
        x_new = x + v
        q, r = jnp.linalg.qr(x_new)
        # Ensure positive diagonals for consistency
        signs = jnp.sign(jnp.diag(r))
        q = q * signs
        return q
    
    def riemannian_gradient(self, x: Array, euclidean_grad: Array) -> Array:
        """Riemannian gradient on Stiefel manifold."""
        return self.proj_tangent(x, euclidean_grad)


class PositiveDefiniteManifold(ManifoldAdapter):
    """
    Symmetric positive definite matrices: {X ∈ ℝⁿˣⁿ | X = X^T, X ≻ 0}
    
    Useful for covariance matrices, metrics, etc.
    """
    
    def proj_tangent(self, x: Array, v: Array) -> Array:
        """Project onto tangent space: symmetrize(v)"""
        return (v + v.T) / 2
    
    def retract(self, x: Array, v: Array) -> Array:
        """Matrix exponential retraction: exp(v) @ x"""
        return jax.scipy.linalg.expm(v) @ x
    
    def riemannian_gradient(self, x: Array, euclidean_grad: Array) -> Array:
        """Riemannian gradient on SPD manifold."""
        return self.proj_tangent(x, euclidean_grad)


# Registry of available manifolds
MANIFOLD_REGISTRY: Dict[str, type[ManifoldAdapter]] = {
    'euclidean': EuclideanManifold,
    'sphere': SphereManifold,
    'stiefel': StiefelManifold,
    'spd': PositiveDefiniteManifold,
}


def create_manifold(manifold_type: str, **params) -> ManifoldAdapter:
    """
    Factory function to create manifold adapters.
    
    Args:
        manifold_type: Type of manifold ('euclidean', 'sphere', etc.)
        **params: Parameters for manifold construction
        
    Returns:
        Configured manifold adapter
        
    Raises:
        ValueError: If manifold_type is not registered
    """
    if manifold_type not in MANIFOLD_REGISTRY:
        available = list(MANIFOLD_REGISTRY.keys())
        raise ValueError(f"Unknown manifold type: {manifold_type}. Available: {available}")
    
    manifold_class = MANIFOLD_REGISTRY[manifold_type]
    return manifold_class(**params)


def register_manifold(manifold_type: str, manifold_class: type[ManifoldAdapter]):
    """Register a new manifold type."""
    MANIFOLD_REGISTRY[manifold_type] = manifold_class