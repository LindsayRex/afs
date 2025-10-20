"""
Certificate estimators for Flow Dynamic Analysis (FDA).
"""
from typing import Callable
import jax
import jax.numpy as jnp

def estimate_gamma(L_apply: Callable, key: jnp.ndarray, input_shape: tuple, num_iterations: int = 20) -> float:
    """
    Estimates the spectral gap (smallest eigenvalue) of a linear operator using the inverse power method.
    """
    dim = input_shape[0]
    L_matrix = jax.vmap(L_apply)(jnp.eye(dim))

    v = jax.random.normal(key, input_shape)
    v = v / jnp.linalg.norm(v)

    for _ in range(num_iterations):
        # Solve L*w = v for w
        w = jax.scipy.linalg.solve(L_matrix, v)
        v = w / jnp.linalg.norm(w)

    # The smallest eigenvalue is 1 / (v^T * L * v)
    # Since L*v is not available, we use v^T * w, where w = L_inv*v
    # and eigenvalue of L_inv is 1/eigenvalue of L
    # so smallest eigenvalue of L is 1/largest eigenvalue of L_inv
    # Rayleigh quotient for L_inv is (v^T * L_inv * v) / (v^T * v) = v^T * w
    largest_eigval_inv = jnp.dot(v, w)
    
    return float(1.0 / largest_eigval_inv)

def estimate_eta_dd(L_apply: Callable, input_shape: tuple, eps: float = 1e-9) -> float:
    """
    Estimates the diagonal dominance (eta) of a linear operator.
    """
    # Construct the matrix representation of the linear operator L
    dim = input_shape[0]
    basis_vectors = jnp.eye(dim)
    L_matrix = jax.vmap(L_apply)(basis_vectors)
    
    # Get the absolute values of the diagonal and the full matrix
    abs_L = jnp.abs(L_matrix)
    diag_abs_L = jnp.diag(abs_L)
    
    # Calculate the sum of absolute values of off-diagonal elements for each row
    off_diag_row_sum = jnp.sum(abs_L, axis=1) - diag_abs_L
    
    # Compute the ratio for each row
    # Add epsilon to avoid division by zero
    ratios = off_diag_row_sum / (diag_abs_L + eps)
    
    # The diagonal dominance is the maximum of these ratios
    return float(jnp.max(ratios))
