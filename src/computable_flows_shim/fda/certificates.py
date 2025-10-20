"""
Certificate estimators for Flow Dynamic Analysis (FDA).
"""
from typing import Callable
import jax
import jax.numpy as jnp
from jax.numpy.linalg import eigsh

def estimate_gamma(L_apply: Callable, key: jnp.ndarray, input_shape: tuple) -> float:
    """
    Estimates the spectral gap (smallest eigenvalue) of a linear operator using Lanczos iteration.
    """
    # Generate a random starting vector for the Lanczos algorithm
    v0 = jax.random.normal(key, input_shape)
    
    # Use eigsh to find the smallest eigenvalue ('SA' means smallest algebraic)
    # We ask for k=1 eigenvalue. eigsh returns eigenvalues and eigenvectors.
    eigenvalues, _ = eigsh(lambda v: L_apply(v), v0, k=1, which='SA')
    
    # The result is an array with one element, so we extract it.
    return eigenvalues[0]
