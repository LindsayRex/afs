"""
Certificate estimators for Flow Dynamic Analysis (FDA).
"""
from typing import Callable
import jax
import jax.numpy as jnp

def estimate_gamma(L_apply: Callable, key: jnp.ndarray, input_shape: tuple, num_iterations: int = 20, mode: str = "dense", lanczos_k: int = 20) -> float:
    """
    Estimates the spectral gap (minimum eigenvalue for symmetric matrices, Gershgorin lower bound for non-symmetric).
    """
    dim = input_shape[0]
    # Construct L as a matrix where columns are L_apply(e_i); transpose to row-major
    L_matrix = jax.vmap(L_apply)(jnp.eye(dim)).T

    # Check symmetry
    is_symmetric = jnp.allclose(L_matrix, L_matrix.T, atol=1e-8)
    if mode == "dense":
        if is_symmetric:
            eigvals = jnp.linalg.eigh(L_matrix)[0]
            eigvals_real = jnp.real(eigvals)
            return float(jnp.min(eigvals_real))
        else:
            # Gershgorin fallback for non-symmetric
            diag = jnp.diag(L_matrix)
            off_diag_sum = jnp.array([
                jnp.sum(jnp.abs(L_matrix[i, :])) - jnp.abs(L_matrix[i, i])
                for i in range(dim)
            ])
            gershgorin_bounds = diag - off_diag_sum
            return float(jnp.min(gershgorin_bounds))
    elif mode == "lanczos":
        # Lanczos matrix-free approximation for symmetric operators
        # Use L_apply directly with vector inputs
        def lanczos_min_eig(L_apply, dim, k, key):
            v = jax.random.normal(key, (dim,))
            v = v / jnp.linalg.norm(v)
            alphas = []
            betas = []
            v_prev = jnp.zeros_like(v)
            beta_prev = 0.0
            for i in range(k):
                w = L_apply(v)
                alpha = jnp.dot(v, w)
                w = w - alpha * v - beta_prev * v_prev
                beta = jnp.linalg.norm(w)
                alphas.append(alpha)
                betas.append(beta)
                if beta == 0:
                    break
                v_prev = v
                v = w / beta
                beta_prev = beta
            m = len(alphas)
            T = jnp.zeros((m, m))
            alphas_arr = jnp.array(alphas)
            T = T + jnp.diag(alphas_arr)
            if m > 1:
                off = jnp.array(betas[:-1])
                T = T + jnp.diag(off, 1) + jnp.diag(off, -1)
            eigs = jnp.linalg.eigh(T)[0]
            return float(jnp.min(eigs))

        if not is_symmetric:
            # Lanczos requires symmetry; fall back to Gershgorin
            diag = jnp.diag(L_matrix)
            off_diag_sum = jnp.array([
                jnp.sum(jnp.abs(L_matrix[i, :])) - jnp.abs(L_matrix[i, i])
                for i in range(dim)
            ])
            gershgorin_bounds = diag - off_diag_sum
            return float(jnp.min(gershgorin_bounds))
        # run lanczos
        return lanczos_min_eig(L_apply, dim, lanczos_k, key)
    else:
        # Unknown mode: default to Gershgorin safe fallback
        diag = jnp.diag(L_matrix)
        off_diag_sum = jnp.array([
            jnp.sum(jnp.abs(L_matrix[i, :])) - jnp.abs(L_matrix[i, i])
            for i in range(dim)
        ])
        gershgorin_bounds = diag - off_diag_sum
        return float(jnp.min(gershgorin_bounds))
    # End of function

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
