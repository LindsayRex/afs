"""
Certificate estimators for Flow Dynamic Analysis (FDA).
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Optional

import jax
import jax.numpy as jnp

from computable_flows_shim.core import numerical_stability_check

if TYPE_CHECKING:
    from computable_flows_shim.multi.transform_op import TransformOp


@numerical_stability_check
def estimate_gamma(
    L_apply: Callable,
    key: jnp.ndarray,
    input_shape: tuple,
    num_iterations: int = 20,
    mode: str = "dense",
    lanczos_k: int = 20,
) -> float:
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
            off_diag_sum = jnp.array(
                [
                    jnp.sum(jnp.abs(L_matrix[i, :])) - jnp.abs(L_matrix[i, i])
                    for i in range(dim)
                ]
            )
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
            for _ in range(k):
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
            off_diag_sum = jnp.array(
                [
                    jnp.sum(jnp.abs(L_matrix[i, :])) - jnp.abs(L_matrix[i, i])
                    for i in range(dim)
                ]
            )
            gershgorin_bounds = diag - off_diag_sum
            return float(jnp.min(gershgorin_bounds))
        # run lanczos
        return lanczos_min_eig(L_apply, dim, lanczos_k, key)
    else:
        # Unknown mode: default to Gershgorin safe fallback
        diag = jnp.diag(L_matrix)
        off_diag_sum = jnp.array(
            [
                jnp.sum(jnp.abs(L_matrix[i, :])) - jnp.abs(L_matrix[i, i])
                for i in range(dim)
            ]
        )
        gershgorin_bounds = diag - off_diag_sum
        return float(jnp.min(gershgorin_bounds))
    # End of function


@numerical_stability_check
def estimate_gamma_lanczos(
    L_apply: Callable,
    key: jnp.ndarray,
    input_shape: tuple,
    k: int = 20,
    transform_op: Optional["TransformOp"] = None,
):
    """
    Estimates the spectral gap using matrix-free Lanczos method with JAX lax.scan.

    This implements the Lanczos algorithm for finding the smallest eigenvalue of a
    symmetric positive definite operator. The algorithm builds a tridiagonal matrix
    whose eigenvalues approximate those of the original operator.

    Args:
        L_apply: Matrix-free linear operator L(v) -> vector
        key: JAX PRNG key for random initialization
        input_shape: Shape of input vectors (typically (n,) for n-dimensional problem)
        k: Number of Lanczos iterations (higher k = better approximation but more expensive)
        transform_op: Optional wavelet transform for W-space aware computation.
                     If provided, computes eigenvalues of W^T L W instead of L.

    Returns:
        Estimated minimum eigenvalue (spectral gap)
    """
    from jax import lax

    # Extract dimension - make sure this is done before JIT tracing
    dim = input_shape[0]

    # Initialize random starting vector
    v0 = jax.random.normal(key, (dim,))
    v0 = v0 / jnp.linalg.norm(v0)

    # Create W-space aware L_apply if transform provided
    if transform_op is not None:

        def L_w_space(v):
            # Transform to W-space, apply L, transform back
            coeffs = transform_op.forward(v)

            # For simplicity, assume L_apply works on flattened coefficients
            # This is a common pattern for W-space operators
            if isinstance(coeffs, list):
                coeffs_flat = jnp.concatenate([c.flatten() for c in coeffs])
            else:
                coeffs_flat = coeffs.flatten()

            # Apply the operator in coefficient space
            result_flat = L_apply(coeffs_flat)

            # Reshape back to coefficient structure
            if isinstance(coeffs, list):
                result_coeffs = []
                start_idx = 0
                for c in coeffs:
                    size = c.size
                    result_coeffs.append(
                        result_flat[start_idx : start_idx + size].reshape(c.shape)
                    )
                    start_idx += size
                return transform_op.inverse(result_coeffs)
            else:
                return transform_op.inverse(result_flat.reshape(coeffs.shape))

        effective_L_apply = L_w_space
    else:
        effective_L_apply = L_apply

    # Lanczos algorithm state
    def lanczos_step(carry, _):
        v, v_prev, beta_prev = carry

        # Matrix-vector product
        w = effective_L_apply(v)

        # Compute alpha (Rayleigh quotient)
        alpha = jnp.dot(v, w)

        # Gram-Schmidt orthogonalization
        w = w - alpha * v - beta_prev * v_prev

        # Compute new beta
        beta = jnp.linalg.norm(w)

        # New normalized vector (avoid division by zero)
        v_next = jnp.where(beta > 1e-12, w / beta, jnp.zeros_like(v))

        return (v_next, v, beta), (alpha, beta)

    # Initial carry: (current_v, previous_v, previous_beta)
    init_carry = (v0, jnp.zeros_like(v0), 0.0)

    # Run k Lanczos iterations
    _, (alphas, betas) = lax.scan(lanczos_step, init_carry, jnp.arange(k))

    # Build tridiagonal matrix T (k x k)
    T = jnp.zeros((k, k))

    # Set diagonal elements (alphas)
    T = T.at[jnp.diag_indices(k)].set(alphas)

    # Set off-diagonal elements (betas)
    if k > 1:
        # betas[0] is the first beta (between v0 and v1)
        # betas[1] is between v1 and v2, etc.
        # So we need betas[0:k-1] for the off-diagonals
        off_diag = betas[: k - 1]
        T = T.at[(jnp.arange(1, k), jnp.arange(k - 1))].set(off_diag)
        T = T.at[(jnp.arange(k - 1), jnp.arange(1, k))].set(off_diag)

    # Compute eigenvalues of tridiagonal matrix
    eigenvals = jnp.linalg.eigh(T)[0]

    # Filter out spurious small eigenvalues (numerical artifacts from Lanczos)
    # Use JIT-compatible approach: mask small eigenvalues with large positive value
    threshold = 1e-6
    masked_eigenvals = jnp.where(jnp.abs(eigenvals) > threshold, eigenvals, 1e10)

    # Find the minimum among significant eigenvalues
    # Convert to ensure type checker understands this is an array
    masked_array = jnp.asarray(masked_eigenvals)
    min_significant = jnp.min(masked_array)

    # If all eigenvalues were masked (spurious), return the eigenvalue with smallest absolute value
    # Otherwise return the minimum significant eigenvalue
    abs_eigenvals = jnp.abs(eigenvals)
    min_abs = jnp.min(abs_eigenvals)
    return jnp.where(min_significant < 1e9, min_significant, min_abs)


@numerical_stability_check
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
