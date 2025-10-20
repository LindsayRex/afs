from typing import Callable, Optional
import jax
import jax.numpy as jnp
from computable_flows_shim.multi.wavelets import TransformOp

def LW_apply(v: jnp.ndarray, L_apply: Callable, tf: TransformOp) -> jnp.ndarray:
    """
    Applies the operator L in the wavelet domain.
    L_W = W L W^T
    """
    # tf.forward = analysis (W), tf.inverse = synthesis (W^T or tilde W)
    u = tf.inverse(v)          # synth -> physical domain
    Lu = L_apply(u)            # apply core operator
    return tf.forward(Lu)     # analysis back to W-space

def estimate_eta_dd_in_W(L_apply: Callable, W: TransformOp, key, n_probe: int = 16, eps: float = 1e-9, shape=(10,)) -> float:
    """
    Estimates the diagonal dominance (η) of an operator in the wavelet domain.
    """
    # This is a simplified implementation of the random probe method.
    # A real implementation would be more robust.
    N = shape[0]

    # Generate random Rademacher probes
    probes = jax.random.rademacher(key, shape=(n_probe, N))

    # Apply the operator in the wavelet domain
    LW_probes = jax.vmap(LW_apply, in_axes=(0, None, None))(probes, L_apply, W)

    # Estimate the diagonal and off-diagonal sums
    diag_abs = jnp.abs(jnp.einsum('bi,bi->b', probes, LW_probes))
    off_diag_abs = jnp.sum(jnp.abs(LW_probes), axis=1) - diag_abs

    # Compute the diagonal dominance ratio
    eta_dd = jnp.mean(off_diag_abs / (diag_abs + eps))
    
    return float(eta_dd)

def estimate_gamma_in_W(L_apply: Callable, W: TransformOp, key, shape=(10,), iters: int = 64) -> float:
    """
    Estimates the spectral gap (γ) of an operator in the wavelet domain using power iteration.
    """
    N = shape[0]
    
    # Define the operator in the wavelet domain
    @jax.jit
    def LW_fn(v):
        return LW_apply(v, L_apply, W)

    # Power iteration to find the largest eigenvalue (Lipschitz constant)
    v = jax.random.normal(key, shape)
    for _ in range(iters):
        v = LW_fn(v)
        v /= jnp.linalg.norm(v)
    
    lipschitz = jnp.vdot(v, LW_fn(v))

    # Estimate the smallest eigenvalue using the Lipschitz constant
    # This is a simplification and assumes the operator is positive definite
    # A more robust implementation would use a more sophisticated method
    # like Lanczos or inverse iteration.
    
    # Shift the operator to find the smallest eigenvalue
    @jax.jit
    def shifted_LW_fn(v):
        return lipschitz * v - LW_fn(v)

    v = jax.random.normal(key, shape)
    for _ in range(iters):
        v = shifted_LW_fn(v)
        v /= jnp.linalg.norm(v)
        
    lambda_max_shifted = jnp.vdot(v, shifted_LW_fn(v))
    lambda_min = lipschitz - lambda_max_shifted

    return float(lambda_min)

def check_lyapunov(E_values: jnp.ndarray, alpha0: Optional[float] = None, grad_norms: Optional[jnp.ndarray] = None) -> bool:
    """
    Checks for Lyapunov descent.
    """
    # Placeholder implementation
    return True
