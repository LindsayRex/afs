"""
Tikhonov Atom Implementation.

This module implements the Tikhonov regularized quadratic atom: (1/2)‖Ax - b‖² + (λ/2)‖x‖²
"""

from typing import Dict, Any
import jax.numpy as jnp
from ..base import Atom
from computable_flows_shim.core import numerical_stability_check

# Type aliases
Array = jnp.ndarray
State = Dict[str, Array]


class TikhonovAtom(Atom):
    """
    Tikhonov regularized quadratic atom: (1/2)‖Ax - b‖² + (λ/2)‖x‖²

    This implements Tikhonov regularization for ill-posed inverse problems.
    The regularization parameter λ controls the trade-off between data fidelity and smoothness.
    """

    @property
    def name(self) -> str:
        return "tikhonov"

    @property
    def form(self) -> str:
        return r"\frac{1}{2}\|Ax - b\|_2^2 + \frac{\lambda}{2}\|x\|_2^2"

    @numerical_stability_check
    def energy(self, state: State, params: Dict[str, Any]) -> float:
        """Compute Tikhonov energy: (1/2)‖Ax - b‖² + (λ/2)‖x‖²"""
        A = params['A']
        b = params['b']
        lam = params.get('lambda', 1.0)  # Default regularization parameter
        x = state[params['variable']]

        residual = A @ x - b
        data_fidelity = 0.5 * float(jnp.sum(residual**2))
        regularization = 0.5 * lam * float(jnp.sum(x**2))

        return data_fidelity + regularization

    @numerical_stability_check
    def gradient(self, state: State, params: Dict[str, Any]) -> State:
        """Compute gradient: A^T(Ax - b) + λx"""
        A = params['A']
        b = params['b']
        lam = params.get('lambda', 1.0)
        x = state[params['variable']]

        residual = A @ x - b
        grad_x = A.T @ residual + lam * x

        return {params['variable']: grad_x}

    @numerical_stability_check
    def prox(self, state: State, step_size: float, params: Dict[str, Any]) -> State:
        """
        Proximal operator for Tikhonov regularization.

        Solves: argmin_x (1/2)‖Ax - b‖² + (λ/2)‖x‖² + (1/(2τ))‖x - x₀‖²
        Solution: (A^T A + λI + I/τ) x = A^T b + x₀/τ
        """
        A = params['A']
        b = params['b']
        lam = params.get('lambda', 1.0)
        x = state[params['variable']]

        # Form the regularized system: (A^T A + λI + I/step_size)
        ATA = A.T @ A
        ATb = A.T @ b

        # Add regularization and proximal regularization
        regularization_matrix = lam * jnp.eye(ATA.shape[0])
        proximal_matrix = jnp.eye(ATA.shape[0]) / step_size

        lhs = ATA + regularization_matrix + proximal_matrix
        rhs = ATb + x / step_size

        x_new = jnp.linalg.solve(lhs, rhs)

        return {params['variable']: x_new}

    def certificate_contributions(self, params: Dict[str, Any]) -> Dict[str, float]:
        """
        Certificate contributions for Tikhonov atom.

        The regularization improves conditioning and provides better certificates.
        """
        A = params['A']
        lam = params.get('lambda', 1.0)

        # Effective Lipschitz constant: σ_max(A^T A + λI)
        ATA = A.T @ A
        regularization_matrix = lam * jnp.eye(ATA.shape[0])
        effective_matrix = ATA + regularization_matrix
        lipschitz = float(jnp.linalg.norm(effective_matrix, ord=2))

        return {
            'lipschitz': lipschitz,
            'eta_dd_contribution': lam,  # Regularization improves diagonal dominance
            'gamma_contribution': -lipschitz  # Still contributes negatively, but less than unregularized
        }