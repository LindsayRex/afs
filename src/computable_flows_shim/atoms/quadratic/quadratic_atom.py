"""
Quadratic Atom Implementation.

This module implements the quadratic data fidelity atom: (1/2)‖Ax - b‖²
"""

from typing import Any

import jax.numpy as jnp

from computable_flows_shim.core import numerical_stability_check

from ..base import Atom

# Type aliases
Array = jnp.ndarray
State = dict[str, Array]


class QuadraticAtom(Atom):
    """
    Quadratic data fidelity atom: (1/2)‖Ax - b‖²

    This is the fundamental atom for least squares and Gaussian likelihoods.
    Mathematical properties:
    - Convex and differentiable
    - Lipschitz gradient with constant σ_max(A^T A)
    - Proximal operator has closed-form solution
    """

    @property
    def name(self) -> str:
        return "quadratic"

    @property
    def form(self) -> str:
        return r"\frac{1}{2}\|Ax - b\|_2^2"

    @numerical_stability_check
    def energy(self, state: State, params: dict[str, Any]) -> float:
        """Compute quadratic energy: (1/2)‖Ax - b‖²"""
        A = params["A"]
        b = params["b"]
        x = state[params["variable"]]

        residual = A @ x - b
        return 0.5 * float(jnp.sum(residual**2))

    @numerical_stability_check
    def gradient(self, state: State, params: dict[str, Any]) -> State:
        """Compute gradient: A^T(Ax - b)"""
        A = params["A"]
        b = params["b"]
        x = state[params["variable"]]

        residual = A @ x - b
        grad_x = A.T @ residual

        return {params["variable"]: grad_x}

    @numerical_stability_check
    def prox(self, state: State, step_size: float, params: dict[str, Any]) -> State:
        """
        Proximal operator for quadratic (exact solution via linear system).

        Solves: argmin_x (1/2)‖Ax - b‖² + (1/(2τ))‖x - x₀‖²
        Solution: (A^T A + I/τ) x = A^T b + x₀/τ
        """
        A = params["A"]
        b = params["b"]
        x = state[params["variable"]]

        # Form the regularized system: (A^T A + I/step_size)
        ATA = A.T @ A
        ATb = A.T @ b

        # Regularized system matrix and RHS
        lhs = ATA + jnp.eye(ATA.shape[0]) / step_size
        rhs = ATb + x / step_size

        # Solve the linear system
        x_new = jnp.linalg.solve(lhs, rhs)

        return {params["variable"]: x_new}

    def certificate_contributions(self, params: dict[str, Any]) -> dict[str, float]:
        """
        Certificate contributions for quadratic atom.

        Returns Lipschitz constant and diagonal dominance contributions.
        For quadratic atoms, the Lipschitz constant is σ_max(A^T A).
        """
        A = params["A"]

        # Compute spectral norm of A^T A (Lipschitz constant of gradient)
        ATA = A.T @ A
        lipschitz = float(jnp.linalg.norm(ATA, ord=2))

        return {
            "lipschitz": lipschitz,
            "eta_dd_contribution": 0.0,  # Quadratic terms don't affect diagonal dominance
            "gamma_contribution": -lipschitz,  # Contributes negatively to spectral gap
        }
