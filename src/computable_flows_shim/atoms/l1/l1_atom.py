"""
L1 Atom Implementation.

This module implements the L1 regularization atom: λ‖x‖₁
"""

from typing import Dict, Any
import jax.numpy as jnp
from ..base import Atom
from computable_flows_shim.core import numerical_stability_check

# Type aliases
Array = jnp.ndarray
State = Dict[str, Array]


class L1Atom(Atom):
    """
    L1 regularization atom: λ‖x‖₁

    This implements L1 regularization for sparse recovery and compressed sensing.
    The proximal operator is the soft-thresholding function.
    """

    @property
    def name(self) -> str:
        return "l1"

    @property
    def form(self) -> str:
        return r"\lambda\|x\|_1"

    @numerical_stability_check
    def energy(self, state: State, params: Dict[str, Any]) -> float:
        """Compute L1 energy: λ‖x‖₁"""
        lam = params.get('lambda', 1.0)
        x = state[params['variable']]

        return lam * float(jnp.sum(jnp.abs(x)))

    @numerical_stability_check
    def gradient(self, state: State, params: Dict[str, Any]) -> State:
        """L1 regularization is not differentiable, but subgradient exists."""
        # L1 is not differentiable at zero, so we return a subgradient
        # For practical purposes, we can return the sign function
        lam = params.get('lambda', 1.0)
        x = state[params['variable']]

        # Subgradient of λ‖x‖₁ is λ*sign(x) where sign(0) can be any value in [-1, 1]
        subgrad_x = lam * jnp.sign(x)

        return {params['variable']: subgrad_x}

    @numerical_stability_check
    def prox(self, state: State, step_size: float, params: Dict[str, Any]) -> State:
        """
        Proximal operator for L1 regularization: soft-thresholding.

        prox_τ^g(x) where g(y) = λ‖y‖₁ is the soft-thresholding operator:
        S_λτ(x_i) = sign(x_i) * max(|x_i| - λτ, 0)
        """
        lam = params.get('lambda', 1.0)
        x = state[params['variable']]

        # Soft-thresholding: S_λτ(x) = sign(x) * max(|x| - λτ, 0)
        threshold = lam * step_size
        x_new = jnp.sign(x) * jnp.maximum(jnp.abs(x) - threshold, 0)

        return {params['variable']: x_new}

    def certificate_contributions(self, params: Dict[str, Any]) -> Dict[str, float]:
        """
        Certificate contributions for L1 atom.

        L1 regularization doesn't contribute to Lipschitz constants but affects convergence.
        """
        lam = params.get('lambda', 1.0)

        return {
            'lipschitz': 0.0,  # L1 doesn't contribute to gradient Lipschitz
            'eta_dd_contribution': 0.0,  # No diagonal dominance contribution
            'gamma_contribution': 0.0   # No spectral contribution (nonsmooth)
        }