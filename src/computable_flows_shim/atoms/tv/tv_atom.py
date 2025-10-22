"""
TV Atom Implementation.

This module implements the total variation regularization atom: λ‖Dx‖₁
"""

from typing import Dict, Any
import jax.numpy as jnp
from ..base import Atom

# Type aliases
Array = jnp.ndarray
State = Dict[str, Array]


class TVAtom(Atom):
    """
    Total Variation regularization atom: λ‖Dx‖₁

    This implements anisotropic total variation regularization for piecewise-constant
    signals/images. The finite difference operator D creates differences between
    neighboring elements. The proximal operator uses shrinkage on these differences.
    """

    @property
    def name(self) -> str:
        return "tv"

    @property
    def form(self) -> str:
        return r"\lambda\|Dx\|_1"

    def energy(self, state: State, params: Dict[str, Any]) -> float:
        """Compute TV energy: λ‖Dx‖₁"""
        lam = params.get('lambda', 1.0)
        x = state[params['variable']]

        # Compute finite differences (anisotropic TV)
        if x.ndim == 1:
            # 1D signal: forward differences
            diff = x[1:] - x[:-1]
        else:
            # Multi-dimensional: anisotropic differences along each axis
            diff = jnp.zeros_like(x)
            for axis in range(x.ndim):
                slices = [slice(None)] * x.ndim
                slices[axis] = slice(1, None)
                diff = diff + (x[tuple(slices)] - x[tuple(slices[:-1] + [slice(None, -1)])])**2
            diff = jnp.sqrt(diff)

        return lam * float(jnp.sum(jnp.abs(diff)))

    def gradient(self, state: State, params: Dict[str, Any]) -> State:
        """Compute subgradient of TV regularization."""
        lam = params.get('lambda', 1.0)
        x = state[params['variable']]

        if x.ndim == 1:
            # 1D TV subgradient
            diff = x[1:] - x[:-1]
            subgrad = jnp.zeros_like(x)
            subgrad = subgrad.at[:-1].add(-lam * jnp.sign(diff))
            subgrad = subgrad.at[1:].add(lam * jnp.sign(diff))
        else:
            # Multi-D anisotropic TV subgradient
            subgrad = jnp.zeros_like(x)
            for axis in range(x.ndim):
                # Forward differences along this axis
                slices_fwd = [slice(None)] * x.ndim
                slices_fwd[axis] = slice(1, None)
                slices_bwd = [slice(None)] * x.ndim
                slices_bwd[axis] = slice(None, -1)

                diff = x[tuple(slices_fwd)] - x[tuple(slices_bwd)]
                sign_diff = jnp.sign(diff)

                # Add to subgradient
                subgrad = subgrad.at[tuple(slices_bwd)].add(-lam * sign_diff)
                subgrad = subgrad.at[tuple(slices_fwd)].add(lam * sign_diff)

        return {params['variable']: subgrad}

    def prox(self, state: State, step_size: float, params: Dict[str, Any]) -> State:
        """
        Proximal operator for TV regularization.

        For 1D TV, this uses the taut-string algorithm or iterative shrinkage.
        For simplicity, we implement a basic iterative proximal method.
        """
        lam = params.get('lambda', 1.0)
        x = state[params['variable']]

        if x.ndim == 1:
            # 1D TV prox using iterative soft-thresholding on differences
            # This is a simplified implementation - full TV prox is more complex
            tau = step_size

            # Iterative proximal for TV (simplified)
            for _ in range(5):  # Few iterations for approximation
                # Compute differences
                diff = x[1:] - x[:-1]
                # Soft-threshold differences
                thresholded_diff = jnp.sign(diff) * jnp.maximum(jnp.abs(diff) - lam * tau, 0)
                # Reconstruct signal
                x = jnp.cumsum(jnp.concatenate([x[:1], thresholded_diff]))
                # Project back to maintain mean (simplified TV prox)
                x = x - jnp.mean(x) + jnp.mean(state[params['variable']])
        else:
            # Multi-D: simplified anisotropic TV prox
            # This is a very basic approximation - real TV prox needs more sophisticated methods
            tau = step_size
            for axis in range(x.ndim):
                # Apply 1D TV prox along each axis
                for _ in range(3):  # Few iterations
                    slices = [slice(None)] * x.ndim
                    slices[axis] = slice(1, None)
                    diff = x[tuple(slices)] - x[tuple([slice(None) if i != axis else slice(None, -1) for i in range(x.ndim)])]
                    thresholded_diff = jnp.sign(diff) * jnp.maximum(jnp.abs(diff) - lam * tau, 0)

                    # Reconstruct along this axis (simplified)
                    cumsum_axis = jnp.cumsum(jnp.concatenate([x[tuple([slice(None) if i != axis else slice(1) for i in range(x.ndim)])], thresholded_diff]), axis=axis)
                    x = cumsum_axis

        return {params['variable']: x}

    def certificate_contributions(self, params: Dict[str, Any]) -> Dict[str, float]:
        """
        Certificate contributions for TV atom.

        TV regularization is nonsmooth and doesn't contribute to Lipschitz constants.
        """
        return {
            'lipschitz': 0.0,  # TV doesn't contribute to gradient Lipschitz
            'eta_dd_contribution': 0.0,  # No diagonal dominance contribution
            'gamma_contribution': 0.0   # No spectral contribution (nonsmooth)
        }