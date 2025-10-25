"""
Wavelet L1 Atom Implementation.

This module implements the wavelet L1 regularization atom: λ‖Wx‖₁
"""

from typing import Any

import jax.numpy as jnp

from computable_flows_shim.core import numerical_stability_check

from ..base import Atom

# Type aliases
Array = jnp.ndarray
State = dict[str, Array]


class WaveletL1Atom(Atom):
    """
    Wavelet L1 regularization atom: λ‖Wx‖₁

    This implements L1 regularization in wavelet space for sparse recovery
    with multiscale representations. Uses TransformOp for frame-aware transforms.
    """

    @property
    def name(self) -> str:
        return "wavelet_l1"

    @property
    def form(self) -> str:
        return r"\lambda\|Wx\|_1"

    @numerical_stability_check
    def energy(self, state: State, params: dict[str, Any]) -> float:
        """Compute wavelet L1 energy: λ‖Wx‖₁"""
        from computable_flows_shim.multi.transform_op import make_transform

        lam = params.get("lambda", 1.0)
        transform = make_transform(
            params.get("wavelet", "haar"),
            params.get("levels", 2),
            params.get("ndim", 1),
        )

        x = state[params["variable"]]
        coeffs = transform.forward(x)

        # Sum L1 norm over all coefficient arrays
        total_l1 = 0.0
        for coeff in coeffs:
            total_l1 += float(jnp.sum(jnp.abs(coeff)))

        return lam * total_l1

    @numerical_stability_check
    def gradient(self, state: State, params: dict[str, Any]) -> State:
        """Compute subgradient of wavelet L1 regularization."""
        from computable_flows_shim.multi.transform_op import make_transform

        lam = params.get("lambda", 1.0)
        transform = make_transform(
            params.get("wavelet", "haar"),
            params.get("levels", 2),
            params.get("ndim", 1),
        )

        x = state[params["variable"]]
        coeffs = transform.forward(x)

        # Subgradient in wavelet space: λ * sign(Wx)
        subgrad_coeffs = [lam * jnp.sign(coeff) for coeff in coeffs]

        # Transform back to original space
        subgrad_x = transform.inverse(subgrad_coeffs)

        return {params["variable"]: subgrad_x}

    @numerical_stability_check
    def prox(self, state: State, step_size: float, params: dict[str, Any]) -> State:
        """
        Proximal operator for wavelet L1 regularization.

        Solves: argmin_x λ‖Wx‖₁ + (1/(2τ))‖x - x₀‖²
        Solution: x = W^T prox_λτ( W x₀ )
        """
        from computable_flows_shim.multi.transform_op import make_transform

        lam = params.get("lambda", 1.0)
        transform = make_transform(
            params.get("wavelet", "haar"),
            params.get("levels", 2),
            params.get("ndim", 1),
        )

        x = state[params["variable"]]

        # Analysis: transform to wavelet space
        coeffs = transform.forward(x)

        # Soft-thresholding in wavelet space
        threshold = lam * step_size
        thresholded_coeffs = [
            jnp.sign(coeff) * jnp.maximum(jnp.abs(coeff) - threshold, 0)
            for coeff in coeffs
        ]

        # Synthesis: transform back to original space
        x_new = transform.inverse(thresholded_coeffs)

        return {params["variable"]: x_new}

    def certificate_contributions(self, params: dict[str, Any]) -> dict[str, float]:
        """
        Certificate contributions for wavelet L1 atom.

        Wavelet transforms are frame operators with frame bounds.
        """
        from computable_flows_shim.multi.transform_op import make_transform

        transform = make_transform(
            params.get("wavelet", "haar"),
            params.get("levels", 2),
            params.get("ndim", 1),
        )

        # Frame constant affects conditioning
        frame_constant = transform.c

        return {
            "lipschitz": 0.0,  # L1 doesn't contribute to gradient Lipschitz
            "eta_dd_contribution": 0.0,  # No diagonal dominance contribution
            "gamma_contribution": 0.0,  # No spectral contribution (nonsmooth)
            "frame_constant": frame_constant,  # For W-space analysis
        }
