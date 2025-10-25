"""
The main step function for executing a compiled flow.
"""

from typing import Any

import jax.numpy as jnp

from computable_flows_shim.core import numerical_stability_check
from computable_flows_shim.energy.compile import CompiledEnergy
from computable_flows_shim.runtime.primitives import F_Dis, F_Multi, F_Proj


@numerical_stability_check
def run_flow_step(
    state: dict[str, jnp.ndarray],
    compiled: CompiledEnergy,
    step_alpha: float,
    manifolds: dict[str, Any] | None = None,
    W: Any | None = None,
) -> dict[str, jnp.ndarray]:
    """
    Runs one full step of a Forward-Backward Splitting flow.

    If W is provided, includes multiscale transforms:
    F_Dis → F_Multi_forward → F_Proj → F_Multi_inverse
    Otherwise, simple: F_Dis → F_Proj
    """
    if manifolds is None:
        manifolds = {}

    # Forward step (dissipative) - always in physical domain
    state_after_dis = F_Dis(state, compiled.f_grad, step_alpha, manifolds)

    if W is not None:
        # Multiscale: transform to W-space
        u = F_Multi(state_after_dis["x"], W, "forward")

        # Check if compiled energy supports W-space prox
        if (
            hasattr(compiled, "g_prox_in_W")
            and compiled.compile_report is not None
            and compiled.compile_report.get("w_space_aware", False)
        ):
            # Use W-space aware proximal operator
            u_proj_coeffs = compiled.g_prox_in_W(u, step_alpha)
            u_proj = u_proj_coeffs  # Already in coefficient form
        else:
            # Fallback: apply prox in physical space (inefficient but compatible)
            u_proj = compiled.g_prox({"x": u}, step_alpha)["x"]

        # Transform back to physical domain
        x_new = F_Multi(u_proj, W, "inverse")
        state_after_proj = {"x": x_new, "y": state["y"]}  # Keep y unchanged
    else:
        # Simple flow: projective in physical domain
        state_after_proj = F_Proj(state_after_dis, compiled.g_prox, step_alpha)

    return state_after_proj
