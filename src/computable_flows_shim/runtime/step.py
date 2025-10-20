"""
The main step function for executing a compiled flow.
"""
from typing import Dict, Any
import jax.numpy as jnp
from computable_flows_shim.energy.compile import CompiledEnergy
from computable_flows_shim.runtime.primitives import F_Dis, F_Proj

def run_flow_step(
    state: Dict[str, jnp.ndarray],
    compiled: CompiledEnergy,
    step_alpha: float,
    manifolds: Dict[str, Any] = {}
) -> Dict[str, jnp.ndarray]:
    """
    Runs one full step of a Forward-Backward Splitting flow.
    
    This corresponds to: z_{k+1} = F_Proj(F_Dis(z_k))
    """
    if manifolds is None:
        manifolds = {}

    # Forward step (dissipative)
    state_after_dis = F_Dis(state, compiled.f_grad, step_alpha, manifolds)
    
    # Backward step (projective/proximal)
    state_after_proj = F_Proj(state_after_dis, compiled.g_prox, step_alpha)
    
    return state_after_proj
