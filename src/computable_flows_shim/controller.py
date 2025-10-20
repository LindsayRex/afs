"""
The Flight Controller for the Computable Flows Shim.
"""
from typing import Dict, Any, Callable, Optional
import jax
import jax.numpy as jnp
from computable_flows_shim.energy.compile import CompiledEnergy
from computable_flows_shim.runtime.step import run_flow_step
from computable_flows_shim.fda.certificates import estimate_gamma

def run_certified(
    initial_state: Dict[str, jnp.ndarray],
    compiled: CompiledEnergy,
    num_iterations: int,
    step_alpha: float,
    _step_function_for_testing: Optional[Callable] = None
) -> Dict[str, jnp.ndarray]:
    """
    Runs a full flow for a fixed number of iterations, enforcing certificates.
    """
    # --- Phase 1: Certificate Checks ---
    key = jax.random.PRNGKey(0) # A dummy key for now.
    
    # For now, we assume the variable to check is 'x' and get its shape.
    # This will need to be generalized later.
    input_shape = initial_state['x'].shape
    
    gamma = estimate_gamma(compiled.L_apply, key, input_shape)
    
    if gamma <= 0:
        raise ValueError(f"System is unstable: Spectral gap (gamma) is non-positive ({gamma:.4f}).")

    # --- Phase 2: Main Loop ---
    state = initial_state
    energy = compiled.f_value(state)
    
    step_func = _step_function_for_testing or run_flow_step

    for _ in range(num_iterations):
        state = step_func(state, compiled, step_alpha)
        new_energy = compiled.f_value(state)
        
        if new_energy > energy:
            raise ValueError("Lyapunov descent violated: Energy increased.")
            
        energy = new_energy
        
    return state
