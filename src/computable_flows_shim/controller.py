"""
The Flight Controller for the Computable Flows Shim.
"""
from typing import Dict, Any, Callable, Optional
import jax
import jax.numpy as jnp
from computable_flows_shim.energy.compile import CompiledEnergy
from computable_flows_shim.runtime.step import run_flow_step

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
    state = initial_state
    energy = compiled.f_value(state)
    
    # Use the provided step function for testing, or the real one by default
    step_func = _step_function_for_testing or run_flow_step

    for _ in range(num_iterations):
        state = step_func(state, compiled, step_alpha)
        new_energy = compiled.f_value(state)
        
        # Lyapunov Descent Certificate Check
        if new_energy > energy:
            raise ValueError("Lyapunov descent violated: Energy increased.")
            
        energy = new_energy
        
    return state
