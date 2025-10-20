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
    
    This is the main entry point for the flight controller.
    """
    state = initial_state
    energy = compiled.f_value(state)
    
    step_function = _step_function_for_testing or run_flow_step
    
    # This is a simple, non-JITted loop for now.
    for _ in range(num_iterations):
        state = step_function(state, compiled, step_alpha)
        new_energy = compiled.f_value(state)
        
        # Certificate Check 1: Lyapunov Descent
        if new_energy > energy:
            raise ValueError(f"Lyapunov descent violated: Energy increased from {energy} to {new_energy}")
            
        energy = new_energy
        
    return state
