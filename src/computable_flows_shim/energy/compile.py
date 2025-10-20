"""
Compiles a declarative energy specification into JAX-jittable functions.
"""
from typing import Callable, Dict, Any, NamedTuple
import jax
import jax.numpy as jnp
from computable_flows_shim.energy.specs import EnergySpec

class CompiledEnergy(NamedTuple):
    f_value: Callable
    f_grad: Callable
    g_prox: Callable

def compile_energy(spec: EnergySpec, op_registry: Dict[str, Any]) -> CompiledEnergy:
    """
    Compiles an energy specification.
    """
    
    def f_value(state: Dict[str, jnp.ndarray]) -> Any:
        total_energy = 0.0
        for term in spec.terms:
            if term.type == 'quadratic':
                op = op_registry[term.op]
                x = state[term.variable]
                if term.target is not None:
                    y = state[term.target]
                    residual = op(x) - y
                    total_energy += term.weight * 0.5 * jnp.sum(residual**2)
            elif term.type == 'tikhonov':
                op = op_registry[term.op]
                x = state[term.variable]
                residual = op(x)
                total_energy += term.weight * 0.5 * jnp.sum(residual**2)
        return total_energy

    f_grad = jax.grad(f_value)

    def g_prox(state: Dict[str, jnp.ndarray], step_alpha: float) -> Dict[str, jnp.ndarray]:
        new_state = state.copy()
        for term in spec.terms:
            if term.type == 'l1':
                op = op_registry[term.op]
                x = state[term.variable]
                
                # Soft-thresholding operator for L1 norm
                threshold = step_alpha * term.weight
                transformed_x = op(x)
                thresholded_x = jnp.sign(transformed_x) * jnp.maximum(jnp.abs(transformed_x) - threshold, 0)
                
                # This assumes the op is its own inverse, like Identity or a unitary transform.
                # A full implementation would need W.inverse(thresholded_x).
                new_state[term.variable] = thresholded_x
                
        return new_state

    return CompiledEnergy(
        f_value=jax.jit(f_value),
        f_grad=jax.jit(f_grad),
        g_prox=jax.jit(g_prox)
    )
