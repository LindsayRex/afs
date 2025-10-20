from .specs import EnergySpec, TermSpec
from computable_flows_shim.runtime.step import CompiledEnergy
from computable_flows_shim.ops import Op
from typing import Dict, List, Callable
import jax
import jax.numpy as jnp

# A simple mapping to determine if a term is smooth or requires a proximal operator.
# This would be more sophisticated in a real implementation.
SMOOTH_TERMS = {"quadratic"}
PROXIMAL_TERMS = {"l1"}

def _partition_terms(terms: List[TermSpec]) -> (List[TermSpec], List[TermSpec]):
    """Partitions terms into smooth and proximal lists."""
    smooth_terms = [t for t in terms if t.type in SMOOTH_TERMS]
    proximal_terms = [t for t in terms if t.type in PROXIMAL_TERMS]
    return smooth_terms, proximal_terms

def _compile_smooth_term_function(term: TermSpec, op: Op) -> Callable:
    """Creates a JAX function for a single smooth term."""
    if term.type == "quadratic":
        def term_fn(state):
            # E.g., weight * ||Op(x) - y||^2
            x = state[term.op] # This is a simplification, state mapping will be more complex
            y = state[term.target]
            residual = op(x) - y
            return term.weight * jnp.sum(residual ** 2)
        return term_fn
    else:
        raise NotImplementedError(f"Smooth term type '{term.type}' not implemented.")

def _compile_proximal_term_function(term: TermSpec, op: Op) -> Callable:
    """Creates a JAX proximal operator for a single proximal term."""
    if term.type == "l1":
        def prox_fn(x, alpha):
            # soft thresholding: sign(x) * max(|x| - alpha, 0)
            return jnp.sign(x) * jnp.maximum(jnp.abs(x) - alpha * term.weight, 0)
        return prox_fn
    else:
        raise NotImplementedError(f"Proximal term type '{term.type}' not implemented.")


def compile_energy(spec: EnergySpec, op_registry: Dict[str, Op]) -> CompiledEnergy:
    """
    Compiles an EnergySpec into a JAX-jittable CompiledEnergy object.
    """
    smooth_terms, proximal_terms = _partition_terms(spec.terms)

    # --- Compile f(x) and its gradient ---
    smooth_term_fns = [
        _compile_smooth_term_function(t, op_registry[t.op]) for t in smooth_terms
    ]

    def f_value(state):
        total_energy = 0.0
        for fn in smooth_term_fns:
            total_energy += fn(state)
        return total_energy

    f_grad = jax.grad(f_value)


    # --- Compile g(x)'s proximal operator ---
    proximal_term_fns = [
        _compile_proximal_term_function(t, op_registry[t.op]) for t in proximal_terms
    ]
    
    def g_prox(state, step_alpha, W):
        # This is a simplified prox application. A real implementation would
        # handle compositions and transforms correctly.
        x = state['main'] # Simplification
        for prox_fn in proximal_term_fns:
            x = prox_fn(x, step_alpha)
        return {'main': x}


    # --- Placeholder for L_apply and W ---
    def L_apply(v):
        # Placeholder for the operator used in FDA
        return v

    from computable_flows_shim.multi.wavelets import TransformOp
    # A real implementation would select this from the spec.transforms
    W = TransformOp(
        name="placeholder",
        forward=lambda x: x,
        inverse=lambda x: x
    )

    return CompiledEnergy(
        f_value=f_value,
        f_grad=f_grad,
        g_prox=g_prox,
        W=W,
        L_apply=L_apply
    )
