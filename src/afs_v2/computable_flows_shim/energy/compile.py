from .specs import EnergySpec, TermSpec
from computable_flows_shim.runtime.step import CompiledEnergy
from computable_flows_shim.ops import Op
from typing import Dict, List, Callable, Tuple
import jax
import jax.numpy as jnp
from .atoms import ATOM_REGISTRY

def _partition_and_compile_terms(
    terms: List[TermSpec], op_registry: Dict[str, Op]
) -> Tuple[List[Callable], List[Callable]]:
    """
    Partitions terms and compiles them using the ATOM_REGISTRY.
    Returns a list of smooth functions and a list of proximal functions.
    """
    smooth_term_fns = []
    proximal_term_fns = []

    for term in terms:
        atom = ATOM_REGISTRY.get(term.type)
        if not atom:
            raise NotImplementedError(f"Atom type '{term.type}' not found in registry.")

        op = op_registry.get(term.op)
        if not op:
            raise ValueError(f"Operator '{term.op}' not found in op_registry.")

        if atom.kind == 'smooth':
            smooth_term_fns.append(atom.compiler(term, op))
        elif atom.kind == 'nonsmooth':
            proximal_term_fns.append(atom.compiler(term, op))

    return smooth_term_fns, proximal_term_fns


def compile_energy(spec: EnergySpec, op_registry: Dict[str, Op]) -> CompiledEnergy:
    """
    Compiles an EnergySpec into a JAX-jittable CompiledEnergy object.
    """
    smooth_term_fns, proximal_term_fns = _partition_and_compile_terms(spec.terms, op_registry)

    # --- Compile f(x) and its gradient ---
    def f_value(state):
        total_energy = 0.0
        for fn in smooth_term_fns:
            total_energy += fn(state)
        return total_energy

    f_grad = jax.grad(f_value)

    # --- Compile g(x)'s proximal operator ---
    def g_prox(state, step_alpha, W):
        x = state['main'] # Simplification
        # Proximal operators are composed by applying them sequentially
        for prox_fn in reversed(proximal_term_fns):
            x = prox_fn(x, step_alpha)
        return {'main': x}

    # --- Placeholder for L_apply and W ---
    def L_apply(v):
        # Placeholder for the operator used in FDA
        return v

    from computable_flows_shim.multi.wavelets import TransformOp
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

def _compile_smooth_term_function(term: TermSpec, op: Op) -> Callable:
    """Creates a JAX function for a single smooth term."""
    if term.type == "quadratic":
        def term_fn(state):
            # E.g., weight * ||Op(x) - y||^2
            # This is a simplification, state and op mapping will be more complex
            x = state.get(term.variable) 
            if x is None: return 0.0 # Or handle error appropriately
            
            target = state.get(term.target)
            if target is None: return 0.0 # Or handle error

            residual = op(x) - target
            return term.weight * jnp.sum(residual ** 2)
        return term_fn
