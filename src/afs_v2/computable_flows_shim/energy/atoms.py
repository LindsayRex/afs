from dataclasses import dataclass
from typing import Callable, Dict, Any

@dataclass(frozen=True)
class Atom:
    """Represents a mathematical object from the atoms library."""
    kind: str  # 'smooth' or 'nonsmooth'
    compiler: Callable
    # We can add more fields here later, like certificate hooks

# Global registry for all available atoms
ATOM_REGISTRY: Dict[str, Atom] = {}

def register_atom(name: str, kind: str):
    """A decorator to register a new atom compiler."""
    def decorator(compiler_fn: Callable):
        ATOM_REGISTRY[name] = Atom(kind=kind, compiler=compiler_fn)
        return compiler_fn
    return decorator

# --- Compiler Implementations for Core Atoms ---

@register_atom("quadratic", "smooth")
def compile_quadratic(term, op):
    """Compiler for the quadratic term: weight * ||Op(x) - y||^2"""
    import jax.numpy as jnp
    def term_fn(state):
        x = state.get(term.variable)
        if x is None: return 0.0
        
        target = state.get(term.target)
        if target is None: return 0.0

        residual = op(x) - target
        return term.weight * jnp.sum(residual ** 2)
    return term_fn

@register_atom("l1", "nonsmooth")
def compile_l1(term, op):
    """Compiler for the L1 proximal operator: soft_thresholding"""
    import jax.numpy as jnp
    def prox_fn(x, alpha):
        # soft thresholding: sign(x) * max(|x| - alpha, 0)
        return jnp.sign(x) * jnp.maximum(jnp.abs(x) - alpha * term.weight, 0)
    return prox_fn
