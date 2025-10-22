"""
Compiles a declarative energy specification into JAX-jittable functions.
"""
from typing import Callable, Dict, Any, NamedTuple, Optional
from dataclasses import dataclass
import jax
import jax.numpy as jnp
from computable_flows_shim.energy.specs import EnergySpec

class CompiledEnergy(NamedTuple):
    f_value: Callable
    f_grad: Callable
    g_prox: Callable
    L_apply: Callable
    # Optional compile-time metadata
    compile_report: Optional[Dict[str, Any]] = None


@dataclass
class CompileReport:
    lens_name: str
    unit_normalization_table: Dict[str, float]
    # Track lens selections per term for W-space analysis
    term_lenses: Dict[str, str]

def compile_energy(spec: EnergySpec, op_registry: Dict[str, Any]) -> CompiledEnergy:
    """
    Compiles an energy specification.
    
    Args:
        spec: Energy specification to compile
        op_registry: Registry of operator functions
        
    Returns:
        Compiled energy functions
        
    Raises:
        ValueError: If unknown atom type is encountered
    """
    
    # Validate atom types
    known_atom_types = {'quadratic', 'tikhonov', 'l1', 'wavelet_l1'}
    for term in spec.terms:
        if term.type not in known_atom_types:
            raise ValueError(f"Unknown atom type: {term.type}. Known types: {known_atom_types}")
    
    # --- Compile the smooth part (f) ---
    
    # --- Compile the smooth part (f) ---
    def f_value(state: Dict[str, jnp.ndarray]) -> Any:
        total_energy = 0.0
        for term in spec.terms:
            if term.type == 'quadratic' or term.type == 'tikhonov':
                op = op_registry[term.op]
                x = state[term.variable]
                
                if term.target is not None:
                    y = state[term.target]
                    residual = op(x) - y
                else:
                    residual = op(x)
                    
                total_energy += term.weight * 0.5 * jnp.sum(residual**2)
        return total_energy

    f_grad = jax.grad(f_value)

    # --- Compile the non-smooth part (g) ---
    def g_prox(state: Dict[str, jnp.ndarray], step_alpha: float) -> Dict[str, jnp.ndarray]:
        new_state = state.copy()
        for term in spec.terms:
            if term.type == 'l1':
                op = op_registry[term.op]
                x = state[term.variable]
                
                threshold = step_alpha * term.weight
                transformed_x = op(x)
                thresholded_x = jnp.sign(transformed_x) * jnp.maximum(jnp.abs(transformed_x) - threshold, 0)
                
                # This assumes op is its own inverse for now.
                new_state[term.variable] = thresholded_x
            elif term.type == 'wavelet_l1':
                # For wavelet L1, we need to use TransformOp for proper analysis/synthesis
                from computable_flows_shim.atoms.library import create_atom
                
                atom = create_atom('wavelet_l1')
                # Get wavelet parameters from term
                wavelet_params = {
                    'lambda': term.weight,
                    'wavelet': term.wavelet or 'haar',
                    'levels': term.levels or 2,
                    'ndim': term.ndim or 1,
                    'variable': term.variable
                }
                new_state = atom.prox(new_state, step_alpha, wavelet_params)
                
        return new_state

    # --- Compile the linear operator (L) for FDA ---
    # For now, we assume the dominant linear operator comes from the first quadratic/tikhonov term.
    # This is a simplification and will be improved later.
    L_op = None
    for term in spec.terms:
        if term.type in ['quadratic', 'tikhonov']:
            L_op = op_registry[term.op]
            break
    
    def L_apply(v: jnp.ndarray) -> jnp.ndarray:
        if L_op is None:
            # If no linear operator is found, assume identity.
            return v
        return L_op(v)

    return CompiledEnergy(
        f_value=jax.jit(f_value),
        f_grad=jax.jit(f_grad),
        g_prox=jax.jit(g_prox),
        L_apply=L_apply,
        compile_report={
            'lens_name': 'identity',
            'unit_normalization_table': {
                term.variable: float(jnp.sqrt(term.weight) if hasattr(term, 'weight') else 1.0)
                for term in spec.terms
            },
            'term_lenses': {
                f"{term.variable}_{term.type}": 'wavelet' if term.type == 'wavelet_l1' else 'identity'
                for term in spec.terms
            }
        }
    )
