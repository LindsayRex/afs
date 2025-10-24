"""
Compiles a declarative energy specification into JAX-jittable functions.
"""
from typing import Callable, Dict, Any, NamedTuple, Optional, List
from dataclasses import dataclass
import jax
import jax.numpy as jnp
from jax import random
from computable_flows_shim.energy.specs import EnergySpec
from computable_flows_shim.core import numerical_stability_check

class CompiledEnergy(NamedTuple):
    f_value: Callable
    f_grad: Callable
    g_prox: Callable
    g_prox_in_W: Callable  # W-space proximal operator
    L_apply: Callable
    # Optional compile-time metadata
    compile_report: Optional[Dict[str, Any]] = None


@dataclass
class CompileReport:
    lens_name: str
    unit_normalization_table: Dict[str, float]
    # Track lens selections per term for W-space analysis
    term_lenses: Dict[str, str]


def _run_lens_probe_if_needed(spec: EnergySpec) -> Optional[Dict[str, Any]]:
    """
    Run lens probe in builder mode if the spec contains multiscale terms.

    Args:
        spec: Energy specification to analyze

    Returns:
        Lens probe results if multiscale terms are present, None otherwise
    """
    # Check if we have any wavelet-based terms that would benefit from lens selection
    multiscale_terms = [term for term in spec.terms if term.type == 'wavelet_l1']

    if not multiscale_terms:
        return None

    # Generate sample data for lens probe based on state shapes
    sample_data = _generate_sample_data_for_lens_probe(spec.state)

    # Get candidate transforms from the multiscale terms
    candidates = []
    for term in multiscale_terms:
        from computable_flows_shim.multi.transform_op import make_transform
        try:
            transform = make_transform(
                wavelet=term.wavelet or 'haar',
                levels=term.levels or 2,
                ndim=term.ndim or 1
            )
            candidates.append(transform)
        except Exception:
            # Skip invalid transforms
            continue

    # Add some default candidates if we don't have many
    if len(candidates) < 2:
        from computable_flows_shim.multi.transform_op import make_transform
        try:
            candidates.append(make_transform('haar', levels=3, ndim=1))
            candidates.append(make_transform('db4', levels=3, ndim=1))
        except Exception:
            pass

    if not candidates:
        return None

    # Run lens probe
    try:
        from computable_flows_shim.multi.lens_probe import run_lens_probe
        probe_results = run_lens_probe(
            data=sample_data,
            candidates=candidates,
            target_sparsity=0.8,
            selection_rule='min_reconstruction_error'
        )
        return probe_results
    except Exception:
        # Lens probe failed, return None
        return None


def _generate_sample_data_for_lens_probe(state_spec) -> jnp.ndarray:
    """
    Generate sample data for lens probe based on state specification.

    Uses a fixed seed for reproducible probe results.
    """
    key = random.PRNGKey(12345)  # Fixed seed for reproducible results

    # For now, assume we probe on the first variable
    # In practice, this could be more sophisticated
    if state_spec.shapes:
        first_var = list(state_spec.shapes.keys())[0]
        shape = state_spec.shapes[first_var]

        # Generate random data with some structure (not just noise)
        if len(shape) == 1:
            # 1D signal
            x = jnp.linspace(0, 1, shape[0])
            data = jnp.sin(2 * jnp.pi * x) + 0.5 * jnp.sin(4 * jnp.pi * x)
        elif len(shape) == 2:
            # 2D image-like data
            data = random.normal(random.PRNGKey(12346), shape) * 0.1
            # Add some structure
            x = jnp.linspace(0, 1, shape[0])
            y = jnp.linspace(0, 1, shape[1])
            X, Y = jnp.meshgrid(x, y)
            data += jnp.sin(2 * jnp.pi * X) * jnp.cos(2 * jnp.pi * Y)
        else:
            # Higher dimensional - just use random data
            data = random.normal(key, shape) * 0.1

        return data

    # Fallback: generate a default 1D signal
    x = jnp.linspace(0, 1, 256)
    return jnp.sin(2 * jnp.pi * x) + 0.5 * jnp.sin(4 * jnp.pi * x)


def _create_compile_report(spec: EnergySpec, lens_probe_results: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Create the compile report with lens probe results integrated.

    Args:
        spec: The energy specification
        lens_probe_results: Results from lens probe, or None if not run

    Returns:
        Compile report dictionary
    """
    # Determine selected lens
    selected_lens = 'identity'
    if lens_probe_results:
        selected_lens = lens_probe_results.get('selected_lens', 'identity')

    # Create term lenses mapping
    term_lenses = {}
    for term in spec.terms:
        if term.type == 'wavelet_l1':
            # Use selected lens for wavelet terms
            term_lenses[f"{term.variable}_{term.type}"] = selected_lens
        else:
            term_lenses[f"{term.variable}_{term.type}"] = 'identity'

    compile_report = {
        'lens_name': selected_lens,
        'unit_normalization_table': {
            term.variable: float(jnp.sqrt(term.weight) if hasattr(term, 'weight') else 1.0)
            for term in spec.terms
        },
        'term_lenses': term_lenses,
        'w_space_aware': any(term.type in ['wavelet_l1'] for term in spec.terms)
    }

    # Add lens probe results if available
    if lens_probe_results:
        compile_report['lens_probe'] = {
            'selected_lens': lens_probe_results['selected_lens'],
            'candidate_results': lens_probe_results['candidate_results'],
            'selection_criteria': lens_probe_results['selection_criteria'],
            'target_sparsity': lens_probe_results['target_sparsity'],
            'probe_data_shape': lens_probe_results['data_shape'],
            'probe_data_dtype': lens_probe_results['data_dtype']
        }

    return compile_report


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

    # Run lens probe for multiscale terms (builder mode)
    lens_probe_results = _run_lens_probe_if_needed(spec)

    # --- Compile the smooth part (f) ---

    # --- Compile the smooth part (f) ---
    @numerical_stability_check
    def f_value(state: Dict[str, jnp.ndarray]) -> Any:
        total_energy = 0.0
        for term in spec.terms:
            if term.type in ('quadratic', 'tikhonov'):
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
    @numerical_stability_check
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
                from computable_flows_shim.atoms import create_atom

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

    # --- Compile the non-smooth part (g) in W-space ---
    @numerical_stability_check
    def g_prox_in_W(coeffs: List[jnp.ndarray], step_alpha: float) -> List[jnp.ndarray]:
        """
        Apply proximal operators directly in W-space (wavelet coefficient space).

        Args:
            coeffs: List of wavelet coefficient arrays
            step_alpha: Step size parameter

        Returns:
            List of proximal wavelet coefficient arrays
        """
        new_coeffs = []
        coeff_idx = 0  # Track which coefficient array we're processing

        for term in spec.terms:
            if term.type == 'l1':
                # For L1 in W-space, apply soft-thresholding directly to coefficients
                # This assumes the coefficients correspond to the variable
                if coeff_idx < len(coeffs):
                    coeff_array = coeffs[coeff_idx]
                    threshold = step_alpha * term.weight
                    thresholded = jnp.sign(coeff_array) * jnp.maximum(jnp.abs(coeff_array) - threshold, 0)
                    new_coeffs.append(thresholded)
                    coeff_idx += 1
                else:
                    # If we run out of coeffs, append unchanged
                    new_coeffs.append(coeffs[coeff_idx] if coeff_idx < len(coeffs) else jnp.array([]))
                    coeff_idx += 1

            elif term.type == 'wavelet_l1':
                # For wavelet L1, apply soft-thresholding to ALL coefficient arrays
                # This is the natural W-space proximal operator
                threshold = step_alpha * term.weight
                for i, coeff_array in enumerate(coeffs):
                    if i >= len(new_coeffs):
                        thresholded = jnp.sign(coeff_array) * jnp.maximum(jnp.abs(coeff_array) - threshold, 0)
                        new_coeffs.append(thresholded)

            else:
                # For terms that don't have W-space prox, pass coefficients through unchanged
                for i, coeff_array in enumerate(coeffs):
                    if i >= len(new_coeffs):
                        new_coeffs.append(coeff_array)

        # Ensure we have the same number of coefficient arrays
        while len(new_coeffs) < len(coeffs):
            new_coeffs.append(coeffs[len(new_coeffs)])

        return new_coeffs
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
        g_prox_in_W=jax.jit(g_prox_in_W),
        L_apply=L_apply,
        compile_report=_create_compile_report(spec, lens_probe_results)
    )
