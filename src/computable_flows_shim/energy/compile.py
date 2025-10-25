"""
Compiles a declarative energy specification into JAX-jittable functions.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import random

from computable_flows_shim.core import numerical_stability_check
from computable_flows_shim.energy.specs import EnergySpec


class CompiledEnergy(NamedTuple):
    f_value: Callable
    f_grad: Callable
    g_prox: Callable
    g_prox_in_w: Callable  # W-space proximal operator
    L_apply: Callable
    # Optional compile-time metadata
    compile_report: dict[str, Any] | None = None
    invariants_spec: dict | None = None  # FDA invariants specification


@dataclass
class CompileReport:
    lens_name: str
    unit_normalization_table: dict[str, float]
    # Track lens selections per term for W-space analysis
    term_lenses: dict[str, str]


def _compute_unit_normalization(
    spec: EnergySpec, op_registry: dict[str, Any]
) -> dict[str, float]:
    """
    Compute per-term unit normalization factors using energy evaluation.

    This replaces statistical normalization with energy-based normalization to maintain
    the pure flow-based paradigm. Each term's energy contribution on sample data
    provides the normalization factor.

    Args:
        spec: Energy specification
        op_registry: Registry of operator functions

    Returns:
        Dictionary mapping term identifiers to normalization factors
    """
    normalization_factors = {}

    # Generate sample data for each state variable
    sample_state = {}
    for var_name, shape in spec.state.shapes.items():
        # Use normal distribution with unit variance for sample data
        key = random.PRNGKey(42)  # Fixed seed for reproducibility
        # Default to float64 for normalization computation
        sample_state[var_name] = random.normal(key, shape=shape, dtype=jnp.float64)

    # For each term, evaluate its energy contribution directly
    for i, term in enumerate(spec.terms):
        term_key = f"{term.variable}_{term.type}_{i}"

        # Evaluate term energy contribution using the energy functional approach
        try:
            if term.type in ("quadratic", "tikhonov"):
                op = op_registry[term.op]
                x = sample_state[term.variable]

                if term.target is not None and term.target in sample_state:
                    y = sample_state[term.target]
                    residual = op(x) - y
                else:
                    residual = op(x)

                # Energy contribution: 0.5 * ||residual||^2
                # Use this as the normalization factor (energy scale)
                energy_contribution = 0.5 * jnp.sum(residual**2)
                normalization_factors[term_key] = float(
                    jnp.maximum(energy_contribution, 1e-8)
                )

            elif term.type == "l1":
                op = op_registry[term.op]
                x = sample_state[term.variable]
                transformed_x = op(x)

                # For L1 terms, use the L1 norm of the transformed variable
                # This keeps it in the energy/functional domain
                l1_contribution = jnp.sum(jnp.abs(transformed_x))
                normalization_factors[term_key] = float(
                    jnp.maximum(l1_contribution, 1e-8)
                )

            elif term.type == "wavelet_l1":
                # For wavelet L1 terms, compute normalization based on wavelet transform energy scale
                # This provides proper energy-based normalization instead of fallback 1.0
                try:
                    from computable_flows_shim.multi.transform_op import make_transform

                    # Use the wavelet specified in the term, or default to 'haar'
                    wavelet_name = term.wavelet or "haar"
                    levels = term.levels or 2
                    ndim = term.ndim or 1

                    # Create the wavelet transform
                    transform = make_transform(wavelet_name, levels=levels, ndim=ndim)

                    # Apply wavelet transform to sample data
                    coeffs = transform.forward(sample_state[term.variable])

                    # Compute L1 norm of all wavelet coefficients
                    # This gives the energy scale of the wavelet transform
                    if isinstance(coeffs, list):
                        # 1D case: list of coefficient arrays
                        total_l1 = sum(
                            float(jnp.sum(jnp.abs(coeff_array)))
                            for coeff_array in coeffs
                        )
                    else:
                        # 2D case: nested structure, flatten and sum
                        flat_coeffs = []

                        def _flatten(obj, flat_coeffs):
                            if isinstance(obj, (list, tuple)):
                                for item in obj:
                                    _flatten(item, flat_coeffs)
                            else:
                                flat_coeffs.append(obj)

                        _flatten(coeffs, flat_coeffs)
                        total_l1 = sum(
                            float(jnp.sum(jnp.abs(coeff_array)))
                            for coeff_array in flat_coeffs
                        )

                    # Use L1 norm as normalization factor, with minimum to avoid division by very small numbers
                    normalization_factors[term_key] = float(jnp.maximum(total_l1, 1e-8))

                except (ValueError, RuntimeError, TypeError) as e:
                    # If wavelet transform fails, fall back to a reasonable default based on signal size
                    signal_size = jnp.prod(jnp.array(sample_state[term.variable].shape))
                    fallback_scale = float(
                        jnp.sqrt(signal_size)
                    )  # Rough estimate based on signal magnitude
                    normalization_factors[term_key] = fallback_scale
                    print(
                        f"Warning: Wavelet normalization failed for {term_key}, using fallback: {e}"
                    )

            else:
                # Fallback for unknown term types
                normalization_factors[term_key] = 1.0

        except (ValueError, RuntimeError, KeyError) as e:
            # If evaluation fails, use fallback normalization
            print(f"Warning: Could not compute normalization for term {term_key}: {e}")
            normalization_factors[term_key] = 1.0

    return normalization_factors


def _run_lens_probe_if_needed(spec: EnergySpec) -> dict[str, Any] | None:
    """
    Run lens probe in builder mode if the spec contains multiscale terms.

    Args:
        spec: Energy specification to analyze

    Returns:
        Lens probe results if multiscale terms are present, None otherwise
    """
    # Check if we have any wavelet-based terms that would benefit from lens selection
    multiscale_terms = [term for term in spec.terms if term.type == "wavelet_l1"]

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
                wavelet=term.wavelet or "haar",
                levels=term.levels or 2,
                ndim=term.ndim or 1,
            )
            candidates.append(transform)
        except (ValueError, ImportError, AttributeError):
            # Skip invalid transforms
            continue

    # Add some default candidates if we don't have many
    if len(candidates) < 2:
        from computable_flows_shim.multi.transform_op import make_transform

        try:
            candidates.append(make_transform("haar", levels=3, ndim=1))
            candidates.append(make_transform("db4", levels=3, ndim=1))
        except (ValueError, ImportError, AttributeError):
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
            selection_rule="min_reconstruction_error",
        )
        return probe_results
    except (ValueError, RuntimeError, ImportError):
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
        first_var = next(iter(state_spec.shapes.keys()))
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


def _create_compile_report(
    spec: EnergySpec,
    lens_probe_results: dict[str, Any] | None,
    unit_normalization_table: dict[str, float],
) -> dict[str, Any]:
    """
    Create the compile report with lens probe results integrated.

    Args:
        spec: The energy specification
        lens_probe_results: Results from lens probe, or None if not run

    Returns:
        Compile report dictionary
    """
    # Determine selected lens
    selected_lens = "identity"
    if lens_probe_results:
        selected_lens = lens_probe_results.get("selected_lens", "identity")

    # Create term lenses mapping
    term_lenses = {}
    for term in spec.terms:
        if term.type == "wavelet_l1":
            # Use selected lens for wavelet terms
            term_lenses[f"{term.variable}_{term.type}"] = selected_lens
        else:
            term_lenses[f"{term.variable}_{term.type}"] = "identity"

    compile_report = {
        "lens_name": selected_lens,
        "unit_normalization_table": unit_normalization_table,
        "term_lenses": term_lenses,
        "w_space_aware": any(term.type in ["wavelet_l1"] for term in spec.terms),
    }

    # Add lens probe results if available
    if lens_probe_results:
        compile_report["lens_probe"] = {
            "selected_lens": lens_probe_results["selected_lens"],
            "candidate_results": lens_probe_results["candidate_results"],
            "selection_criteria": lens_probe_results["selection_criteria"],
            "target_sparsity": lens_probe_results["target_sparsity"],
            "probe_data_shape": lens_probe_results["data_shape"],
            "probe_data_dtype": lens_probe_results["data_dtype"],
        }

    return compile_report


def compile_energy(spec: EnergySpec, op_registry: dict[str, Any]) -> CompiledEnergy:
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

    # Run lens probe for multiscale terms (builder mode)
    lens_probe_results = _run_lens_probe_if_needed(spec)

    # Compute proper unit normalization factors
    unit_normalization_table = _compute_unit_normalization(spec, op_registry)

    # --- Compile the smooth part (f) ---

    # --- Compile the smooth part (f) ---
    @numerical_stability_check
    def f_value(state: dict[str, jnp.ndarray]) -> Any:
        total_energy = 0.0
        for term in spec.terms:
            if term.type in ("quadratic", "tikhonov"):
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
    def g_prox(
        state: dict[str, jnp.ndarray], step_alpha: float
    ) -> dict[str, jnp.ndarray]:
        new_state = state.copy()
        for term in spec.terms:
            if term.type == "l1":
                op = op_registry[term.op]
                x = state[term.variable]

                threshold = step_alpha * term.weight
                transformed_x = op(x)
                thresholded_x = jnp.sign(transformed_x) * jnp.maximum(
                    jnp.abs(transformed_x) - threshold, 0
                )

                # This assumes op is its own inverse for now.
                new_state[term.variable] = thresholded_x
            elif term.type == "wavelet_l1":
                # For wavelet L1, we need to use TransformOp for proper analysis/synthesis
                from computable_flows_shim.atoms import create_atom

                atom = create_atom("wavelet_l1")
                # Get wavelet parameters from term
                wavelet_params = {
                    "lambda": term.weight,
                    "wavelet": term.wavelet or "haar",
                    "levels": term.levels or 2,
                    "ndim": term.ndim or 1,
                    "variable": term.variable,
                }
                new_state = atom.prox(new_state, step_alpha, wavelet_params)

        return new_state

    # --- Compile the non-smooth part (g) in W-space ---
    @numerical_stability_check
    def g_prox_in_w(coeffs: list[jnp.ndarray], step_alpha: float) -> list[jnp.ndarray]:
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
            if term.type == "l1":
                # For L1 in W-space, apply soft-thresholding directly to coefficients
                # This assumes the coefficients correspond to the variable
                if coeff_idx < len(coeffs):
                    coeff_array = coeffs[coeff_idx]
                    threshold = step_alpha * term.weight
                    thresholded = jnp.sign(coeff_array) * jnp.maximum(
                        jnp.abs(coeff_array) - threshold, 0
                    )
                    new_coeffs.append(thresholded)
                    coeff_idx += 1
                else:
                    # If we run out of coeffs, append unchanged
                    new_coeffs.append(
                        coeffs[coeff_idx] if coeff_idx < len(coeffs) else jnp.array([])
                    )
                    coeff_idx += 1

            elif term.type == "wavelet_l1":
                # For wavelet L1, apply soft-thresholding to ALL coefficient arrays
                # This is the natural W-space proximal operator
                threshold = step_alpha * term.weight
                for i, coeff_array in enumerate(coeffs):
                    if i >= len(new_coeffs):
                        thresholded = jnp.sign(coeff_array) * jnp.maximum(
                            jnp.abs(coeff_array) - threshold, 0
                        )
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
        if term.type in ["quadratic", "tikhonov"]:
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
        g_prox_in_w=jax.jit(g_prox_in_w),
        L_apply=L_apply,
        compile_report=_create_compile_report(
            spec, lens_probe_results, unit_normalization_table
        ),
        invariants_spec=getattr(spec.state, "invariants", None),
    )
