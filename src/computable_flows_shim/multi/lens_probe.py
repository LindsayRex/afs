"""
Lens Probe: Compressibility analysis and lens selection for multiscale transforms.

Provides builder mode functionality to analyze wavelet compressibility and select
optimal transforms based on sparsity and reconstruction quality metrics.
"""

from typing import Dict, List, Any, Union, Tuple
import jax.numpy as jnp
from computable_flows_shim.multi.transform_op import TransformOp


def calculate_compressibility(
    coeffs: Union[List[jnp.ndarray], Any],
    threshold: float = 1e-8
) -> Dict[str, Any]:
    """
    Calculate compressibility metrics for wavelet coefficients.

    Args:
        coeffs: Wavelet coefficient arrays (list for 1D, nested structure for 2D)
        threshold: Threshold for considering coefficients as non-zero

    Returns:
        Dict with compressibility metrics:
        - overall_sparsity: fraction of coefficients above threshold
        - band_sparsity: list of sparsity per decomposition band
        - energy_distribution: list of energy fraction per band
    """
    # Handle both 1D (list of arrays) and 2D (nested structure) cases
    if isinstance(coeffs, list):
        # 1D case: coeffs is [approx, detail_level1, detail_level2, ...]
        band_arrays = coeffs
    else:
        # 2D case: coeffs is nested structure, flatten to list of arrays
        band_arrays = _flatten_coeff_structure(coeffs)

    # Calculate sparsity for each band
    band_sparsity = []
    band_energy = []
    total_coeffs = 0
    total_nonzero = 0

    for band_coeffs in band_arrays:
        # Flatten the band coefficients
        flat_coeffs = jnp.ravel(band_coeffs)

        # Count total coefficients and non-zero coefficients
        band_total = flat_coeffs.shape[0]
        band_nonzero = jnp.sum(jnp.abs(flat_coeffs) > threshold)

        # Calculate sparsity (fraction of non-zero coefficients)
        sparsity = float(band_nonzero / band_total) if band_total > 0 else 0.0
        band_sparsity.append(sparsity)

        # Calculate energy in this band
        energy = float(jnp.sum(flat_coeffs**2))
        band_energy.append(energy)

        total_coeffs += band_total
        total_nonzero += band_nonzero

    # Overall sparsity
    overall_sparsity = float(total_nonzero / total_coeffs) if total_coeffs > 0 else 0.0

    # Energy distribution (fraction of total energy in each band)
    total_energy = sum(band_energy)
    energy_distribution = [
        energy / total_energy if total_energy > 0 else 0.0
        for energy in band_energy
    ]

    return {
        'overall_sparsity': overall_sparsity,
        'band_sparsity': band_sparsity,
        'energy_distribution': energy_distribution,
        'total_coefficients': total_coeffs,
        'nonzero_coefficients': int(total_nonzero)
    }


def calculate_reconstruction_error(
    original: jnp.ndarray,
    reconstruction: jnp.ndarray
) -> Dict[str, float]:
    """
    Calculate reconstruction error metrics.

    Args:
        original: Original data array
        reconstruction: Reconstructed data array

    Returns:
        Dict with error metrics:
        - mse: Mean squared error
        - rmse: Root mean squared error
        - relative_error: Relative L2 error
        - max_error: Maximum absolute error
    """
    error = original - reconstruction

    mse = float(jnp.mean(error**2))
    rmse = float(jnp.sqrt(mse))
    relative_error = float(jnp.linalg.norm(error) / jnp.linalg.norm(original))
    max_error = float(jnp.max(jnp.abs(error)))

    return {
        'mse': mse,
        'rmse': rmse,
        'relative_error': relative_error,
        'max_error': max_error
    }


def run_lens_probe(
    data: jnp.ndarray,
    candidates: List[TransformOp],
    target_sparsity: float = 0.8,
    selection_rule: str = 'min_reconstruction_error'
) -> Dict[str, Any]:
    """
    Run lens probe to evaluate transform candidates and select optimal lens.

    Args:
        data: Sample data for compressibility analysis
        candidates: List of TransformOp candidates to evaluate
        target_sparsity: Target sparsity level for optimization (0-1)
        selection_rule: How to select best lens ('min_reconstruction_error', 'max_sparsity_at_target')

    Returns:
        Dict with probe results:
        - selected_lens: Name of selected transform
        - candidate_results: Dict of results per candidate
        - selection_criteria: Criteria used for selection
    """
    if not candidates:
        raise ValueError("No transform candidates provided")

    if data.size == 0:
        raise ValueError("Empty data provided for lens probe")

    candidate_results = {}

    # Evaluate each candidate
    for transform in candidates:
        try:
            # Forward transform
            coeffs = transform.forward(data)

            # Calculate compressibility metrics
            compressibility = calculate_compressibility(coeffs)

            # Calculate reconstruction error (perfect reconstruction test)
            reconstruction = transform.inverse(coeffs)
            reconstruction_error = calculate_reconstruction_error(data, reconstruction)

            # Calculate sparsity achieved at target threshold
            # Find threshold that gives approximately target_sparsity
            sparsity_at_target = _calculate_sparsity_at_target(coeffs, target_sparsity)

            candidate_results[transform.name] = {
                'compressibility': compressibility,
                'reconstruction_error': reconstruction_error,
                'sparsity_at_target': sparsity_at_target,
                'transform_levels': transform.levels,
                'frame_type': transform.frame,
                'frame_constant': transform.c
            }

        except Exception as e:
            # Log failed transform but continue with others
            candidate_results[transform.name] = {
                'error': str(e),
                'compressibility': None,
                'reconstruction_error': None,
                'sparsity_at_target': 0.0
            }

    # Select best lens based on selection rule
    selected_lens = _select_best_lens(candidate_results, selection_rule)

    return {
        'selected_lens': selected_lens,
        'candidate_results': candidate_results,
        'selection_criteria': selection_rule,
        'target_sparsity': target_sparsity,
        'data_shape': data.shape,
        'data_dtype': str(data.dtype)
    }


def _flatten_coeff_structure(coeffs: Any) -> List[jnp.ndarray]:
    """
    Flatten 2D wavelet coefficient structure into list of arrays.

    For 2D transforms, jaxwt returns nested tuples/lists.
    This flattens them into a simple list for uniform processing.
    """
    flattened = []

    def _flatten_recursive(obj):
        if isinstance(obj, (list, tuple)):
            for item in obj:
                _flatten_recursive(item)
        else:
            flattened.append(obj)

    _flatten_recursive(coeffs)
    return flattened


def _calculate_sparsity_at_target(
    coeffs: Union[List[jnp.ndarray], Any],
    target_sparsity: float
) -> float:
    """
    Calculate what sparsity level is achieved at the target threshold.

    This is a simplified version - in practice you'd solve for the threshold
    that gives exactly target_sparsity, but here we use a heuristic.
    """
    # For now, use a simple threshold based on coefficient magnitudes
    if isinstance(coeffs, list):
        all_coeffs = jnp.concatenate([jnp.ravel(c) for c in coeffs])
    else:
        flat_coeffs = _flatten_coeff_structure(coeffs)
        all_coeffs = jnp.concatenate([jnp.ravel(c) for c in flat_coeffs])

    # Sort coefficients by absolute value
    sorted_coeffs = jnp.sort(jnp.abs(all_coeffs))

    # Find threshold that would keep target_sparsity fraction of coefficients
    n_keep = int(target_sparsity * len(sorted_coeffs))
    if n_keep >= len(sorted_coeffs):
        threshold = 0.0
    else:
        threshold = float(sorted_coeffs[-(n_keep + 1)])

    # Calculate actual sparsity achieved
    actual_sparsity = float(jnp.sum(jnp.abs(all_coeffs) > threshold) / len(all_coeffs))

    return actual_sparsity


def _select_best_lens(
    candidate_results: Dict[str, Any],
    selection_rule: str
) -> str:
    """
    Select the best lens based on the given selection rule.
    """
    # Filter out failed candidates
    valid_candidates = {
        name: results for name, results in candidate_results.items()
        if results.get('reconstruction_error') is not None
    }

    if not valid_candidates:
        # If all failed, return the first one
        return list(candidate_results.keys())[0]

    if selection_rule == 'min_reconstruction_error':
        # Select lens with minimum relative reconstruction error
        return min(
            valid_candidates.keys(),
            key=lambda name: valid_candidates[name]['reconstruction_error']['relative_error']
        )

    elif selection_rule == 'max_sparsity_at_target':
        # Select lens with maximum sparsity at target level
        return max(
            valid_candidates.keys(),
            key=lambda name: valid_candidates[name]['sparsity_at_target']
        )

    # Default to min reconstruction error
    return min(
        valid_candidates.keys(),
        key=lambda name: valid_candidates[name]['reconstruction_error']['relative_error']
    )
