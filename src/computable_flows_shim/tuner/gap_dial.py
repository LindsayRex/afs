"""
Gap Dial Auto-Tuner for in-run parameter optimization.

The Gap Dial monitors the spectral gap of the Hessian operator during flow execution
and adaptively tunes regularization parameters to maintain numerical stability and
optimal convergence properties.
"""

from typing import Dict, Any, Optional, Callable, Tuple
import jax
import jax.numpy as jnp
from dataclasses import dataclass
from ..energy.compile import CompiledEnergy


@dataclass
class GapDialConfig:
    """Configuration for Gap Dial tuner."""
    target_gap: float = 0.1  # Target minimum spectral gap
    gap_tolerance: float = 0.05  # Tolerance for gap maintenance
    lambda_min: float = 1e-6  # Minimum regularization strength
    lambda_max: float = 1e3  # Maximum regularization strength
    adaptation_rate: float = 0.1  # How aggressively to adapt parameters
    monitoring_interval: int = 5  # Check gap every N iterations
    warmup_iterations: int = 10  # Initial iterations before tuning starts


class GapDialTuner:
    """
    In-run parameter optimizer that maintains spectral gap through adaptive regularization.

    The Gap Dial continuously monitors the spectral properties of the energy functional's
    Hessian and adjusts regularization parameters to ensure numerical stability and
    convergence guarantees.
    """

    def __init__(self, config: Optional[GapDialConfig] = None):
        self.config = config or GapDialConfig()
        self.current_lambda = 1.0  # Default regularization strength
        self.gap_history = []
        self.lambda_history = []
        self.iteration_count = 0
        self.last_gap_check = 0

    def should_check_gap(self, iteration: int) -> bool:
        """Determine if gap should be checked at this iteration."""
        if iteration < self.config.warmup_iterations:
            return False
        return (iteration - self.last_gap_check) >= self.config.monitoring_interval

    def estimate_spectral_gap(self, compiled: CompiledEnergy, state: Dict[str, jnp.ndarray]) -> float:
        """
        Estimate the spectral gap of the Hessian at current state.

        Returns the minimum eigenvalue (spectral gap) of the Hessian operator.
        """
        # Use the compiled energy's L_apply function for gap estimation
        key = jax.random.PRNGKey(42 + self.iteration_count)  # Vary key per iteration
        input_shape = state['x'].shape

        # For efficiency, use a lightweight gap estimation
        # This could be enhanced with more sophisticated spectral methods
        dim = input_shape[0]

        # Construct small Hessian matrix for gap estimation
        # In practice, this would use matrix-free methods for large problems
        if dim <= 50:  # Small problem: compute full Hessian
            L_matrix = jax.vmap(compiled.L_apply)(jnp.eye(dim)).T
            if jnp.allclose(L_matrix, L_matrix.T, atol=1e-8):  # Symmetric
                eigvals = jnp.linalg.eigh(L_matrix)[0]
                return float(jnp.min(jnp.real(eigvals)))
            else:
                # Gershgorin bound for non-symmetric case
                diag = jnp.diag(L_matrix)
                off_diag_sum = jnp.array([
                    jnp.sum(jnp.abs(L_matrix[i, :])) - jnp.abs(L_matrix[i, i])
                    for i in range(dim)
                ])
                gershgorin_bounds = diag - off_diag_sum
                return float(jnp.min(gershgorin_bounds))
        else:
            # Large problem: use stochastic estimation or power method
            # For now, use a simplified approach
            return self._estimate_gap_stochastic(compiled.L_apply, input_shape, key)

    def _estimate_gap_stochastic(self, L_apply: Callable, input_shape: Tuple, key: jnp.ndarray) -> float:
        """Stochastic spectral gap estimation for large problems."""
        dim = input_shape[0]

        # Simple power method for minimum eigenvalue estimation
        # This is a simplified version - production would use more robust methods
        v = jax.random.normal(key, (dim,))
        v = v / jnp.linalg.norm(v)

        # Run a few power iterations
        for _ in range(10):
            Lv = L_apply(v)
            v = Lv / jnp.linalg.norm(Lv)

        # Rayleigh quotient estimate
        Lv = L_apply(v)
        rayleigh = jnp.dot(v, Lv) / jnp.dot(v, v)

        return float(rayleigh)

    def adapt_parameters(self, current_gap: float, compiled: CompiledEnergy) -> Dict[str, Any]:
        """
        Adapt regularization parameters based on current spectral gap.

        Returns a dictionary of parameter updates to apply to the energy functional.
        """
        self.gap_history.append(current_gap)
        self.lambda_history.append(self.current_lambda)

        gap_error = self.config.target_gap - current_gap

        # Adaptive regularization tuning
        if abs(gap_error) > self.config.gap_tolerance:
            # Gap too small: increase regularization
            if current_gap < self.config.target_gap:
                lambda_factor = 1.0 + self.config.adaptation_rate * abs(gap_error)
            # Gap too large: decrease regularization (less common)
            else:
                lambda_factor = 1.0 - self.config.adaptation_rate * abs(gap_error) * 0.1

            # Update lambda with bounds
            new_lambda = self.current_lambda * lambda_factor
            new_lambda = jnp.clip(new_lambda, self.config.lambda_min, self.config.lambda_max)
            self.current_lambda = float(new_lambda)

        # In a full implementation, this would modify the compiled energy functional
        # For now, return parameter suggestions
        return {
            'lambda_regularization': self.current_lambda,
            'current_gap': current_gap,
            'target_gap': self.config.target_gap,
            'gap_error': gap_error,
            'adaptation_applied': abs(gap_error) > self.config.gap_tolerance
        }

    def get_tuning_status(self) -> Dict[str, Any]:
        """Get current tuning status and history."""
        return {
            'current_lambda': self.current_lambda,
            'gap_history': self.gap_history[-10:],  # Last 10 measurements
            'lambda_history': self.lambda_history[-10:],
            'iteration_count': self.iteration_count,
            'config': {
                'target_gap': self.config.target_gap,
                'adaptation_rate': self.config.adaptation_rate,
                'monitoring_interval': self.config.monitoring_interval
            }
        }

    def reset(self):
        """Reset tuner state."""
        self.current_lambda = 1.0
        self.gap_history.clear()
        self.lambda_history.clear()
        self.iteration_count = 0
        self.last_gap_check = 0


def create_gap_dial_tuner(target_gap: float = 0.1, adaptation_rate: float = 0.1) -> GapDialTuner:
    """
    Factory function to create a Gap Dial tuner with common settings.

    Args:
        target_gap: Target minimum spectral gap for stability
        adaptation_rate: How aggressively to adapt parameters (0.1 = 10% change per adaptation)

    Returns:
        Configured GapDialTuner instance
    """
    config = GapDialConfig(
        target_gap=target_gap,
        adaptation_rate=adaptation_rate
    )
    return GapDialTuner(config)