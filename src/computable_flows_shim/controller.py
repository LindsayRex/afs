"""
The Flight Controller for the Computable Flows Shim.

Implements the RED/AMBER/GREEN phase machine for certificate-gated parameter tuning
with rollback capability and budget enforcement.
"""
from typing import Dict, Any, Callable, Optional, NamedTuple, Tuple
import jax
import jax.numpy as jnp
import time
from enum import Enum
from dataclasses import dataclass
from computable_flows_shim.energy.compile import CompiledEnergy
from computable_flows_shim.runtime.step import run_flow_step
from computable_flows_shim.fda.certificates import estimate_gamma_lanczos, estimate_eta_dd
from computable_flows_shim.runtime.checkpoint import CheckpointManager
from .telemetry import TelemetryManager
from .tuner.gap_dial import GapDialTuner
from .logging import get_logger, log_performance


class Phase(Enum):
    """Controller operational phases."""
    RED = "RED"        # Spec/units/ops invalid
    AMBER = "AMBER"    # Spec OK, not certified
    GREEN = "GREEN"    # Certified, tuner allowed


@dataclass
class ControllerConfig:
    """Configuration for the Flight Controller."""
    # Certificate thresholds
    eta_max: float = 0.9
    gamma_min: float = 1e-6

    # Remediation settings
    max_remediation_attempts: int = 3
    alpha_reduction_factor: float = 0.5

    # Budget limits
    max_wall_time_ms: Optional[float] = None
    max_iterations: Optional[int] = None
    max_tuner_moves: int = 10

    # Rollback settings
    max_rollbacks: int = 3
    rollback_on_cert_failure: bool = True

    # Step settings
    max_step_attempts: int = 3
    step_alpha_reduction_factor: float = 0.5


@dataclass
class Checkpoint:
    """Controller checkpoint for rollback capability."""
    iteration: int
    state: Dict[str, jnp.ndarray]
    energy: float
    alpha: float
    certificates: Dict[str, float]
    tuner_state: Optional[Dict[str, Any]] = None
    timestamp: Optional[float] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class FlightController:
    """
    Flight Controller implementing RED/AMBER/GREEN phase machine.

    Manages certificate-gated parameter tuning with rollback capability
    and budget enforcement for safe, automated optimization.
    """

    def __init__(self, config: Optional[ControllerConfig] = None):
        self.config = config or ControllerConfig()
        self.logger = get_logger(__name__)
        self.phase = Phase.RED
        self.checkpoints: list[Checkpoint] = []
        self.rollback_count = 0
        self.tuner_move_count = 0
        self.start_time = time.time()
        
        self.logger.debug("FlightController initialized", extra={
            'config': {
                'eta_max': self.config.eta_max,
                'gamma_min': self.config.gamma_min,
                'max_iterations': self.config.max_iterations,
                'max_wall_time_ms': self.config.max_wall_time_ms
            }
        })

    def get_status(self) -> Dict[str, Any]:
        """Get current controller status."""
        return {
            'phase': self.phase.value,
            'rollback_count': self.rollback_count,
            'tuner_move_count': self.tuner_move_count,
            'wall_time_ms': (time.time() - self.start_time) * 1000,
            'checkpoints_available': len(self.checkpoints)
        }

    def check_budget_limits(self, iteration: int) -> bool:
        """Check if any budget limits have been exceeded."""
        if self.config.max_iterations and iteration >= self.config.max_iterations:
            return False

        if self.config.max_wall_time_ms:
            elapsed = (time.time() - self.start_time) * 1000
            if elapsed >= self.config.max_wall_time_ms:
                return False

        if self.tuner_move_count >= self.config.max_tuner_moves:
            return False

        return True

    def assess_certificates(self, compiled: CompiledEnergy, input_shape: tuple,
                          key: jnp.ndarray) -> Tuple[float, float, bool]:
        """
        Assess certificate feasibility.

        Returns:
            eta_dd, gamma, is_feasible
        """
        self.logger.debug("Assessing certificates", extra={
            'input_shape': input_shape
        })
        
        eta = estimate_eta_dd(compiled.L_apply, input_shape)
        gamma = estimate_gamma_lanczos(compiled.L_apply, key, input_shape)

        # Convert to scalars for boolean logic
        eta_scalar = float(eta)
        gamma_scalar = float(gamma)
        is_feasible = (eta_scalar <= self.config.eta_max and gamma_scalar >= self.config.gamma_min)
        
        self.logger.debug("Certificate assessment completed", extra={
            'eta_dd': eta_scalar,
            'gamma': gamma_scalar,
            'is_feasible': is_feasible,
            'eta_max': self.config.eta_max,
            'gamma_min': self.config.gamma_min
        })
        
        return eta_scalar, gamma_scalar, is_feasible

    def create_checkpoint(self, iteration: int, state: Dict[str, jnp.ndarray],
                         energy: float, alpha: float, certificates: Dict[str, float],
                         tuner_state: Optional[Dict[str, Any]] = None) -> Checkpoint:
        """Create a checkpoint for potential rollback."""
        checkpoint = Checkpoint(
            iteration=iteration,
            state=state.copy(),
            energy=energy,
            alpha=alpha,
            certificates=certificates.copy(),
            tuner_state=tuner_state.copy() if tuner_state else None
        )
        self.checkpoints.append(checkpoint)

        # Keep only last 5 checkpoints to prevent memory bloat
        if len(self.checkpoints) > 5:
            self.checkpoints.pop(0)

        return checkpoint

    def rollback_to_checkpoint(self, target_checkpoint: Checkpoint,
                             telemetry_manager: Optional[TelemetryManager] = None,
                             run_id: str = "") -> Tuple[Dict[str, jnp.ndarray], float, Optional[Dict[str, Any]]]:
        """
        Rollback to a previous checkpoint.

        Returns:
            state, alpha, tuner_state
        """
        if self.rollback_count >= self.config.max_rollbacks:
            raise ValueError(f"Maximum rollbacks ({self.config.max_rollbacks}) exceeded")

        self.rollback_count += 1

        if telemetry_manager:
            telemetry_manager.flight_recorder.log_event(
                run_id=run_id,
                event="ROLLBACK",
                payload={
                    'rollback_count': self.rollback_count,
                    'target_iteration': target_checkpoint.iteration,
                    'reason': 'certificate_regression'
                }
            )

        return target_checkpoint.state, target_checkpoint.alpha, target_checkpoint.tuner_state

    def transition_phase(self, new_phase: Phase, telemetry_manager: Optional[TelemetryManager] = None,
                        run_id: str = "", **kwargs):
        """Transition to a new phase with telemetry logging."""
        old_phase = self.phase
        self.phase = new_phase

        if telemetry_manager:
            telemetry_manager.flight_recorder.log_event(
                run_id=run_id,
                event="PHASE_TRANSITION",
                payload={
                    'from': old_phase.value,
                    'to': new_phase.value,
                    **kwargs
                }
            )

    def run_certified_flow(
        self,
        initial_state: Dict[str, jnp.ndarray],
        compiled: CompiledEnergy,
        num_iterations: int,
        initial_alpha: float,
        telemetry_manager: Optional[TelemetryManager] = None,
        flow_name: str = "",
        run_id: str = "",
        gap_dial_tuner: Optional[GapDialTuner] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
        _step_function_for_testing: Optional[Callable] = None
    ) -> Dict[str, jnp.ndarray]:
        """
        Run the certified flow with full phase machine control.

        Implements RED/AMBER/GREEN phases with certificate gating,
        rollback capability, and budget enforcement.
        """
        start_time_perf = time.perf_counter()
        try:
            self.logger.info("Starting certified flow execution", extra={
                'flow_name': flow_name,
                'run_id': run_id,
                'num_iterations': num_iterations,
                'initial_alpha': initial_alpha,
                'input_shape': initial_state['x'].shape
            })
            
            # Initialize
            state = initial_state.copy()
            current_alpha = initial_alpha
            step_func = _step_function_for_testing or run_flow_step
            key = jax.random.PRNGKey(42)  # For certificate estimation
            input_shape = state['x'].shape

            # Phase 0: Initial certificate assessment (RED/AMBER)
            self.transition_phase(Phase.RED, telemetry_manager, run_id, stage="initial_assessment")

            eta, gamma, is_feasible = self.assess_certificates(compiled, input_shape, key)

            # Log initial certificate check
            if telemetry_manager:
                telemetry_manager.flight_recorder.log_event(
                    run_id=run_id,
                    event="CERT_CHECK",
                    payload={"eta_dd": eta, "gamma": gamma, "feasible": is_feasible}
                )

            if not is_feasible:
                self.logger.warning("Initial certificate assessment failed, attempting remediation", extra={
                    'eta_dd': float(eta),
                    'gamma': float(gamma),
                    'eta_max': self.config.eta_max,
                    'gamma_min': self.config.gamma_min
                })
                # Try remediation by reducing alpha
                for attempt in range(self.config.max_remediation_attempts):
                    current_alpha *= self.config.alpha_reduction_factor
                    eta, gamma, is_feasible = self.assess_certificates(compiled, input_shape, key)

                    if telemetry_manager:
                        telemetry_manager.flight_recorder.log_event(
                            run_id=run_id,
                            event="CERT_REMEDIATION",
                            payload={
                                'attempt': attempt,
                                'alpha': current_alpha,
                                'eta': eta,
                                'gamma': gamma,
                                'feasible': is_feasible
                            }
                        )

                    if is_feasible:
                        self.logger.info("Certificate remediation successful", extra={
                            'attempt': attempt,
                            'final_alpha': current_alpha,
                            'eta_dd': float(eta),
                            'gamma': float(gamma)
                        })
                        break

            if not is_feasible:
                # Still not feasible - fail
                self.logger.error("Certificate assessment failed after all remediation attempts", extra={
                    'max_attempts': self.config.max_remediation_attempts,
                    'final_eta': float(eta),
                    'final_gamma': float(gamma)
                })
                self.transition_phase(Phase.RED, telemetry_manager, run_id,
                                    stage="certification_failed", eta=eta, gamma=gamma)
                raise ValueError(
                    f"System failed certification after {self.config.max_remediation_attempts} "
                    f"remediation attempts. Final eta={eta:.4f}, gamma={gamma:.4f}."
                )

            # Phase 1: Certification passed - enter GREEN phase
            self.logger.info("Certificate assessment passed, entering GREEN phase", extra={
                'final_alpha': current_alpha,
                'eta_dd': float(eta),
                'gamma': float(gamma)
            })
            self.transition_phase(Phase.GREEN, telemetry_manager, run_id,
                                stage="certification_passed", eta=eta, gamma=gamma)

            # Initialize Gap Dial tuner if provided
            if gap_dial_tuner:
                gap_dial_tuner.reset()
                if telemetry_manager:
                    telemetry_manager.flight_recorder.log_event(
                        run_id=run_id,
                        event="GAP_DIAL_ENABLED",
                        payload=gap_dial_tuner.get_tuning_status()
                    )

            # Phase 2: Main optimization loop with tuning and rollback
            energy = compiled.f_value(state)
            last_good_checkpoint = self.create_checkpoint(
                0, state, energy, current_alpha,
                {'eta_dd': float(eta), 'gamma': float(gamma)},
                gap_dial_tuner.get_tuning_status() if gap_dial_tuner else None
            )

            for i in range(num_iterations):
                # Check budget limits
                if not self.check_budget_limits(i):
                    self.logger.warning("Budget limits exceeded, terminating optimization", extra={
                        'iteration': i,
                        'budget_status': self.get_status()
                    })
                    if telemetry_manager:
                        telemetry_manager.flight_recorder.log_event(
                            run_id=run_id,
                            event="BUDGET_EXCEEDED",
                            payload={'iteration': i, 'status': self.get_status()}
                        )
                    break

                # Compute telemetry fields
                t_wall_ms = (time.time() - self.start_time) * 1000.0
                grad = compiled.f_grad(state)
                grad_norm = float(jnp.linalg.norm(grad['x']))
                eta = estimate_eta_dd(compiled.L_apply, input_shape)
                gamma = estimate_gamma_lanczos(compiled.L_apply, key, input_shape)

                # Sparsity computation
                x = state['x']
                l1_norm = float(jnp.linalg.norm(x, ord=1))
                l2_norm = float(jnp.linalg.norm(x, ord=2))
                n = float(jnp.prod(jnp.array(x.shape)))
                sparsity_wx = l1_norm / (l2_norm * float(jnp.sqrt(n))) if l2_norm > 0 else 0.0

                # Gap Dial: Monitor and adapt parameters
                gap_dial_status = None
                if gap_dial_tuner and gap_dial_tuner.should_check_gap(i):
                    current_gap = gap_dial_tuner.estimate_spectral_gap(compiled, state)
                    gap_dial_status = gap_dial_tuner.adapt_parameters(current_gap, compiled)

                    if gap_dial_status['adaptation_applied']:
                        self.tuner_move_count += 1
                        self.logger.debug("Gap Dial adaptation applied", extra={
                            'iteration': i,
                            'current_gap': float(current_gap),
                            'lambda_regularization': gap_dial_status['lambda_regularization']
                        })

                        # Validate certificates after tuner move
                        eta_after, gamma_after, still_feasible = self.assess_certificates(
                            compiled, input_shape, key
                        )

                        if not still_feasible and self.config.rollback_on_cert_failure:
                            # Rollback to last good checkpoint
                            self.logger.warning("Certificate violation after tuner adaptation, rolling back", extra={
                                'eta_after': float(eta_after),
                                'gamma_after': float(gamma_after)
                            })
                            state, current_alpha, tuner_state = self.rollback_to_checkpoint(
                                last_good_checkpoint, telemetry_manager, run_id
                            )
                            if gap_dial_tuner and tuner_state:
                                # Restore tuner state
                                gap_dial_tuner.current_lambda = tuner_state.get('current_lambda', 1.0)
                            energy = compiled.f_value(state)
                            continue  # Skip this iteration and retry

                        if telemetry_manager:
                            telemetry_manager.flight_recorder.log_event(
                                run_id=run_id,
                                event="GAP_DIAL_ADAPTATION",
                                payload={
                                    'iteration': i,
                                    'gap': current_gap,
                                    'lambda': gap_dial_status['lambda_regularization'],
                                    'eta_after': eta_after,
                                    'gamma_after': gamma_after
                                }
                            )

                # Log telemetry
                if telemetry_manager:
                    telemetry_manager.flight_recorder.log(
                        run_id=run_id,
                        flow_name=flow_name,
                        phase=self.phase.value,
                        iter=i,
                        t_wall_ms=t_wall_ms,
                        E=float(energy),
                        grad_norm=grad_norm,
                        eta_dd=float(eta),
                        gamma=float(gamma),
                        alpha=float(current_alpha),
                        phi_residual=float(jnp.nan),  # Placeholder
                        invariant_drift_max=float(jnp.nan),  # Placeholder
                        sparsity_wx=sparsity_wx
                    )

                # Try step with current alpha, with remediation if energy increases
                step_succeeded = False
                step_alpha_local = current_alpha

                for step_attempt in range(self.config.max_step_attempts):
                    candidate_state = step_func(state, compiled, step_alpha_local)
                    new_energy = compiled.f_value(candidate_state)

                    if new_energy <= energy:
                        # Success - accept the step
                        state = candidate_state
                        energy = new_energy
                        step_succeeded = True

                        # Update last good checkpoint
                        last_good_checkpoint = self.create_checkpoint(
                            i + 1, state, energy, current_alpha,
                            {'eta_dd': float(eta), 'gamma': float(gamma)},
                            gap_dial_tuner.get_tuning_status() if gap_dial_tuner else None
                        )
                        break
                    else:
                        # Energy increased - reduce alpha and retry
                        if telemetry_manager:
                            telemetry_manager.flight_recorder.log_event(
                                run_id=run_id,
                                event="STEP_REMEDIATION",
                                payload={
                                    'iter': i,
                                    'attempt': step_attempt,
                                    'E_prev': float(energy),
                                    'E_new': float(new_energy),
                                    'alpha': float(step_alpha_local)
                                }
                            )
                        step_alpha_local *= self.config.step_alpha_reduction_factor

                if not step_succeeded:
                    # All step attempts failed
                    self.logger.error("Step failed to decrease energy after all remediation attempts", extra={
                        'iteration': i,
                        'max_attempts': self.config.max_step_attempts,
                        'previous_energy': float(energy)
                    })
                    if telemetry_manager:
                        telemetry_manager.flight_recorder.log_event(
                            run_id=run_id,
                            event="STEP_FAIL",
                            payload={
                                'iter': i,
                                'attempts': self.config.max_step_attempts,
                                'E_prev': float(energy)
                            }
                        )
                    raise ValueError(
                        f"Step failed to decrease energy after {self.config.max_step_attempts} "
                        f"remediation attempts at iteration {i}."
                    )

            # Phase 3: Finalization
            self.logger.info("Certified flow execution completed", extra={
                'total_iterations': min(i + 1, num_iterations),
                'final_energy': float(energy),
                'rollbacks': self.rollback_count,
                'tuner_moves': self.tuner_move_count,
                'final_phase': self.phase.value
            })
            if telemetry_manager:
                telemetry_manager.flight_recorder.log_event(
                    run_id=run_id,
                    event="RUN_FINISHED",
                    payload={
                        'final_phase': self.phase.value,
                        'total_iterations': min(i + 1, num_iterations),
                        'final_energy': float(energy),
                        'rollbacks': self.rollback_count,
                        'tuner_moves': self.tuner_move_count
                    }
                )
                telemetry_manager.flush()

            duration = time.perf_counter() - start_time_perf
            self.logger.debug("run_certified_flow completed",
                             extra={'duration_ms': duration * 1000, 'success': True})
            return state
        except Exception as e:
            duration = time.perf_counter() - start_time_perf
            self.logger.error("run_certified_flow failed",
                             extra={'duration_ms': duration * 1000, 'error': str(e), 'success': False})
            raise




