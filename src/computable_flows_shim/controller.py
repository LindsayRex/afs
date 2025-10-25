"""
The Flight Controller for the Computable Flows Shim.

Implements the RED/AMBER/GREEN phase machine for certificate-gated parameter tuning
with rollback capability and budget enforcement.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

import jax
import jax.numpy as jnp

from computable_flows_shim.energy.compile import CompiledEnergy
from computable_flows_shim.energy.policies import FlowPolicy, MultiscaleSchedule
from computable_flows_shim.fda.certificates import (
    estimate_eta_dd,
    estimate_gamma_lanczos,
)
from computable_flows_shim.runtime.checkpoint import CheckpointManager
from computable_flows_shim.runtime.step import run_flow_step

from .logging import get_logger
from .telemetry import TelemetryManager
from .tuner.gap_dial import GapDialTuner


class Phase(Enum):
    """Controller operational phases."""

    RED = "RED"  # Spec/units/ops invalid
    AMBER = "AMBER"  # Spec OK, not certified
    GREEN = "GREEN"  # Certified, tuner allowed


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
    max_wall_time_ms: float | None = None
    max_iterations: int | None = None
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
    state: dict[str, jnp.ndarray]
    energy: float
    alpha: float
    certificates: dict[str, float]
    tuner_state: dict[str, Any] | None = None
    timestamp: float | None = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class FlightController:
    """
    Flight Controller implementing RED/AMBER/GREEN phase machine.

    Manages certificate-gated parameter tuning with rollback capability
    and budget enforcement for safe, automated optimization.
    """

    def __init__(self, config: ControllerConfig | None = None):
        self.config = config or ControllerConfig()
        self.logger = get_logger(__name__)
        self.phase = Phase.RED
        self.checkpoints: list[Checkpoint] = []
        self.rollback_count = 0
        self.tuner_move_count = 0
        self.start_time = time.time()

        self.logger.debug(
            "FlightController initialized",
            extra={
                "config": {
                    "eta_max": self.config.eta_max,
                    "gamma_min": self.config.gamma_min,
                    "max_iterations": self.config.max_iterations,
                    "max_wall_time_ms": self.config.max_wall_time_ms,
                }
            },
        )

    def get_status(self) -> dict[str, Any]:
        """Get current controller status."""
        return {
            "phase": self.phase.value,
            "rollback_count": self.rollback_count,
            "tuner_move_count": self.tuner_move_count,
            "wall_time_ms": (time.time() - self.start_time) * 1000,
            "checkpoints_available": len(self.checkpoints),
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

    def assess_certificates(
        self, compiled: CompiledEnergy, input_shape: tuple, key: jnp.ndarray
    ) -> tuple[float, float, bool]:
        """
        Assess certificate feasibility.

        Returns:
            eta_dd, gamma, is_feasible
        """
        self.logger.debug("Assessing certificates", extra={"input_shape": input_shape})

        eta = estimate_eta_dd(compiled.L_apply, input_shape)
        gamma = estimate_gamma_lanczos(compiled.L_apply, key, input_shape)

        # Convert to scalars for boolean logic
        eta_scalar = float(eta)
        gamma_scalar = float(gamma)
        is_feasible = (
            eta_scalar <= self.config.eta_max and gamma_scalar >= self.config.gamma_min
        )

        self.logger.debug(
            "Certificate assessment completed",
            extra={
                "eta_dd": eta_scalar,
                "gamma": gamma_scalar,
                "is_feasible": is_feasible,
                "eta_max": self.config.eta_max,
                "gamma_min": self.config.gamma_min,
            },
        )

        return eta_scalar, gamma_scalar, is_feasible

    def create_checkpoint(
        self,
        iteration: int,
        state: dict[str, jnp.ndarray],
        energy: float,
        alpha: float,
        certificates: dict[str, float],
        tuner_state: dict[str, Any] | None = None,
    ) -> Checkpoint:
        """Create a checkpoint for potential rollback."""
        checkpoint = Checkpoint(
            iteration=iteration,
            state=state.copy(),
            energy=energy,
            alpha=alpha,
            certificates=certificates.copy(),
            tuner_state=tuner_state.copy() if tuner_state else None,
        )
        self.checkpoints.append(checkpoint)

        # Keep only last 5 checkpoints to prevent memory bloat
        if len(self.checkpoints) > 5:
            self.checkpoints.pop(0)

        return checkpoint

    def rollback_to_checkpoint(
        self,
        target_checkpoint: Checkpoint,
        telemetry_manager: TelemetryManager | None = None,
        run_id: str = "",
    ) -> tuple[dict[str, jnp.ndarray], float, dict[str, Any] | None]:
        """
        Rollback to a previous checkpoint.

        Returns:
            state, alpha, tuner_state
        """
        if self.rollback_count >= self.config.max_rollbacks:
            raise ValueError(
                f"Maximum rollbacks ({self.config.max_rollbacks}) exceeded"
            )

        self.rollback_count += 1

        if telemetry_manager:
            telemetry_manager.flight_recorder.log_event(
                run_id=run_id,
                event="ROLLBACK",
                payload={
                    "rollback_count": self.rollback_count,
                    "target_iteration": target_checkpoint.iteration,
                    "reason": "certificate_regression",
                },
            )

        return (
            target_checkpoint.state,
            target_checkpoint.alpha,
            target_checkpoint.tuner_state,
        )

    def transition_phase(
        self,
        new_phase: Phase,
        telemetry_manager: TelemetryManager | None = None,
        run_id: str = "",
        **kwargs,
    ):
        """Transition to a new phase with telemetry logging."""
        old_phase = self.phase
        self.phase = new_phase

        if telemetry_manager:
            telemetry_manager.flight_recorder.log_event(
                run_id=run_id,
                event="PHASE_TRANSITION",
                payload={"from": old_phase.value, "to": new_phase.value, **kwargs},
            )

    def run_certified_flow(
        self,
        initial_state: dict[str, jnp.ndarray],
        compiled: CompiledEnergy,
        num_iterations: int,
        initial_alpha: float,
        telemetry_manager: TelemetryManager | None = None,
        flow_name: str = "",
        run_id: str = "",
        gap_dial_tuner: GapDialTuner | None = None,
        checkpoint_manager: CheckpointManager | None = None,
        flow_policy: FlowPolicy | None = None,
        multiscale_schedule: MultiscaleSchedule | None = None,
        _step_function_for_testing: Callable | None = None,
    ) -> dict[str, jnp.ndarray]:
        """
        Run the certified flow with full phase machine control.

        Implements RED/AMBER/GREEN phases with certificate gating,
        rollback capability, and budget enforcement.
        """
        start_time_perf = time.perf_counter()
        try:
            self.logger.info(
                "Starting certified flow execution",
                extra={
                    "flow_name": flow_name,
                    "run_id": run_id,
                    "num_iterations": num_iterations,
                    "initial_alpha": initial_alpha,
                    "input_shape": initial_state["x"].shape,
                },
            )

            # Initialize
            state = initial_state.copy()
            current_alpha = initial_alpha
            step_func = _step_function_for_testing or run_flow_step
            key = jax.random.PRNGKey(42)  # For certificate estimation
            input_shape = state["x"].shape

            # Phase 0: Initial certificate assessment (RED/AMBER)
            self.transition_phase(
                Phase.RED, telemetry_manager, run_id, stage="initial_assessment"
            )

            eta, gamma, is_feasible = self.assess_certificates(
                compiled, input_shape, key
            )

            # Log initial certificate check
            if telemetry_manager:
                telemetry_manager.flight_recorder.log_event(
                    run_id=run_id,
                    event="CERT_CHECK",
                    payload={"eta_dd": eta, "gamma": gamma, "feasible": is_feasible},
                )

            if not is_feasible:
                self.logger.warning(
                    "Initial certificate assessment failed, attempting remediation",
                    extra={
                        "eta_dd": float(eta),
                        "gamma": float(gamma),
                        "eta_max": self.config.eta_max,
                        "gamma_min": self.config.gamma_min,
                    },
                )
                # Try remediation by reducing alpha
                for attempt in range(self.config.max_remediation_attempts):
                    current_alpha *= self.config.alpha_reduction_factor
                    eta, gamma, is_feasible = self.assess_certificates(
                        compiled, input_shape, key
                    )

                    if telemetry_manager:
                        telemetry_manager.flight_recorder.log_event(
                            run_id=run_id,
                            event="CERT_REMEDIATION",
                            payload={
                                "attempt": attempt,
                                "alpha": current_alpha,
                                "eta": eta,
                                "gamma": gamma,
                                "feasible": is_feasible,
                            },
                        )

                    if is_feasible:
                        self.logger.info(
                            "Certificate remediation successful",
                            extra={
                                "attempt": attempt,
                                "final_alpha": current_alpha,
                                "eta_dd": float(eta),
                                "gamma": float(gamma),
                            },
                        )
                        break

            if not is_feasible:
                # Still not feasible - fail
                self.logger.error(
                    "Certificate assessment failed after all remediation attempts",
                    extra={
                        "max_attempts": self.config.max_remediation_attempts,
                        "final_eta": float(eta),
                        "final_gamma": float(gamma),
                    },
                )
                self.transition_phase(
                    Phase.RED,
                    telemetry_manager,
                    run_id,
                    stage="certification_failed",
                    eta=eta,
                    gamma=gamma,
                )
                raise ValueError(
                    f"System failed certification after {self.config.max_remediation_attempts} "
                    f"remediation attempts. Final eta={eta:.4f}, gamma={gamma:.4f}."
                )

            # Phase 1: Certification passed - enter GREEN phase
            self.logger.info(
                "Certificate assessment passed, entering GREEN phase",
                extra={
                    "final_alpha": current_alpha,
                    "eta_dd": float(eta),
                    "gamma": float(gamma),
                },
            )
            self.transition_phase(
                Phase.GREEN,
                telemetry_manager,
                run_id,
                stage="certification_passed",
                eta=eta,
                gamma=gamma,
            )

            # Log flow policy application
            if flow_policy and telemetry_manager:
                telemetry_manager.flight_recorder.log_event(
                    run_id=run_id,
                    event="FLOW_POLICY_APPLIED",
                    payload={
                        "family": flow_policy.family,
                        "discretization": flow_policy.discretization,
                        "preconditioner": flow_policy.preconditioner,
                    },
                )

            # Initialize multiscale state if schedule provided
            current_level = 0
            max_level = multiscale_schedule.levels if multiscale_schedule else 0

            if multiscale_schedule and telemetry_manager:
                telemetry_manager.flight_recorder.log_event(
                    run_id=run_id,
                    event="MULTISCALE_SCHEDULE_INIT",
                    payload={
                        "mode": multiscale_schedule.mode,
                        "levels": multiscale_schedule.levels,
                        "activate_rule": multiscale_schedule.activate_rule,
                    },
                )

            # Phase 2: Main optimization loop with tuning and rollback
            energy = compiled.f_value(state)
            last_energy = energy  # Initialize for multiscale activation
            last_good_checkpoint = self.create_checkpoint(
                0,
                state,
                energy,
                current_alpha,
                {"eta_dd": float(eta), "gamma": float(gamma)},
                gap_dial_tuner.get_tuning_status() if gap_dial_tuner else None,
            )

            for i in range(num_iterations):
                # Check budget limits
                if not self.check_budget_limits(i):
                    self.logger.warning(
                        "Budget limits exceeded, terminating optimization",
                        extra={"iteration": i, "budget_status": self.get_status()},
                    )
                    if telemetry_manager:
                        telemetry_manager.flight_recorder.log_event(
                            run_id=run_id,
                            event="BUDGET_EXCEEDED",
                            payload={"iteration": i, "status": self.get_status()},
                        )
                    break

                # Compute telemetry fields
                t_wall_ms = (time.time() - self.start_time) * 1000.0
                grad = compiled.f_grad(state)
                grad_norm = float(jnp.linalg.norm(grad["x"]))
                eta = estimate_eta_dd(compiled.L_apply, input_shape)
                gamma = estimate_gamma_lanczos(compiled.L_apply, key, input_shape)

                # Sparsity computation
                x = state["x"]
                l1_norm = float(jnp.linalg.norm(x, ord=1))
                l2_norm = float(jnp.linalg.norm(x, ord=2))
                n = float(jnp.prod(jnp.array(x.shape)))
                sparsity_wx = (
                    l1_norm / (l2_norm * float(jnp.sqrt(n))) if l2_norm > 0 else 0.0
                )

                # Gap Dial: Monitor and adapt parameters
                gap_dial_status = None
                if gap_dial_tuner and gap_dial_tuner.should_check_gap(i):
                    current_gap = gap_dial_tuner.estimate_spectral_gap(compiled, state)
                    gap_dial_status = gap_dial_tuner.adapt_parameters(
                        current_gap, compiled
                    )

                    # Update tuner state
                    gap_dial_tuner.iteration_count = i + 1
                    gap_dial_tuner.last_gap_check = i

                    if gap_dial_status["adaptation_applied"]:
                        self.tuner_move_count += 1
                        self.logger.debug(
                            "Gap Dial adaptation applied",
                            extra={
                                "iteration": i,
                                "current_gap": float(current_gap),
                                "lambda_regularization": gap_dial_status[
                                    "lambda_regularization"
                                ],
                            },
                        )

                        # Validate certificates after tuner move
                        eta_after, gamma_after, still_feasible = (
                            self.assess_certificates(compiled, input_shape, key)
                        )

                        if not still_feasible and self.config.rollback_on_cert_failure:
                            # Rollback to last good checkpoint
                            self.logger.warning(
                                "Certificate violation after tuner adaptation, rolling back",
                                extra={
                                    "eta_after": float(eta_after),
                                    "gamma_after": float(gamma_after),
                                },
                            )
                            state, current_alpha, tuner_state = (
                                self.rollback_to_checkpoint(
                                    last_good_checkpoint, telemetry_manager, run_id
                                )
                            )
                            if gap_dial_tuner and tuner_state:
                                # Restore tuner state
                                gap_dial_tuner.current_lambda = tuner_state.get(
                                    "current_lambda", 1.0
                                )
                            energy = compiled.f_value(state)
                            continue  # Skip this iteration and retry

                        if telemetry_manager:
                            telemetry_manager.flight_recorder.log_event(
                                run_id=run_id,
                                event="GAP_DIAL_ADAPTATION",
                                payload={
                                    "iteration": i,
                                    "gap": current_gap,
                                    "lambda": gap_dial_status["lambda_regularization"],
                                    "eta_after": eta_after,
                                    "gamma_after": gamma_after,
                                },
                            )

                # Policy-driven primitive selection and multiscale activation
                flow_family = "gradient"  # Default
                lens_name = "identity"  # Default
                level_active_max = 0  # Default

                if flow_policy:
                    # Select primitive family based on policy
                    if flow_policy.family == "basic":
                        flow_family = "gradient"
                    elif flow_policy.family == "preconditioned":
                        flow_family = "preconditioned"
                    elif flow_policy.family == "accelerated":
                        flow_family = "accelerated"

                    # Select discretization based on policy
                    if flow_policy.discretization == "explicit":
                        lens_name = "explicit_euler"
                    elif flow_policy.discretization == "implicit":
                        lens_name = "implicit_euler"
                    elif flow_policy.discretization == "symplectic":
                        lens_name = "symplectic_euler"

                if multiscale_schedule:
                    # Determine if we should activate a new level
                    should_activate = False

                    if multiscale_schedule.mode == "fixed_schedule":
                        # Parse activate_rule for fixed schedule (e.g., 'iteration%2==0' means every 2 iterations)
                        if (
                            "iteration%" in multiscale_schedule.activate_rule
                            and "==0" in multiscale_schedule.activate_rule
                        ):
                            # Extract the divisor from pattern like 'iteration%N==0'
                            import re

                            match = re.search(
                                r"iteration%(\d+)==0", multiscale_schedule.activate_rule
                            )
                            if match:
                                interval = int(match.group(1))
                                if i % interval == 0 and i > 0:
                                    should_activate = True
                                    current_level = min(
                                        current_level + 1,
                                        multiscale_schedule.levels - 1,
                                    )
                        else:
                            # Default: activate one level per iteration until max reached
                            if current_level < multiscale_schedule.levels - 1:
                                should_activate = True
                                current_level += 1

                    elif multiscale_schedule.mode == "residual_driven":
                        # Activate based on residual reduction
                        try:
                            residual_threshold = float(
                                multiscale_schedule.activate_rule
                            )
                            if i > 0 and (energy / last_energy) < residual_threshold:
                                should_activate = True
                                current_level = min(
                                    current_level + 1, multiscale_schedule.levels - 1
                                )
                        except ValueError:
                            # If activate_rule is not a number, skip activation
                            pass

                    elif multiscale_schedule.mode == "energy_driven":
                        # Activate based on energy improvement
                        try:
                            energy_threshold = float(multiscale_schedule.activate_rule)
                            if i > 0 and abs(energy - last_energy) < energy_threshold:
                                should_activate = True
                                current_level = min(
                                    current_level + 1, multiscale_schedule.levels - 1
                                )
                        except ValueError:
                            # If activate_rule is not a number, skip activation
                            pass

                    if should_activate:
                        level_active_max = current_level
                        if telemetry_manager:
                            telemetry_manager.flight_recorder.log_event(
                                run_id=run_id,
                                event="SCALE_ACTIVATED",
                                payload={
                                    "iteration": i,
                                    "level": current_level,
                                    "mode": multiscale_schedule.mode,
                                    "energy": float(energy),
                                },
                            )

                # Store last energy for residual/energy driven activation
                last_energy = energy

                # Log telemetry
                if telemetry_manager:
                    telemetry_manager.flight_recorder.log(
                        run_id=run_id,
                        flow_name=flow_name,
                        phase=self.phase.value,
                        iter=i,
                        trial_id="",  # Placeholder for tuner trials
                        t_wall_ms=t_wall_ms,
                        alpha=float(current_alpha),
                        **{
                            "lambda": float(
                                gap_dial_tuner.current_lambda if gap_dial_tuner else 1.0
                            )
                        },  # Use dict unpacking for reserved keyword
                        lambda_j="{}",  # Placeholder for per-scale lambdas
                        E=float(energy),
                        grad_norm=grad_norm,
                        eta_dd=float(eta),
                        gamma=float(gamma),
                        sparsity_wx=sparsity_wx,
                        metric_ber=0.0,  # Placeholder metric
                        warnings="",  # Placeholder for warnings
                        notes="",  # Placeholder for notes
                        invariant_drift_max=float(jnp.nan),  # Placeholder
                        phi_residual=float(jnp.nan),  # Placeholder
                        lens_name=lens_name,  # Policy-driven lens selection
                        level_active_max=level_active_max,  # Policy-driven multiscale levels
                        sparsity_mode="l1",  # Default sparsity mode
                        flow_family=flow_family,  # Policy-driven flow family
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
                            i + 1,
                            state,
                            energy,
                            current_alpha,
                            {"eta_dd": float(eta), "gamma": float(gamma)},
                            gap_dial_tuner.get_tuning_status()
                            if gap_dial_tuner
                            else None,
                        )
                        break
                    else:
                        # Energy increased - reduce alpha and retry
                        if telemetry_manager:
                            telemetry_manager.flight_recorder.log_event(
                                run_id=run_id,
                                event="STEP_REMEDIATION",
                                payload={
                                    "iter": i,
                                    "attempt": step_attempt,
                                    "E_prev": float(energy),
                                    "E_new": float(new_energy),
                                    "alpha": float(step_alpha_local),
                                },
                            )
                        step_alpha_local *= self.config.step_alpha_reduction_factor

                if not step_succeeded:
                    # All step attempts failed
                    self.logger.error(
                        "Step failed to decrease energy after all remediation attempts",
                        extra={
                            "iteration": i,
                            "max_attempts": self.config.max_step_attempts,
                            "previous_energy": float(energy),
                        },
                    )
                    if telemetry_manager:
                        telemetry_manager.flight_recorder.log_event(
                            run_id=run_id,
                            event="STEP_FAIL",
                            payload={
                                "iter": i,
                                "attempts": self.config.max_step_attempts,
                                "E_prev": float(energy),
                            },
                        )
                    raise ValueError(
                        f"Step failed to decrease energy after {self.config.max_step_attempts} "
                        f"remediation attempts at iteration {i}."
                    )

            # Phase 3: Finalization
            self.logger.info(
                "Certified flow execution completed",
                extra={
                    "total_iterations": min(i + 1, num_iterations),
                    "final_energy": float(energy),
                    "rollbacks": self.rollback_count,
                    "tuner_moves": self.tuner_move_count,
                    "final_phase": self.phase.value,
                },
            )
            if telemetry_manager:
                telemetry_manager.flight_recorder.log_event(
                    run_id=run_id,
                    event="RUN_FINISHED",
                    payload={
                        "final_phase": self.phase.value,
                        "total_iterations": min(i + 1, num_iterations),
                        "final_energy": float(energy),
                        "rollbacks": self.rollback_count,
                        "tuner_moves": self.tuner_move_count,
                    },
                )
                telemetry_manager.flush()

            duration = time.perf_counter() - start_time_perf
            self.logger.debug(
                "run_certified_flow completed",
                extra={"duration_ms": duration * 1000, "success": True},
            )
            return state
        except Exception as e:
            duration = time.perf_counter() - start_time_perf
            self.logger.error(
                "run_certified_flow failed",
                extra={
                    "duration_ms": duration * 1000,
                    "error": str(e),
                    "success": False,
                },
            )
            raise
