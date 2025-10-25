from typing import Any

import jax.numpy as jnp

from ..energy.compile import CompiledEnergy  # Import canonical CompiledEnergy
from ..energy.policies import FlowPolicy, MultiscaleSchedule
from ..telemetry import TelemetryManager
from .checkpoint import CheckpointManager
from .primitives import F_Dis, F_Dis_Preconditioned, F_Multi, F_Proj

# JAX types for clarity
Array = jnp.ndarray
State = dict[str, Array]


def run_flow_step(
    state: dict[str, jnp.ndarray],
    compiled: CompiledEnergy,
    step_alpha: float,
    manifolds: dict[str, Any] | None = None,
    W: Any | None = None,
    flow_policy: FlowPolicy | None = None,
    multiscale_schedule: MultiscaleSchedule | None = None,
    iteration: int = 0,
    telemetry_manager: TelemetryManager | None = None,
    previous_active_levels: list[int] | None = None,
) -> dict[str, jnp.ndarray]:
    """
    Runs one full step of a Forward-Backward Splitting flow.

    If W is provided, includes multiscale transforms:
    F_Dis → F_Multi_forward → F_Proj → F_Multi_inverse
    Otherwise, simple: F_Dis → F_Proj

    Policy-driven execution:
    - flow_policy controls primitive selection (basic/preconditioned/accelerated)
    - multiscale_schedule controls multiscale activation

    Args:
        state: Current optimization state
        compiled: Compiled energy functional
        step_alpha: Step size for optimization
        manifolds: Optional manifold constraints
        W: Optional wavelet transform for multiscale
        flow_policy: Optional flow policy for primitive selection
        multiscale_schedule: Optional multiscale schedule for level activation
        iteration: Current iteration number
        telemetry_manager: Optional telemetry manager for event logging
        previous_active_levels: Mutable list to track previous active levels for event emission

    Returns:
        Updated optimization state
    """
    if manifolds is None:
        manifolds = {}

    # Default to basic policy if none provided
    if flow_policy is None:
        flow_policy = FlowPolicy()

    # Select dissipative primitive based on policy
    if flow_policy.family == "basic":
        dissipative_fn = F_Dis
        dissipative_kwargs = {}
    elif flow_policy.family == "preconditioned":
        dissipative_fn = F_Dis_Preconditioned
        dissipative_kwargs = {"preconditioner": flow_policy.preconditioner}
    elif flow_policy.family == "accelerated":
        # TODO: Implement accelerated F_Dis variant
        raise NotImplementedError("Accelerated family not yet implemented")
    else:
        raise ValueError(f"Unknown flow family: {flow_policy.family}")

    if multiscale_schedule is not None and W is not None:
        # Determine active levels for multiscale schedule
        active_levels = _determine_active_levels(
            multiscale_schedule, state, compiled, iteration
        )

        # Emit SCALE_ACTIVATED event if levels increased
        if previous_active_levels is not None and telemetry_manager is not None:
            prev_levels = previous_active_levels[0] if previous_active_levels else 1
            if active_levels > prev_levels:
                telemetry_manager.flight_recorder.log_event(
                    run_id=getattr(telemetry_manager, "run_id", "unknown"),
                    event="SCALE_ACTIVATED",
                    payload={
                        "previous_levels": prev_levels,
                        "new_levels": active_levels,
                        "iteration": iteration,
                        "mode": multiscale_schedule.mode,
                    },
                )
            previous_active_levels[0] = active_levels

        # Create level-limited transform if schedule requires it
        if active_levels < W.levels:
            W_active = _create_level_limited_transform(W, active_levels)
        else:
            W_active = W

        # Multiscale flow in wavelet space
        # Transform to W-space
        u = F_Multi(state["x"], W_active, "forward")
        # Dissipative step in W-space
        grad_x = compiled.f_grad(state)
        grad_u = F_Multi(grad_x["x"], W_active, "forward")
        u_after_dis = [u[i] - step_alpha * grad_u[i] for i in range(len(u))]
        # Projective step in W-space (soft-thresholding for L1)
        threshold = step_alpha * 0.5  # Assuming weight=0.5 for L1
        u_proj = [
            jnp.where(jnp.abs(c) > threshold, c - threshold * jnp.sign(c), 0)
            for c in u_after_dis
        ]
        # Transform back to physical domain
        x_new = F_Multi(u_proj, W_active, "inverse")
        state_after_proj = {"x": x_new, "y": state["y"]}
    else:
        # Simple flow: dissipative in physical, projective in physical
        state_after_dis = dissipative_fn(
            state, compiled.f_grad, step_alpha, manifolds, **dissipative_kwargs
        )
        state_after_proj = F_Proj(state_after_dis, compiled.g_prox, step_alpha)

    return state_after_proj


def resume_flow(
    checkpoint_id: str,
    checkpoint_manager: CheckpointManager,
    compiled: CompiledEnergy,
    remaining_iters: int,
    step_alpha: float,
    telemetry_manager: TelemetryManager | None = None,
    checkpoint_interval: int = 100,
) -> dict[str, jnp.ndarray]:
    """
    Resume a computable flow from a checkpoint.

    Args:
        checkpoint_id: ID of checkpoint to resume from
        checkpoint_manager: Checkpoint manager instance
        compiled: Compiled energy specification
        remaining_iters: Number of additional iterations to run
        step_alpha: Step size for optimization
        telemetry_manager: Optional telemetry manager for logging
        checkpoint_interval: How often to create checkpoints

    Returns:
        Final optimization state
    """
    # Load checkpoint
    checkpoint_data = checkpoint_manager.load_checkpoint(checkpoint_id)

    # Extract state and metadata
    state = checkpoint_data["state"]
    start_iteration = checkpoint_data["iteration"]
    run_id = checkpoint_data["run_id"]
    flow_config = checkpoint_data["flow_config"]

    # Update telemetry manager run ID if provided
    if telemetry_manager:
        telemetry_manager.run_id = run_id

    # Log resume event
    if telemetry_manager:
        telemetry_manager.flight_recorder.log_event(
            run_id=run_id,
            event="FLOW_RESUMED",
            payload={
                "checkpoint_id": checkpoint_id,
                "start_iteration": start_iteration,
                "remaining_iters": remaining_iters,
            },
        )

    # Continue running from checkpoint
    for i in range(start_iteration, start_iteration + remaining_iters):
        state = run_flow_step(state, compiled, step_alpha)

        # Create checkpoint if requested and at interval
        if (i + 1) % checkpoint_interval == 0:
            certificates = checkpoint_data.get("certificates", {})
            new_checkpoint_id = checkpoint_manager.create_checkpoint(
                run_id=run_id,
                iteration=i + 1,
                state=state,
                flow_config=flow_config,
                certificates=certificates,
                telemetry_history=None,  # TODO: Implement telemetry history extraction
            )

            # Log checkpoint creation
            if telemetry_manager:
                telemetry_manager.flight_recorder.log_event(
                    run_id=run_id,
                    event="CHECKPOINT_CREATED",
                    payload={"checkpoint_id": new_checkpoint_id, "iteration": i + 1},
                )

        # Log telemetry
        if telemetry_manager:
            telemetry_manager.flight_recorder.log(
                iter=i, E=float(compiled.f_value(state))
            )

    return state


def _determine_active_levels(
    multiscale_schedule: MultiscaleSchedule | None,
    state: State,
    compiled: CompiledEnergy,
    iteration: int,
) -> int:
    """
    Determine how many multiscale levels should be active based on the schedule.

    Args:
        multiscale_schedule: The multiscale schedule specification
        state: Current optimization state
        compiled: Compiled energy functional
        iteration: Current iteration number

    Returns:
        Number of active levels
    """
    if multiscale_schedule is None:
        return 1  # Default to all levels if no schedule

    if multiscale_schedule.mode == "fixed_schedule":
        # All levels active from the start
        return multiscale_schedule.levels

    elif multiscale_schedule.mode == "residual_driven":
        # Activate levels based on residual magnitude
        residual = jnp.linalg.norm(compiled.f_grad(state)["x"])
        threshold = _parse_threshold(multiscale_schedule.activate_rule, "residual")

        # Simple progressive activation: more levels as residual decreases
        if residual > threshold:
            return 1  # Only coarse level
        else:
            # Progressively activate more levels as residual decreases
            progress = min(1.0, threshold / max(residual, 1e-10))
            active_levels = max(1, int(progress * multiscale_schedule.levels))
            return min(active_levels, multiscale_schedule.levels)

    elif multiscale_schedule.mode == "energy_driven":
        # Activate levels based on energy decrease rate
        threshold = _parse_threshold(
            multiscale_schedule.activate_rule, "energy_decrease"
        )

        # For energy-driven, we need to track energy history
        # For now, use a simple heuristic based on iteration
        # TODO: Implement proper energy decrease tracking
        progress = min(1.0, iteration / 100.0)  # Simple time-based progression
        active_levels = max(1, int(progress * multiscale_schedule.levels))
        return min(active_levels, multiscale_schedule.levels)

    else:
        raise ValueError(f"Unknown multiscale mode: {multiscale_schedule.mode}")


def _parse_threshold(rule: str, expected_var: str) -> float:
    """
    Parse a threshold value from an activation rule.

    Args:
        rule: Activation rule string (e.g., "residual>0.01")
        expected_var: Expected variable name

    Returns:
        Threshold value
    """
    # Simple parsing for now - assumes format "var>value" or "var<value"
    if ">" in rule:
        var, threshold_str = rule.split(">", 1)
        if var.strip() == expected_var:
            return float(threshold_str.strip())
    elif "<" in rule:
        var, threshold_str = rule.split("<", 1)
        if var.strip() == expected_var:
            return float(threshold_str.strip())

    # Default threshold if parsing fails
    return 0.01


def _create_level_limited_transform(W: Any, active_levels: int) -> Any:
    """
    Create a transform that limits the number of active levels.

    Args:
        W: Original transform object
        active_levels: Number of levels to keep active

    Returns:
        Modified transform object
    """
    if active_levels >= W.levels:
        return W

    # Create a wrapper that truncates coefficients to active levels
    class LevelLimitedTransform:
        def __init__(self, original_transform, active_levels):
            self.original = original_transform
            self.levels = active_levels
            self.name = original_transform.name
            self.ndim = original_transform.ndim
            self.frame = original_transform.frame
            self.c = original_transform.c

        def forward(self, x):
            coeffs = self.original.forward(x)
            # Truncate to active levels (keep approximation + active details)
            return coeffs[: active_levels + 1]  # +1 for approximation coefficient

        def inverse(self, coeffs):
            # Pad with zeros for inactive levels
            full_coeffs = coeffs + [
                jnp.zeros_like(coeffs[-1]) for _ in range(W.levels - active_levels)
            ]
            return self.original.inverse(full_coeffs)

    return LevelLimitedTransform(W, active_levels)
