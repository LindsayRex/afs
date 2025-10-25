from typing import Any

import jax.numpy as jnp

from ..energy.compile import CompiledEnergy  # Import canonical CompiledEnergy
from ..telemetry import TelemetryManager
from .checkpoint import CheckpointManager
from .primitives import F_Dis, F_Multi, F_Proj

# JAX types for clarity
Array = jnp.ndarray
State = dict[str, Array]


def run_flow_step(
    state: dict[str, jnp.ndarray],
    compiled: CompiledEnergy,
    step_alpha: float,
    manifolds: dict[str, Any] = {},
    W: Any | None = None,
) -> dict[str, jnp.ndarray]:
    """
    Runs one full step of a Forward-Backward Splitting flow.

    If W is provided, includes multiscale transforms:
    F_Dis → F_Multi_forward → F_Proj → F_Multi_inverse
    Otherwise, simple: F_Dis → F_Proj
    """
    if manifolds is None:
        manifolds = {}

    # Forward step (dissipative) - always in physical domain
    state_after_dis = F_Dis(state, compiled.f_grad, step_alpha, manifolds)

    if W is not None:
        # Multiscale: transform to W-space
        u = F_Multi(state_after_dis["x"], W, "forward")
        # Projective step in W-space
        # For now, assume prox is for the transformed space
        # TODO: Update compiler to handle W-space prox
        u_proj = compiled.g_prox({"x": u}, step_alpha)["x"]
        # Transform back to physical domain
        x_new = F_Multi(u_proj, W, "inverse")
        state_after_proj = {"x": x_new, "y": state["y"]}  # Keep y unchanged
    else:
        # Simple flow: projective in physical domain
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
