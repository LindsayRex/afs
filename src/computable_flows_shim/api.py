"""
Core API for the Computable Flows Shim.

This module provides the main user-facing API for running certified optimization flows.
All functions follow the Functional Core/Imperative Shell pattern:
- Functional Core: Pure computational logic (energy functions, certificates, etc.)
- Imperative Shell: Orchestration, I/O, telemetry, error handling (this API layer)
"""

import os
from typing import Any, Protocol

from computable_flows_shim.controller import FlightController
from computable_flows_shim.telemetry import TelemetryManager


class Op(Protocol):
    """Protocol for a linear operator used in energy specifications.

    Linear operators define transformations applied to variables in energy functionals.
    They must be callable and support JAX operations for automatic differentiation.

    Args:
        x: Input array to transform

    Returns:
        Transformed array of same shape as input

    Example:
        class IdentityOp(Op):
            def __call__(self, x):
                return x
    """

    def __call__(self, x: Any) -> Any: ...


def run_certified_with_telemetry(
    initial_state: dict[str, Any],
    compiled: Any,  # CompiledEnergy from energy.compile
    num_iterations: int,
    step_alpha: float,
    flow_name: str,
    run_id: str,
    out_dir: str,
    schema_version: int = 3,
    residual_details: dict[str, Any] | None = None,
    extra_manifest: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], TelemetryManager]:
    """
    Run a certified optimization flow with full telemetry recording.

    This is the main API function for executing optimization flows with certificate
    validation, parameter tuning, and comprehensive observability. Implements the
    RED/AMBER/GREEN phase machine with automatic rollback and remediation.

    **Design Pattern**: Imperative Shell - orchestrates the optimization flow,
    handles I/O, telemetry, and error recovery while delegating pure computation
    to the Functional Core.

    Args:
        initial_state: Initial state dictionary with JAX arrays for all variables.
            Keys should match variable names in the energy specification.
            Example: {'x': jnp.array([1.0, 2.0]), 'y': jnp.array([0.0])}

        compiled: Compiled energy functional from compile_energy().
            Contains jitted f_value, f_grad, g_prox functions and metadata.

        num_iterations: Maximum number of optimization iterations to run.
            May terminate early due to convergence, budget limits, or errors.

        step_alpha: Initial step size for gradient descent. Will be automatically
            adjusted via certificate validation and remediation if needed.

        flow_name: Human-readable name for this optimization flow.
            Used in telemetry and manifest metadata.

        run_id: Unique identifier for this run (typically timestamp-based).
            Must be filesystem-safe. Example: "20251025_143022"

        out_dir: Directory path where telemetry files will be written.
            Will be created if it doesn't exist. Structure:
            out_dir/
            ├── fda_run_{run_id}/
            │   ├── telemetry.parquet
            │   ├── events.parquet
            │   └── manifest.toml

        schema_version: Telemetry schema version (default: 3).
            Controls column format in telemetry files.

        residual_details: Optional metadata about residual computation.
            Example: {"method": "l2_norm", "norm": "L2", "notes": "computed from data"}

        extra_manifest: Optional additional metadata to include in manifest.toml.
            Will be merged with standard manifest fields.

    Returns:
        Tuple of (final_state, telemetry_manager):
        - final_state: Final optimized state dictionary with same structure as initial_state
        - telemetry_manager: TelemetryManager instance with access to all recorded data

    Raises:
        ValueError: If certificate validation fails after all remediation attempts
        RuntimeError: If step function fails to decrease energy after remediation
        OSError: If telemetry directory cannot be created or written to

    Example:
        ```python
        # Compile energy specification
        spec = EnergySpec(
            terms=[TermSpec(type='quadratic', op='I', weight=1.0, variable='x', target='y')],
            state=StateSpec(shapes={'x': [2], 'y': [2]})
        )
        op_registry = {'I': IdentityOp()}
        compiled = compile_energy(spec, op_registry)

        # Run optimization with telemetry
        initial_state = {'x': jnp.array([10.0, 5.0]), 'y': jnp.array([0.0, 0.0])}
        final_state, telemetry = run_certified_with_telemetry(
            initial_state=initial_state,
            compiled=compiled,
            num_iterations=100,
            step_alpha=0.1,
            flow_name="quadratic_optimization",
            run_id="20251025_test",
            out_dir="./runs"
        )

        # Access telemetry data
        print(f"Final energy: {compiled.f_value(final_state)}")
        print(f"Run completed in phase: {telemetry.flight_recorder.get_status()}")
        ```

    Notes:
        - Implements RED/AMBER/GREEN phase machine with certificate gating
        - Automatic parameter tuning via GapDial when certificates allow
        - Rollback to last good checkpoint on certificate violations
        - Comprehensive telemetry with schema v3 compliance
        - Atomic file writes to prevent corruption on interruption
    """
    os.makedirs(out_dir, exist_ok=True)

    # Initialize telemetry and controller
    telemetry_manager = TelemetryManager(
        base_path=out_dir, flow_name=flow_name, run_id=run_id
    )

    controller = FlightController()

    # Run the certified flow
    final_state = controller.run_certified_flow(
        initial_state=initial_state,
        compiled=compiled,
        num_iterations=num_iterations,
        initial_alpha=step_alpha,
        telemetry_manager=telemetry_manager,
        flow_name=flow_name,
        run_id=run_id,
    )

    # Finalize telemetry
    telemetry_manager.flush()

    # Write manifest with metadata
    if residual_details is None:
        residual_details = {"method": "unknown", "norm": "L2", "notes": "not provided"}

    telemetry_manager.write_run_manifest(
        schema_version=schema_version,
        residual_details=residual_details,
        extra=extra_manifest or {},
    )

    return final_state, telemetry_manager
