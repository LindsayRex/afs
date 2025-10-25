"""
Core API for the Computable Flows Shim.

This module provides the main user-facing API for running certified optimization flows.
All functions follow the Functional Core/Imperative Shell pattern:
- Functional Core: Pure computational logic (energy functions, certificates, etc.)
- Imperative Shell: Orchestration, I/O, telemetry, error handling (this API layer)
"""

import os
from typing import Any, Protocol

from pydantic import BaseModel, Field, field_validator

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


class AtomSpec(BaseModel):
    """Specification for a single atom in the energy functional.

    An atom represents a fundamental building block of energy functionals with
    well-defined mathematical properties and computational implementations.
    """

    type: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Atom type (e.g., 'quadratic', 'l1', 'wavelet_l1')",
    )
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters specific to this atom type",
    )
    weight: float = Field(..., gt=0, le=1e6, description="Positive weight coefficient")
    variable: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Variable name this atom applies to",
    )

    @field_validator("type")
    @classmethod
    def validate_atom_type(cls, v):
        """Validate atom type is one of the supported atom types."""
        from computable_flows_shim.atoms import ATOM_REGISTRY

        known_types = set(ATOM_REGISTRY.keys())
        if v not in known_types:
            raise ValueError(
                f"Unknown atom type '{v}'. Supported types: {sorted(known_types)}"
            )
        return v


class AtomBasedSpec(BaseModel):
    """Complete specification for atom-based optimization problems.

    This provides a high-level interface for specifying optimization problems
    using atoms directly, rather than constructing EnergySpec manually.
    """

    atoms: list[AtomSpec] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="List of atoms in the energy functional",
    )
    state: dict[str, dict[str, Any]] = Field(
        ..., min_length=1, description="State variable specifications with shapes"
    )
    initial_state: dict[str, Any] = Field(
        ..., description="Initial state dictionary with JAX arrays"
    )
    num_iterations: int = Field(
        ..., gt=0, le=10000, description="Maximum number of optimization iterations"
    )
    step_alpha: float = Field(
        ..., gt=0, le=10, description="Initial step size for gradient descent"
    )
    flow_name: str = Field(
        default="atom_based_flow",
        min_length=1,
        max_length=100,
        description="Human-readable name for this optimization flow",
    )
    run_id: str = Field(
        default=None,
        min_length=1,
        max_length=100,
        description="Unique identifier for this run",
    )
    out_dir: str = Field(
        default="./runs",
        description="Directory path where telemetry files will be written",
    )
    schema_version: int = Field(
        default=3, ge=1, le=10, description="Telemetry schema version"
    )

    @field_validator("state")
    @classmethod
    def validate_state_shapes(cls, v):
        """Validate state specifications contain shape information."""
        for var_name, var_spec in v.items():
            if "shape" not in var_spec:
                raise ValueError(f"Variable '{var_name}' must specify a 'shape' field")
            shape = var_spec["shape"]
            if not isinstance(shape, list) or not shape:
                raise ValueError(
                    f"Shape for variable '{var_name}' must be non-empty list"
                )
            if not all(isinstance(dim, int) and dim > 0 for dim in shape):
                raise ValueError(
                    f"All dimensions for variable '{var_name}' must be positive integers"
                )
        return v

    @field_validator("initial_state")
    @classmethod
    def validate_initial_state(cls, v, info):
        """Validate initial state is consistent with state specification."""
        if "state" in info.data:
            state_spec = info.data["state"]
            for var_name in state_spec:
                if var_name not in v:
                    raise ValueError(f"Initial state missing variable '{var_name}'")
        return v


def atom_spec_to_energy_spec(atom_spec: AtomBasedSpec) -> tuple[Any, dict[str, Any]]:
    """
    Convert an AtomBasedSpec to EnergySpec and operator registry.

    This function translates the high-level atom-based specification into the
    lower-level EnergySpec format that the compilation pipeline expects.

    Args:
        atom_spec: Atom-based specification to convert

    Returns:
        Tuple of (EnergySpec, op_registry) where op_registry maps operator
        names to callable operator functions.

    Raises:
        ValueError: If atom specification cannot be converted
    """
    from computable_flows_shim.energy.specs import EnergySpec, StateSpec, TermSpec

    # Convert state specification
    state_shapes = {}
    for var_name, var_spec in atom_spec.state.items():
        state_shapes[var_name] = var_spec["shape"]

    state_spec = StateSpec(shapes=state_shapes)

    # Convert atoms to terms
    terms = []
    op_registry = {}

    for i, atom in enumerate(atom_spec.atoms):
        # Create a unique term identifier
        term_id = f"{atom.variable}_{atom.type}_{i}"

        # Map atom types to term types and operators
        if atom.type == "quadratic":
            # Quadratic atoms need A and b matrices/vectors
            if "A" not in atom.params or "b" not in atom.params:
                raise ValueError("Quadratic atom requires 'A' and 'b' parameters")

            # Create operator that computes Ax - b
            class QuadraticResidualOp:
                def __init__(self, A, b):
                    self.A = A
                    self.b = b

                def __call__(self, x):
                    return self.A @ x - self.b

            op_name = f"residual_{term_id}"
            op_registry[op_name] = QuadraticResidualOp(
                atom.params["A"], atom.params["b"]
            )

            term = TermSpec(
                type="quadratic",
                op=op_name,
                weight=atom.weight,
                variable=atom.variable,
                target=None,  # No target since residual includes the constant
            )

        elif atom.type in ["l1", "tikhonov", "tv", "wavelet_l1"]:
            # These atoms use identity operator
            op_name = f"I_{term_id}"
            op_registry[op_name] = lambda x: x  # Identity operator

            # Map atom types to term types
            term_type_map = {
                "l1": "l1",
                "tikhonov": "tikhonov",
                "tv": "tv",
                "wavelet_l1": "wavelet_l1",
            }

            term = TermSpec(
                type=term_type_map[atom.type],
                op=op_name,
                weight=atom.weight,
                variable=atom.variable,
                wavelet=atom.params.get("wavelet"),
                levels=atom.params.get("levels"),
                ndim=atom.params.get("ndim"),
            )

        else:
            raise ValueError(f"Unsupported atom type for conversion: {atom.type}")

        terms.append(term)

    energy_spec = EnergySpec(terms=terms, state=state_spec)
    return energy_spec, op_registry


def run_certified(
    atom_spec: AtomBasedSpec | dict[str, Any],
) -> tuple[dict[str, Any], TelemetryManager]:
    """
    Run a certified optimization flow using atom-based specification.

    This is the high-level API function for running optimization flows specified
    using atoms directly. It internally converts the atom specification to
    EnergySpec format and runs the optimization with full telemetry.

    **Design Pattern**: Imperative Shell - orchestrates the optimization flow,
    handles I/O, telemetry, and error recovery while delegating pure computation
    to the Functional Core.

    Args:
        atom_spec: AtomBasedSpec instance or dictionary specifying the optimization problem.
            Should contain atoms, state, initial_state, and optimization parameters.

    Returns:
        Tuple of (final_state, telemetry_manager):
        - final_state: Final optimized state dictionary with same structure as initial_state
        - telemetry_manager: TelemetryManager instance with access to all recorded data

    Raises:
        ValueError: If atom specification is invalid or optimization fails
        RuntimeError: If step function fails to decrease energy after remediation

    Example:
        ```python
        from computable_flows_shim.api import run_certified
        import jax.numpy as jnp

        # Define optimization problem using atoms
        spec = {
            "atoms": [
                {
                    "type": "quadratic",
                    "params": {"A": jnp.eye(2), "b": jnp.array([1.0, 2.0])},
                    "weight": 1.0,
                    "variable": "x"
                },
                {
                    "type": "l1",
                    "params": {"lambda": 0.1},
                    "weight": 0.5,
                    "variable": "x"
                }
            ],
            "state": {"x": {"shape": [2]}},
            "initial_state": {"x": jnp.array([10.0, 5.0])},
            "num_iterations": 100,
            "step_alpha": 0.1,
            "flow_name": "sparse_reconstruction",
            "run_id": "run_001"
        }

        final_state, telemetry = run_certified(spec)
        print(f"Final solution: {final_state['x']}")
        ```
    """
    # Convert dict to AtomBasedSpec if needed
    if isinstance(atom_spec, dict):
        atom_spec = AtomBasedSpec(**atom_spec)
    elif not isinstance(atom_spec, AtomBasedSpec):
        raise ValueError("atom_spec must be AtomBasedSpec instance or dictionary")

    # Convert atom spec to energy spec
    energy_spec, op_registry = atom_spec_to_energy_spec(atom_spec)

    # Compile the energy functional
    from computable_flows_shim.energy.compile import compile_energy

    compiled = compile_energy(energy_spec, op_registry)

    # Set up telemetry
    os.makedirs(atom_spec.out_dir, exist_ok=True)

    telemetry_manager = TelemetryManager(
        base_path=atom_spec.out_dir,
        flow_name=atom_spec.flow_name,
        run_id=atom_spec.run_id,
    )

    # Run the certified flow
    controller = FlightController()
    final_state = controller.run_certified_flow(
        initial_state=atom_spec.initial_state,
        compiled=compiled,
        num_iterations=atom_spec.num_iterations,
        initial_alpha=atom_spec.step_alpha,
        telemetry_manager=telemetry_manager,
        flow_name=atom_spec.flow_name,
        run_id=atom_spec.run_id,
    )

    # Finalize telemetry
    telemetry_manager.flush()

    # Write manifest
    telemetry_manager.write_run_manifest(
        schema_version=atom_spec.schema_version,
        residual_details={
            "method": "atom_based",
            "norm": "L2",
            "notes": f"{len(atom_spec.atoms)} atoms",
        },
        extra={"atom_types": [atom.type for atom in atom_spec.atoms]},
    )

    return final_state, telemetry_manager


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
