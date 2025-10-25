# AFS Shim API Reference

This document provides a comprehensive reference for the AFS Shim API functions. All APIs follow the **Functional Core/Imperative Shell** design pattern:

- **Functional Core**: Pure mathematical computations (energy functions, certificates, optimization steps)
- **Imperative Shell**: Orchestration, I/O, telemetry, error handling, user interaction

## Core API Functions

### `run_certified()`

**Location**: `src/computable_flows_shim/api.py`

**Purpose**: Execute certified optimization flows using atom-based specifications.

**Signature**:
```python
def run_certified(
    atom_spec: AtomBasedSpec | dict[str, Any]
) -> tuple[dict[str, Any], TelemetryManager]:
```

**Key Features**:
- High-level atom-based API for optimization problems
- Automatic conversion from atom specifications to EnergySpec
- Full telemetry recording with schema v3 compliance
- RED/AMBER/GREEN phase machine with certificate validation
- Automatic parameter tuning and rollback capability

**Integration Points**:
- Accepts `AtomBasedSpec` instances or dictionaries
- Internally calls `atom_spec_to_energy_spec()` for conversion
- Uses `FlightController.run_certified_flow()` for execution
- Returns final state and `TelemetryManager` instance

**Error Conditions**:
- `ValueError`: Invalid atom specification or optimization failure
- `ValidationError`: Unknown atom types or malformed specifications

**Example**:
```python
from computable_flows_shim.api import run_certified
import jax.numpy as jnp

# Define optimization using atoms
atom_spec = {
    "atoms": [
        {"type": "l1", "params": {"lambda": 0.1}, "weight": 1.0, "variable": "x"}
    ],
    "state": {"x": {"shape": [10]}},
    "initial_state": {"x": jnp.ones(10)},
    "num_iterations": 100,
    "step_alpha": 0.1
}

final_state, telemetry = run_certified(atom_spec)
```

---

### `atom_spec_to_energy_spec()`

**Location**: `src/computable_flows_shim/api.py`

**Purpose**: Convert atom-based specifications to EnergySpec format.

**Signature**:
```python
def atom_spec_to_energy_spec(
    atom_spec: AtomBasedSpec
) -> tuple[Any, dict[str, Any]]:
```

**Key Features**:
- Translates high-level atom specifications to low-level EnergySpec
- Supports quadratic, l1, tikhonov, tv, and wavelet_l1 atoms
- Generates appropriate operator registry for compilation
- Maintains mathematical equivalence between specifications

**Integration Points**:
- Called internally by `run_certified()`
- Returns tuple of (EnergySpec, op_registry)
- Compatible with existing `compile_energy()` function

---

### `AtomBasedSpec` Model

**Location**: `src/computable_flows_shim/api.py`

**Purpose**: Pydantic model for atom-based optimization specifications.

**Key Fields**:
- `atoms`: List of `AtomSpec` objects defining energy terms
- `state`: Dictionary mapping variable names to shape specifications
- `initial_state`: Dictionary with JAX arrays for initialization
- `num_iterations`: Maximum optimization iterations
- `step_alpha`: Initial step size for gradient descent
- `flow_name`: Human-readable flow identifier
- `run_id`: Unique run identifier
- `out_dir`: Telemetry output directory

**Validation**:
- Atom types validated against ATOM_REGISTRY
- State shapes verified as positive integers
- Initial state consistency checked against state specification

---

### `AtomSpec` Model

**Location**: `src/computable_flows_shim/api.py`

**Purpose**: Specification for individual energy function atoms.

**Key Fields**:
- `type`: Atom type (quadratic, l1, tikhonov, tv, wavelet_l1)
- `params`: Type-specific parameters (A, b matrices for quadratic, lambda for regularization)
- `weight`: Positive coefficient for the energy term
- `variable`: Variable name this atom applies to

## Component APIs

### Energy Compilation

**Function**: `compile_energy()`
**Location**: `src/computable_flows_shim/energy/compile.py`

**Purpose**: Compile Python DSL energy specifications into optimized JAX functions.

**Integration**: Called before `run_certified_with_telemetry()`

### Flight Controller

**Class**: `FlightController`
**Location**: `src/computable_flows_shim/controller.py`

**Purpose**: Orchestrate optimization execution with certificate gating and rollback.

**Key Methods**:
- `run_certified_flow()`: Main execution method
- `get_status()`: Get current phase and statistics
- `assess_certificates()`: Evaluate convergence guarantees

### Telemetry System

**Class**: `TelemetryManager`
**Location**: `src/computable_flows_shim/telemetry/telemetry_manager.py`

**Purpose**: Manage telemetry recording and persistence.

**Key Methods**:
- `flight_recorder`: Access to `FlightRecorder` instance
- `write_run_manifest()`: Write run metadata
- `flush()`: Ensure all data is written to disk

---

## Data Flow Patterns

### Standard Optimization Flow

```
EnergySpec + Op Registry → compile_energy() → CompiledEnergy
                                      ↓
Initial State → run_certified_with_telemetry() → Final State + Telemetry
                                      ↓
FlightController (RED→AMBER→GREEN phases) + GapDial Tuner + Checkpoints
```

### Atom-Based Optimization Flow

```
AtomBasedSpec → run_certified() → atom_spec_to_energy_spec() → EnergySpec + Op Registry
                                      ↓
compile_energy() → CompiledEnergy → FlightController → Final State + Telemetry
                                      ↓
RED→AMBER→GREEN phases + TelemetryManager + Manifest Writer
```

### Telemetry Data Flow

```
run_certified() or run_certified_with_telemetry()
    ↓
TelemetryManager
    ├── FlightRecorder (telemetry.parquet, events.parquet)
    │   └── SCALE_ACTIVATED events for multiscale processing
    └── write_run_manifest() (manifest.toml)
```

---

## Design Pattern Compliance

All API functions follow **Functional Core/Imperative Shell**:

### Imperative Shell (API Layer)
- Input validation and type checking
- Resource management (files, directories)
- Error handling and user feedback
- Orchestration of components
- Side effects (I/O, logging)

### Functional Core (Computation)
- Pure mathematical functions
- No side effects
- Deterministic behavior
- JAX-compatible operations
- Certificate computations

---

## Error Handling Patterns

### Certificate Failures
- Automatic remediation (alpha reduction)
- Rollback to last good checkpoint
- Clear error messages with diagnostic info

### Step Failures
- Automatic alpha reduction
- Multiple remediation attempts
- Detailed telemetry of failure conditions

### I/O Errors
- Atomic file operations
- Graceful degradation
- Clear error messages

---

## Testing Patterns

### API Contract Tests
- Parameter validation for EnergySpec and AtomBasedSpec
- Return type verification and telemetry schema compliance
- Error condition handling (certificate failures, invalid atoms)
- Integration with FlightController and TelemetryManager

### Atom-Based API Tests
- Atom specification validation and conversion
- End-to-end optimization with atom-based specs
- Error handling for unknown atom types
- Mathematical correctness of atom-to-EnergySpec translation

### End-to-End Tests
- Complete optimization flows with telemetry
- Multiscale processing with SCALE_ACTIVATED events
- Manifest correctness and schema validation
- Phase transitions (RED→AMBER→GREEN)

---

## Future API Extensions

### Planned Functions
- `compile_atom_based_energy()`: Direct atom compilation (superseded by `run_certified()`)
- `resume_optimization()`: Checkpoint resumption
- `analyze_telemetry()`: Post-run analysis tools

### Extension Points
- Custom certificate functions
- Plugin tuner implementations
- Alternative telemetry backends
- Custom energy term types
