# AFS Shim API Reference

This document provides a comprehensive reference for the AFS Shim API functions. All APIs follow the **Functional Core/Imperative Shell** design pattern:

- **Functional Core**: Pure mathematical computations (energy functions, certificates, optimization steps)
- **Imperative Shell**: Orchestration, I/O, telemetry, error handling, user interaction

## Core API Functions

### `run_certified_with_telemetry()`

**Location**: `src/computable_flows_shim/api.py`

**Purpose**: Execute certified optimization flows with full telemetry and observability.

**Signature**:
```python
def run_certified_with_telemetry(
    initial_state: Dict[str, Any],
    compiled: CompiledEnergy,
    num_iterations: int,
    step_alpha: float,
    flow_name: str,
    run_id: str,
    out_dir: str,
    schema_version: int = 3,
    residual_details: Optional[Dict[str, Any]] = None,
    extra_manifest: Optional[Dict[str, Any]] = None
) -> Tuple[Dict[str, Any], TelemetryManager]:
```

**Key Features**:
- RED/AMBER/GREEN phase machine with certificate validation
- Automatic parameter tuning via GapDial tuner
- Rollback capability on certificate violations
- Complete telemetry schema v3 recording
- Atomic file operations for reliability

**Integration Points**:
- Calls `FlightController.run_certified_flow()`
- Uses `TelemetryManager` for observability
- Requires `CompiledEnergy` from `compile_energy()`

**Error Conditions**:
- `ValueError`: Certificate validation fails after remediation
- `RuntimeError`: Step function cannot decrease energy
- `OSError`: Cannot create/write telemetry files

---

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

### Telemetry Data Flow

```
run_certified_with_telemetry()
    ↓
TelemetryManager
    ├── FlightRecorder (telemetry.parquet, events.parquet)
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
- Parameter validation
- Return type verification
- Error condition handling
- Integration with components

### End-to-End Tests
- Complete optimization flows
- Telemetry file generation
- Manifest correctness
- Phase transitions

---

## Future API Extensions

### Planned Functions
- `run_certified()`: Simplified version without telemetry
- `compile_atom_based_energy()`: Atom library integration
- `resume_optimization()`: Checkpoint resumption
- `analyze_telemetry()`: Post-run analysis tools

### Extension Points
- Custom certificate functions
- Plugin tuner implementations
- Alternative telemetry backends
- Custom energy term types