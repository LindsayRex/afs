
# Energy Specification & Compilation

---
**Requirements:**
All energy specs must be implemented using JAX and compatible with the official package list (see overview doc above). Use JAX, jaxlib, Optax, and jaxlie for all core computations and manifold operations.

## Declarative Energy Specs
Energies are specified as data in **Python DSL only** (YAML and JSON are deprecated for configs/specs).
Each term: ψ(A(state)), where A is built from registered Ops.
Smooth terms → f (for F_Dis); nonsmooth terms → g (for F_Proj).

## Example Python DSL Spec
```python
terms = [
    dict(type='quadratic', op='H', target='y', weight=1.0),
    dict(type='l1', op='W', weight=0.1),
]
```

## Compilation Flow
1. Parse spec into EnergySpec dataclasses
2. Partition terms into smooth (f) and prox (g)
3. Compile f, grad_f, g_prox, and multiscale W
4. JIT-compile for fast execution

## API Example
```python
from computable_flows_shim.api import compile_energy
compiled = compile_energy(spec, op_registry={"H": H, "W": W})
```

-
## Energy Construction Template

The canonical decomposition for energies is:

\[
\mathcal{E}(x;\mathbf{w}) = \sum_i w_i \mathcal{E}_i^{\text{data}}(x) + \sum_j w_j^{\text{phys}} \mathcal{E}_j^{\text{physics}}(x) + \mathcal{R}(x)\
\]

Spec authors must declare terms under `data_term`, `physics_term`, and `regularizer` with explicit units to enable normalization and comparison across domains.

### Unit normalization
At compile-time, compute energy-based normalization per term and store it in a `unit_normalization_table` inside the `CompileReport`. This table is used to seed initial weights and to make slider/policy knobs meaningful across objectives. Normalization is computed by evaluating each term's energy contribution on sample data, maintaining the pure energy-based paradigm without introducing statistical branching logic.

Example `CompileReport` fields:

- `lens_name`
- `frame_type` and `c`
- `unit_normalization_table` (per-term energy-based normalization)
- `invariants_present` (bool)

All compile-time normalization info is written to the manifest for reproducibility.
- Only Python DSL is supported for new configs/specs. TOML, JSON, and Parquet are used only for auto-generated metadata, telemetry, and agent/UX endpoints.
- All configs/specs must declare and propagate a global dtype (float32 or float64) to all arrays and computations. JAX-specific: set dtype everywhere, avoid silent up/downcasting, and validate dtype consistency.
- Pydantic models can be used for strict validation.
- All configs compile to the same IR.


See the primitives and runtime docs for execution details.

## Telemetry & Controller Integration
- All energy compilation steps, certificate checks, and phase transitions are logged via Flight Recorder events and telemetry rows.
- Controller phases (RED/AMBER/GREEN) enforce build checklist: energy terms are linted and normalized before run; all transitions and results are recorded.
- API: energy compiler is called by the controller, which manages phase logic and telemetry recording.