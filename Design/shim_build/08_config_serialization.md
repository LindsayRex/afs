
# Configuration & Serialization

---
**Config/Serialization Requirements:**
All config manifests must use TOML (via tomllib/tomli) and be compatible with the official package list. Telemetry and artifact serialization must use PyArrow (Parquet) and DuckDB. See the maintained requirements in the overview doc above.

## Config File Formats
- **Python DSL:** The only user-editable config/spec format. All new configs/specs must be written in Python, not YAML.
- **TOML, JSON, Parquet:** Used only for auto-generated metadata, telemetry, and agent/UX endpoints (not for user editing).
- **YAML:** Deprecated for configs/specs and should not be used for new projects.

## Global Dtype and Floating-Point Enforcement
- All configs/specs must declare a global dtype (float32 or float64) and propagate it to all arrays and computations.
- JAX-specific: set dtype everywhere, avoid silent up/downcasting, and validate dtype consistency in all serialization and telemetry.

## Loading & Validation
- Use io_config.py to load and validate configs
- Pydantic models can enforce schema

## Serialization
- Save/load compiled specs, states, and tuner reports
- Use JSON or MsgPack for serialization

## API Example
```python
from computable_flows_shim.serialization import save_run, load_run
save_run("runs/hvac_2025_10_18", spec, init_state, final_state, tuner_result)
spec2, init2, final2, report = load_run("runs/hvac_2025_10_18")
```

## Engineering Notes
- All configs compile to the same IR
- Serialization enables reproducibility and auditability

## Transform and Manifold metadata in configs

Include transform frame info and manifold declaration in your Python DSL specs. Example snippet:

```python
transforms = {
    'W': dict(type='wavelet', name='db4', levels=5, frame='unitary')
}
state = {
    'pose': dict(shape=[N,4,4], manifold='SE3')
}
```

## Flight Recorder Manifest & Capsule
Each run writes a minimal manifest (`manifest.toml`) with identity, versions, shapes, frame type, gates, budgets, seeds, and `schema_version`.
Telemetry and events are stored in Parquet tables (`telemetry.parquet`, `events.parquet`) with strict schemas for all key metrics and state transitions; see `SCHEMA.md` for the canonical column list, types, and the `schema_version` contract.
Checkpoints, artifacts, and logs are included in the run capsule for reproducibility and auditability.

Manifest additions (compile-time):
- `lens_name`: selected transform
- `unit_normalization_table`: per-term RMS/MAD used to seed weights
- `invariants_present`: boolean indicating whether invariants are declared and checkers exist

Pareto trials and fronts are stored in a dedicated sidecar (`runs/<run_id>/pareto/`) as described in `12_pareto_knob_surface.md`.
Optionally, you may use the Tensor Logic front-end (see `13_tensor_logic_frontend.md`) to specify problems as tensor-equation graphs, which are compiled to the same EnergySpec and manifest format.

### Error/Status Enum
Use the standardized event enum values (see runtime doc or `SCHEMA.md`): SPEC_LINT_FAIL, CERT_FAIL, CERT_PASS, TUNER_MOVE_REJECTED, ROLLBACK, TIMEOUT, CANCELLED, RUN_STARTED, RUN_FINISHED.

### Privacy note
Telemetry and artifacts must avoid storing raw/PII data. The `redact_artifacts` manifest flag can be used to suppress large or sensitive artifacts. Only metrics, schema, and events are required for reproducibility and AI parsing.

## Controller Phases in Serialization
Phase stamps (RED/AMBER/GREEN) and certificate results are recorded in events and telemetry for every run.
All config and state transitions are logged for inspection by CLI, notebook, or AI agent.
```

Validation guidance:
- Validate `frame` ∈ {unitary, tight, general} and if `tight`, require `c` numeric.
- Validate `manifold` against available adapters; raise a clear error if not found.

Serialization:
- Save `compiled.wavelet` metadata and `manifolds` declared to the run manifest for re-run reproducibility.


See runtime and example docs for usage patterns.

## Registries & Requirements
See `11_naming_and_layout.md` for registry patterns: `transform_registry`, `manifold_registry`, `op_registry`.

### IO & Telemetry Stack (locked)
We recommend a single, consistent IO stack for telemetry and offline analysis to avoid variability and package creep. The recommended minimal stack is:

- PyArrow (Parquet, Arrow buffers)
- DuckDB (fast in-process analytics and indexing)
- Polars (optional - high-performance DataFrame layer built on Arrow; use for offline dashboards and heavy ETL)

Example pinned versions (suggested; pin concretely in `pyproject.toml` or `requirements.txt`):

- pyarrow==12.0.0
- duckdb==1.7.2
- polars==0.19.10  # optional, for offline analysis and dashboards

Notes:
- Use PyArrow for Parquet serialization and canonical Arrow buffers in telemetry.
- Use DuckDB for interactive querying of the run capsule (Parquet) and for CI checks.
- Polars is optional; it provides convenient, fast manipulation of Arrow tables and integrates zero-copy with PyArrow/DuckDB.
- Avoid introducing additional IO libraries unless a strong justification is provided and approved in a design doc.