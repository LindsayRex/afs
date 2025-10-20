
# Runtime Engine

---
**Runtime Requirements:**
All runtime engine modules must use JAX, jaxlib, and Optax for compute and optimization. Checkpointing should use orbax-checkpoint if advanced features are needed. Telemetry and storage must use DuckDB and PyArrow for all artifact and event logging. See the canonical package list in the overview doc above.

## Purpose
Executes the compiled flow using the four primitives, with hooks for FDA and auto-tuning.

## Execution Flow
1. Build f, g, and W from the spec
2. JIT-compile the inner step:
   - F_Dis → F_Multi → F_Proj → F_Multi⁻¹ (± F_Con)
   - Primitive implementations are manifold-aware: F_Dis and F_Proj will call per-slot manifold adapters (exp/log/retract) when slots declare non-Euclidean manifolds.
3. FDA checks and logs certificates before/during run
4. Tuner adjusts λ, η as needed
3.5 Multiscale schedule & flow policy
   - The runtime accepts a `FlowPolicy` and `MultiscaleSchedule` and will select primitive ordering and activation rules accordingly.
   - Events emitted: `SCALE_ACTIVATED(level)` and `FLOW_POLICY_APPLIED`.
5. Results and logs are serialized

## API Example
```python
from computable_flows_shim.api import run_flow_step
x_next = run_flow_step(x, compiled, eta, W)
```

Runtime extension example for policy/scale:

```python
from computable_flows_shim.controller import run_certified
run_certified(spec, compiled, cert_profile=my_cert, multiscale=my_schedule, flow=my_flow)
```

For integration of certification profiles, multiscale schedules, and GapDial policies, see `11a_fda_hooks.md` for full API wiring and runtime defaults.

## Engineering Notes
- Only the four primitives ever run in the inner loop
- FDA and tuner are called as hooks, not hardwired
- Supports JAX jit for speed

## Runtime responsibilities for manifolds and frames
- The runtime must provide `manifolds: Dict[str, Manifold]` and `transforms: Dict[str, TransformOp]` to the compiled flow.
- FDA hooks receive `L_apply` that is already chart-aware (linearization in tangent coordinates) for manifold slots.

---

See config and serialization docs for I/O details.

## Flight Recorder Capsule
Each run writes a self-contained capsule: `manifest.toml`, `telemetry.parquet`, `events.parquet`, checkpoints, artifacts, logs.
Telemetry: columnar Parquet, append-only, instant DuckDB queries, AI/CLI/Notebook readable.
Events: sparse state transitions, gate changes, tuner moves, certificate passes/fails.

### Error/Status Enum
Standardized events (also recorded in `events.parquet`) should use the following enum values:
- SPEC_LINT_FAIL
- CERT_FAIL
- CERT_PASS
- TUNER_MOVE_REJECTED
- ROLLBACK
- TIMEOUT
- CANCELLED
- RUN_STARTED
- RUN_FINISHED

### Privacy & Telemetry
- Telemetry must not include raw data or PII; only store metrics, aggregated statistics, and small derived artifacts.
- Use `manifest.toml` flag `redact_artifacts = true` to disable storing large artifacts. When enabled, store only summary statistics and minimal provenance metadata.

## Parquet Schema
Telemetry and events schema are defined in `SCHEMA.md`. Include `schema_version` in Parquet metadata and in `manifest.toml` to ensure forward-compatible parsing.

## Manifest
Small TOML: identity, versions, shapes, frame type, gates, budgets, seeds.

## Controller Phases
RED: spec/units/ops invalid; AMBER: spec OK, not certified; GREEN: certified, tuner allowed.
Phase 0: lint/normalize; Phase 1: certificate feasibility; Phase 2: guarded tuning; Phase 3: polish.
Hard gates: no tuning until certified; all moves re-checked; fail-closed on certificate violation.

### Controller Gate Conditions (formal)
- Feasibility gate: require η_dd(L_W) <= η_max **AND** γ(L_W) >= γ_min. Failure policy: attempt auto-remediation (widen λ search or reduce η) up to M attempts; otherwise transition to RED and abort.
- Tuner commit gate: only commit a tuner move if certificates remain within thresholds after a short validation run (rollback on failure).
- Budget guard: enforce per-run budgets (time, trials, wall-time). Exceeding budget emits `TIMEOUT` and halts tuning.

### Runtime Rollback & Checkpoint Policy
- Maintain a last-good-GREEN checkpoint. On certificate regression after a tuner move, automatically rollback to the last-good checkpoint, log `ROLLBACK` event, and mark the move as rejected.
- Limit automatic rollbacks per-run to avoid oscillation (default max 3 rollbacks).

## Global Dtype and Floating-Point Enforcement
- The runtime must enforce a global dtype (float32 or float64) for all arrays and computations, and propagate it to all primitives and modules.
- JAX-specific: set dtype everywhere, avoid silent up/downcasting, and validate dtype consistency in all serialization and telemetry.
- All run capsules must record dtype in manifest and telemetry metadata for reproducibility.

## Checkpointing, Resume, and Determinism
- Checkpoints must be atomic (write temp → rename) and manifest must point to the latest good checkpoint.
- Resume API must restore state arrays, tuner state, phase/iter counters, and RNG keys.
- All runs must be reproducible: set global seeds, document any non-determinism, and provide a `--deterministic` flag for JAX where possible.

### Resume API (example)
Provide a concise resume API that restores runtime and tuner state:

```python
from computable_flows_shim.serialization import load_run, resume_run

# load the manifest and latest checkpoint
manifest = load_run_manifest(run_dir)
# resume_run restores: state arrays, tuner state, phase, iter, rng_key
resume_state = resume_run(run_dir)

# resume_state: {
#   'state': {...},
#   'tuner_state': {...},
#   'phase': 'GREEN',
#   'iter': 123,
#   'rng_key': <jax.random.PRNGKey>
# }

``` 

The resume API must validate the manifest `schema_version` and ensure dtype consistency before restoring.