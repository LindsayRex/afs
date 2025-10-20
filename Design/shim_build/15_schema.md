# Telemetry & Events Schema

This document specifies the columnar telemetry schema, events schema, and versioning for run capsules. It is referenced by the runtime and serialization docs.

schema_version: 2

Telemetry table (`telemetry.parquet`) columns (types, units):
- run_id: string
- phase: string ("RED"|"AMBER"|"GREEN")
- iter: int
- trial_id: string (for tuner trials)
- t_wall_ms: float (ms)
- alpha: float (step size)
- lambda: float (global λ)
- lambda_j: json/dict (per-scale λ_j, stored as JSON string)
- E: float (energy value)
- grad_norm: float (||∇f||)
- eta_dd: float (diagonal dominance metric)
- gamma: float (spectral gap)
- sparsity_wx: float (fraction)
- metric_ber: float (example metric; unitless)
- warnings: string (comma-separated)
- notes: string (free text)
 - invariant_drift_max: float (max |ΔI| across invariants)
 - phi_residual: float (physics residual norm)
 - lens_name: string (selected transform name)
 - level_active_max: int (current finest active level)
 - sparsity_mode: string ("l1"|"group_l1"|"tree")
 - flow_family: string ("gradient"|"preconditioned"|"proximal"|"hamiltonian_damped")

Events table (`events.parquet`) columns:
- run_id: string
- t_wall_ms: float
- event: string (enum - see events list)
- payload: json (arbitrary event payload as JSON)

Events enum (strings):
- SPEC_LINT_FAIL
- CERT_FAIL
- CERT_PASS
- TUNER_MOVE_REJECTED
- ROLLBACK
- TIMEOUT
- CANCELLED
- RUN_STARTED
- RUN_FINISHED
 - LENS_SELECTED
 - SCALE_ACTIVATED

Versioning & compatibility:
- Increment `schema_version` on breaking changes to columns or types.
- Include `schema_version` in Parquet metadata and `manifest.toml`.

Notes:
- `lambda_j` is stored as a JSON string or Parquet nested type for cross-language compatibility.
- All numeric columns are recorded as floats to avoid precision issues across backends.

This file should be kept in sync with any code that writes telemetry or events.
