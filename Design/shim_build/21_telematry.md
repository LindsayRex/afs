## Telemetry & Run Capsule Specification

### Directory Structure
- All telemetry and run provenance data lives in `src/telematry_cfs/`.
- Each run creates a subfolder named `fda_run_{id}` inside `telematry_cfs`.
- Example: `src/telematry_cfs/fda_run_20251020_153045/`

### Run ID Scheme
- Run IDs use reverse date-time format: `YYYYMMDD_HHMMSS` (e.g., `20251020_153045`).
- This ensures newest runs sort to the top in file listings.
- Format: `fda_run_{YYYYMMDD_HHMMSS}`

### Run Folder Contents
- Each `fda_run_{id}` folder contains:
	- `manifest.toml` (run metadata)
	- `telemetry.parquet` (main telemetry table)
	- `events.parquet` (events log)
	- Optionally: checkpoints, config snapshots, QA logs

### DuckDB Placement
- DuckDB database file (`duckdb.db`) should live next to the run folders in `src/telematry_cfs/`.
- This allows efficient querying across all runs without deep nesting.
- Example: `src/telematry_cfs/duckdb.db`

### Parquet File Placement
- All Parquet files (`telemetry.parquet`, `events.parquet`) live inside their respective `fda_run_{id}` folders.

### Naming & User Experience
- "Capsule" is internal; user-facing term is "run" (or "experiment" if preferred).
- Users (researchers, auto-tuners, meta-tuners, flow software, AI agents) access telemetry via `telematry_cfs`, not by diving into Shim internals.
- No deep nesting; everything for a run is in one folder, easy to find and query.

### Example Layout
```
src/telematry_cfs/
	duckdb.db
	fda_run_20251020_153045/
		manifest.toml
		telemetry.parquet
		events.parquet
		checkpoint_0001.pkl
	fda_run_20251020_160000/
		...
```

### Design Principles
- Telemetry is a first-class citizen: discoverable, queryable, versioned.
- No "chaff"—only essential files per run.
- All run folders use the same structure and naming convention.

---
This specification should be kept in sync with implementation and referenced by all code that writes telemetry, events, or manifests.

## User Stories and Write Patterns

### Actors
- Researcher / Developer: iterates quickly on flow specs; runs many short experiments.
- Tuner / Meta-tuner: runs many automated trials with parameter sweeps.
- Production Runner: runs validated flows in production-like configuration for benchmarking.
- AI Agent: consumes telemetry for automated tuning and reporting.

### Story: Developer Iterates on a Flow
- As a researcher, I add a new flow spec named `deconv_wavelet_v1`.
- The system creates (or locates) a per-flow folder at `src/telematry_cfs/flows/deconv_wavelet_v1/`.
- During development, a single parquet file `dev_telemetry.parquet` collects many `run` rows (iter, t_wall_ms, settings, etc.).
- Each invocation appends rows to the same `dev_telemetry.parquet` for that flow to avoid file proliferation.
- Periodically, the researcher can consolidate `dev_telemetry.parquet` into run-level `fda_run_{id}` folders for archival.

### Story: Tuner/Meta-Tuner Trials
- As a tuner, I create many short runs grouped by a `trial_id` and `flow_name`.
- The writer runs in `batch/append` mode: each trial appends into a per-flow `tuner_telemetry.parquet` under `flows/{flow_name}/` with a `trial_id` column to group results.

### Story: Production Runs
- Production runs use immutable per-run folders `fda_run_{YYYYMMDD_HHMMSS}` (as defined above) and write atomically: write to temp file then move into place.

### Aggregation Strategy
- Per-flow aggregation directory: `src/telematry_cfs/flows/{flow_name}/`.
	- Contains: `dev_telemetry.parquet`, `tuner_telemetry.parquet`, and `manifest.toml` documenting schema and flow spec metadata.
- Run isolation directory: `src/telematry_cfs/runs/fda_run_{id}/` for immutable archival runs.

## DuckDB Consolidation and Querying
- The global `duckdb.db` sits at `src/telematry_cfs/duckdb.db` and is the recommended query layer for cross-flow analysis.
- Recommendation: maintain a single telemetry table and events table inside DuckDB that point to Parquet files using DuckDB's parquet_fdw (or built-in parquet read functionality).
- Consolidation process (periodic):
	1. Scan `flows/*/*.parquet` and `runs/*/telemetry.parquet` for new files.
	2. Attach parquet files to DuckDB tables as external tables.
	3. Optionally, vacuum or export a single consolidated `telemetry` Parquet for efficient join queries.

## Implementation Notes
- Writers should support two modes:
	- append (for dev and tuner): append rows to per-flow parquet with small-batch writes.
	- atomic run (for production): write to temp and move into `runs/` folder.
- Use Parquet row-group tuning for append-heavy workloads (smaller row groups) and column scan efficiency for queries (larger row groups) at consolidation time.
- Always include `schema_version` in the manifest and Parquet metadata for forward/backward compatibility.

---
Next steps: prototype a small writer API under `src/telematry_cfs/` providing append and consolidate helpers.

## Telemetry Field Audit & Minimal Contracts

We must capture telemetry that is sufficient for certification, tuning, debugging, and reproducibility. Below is a recommended, layered set of fields (minimal/core, recommended, diagnostic/optional) plus sampling and retention policies.

### Minimal / Core Fields (must always be recorded)
- `run_id` (string): unique run identifier (see Run ID scheme).
- `flow_name` (string): the user-visible flow/spec name.
- `phase` (string): one of RED/AMBER/GREEN.
- `iter` (int): iteration number.
- `t_wall_ms` (float): wall-clock time in milliseconds since run start.
- `E` (float): energy value.
- `grad_norm` (float): norm of ∇f (for Lyapunov checks).
- `eta_dd` (float): diagonal dominance metric (FDA).
- `gamma` (float): spectral gap (FDA).
- `alpha` (float): step-size / learning rate used for the iteration.

### Promotion of Residuals & Invariants to Core
Based on the FDA engineering guidance, residuals and invariant drift metrics are essential for certification and must be treated as core telemetry, not optional diagnostics. The following fields are therefore elevated to core status and must be present for every run sample (subject to the sampling policy):
- `phi_residual` (float): physics residual norm — measures violation of PDE/constraint residuals. Essential for certification.
- `invariant_drift_max` (float): maximum absolute drift across declared invariants since last sample.

Rationale: These values are required for the Certification step in the FDA recipe (see `Flow-Dynamics Ananalysis (FDA).md`) — they directly relate to energy descent, conservation checks, and physical correctness. They must be present for both development and production telemetry writes.

### Recommended Fields (useful for tuning and QA)
- `trial_id` (string): for tuner trials and groupings.
- `lambda` (float): global sparsity parameter.
- `lambda_j` (json): per-scale sparsity parameters.
- `sparsity_wx` (float): sparsity fraction in W-space.
- `level_active_max` (int): current finest active scale.
- `sparsity_mode` (string): 'l1'|'group_l1'|'tree'.
- `flow_family` (string): 'gradient'|'preconditioned'|'proximal'|'hamiltonian_damped'.
- `lens_name` (string): selected transform name.

### Diagnostic / Optional Fields (high-cardinality; sample or log-on-event)
- `phi_residual` (float): physics residual norm.
- `invariant_drift_max` (float): max invariant drift since last sample.
- `metric_ber` (float): domain-specific metric.
- `warnings` (string): comma-separated warnings.
- `notes` (string): free-form notes captured on events.

### Events (sparse, time-stamped) - see `events.parquet`
- Emitted for: SPEC_LINT_FAIL, CERT_FAIL, CERT_PASS, TUNER_MOVE_REJECTED, ROLLBACK, TIMEOUT, CANCELLED, RUN_STARTED, RUN_FINISHED, LENS_SELECTED, SCALE_ACTIVATED.
- Each event row: `run_id`, `t_wall_ms`, `event`, `payload` (json).

### Sampling Strategy
- For dev/tuner append workloads: write every iteration up to a max N (configurable, default N=1000), then sample every k iterations (default k=10). Include `iter` to reconstruct full timeline.
- For production runs: record all iterations for reproducibility but allow a budgeted cap (e.g., 1e6 rows) enforced by `manifest.toml` fields.
- Diagnostic fields: sample at lower frequency or on events (e.g., invariant drift above threshold).

### Retention & Consolidation
- Dev aggregated Parquet: keep rolling 90-day retention by default; consolidated runs moved to `runs/` are archived indefinitely unless manifest flags indicate shorter retention.
- DuckDB: used as an index for fast cross-run queries; maintain periodic compaction/export of consolidated Parquet for efficient queries.

### Schema & Versioning Contract
- The schema in `15_schema.md` is the canonical source. Any change to minimal/core fields must bump `schema_version` and be recorded in `manifest.toml`.
- Writers must embed `schema_version` into Parquet metadata when writing.

### Required Field Contracts
- Core fields must be present (non-null) in entries used for certification or tuning.
- `phi_residual` and `invariant_drift_max` must have units recorded in `manifest.toml` (e.g., L2 norm units or normalized units) and the manifest must declare how they were computed (e.g., operator used, discretization, norm type).

---
Update note: Increase `schema_version` in `15_schema.md` if these core fields are added to the canonical schema. We will bump to schema_version: 3 when we update `15_schema.md` and implement writer updates.

---
I will now mark the audit task completed and move to the next task (update schema contracts).
