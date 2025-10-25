# Gap Dial & Auto-Tuner

## Purpose
Automates the adjustment of sparsity (λ), step size (η), and other parameters to ensure certificates are satisfied and task targets are met.

## Tuning Strategy
- Outer loop sweeps λ (sparsity in W-space), η (step size), and secondary weights
- Uses golden-section or grid search for feasibility
- Monitors certificates (η, γ) and task metrics (e.g., sparsity%, BER)

### Gate & Move Contracts
- Feasibility gate: require η_dd <= η_max and γ >= γ_min before any tuning commits to long runs. Example defaults: η_max = 0.9, γ_min = 1e-6 (tunable per problem).
- If feasibility fails, the tuner should either (a) widen λ search (looser sparsity), (b) reduce step-size η, or (c) abort with `CERT_FAIL` depending on policies.
- Per-trial move caps: limit λ moves to ±20% per trial (configurable, default ±10%). If a proposed move violates local certificate constraints, reject and log `TUNER_MOVE_REJECTED`.
- Termination: stop tuning after N_no_improve trials without certificate or metric improvement (default N=5) or when budget exhausted.
- Rollback semantics: keep a short history of last good GREEN checkpoint; on certificate regression (γ or η violates thresholds), rollback to last good checkpoint and mark trial as rejected.

### GapDial Policy Parameters & Feasibility Sweep

- `GapDial` spec fields:
	- `eta_max`: float (e.g., 0.9)
	- `gamma_min`: float (e.g., 1e-6)
	- `beta_estimator`: 'power' | 'lanczos'
	- `per_scale_seed`: 'median' | 'mad' | callable
	- `lambda_move_cap`: relative cap per trial (default 0.2)

- Feasibility sweep algorithm:
	1. Start with per-scale λ_j seeded from `per_scale_seed`.
	2. Evaluate certificates; if infeasible, widen search by multiplicative factor (e.g., 2x) up to a cap.
	3. If still infeasible after M attempts, mark candidate as `CERT_FAIL` and escalate per policy.

- Per-scale enforcement:
	- Enforce ratio limits between adjacent scales (default [0.5, 2.0]).
	- Re-check certificates after per-scale adjustments; reject moves that reduce certification.

All moves and feasibility steps are logged as events (`TUNER_MOVE_TRY`, `TUNER_MOVE_REJECTED`, `CERT_FAIL`, `ROLLBACK`).

## Frame-aware tuning notes
- When the multiscale transform is a `tight(c)` or `general` frame, adjust prox thresholds and sparsity targets by the frame constant `c` (e.g., effective threshold = tau * c). The tuner must query `compiled.wavelet.frame` and `compiled.wavelet.c`.
- Certificate checks use scaled norms for tight frames to ensure thresholds are consistent across transform types.

### Per-scale λ_j Policy
- Support per-scale λ_j values seeded from robust statistics per band (median or MAD scaling). Default seeding: λ_j = median(|W_j x|) * k (user-configurable k).
- Allowed ratios: enforce λ_j/λ_{j+1} within [0.5, 2.0] by default to prevent extreme per-scale imbalance.
- Re-check certificates (η, γ) after any per-scale change and treat violations as above (rollback or reject move).

## API Example
```python
from computable_flows_shim.tuner import run_with_auto_tuner
result = run_with_auto_tuner(init_state, compiled, spec.compose, spec.auto_tune, metrics_fn=...)
```

## Engineering Notes
- Tuner only adjusts parameters; does not alter flow structure
- Certificates must pass before committing to long runs
- All adjustments are logged and can be serialized


See runtime and FDA docs for integration details.

For multi-objective experiments and automated policy sweeps, see `12_pareto_knob_surface.md` which describes a Pareto Manager that runs certified sweeps over weight vectors and constraints.

## Telemetry & Controller Integration
- All tuning moves, certificate checks, and phase transitions are logged via Flight Recorder events and telemetry rows.
- Controller phases (RED/AMBER/GREEN) enforce build checklist: tuner only runs after certification, with all moves re-checked and recorded.
- API: tuner is called by the controller, which manages phase logic and telemetry recording.
