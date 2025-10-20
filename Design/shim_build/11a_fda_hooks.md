# FDA Hooks Addendum

This addendum formalizes the Flow-Dynamics Analysis (FDA) hooks and spec fields to be included in the Shim spec.

It defines: StateSpec invariants, LensPolicy, FlowPolicy, GapDial, MultiscaleSchedule, SparsityPolicy, CertificationProfile, compile report fields, and telemetry columns.

## StateSpec.invariants
Add to spec:

```python
StateSpec(
    shapes={...},
    invariants={
        'conserved': {'mass': lambda s: compute_mass(s)},
        'constraints': {'balance': lambda s: compute_balance(s)},
        'symmetries': ['translation', 'SE3']
    }
)
```

- Lint rule: fail RED if `invariants` declared without checkers.
- Runtime hook: `validate_invariants(state)` runs at Phase 0 and every k iterations; logs `invariant_drift_max`.

## LensPolicy
Add to spec:

```python
LensPolicy(
    candidates=[TransformRef('db4'), TransformRef('haar')],
    probe_metrics=['compressibility', 'reconstruction_error'],
    selection_rule='min_recon_error @ target_sparsity'
)
```

- Builder Mode compressibility probe ranks candidates and emits `LENS_SELECTED` event.

## FlowPolicy
Add to spec:

```python
FlowPolicy(
    family='preconditioned',
    discretization='symplectic',
    preconditioner=OpRef('diag_precond')
)
```

- Runtime chooses primitive variants accordingly.

## GapDial
Add to spec:

```python
GapDial(
    eta_max=0.9,
    gamma_min=1e-6,
    beta_estimator='lanczos',
    step_rule='c/beta',
    per_scale_init={'median_scaling': True}
)
```

- Feasibility sweep parameters: init lambda, doubling cap, per-scale policies.

## MultiscaleSchedule
Add to spec:

```python
MultiscaleSchedule(
    mode='residual_driven',
    levels=5,
    activate_rule='residual>tau'
)
```

- Runtime wrapper will manage coarse-to-fine unlocking and log `SCALE_ACTIVATED(level)` events.

## SparsityPolicy
Add to spec:

```python
SparsityPolicy(
    penalty='l1',
    thresholding='soft',
    adaptive_rule='residual_adaptive'
)
```

- Prox builder reads this and configures prox operators automatically.

## CertificationProfile
Add to spec:

```python
CertificationProfile(
    checks=['lyapunov', 'physics_residual', 'invariant', 'spectral_gap', 'discretization_independence'],
    tolerances={'lyapunov': 1e-6, 'invariant': 1e-9},
    refinement_test={'enable': True, 'scale_factor': 2}
)
```

- Runtime runs these checks at GREEN and at run end; write pass/fail to `run_card.json`.

## CompileReport changes
`compile_energy` returns `CompiledEnergy` and `CompileReport` with:
- `lens_name`
- `frame_type` and `c`
- `unit_normalization_table` (RMS/MAD per term)
- `invariants` list and checkers present flag

## Telemetry & Schema additions
Append to `SCHEMA.md` the following telemetry columns:
- `invariant_drift_max` (float)
- `phi_residual` (float)
- `lens_name` (string)
- `level_active_max` (int)
- `sparsity_mode` (string)
- `flow_family` (string)

Also add `LENS_SELECTED` and `SCALE_ACTIVATED(level)` to the events enum.

## API changes and wiring
- `controller.run_certified` signature includes cert_profile, multiscale, gap_dial, sparsity, flow (backward-compatible defaults).
- `compile_energy(spec)` returns `CompileReport` to be saved into manifest.

## Tests & Builder Mode probes
Add tests for invariants, lens selection determinism, unit normalization, flow family impact, gap dial failure modes, multiscale activation, sparsity adaptive behavior, and certification pass/fail.

---

This addendum is intentionally minimal and non-invasive: all the hooks map to existing runtime behavior and only formalize the spec, telemetry, and compile-time reporting to make FDA steps repeatable, auditable, and automatable.
