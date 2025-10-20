
# CF Shim Low-Level Readiness Checklist

---
**Readiness Checklist Addition:**
Confirm all modules, configs, and registry patterns use only the official package list (see overview doc above). Remove any legacy or experimental package references not in the official requirements.

This checklist summarizes all requirements for a reliable, reproducible, and AI-friendly CF Shim release. All items are mandatory for release and must be enforced across the codebase, configs, and runtime.

## 1. Reproducibility & Determinism
- Global seeds: one place to set PRNG seeds (JAX PRNGKey), persisted in `manifest.toml`.
- Deterministic numerics: document any non-determinism (e.g., XLA kernels, threading); provide a `--deterministic` flag.
- Shape stability: assert/lock shapes early to avoid silent recompilation.
- Version pins: write an `environment.lock` (exact wheel hashes) for CPU and CUDA stacks.

## 2. Schema/Versioning
Spec version: `spec_version` in manifest; reject unknown major versions.
Telemetry schema version: stored in Parquet metadata; see [`15_schema.md`](./15_schema.md) for column types, units, and enforcement logic (see `telemetry.py`).
Events enum registry: single file listing allowed `event` types + payload keys.

## 3. Validation & Guardrails
- Spec linter: hard errors for missing ops/terms, wrong manifolds, unstable frames; warnings for scale mismatches.
- NaN/Inf sentry: after each primitive, scan for non-finite values; add a `WARN` + store a mini “nan core dump”.
- Unit/scale normalization: automatic RMS/MAD scaling + printed table of effective weights.

## 4. Checkpointing & Resume
- Atomic checkpoints (write temp → rename) with manifest pointer to “latest good”.
- Resume API: must restore state arrays, tuner state, phase/iter counters, RNG keys.
- Rollbacks: auto-rollback to previous GREEN checkpoint on certificate violation.

## 5. Performance Hygiene
- Warmup & cache: explicit JIT warmup; persist XLA cache path per run (optional).
- Profiler hook: context manager for JAX profiler; write trace to `artifacts/profile/`.
- Batch eval for tuner: `vmap` short inner runs over candidate λ’s.

## 6. Failure Modes & Exits
- Graceful cancel (SIGINT): flush telemetry, save checkpoint, emit `CANCELLED` event.
- Budget guards: max wall-time and max iters; hit → `TIMEOUT` event + clean exit.
- Certificate failure policy: configurable “fail-closed” vs “degrade”.

## 7. UX & Observability
- HUD: stable columns & colors; `--watch` mode tails the latest run.
- Run card (machine-readable): after completion, write `run_card.json` (short summary AI can ingest).
- Tiny HTML report: single self-contained page (plots via embedded base64, no server).

## 8. Library Ergonomics
- One-call happy path: `cf.run(spec, init_state)` does RED→AMBER→GREEN, saves capsule, returns RunResult.
- Composable pieces: can call `compile → run_certified → tuner` directly.
- Registries: `op_registry`, `prox_registry`, `transform_registry`, `manifold_registry` with a consistent `register()` decorator.

## 9. Testing Strategy
- Unit tests: primitives, prox, manifolds, FDA (golden values).
- Integration tests: toy quadratic + ℓ1, RED/AMBER/GREEN transitions, resume, telemetry schema integrity.
- Golden notebooks for flagship examples (CPU-only).

## 10. Security & Privacy
- PII/secret scrub: guarantee no raw data leaks into logs (telemetry keeps metrics only).
- Artifact redaction: allow disabling waveform/bitmap dumps.

## 11. Documentation
 Concepts: four primitives, certificates, Gap Dial, frames vs unitary.
 How-to: “add a new prox / transform / manifold”.
 Cookbook: RF, HVAC(15), iFlow(FFT), Hypothesis(UCRC) — each with exact spec + expected certificate values.
 Troubleshooting: why AMBER won’t turn GREEN (common causes, remedies).
 For more details, see `11_naming_and_layout.md`, `14_contributor_guide.md`, and `cf_low_level_readiness_checklist.md`.
## 12. Backward-Compatible Evolutions
- Deprecation policy: warn for one minor version before removing fields/columns.
- Feature flags: new tuner policies, per-scale λ_j, etc., toggleable via manifest.

## 13. AI-Friendliness
- Stable field names in telemetry (snake_case, no renames).
- Short “state machine transcript”: events form a coherent narrative (PHASE_ENTER → CERT_PASS → TUNER_MOVE…).
- Minimal JSON endpoints: `run_card.json`, `latest_event.json` for agent polling.

## 14. Extensibility
- Frame types: `unitary | tight(c) | general` with validation test.
- Manifold slots: per-slot adapters; doc per-slot combination rules.
- Transform families: chirplets, graph wavelets with example constructors.

## 15. Release & Distribution

## 16. Math Proof Obligations & Certification

All mathematical and physical assumptions in primitives, transforms, and flows must be explicitly stated, checked, and documented. Proof obligations are enforced as follows:

- **Analytic/Theoretical:** Each primitive lists analytic assumptions in code docstrings and/or docs. Closed-form properties (convexity, monotonicity, etc.) are cited or derived in `docs/proofs/`.
- **Symbolic/Unit Tests:** Algebraic identities (e.g., A⁻¹A ≈ I, proj(proj(x)) ≈ proj(x)) are checked as property tests with tolerances. Residuals and condition numbers are recorded in telemetry.
- **Numerical Certificates:** Quantities like Lipschitz constant β, diagonal dominance η_dd, spectral gap γ, and Lyapunov decrease are estimated numerically (power/Lanczos/Gershgorin) and logged as certificates. Failures downgrade run to AMBER.
- **Conservation/Invariants:** Physical invariants (energy, mass, etc.) are defined as callables and checked for drift < tolerance during runs. Regression tests inject perturbations and ensure invariants recover.

All proof checks are automated in CI:
| Stage        | Checks                             | Pass condition          | Tooling            |
| ------------ | ---------------------------------- | ----------------------- | ------------------ |
| **Lint/RED** | symbolic / algebraic identities    | all residuals < 1e-8    | pytest-property    |
| **Amber**    | numerical certificates             | η_dd < 0.9, γ > 0.1     | JAX/NumPy/SciPy    |
| **Green**    | invariants + Lyapunov monotonicity | no drift beyond tol     | JAX runtime        |
| **Docs**     | analytic proof references exist    | linked .md file present | Markdown/Sphinx    |

Governance checklist:
- Each primitive lists analytic assumptions in code/doc.
- Each assumption has at least one verification path (analytic, symbolic, numeric, or invariant).
- CI runs all numeric certificates on canonical cases.
- Failing any proof obligation downgrades run to AMBER.
- All proofs/checks are versioned with hash of corresponding operator code.

See `mathematical proof obligations spec.md` for full details and implementation patterns.
## 16. Continuous Integration & Automation

Ensure all checklist items are enforced by CI:
- Run `pytest --strict-markers`
- Run `cf lint` (spec linter)
- Run `cf schema-validate`
- Run `cf readiness-check`

Script these checks into `tests/test_readiness.py` to assert that all readiness sections are programmatically verifiable and release-ready.
