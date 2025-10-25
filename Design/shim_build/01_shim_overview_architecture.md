
# Shim Overview & Architecture


## Official Package Requirements
All modules and engineering plans in this Shim build use the following official package list. This replaces any legacy or experimental package references. All code, config, and registry patterns must be compatible with these packages.

### Core compute & math
- JAX / jaxlib — array compute, JIT, autodiff, vmap/pmap
- NumPy (reference API)
- SciPy (optional)

### Optimizers / solvers (JAX-native)
- Optax — gradient transforms, schedulers, clipping
- Optimistix (optional)

### Manifolds / Lie groups
- jaxlie — SO(2/3), SE(2/3) with exp/log, adjoint
- Pymanopt (optional)

### Multiscale / transforms
- jaxwt — differentiable wavelets in JAX
- CR-Wavelets (optional)
- S2WAV (optional)
- S2BALL (optional)

### Graphs
- NetworkX — graph construction + Laplacians

### Telemetry, storage, and artifacts
- DuckDB (Python)
- PyArrow — Parquet/Arrow I/O

### Config / packaging / docs / QA
- tomllib (stdlib, Py 3.11+) or tomli (backport)
- pytest — tests
- ruff — fast linter/formatter
- MkDocs Material — docs site theme

### Checkpointing
- orbax-checkpoint (optional)

All code, config, and registry patterns must be compatible with these packages. See each package’s documentation for install details and usage.

## Purpose
A high-level summary of the Computable Flow Shim, its role in the physics-based pipeline, and how it integrates with JAX and other frameworks.

## What is the Shim?
The Shim is a reusable runtime adapter that compiles declarative energy specifications into fast, composable flows using four core primitives:

## File Format Policy
- **Python DSL** is the only user-editable config/spec format. All new configs/specs must be written in Python, not YAML.
- **TOML, JSON, and Parquet** are used only for auto-generated metadata, telemetry, and agent/UX endpoints (not for user editing).
- **YAML** is deprecated for configs/specs and should not be used for new projects.

## Global Dtype and Floating-Point Enforcement
- All primitives, configs, and runtime must explicitly declare and enforce a global dtype (e.g., float32 or float64) for all arrays and computations.
- JAX-specific: set and propagate dtype everywhere; avoid silent up/downcasting. Document and test for dtype consistency.
- All serialization and telemetry must record dtype in metadata for reproducibility.

## Low-Level Readiness Checklist
See `cf_low_level_readiness_checklist.md` for the full reproducibility, validation, checkpointing, performance, UX, and extensibility requirements. All items in the checklist are mandatory for release.

It enables forward–backward splitting, multiscale transforms, and certificate-driven tuning in a unified, physics-consistent framework.

## Architecture Diagram
```
[ Spec (Python DSL) ]
        |
   [ Energy Compiler ]
        |
   [ Shim Runtime ]
   /   |   \
F_Dis F_Proj F_Multi (F_Con)
   \   |   /
   [ FDA, Tuner, Logger ]
        |
   [ JAX Backend ]
```

## Full-Stack Shim Diagram (Front-Ends → Compiler → Controller → Primitives → Telemetry → Pareto)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         COMPUTABLE FLOWS — CF SHIM                           │
└──────────────────────────────────────────────────────────────────────────────┘
         (two ways to describe a problem)                     (optional)
┌──────────────────────────────┐                      ┌────────────────────────┐
│  Python DSL (EnergySpec)     │                      │  Tensor Program IR     │
│  - energy terms f,g          │◄───── translator ───►│  (joins/projections)   │
│  - ops, metrics, invariants  │                      │  tensor DAG + objectives│
└──────────────┬───────────────┘                      └───────────┬────────────┘
               │                                                    │
               │                      Builder Mode                  │
               │      (lint, units/scale, lens probe, spec sanity)  │
               ▼                                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                      ENERGY SPEC COMPILATION (to JAX)                        │
│  - resolve ops (Op registry)        - select multiscale lens (TransformOp)   │
│  - manifold adapters (Manifold)      - split into f (smooth) / g (prox)      │
│  - produce:                                                               │
│      CompiledEnergy{ f_value, f_grad, g_prox, W, L_apply }                  │
└─────────────────────────────────────────────┬────────────────────────────────┘
                                              │
                                              │
                                        (RED/AMBER/GREEN gates)
                          ┌───────────────────┴───────────────────┐
                          │           FLIGHT CONTROLLER           │
                          │  Phase 0: Lint/Normalize (RED→AMBER)  │
                          │  Phase 1: Certificates (AMBER→GREEN)  │
                          │   - estimate β → step α=c/β           │
                          │   - Gap Dial λ feasibility            │
                          │   - warm-start Lyapunov check         │
                          │  Phase 2: Task Tuning (GREEN only)    │
                          │   - small guarded moves (TV, per-band)│
                          │   - re-check η_dd, γ every move       │
                          │  Phase 3: Polish / optional 𝓕_Con     │
                          └───────────────┬───────────────────────┘
                                          │
                                          │ (one iteration body)
                                          ▼
                ┌──────────────────────────────────────────────────────┐
                │                  INNER FLOW LOOP                     │
                │  z ← 𝓕_Dis(z; α)                                     │
                │  u ← 𝓕_Multi⁺(z; W)                                  │
                │  u ← prox_{α g_W}(u)  ≡ 𝓕_Proj in W-space            │
                │  z ← 𝓕_Multi⁻(u; W)                                  │
                │  [optional] z ← 𝓕_Con(z; symplectic subslots)        │
                └───────────────────────┬──────────────────────────────┘
                                        │
                                        │ feeds diagnostics
                                        ▼
      ┌─────────────────────────────────┴─────────────────────────────────┐
      │       FLOW DYNAMIC ANALYSIS (FDA) & CERTIFICATES (in W-space)    │
      │   - η_dd (Gershgorin diag-dom)     - γ (Lanczos/power, spectral) │
      │   - Lyapunov descent check         - invariant/physics residual  │
      └─────────────────────────────────┬─────────────────────────────────┘
                                        │
                                        │ decides accept/backtrack & tuner moves
                                        ▼
                   ┌────────────────────────────────────────────────┐
                   │                 GAP DIAL / TUNER               │
                   │  - choose λ (global or per-scale λ_j)          │
                   │  - honor η_dd<η_max, γ≥γ_min (else reject)     │
                   │  - short-run evaluation of task metrics        │
                   └──────────────────────────┬─────────────────────┘
                                              │
                                              │
     ┌─────────────────────────────────────────┴──────────────────────────────────┐
     │                            FLIGHT RECORDER (telemetry)                     │
     │   runs/<id>/                                                               │
     │     manifest.toml    telemetry.parquet   events.parquet   checkpoints/     │
     │     artifacts/ (plots, report.html)     logs.ndjson                        │
     │   columns: iter, phase, E, ||∇f||, η_dd, γ, λ, λ_j, sparsity_Wx, BER,      │
     │            invariant_drift, lens_name, level_active_max, flow_family…      │
     └──────────────────────────────┬────────────────────────────────────────────┘
                                    │
                          ┌─────────┴─────────┐
                          │                   │
                 ┌────────▼────────┐  ┌──────▼────────────────────┐
                 │  CLI / HUD      │  │   PARETO MANAGER (opt)    │
                 │  cf-run / cert  │  │  - generate weight/ε sets │
                 │  cf-hud / viz   │  │  - run certified sweeps   │
                 └────────┬────────┘  │  - build trials/front *.pq│
                          │           └───────────┬────────────────┘
                          │                       │
                          ▼                       ▼
                ┌────────────────┐       ┌──────────────────────────┐
                │  AI Agent      │       │  Business UI / Sliders  │
                │  (reads *.pq)  │       │  (cost/carbon/comfort)  │
                └────────────────┘       └──────────────────────────┘

               (registries & backends available everywhere as adapters)
   ┌───────────────────────────────────────────────────────────────────────────┐
   │ Registries:  Ops • Transforms(𝓦: unitary/tight/general) • Manifolds • Prox │
   │ Backends:   JAX (now) • Torch (later)                                      │
   └───────────────────────────────────────────────────────────────────────────┘
```

**Legend / Notes:**
- 𝓕_Dis, 𝓕_Proj, 𝓕_Multi, 𝓕_Con are the only runtime primitives invoked inside the loop.
- Certificates gate tuning: RED/AMBER/GREEN is enforced by the controller.
- Flight Recorder is Parquet/DuckDB telemetry + checkpoints; HUD/AI read from it.
- Pareto Manager is optional; it sweeps weight vectors (and ε-constraints) but only keeps GREEN runs.
- Tensor Program IR is an optional front-end dialect; it lowers to your EnergySpec; JAX runtime remains unchanged.

## Module Structure
- `ops/` — Operator adapters
- `energy/` — Energy compiler
- `multi/` — Multiscale transforms
- `primitives/` — Core runtime steps
- `fda/` — Flow Dynamic Analysis
- `tuner/` — Gap Dial & auto-tuner
- `runtime/` — Engine
- `io_config/` — Config loading
- `serialization/` — Save/load runs

## Data Flow
1. User specifies energy and ops (Python DSL)
2. Compiler partitions terms into smooth/prox
3. Shim builds and jit-compiles the flow
4. FDA and tuner monitor certificates and adjust parameters
5. Results are logged, serialized, and available for analysis

## Flight Recorder & Flight Controller Integration

### Flight Recorder (Telemetry)
Records high-rate numeric telemetry and sparse events for every run.
Capsule layout: `manifest.toml`, `telemetry.parquet`, `events.parquet`, checkpoints, artifacts, logs.
Parquet schema: strict, typed columns for all key metrics (phase, iter, energy, grad norm, eta_dd, gamma, sparsity, warnings, notes).
API: `TelemetrySink` protocol, `ParquetFlightRecorder` implementation.
Enables instant querying (DuckDB), AI/CLI/Notebook inspection, and reproducible runs.

### Flight Controller (Automaton)
Implements RED/AMBER/GREEN state machine with hard gates.
Phases: lint/normalize (RED→AMBER), certificate feasibility (AMBER→GREEN), guarded tuning (GREEN), polish (optional).
Enforces build checklist: no tuning until certified, small bounded moves, per-scale dials, Lyapunov/metric checks.
API: `run_certified(...)` skeleton, integrates with primitives, FDA, tuner, and telemetry.

### CLI & Builder Mode
CLI HUD, cert, and viz commands for live gauges, attestation, and dashboards.
Builder Mode: stepwise construction, linting, rehearsal, and promotion to run only if certified.

Builder Mode (rehearsal steps):
1. Add terms to spec (Python DSL) and run linter/normalize.
2. Run scale normalization and unit tests (shape/dtype checks).
3. Run a compressibility probe (short run to estimate transform sparsity and per-scale medians).
4. Run FDA certificate rehearsal (small validation run) to check η, γ; fix or adjust terms if needed.
5. Promote to full run (GREEN) only if all checks pass; otherwise iterate or archive the spec with notes.

Builder Mode probes referenced in the multiscale module:
- Lens probe selects the best transform candidate via compressibility + recon-error.
- Coarse-to-fine rehearsal verifies that multiscale schedules preserve certificates when unlocking scales.

Promotion is gated by the same feasibility rules used by the controller (η_max, γ_min). All rehearsal artifacts are stored in the run capsule for reproducibility.

Note: user-editable specs must be written in Python DSL. YAML is deprecated and should not be used.

---

See subsequent documents for detailed module specifications.

See the Pareto knob surface for multi-objective workflows in `12_pareto_knob_surface.md`.
For mathematicians and formal-methods users, an optional Tensor Logic front-end is available (see `13_tensor_logic_frontend.md`). This adapter lets you describe problems as tensor-equation graphs, which are compiled to the same energy-based runtime.

## Naming & Layout
See `11_naming_and_layout.md` for the canonical file/module naming rules, ports-and-adapters layout, and registry conventions. This file defines package/module names, registries, and the dtype policy.

## Runtime Boundaries & IO Stack

Clear boundaries are enforced to keep the runtime certifiable, JIT-compatible, and device-agnostic.

- Inside the Flow (Shim Runtime, Primitives, FDA, Tuner):
     - Use only JAX arrays (`jnp.ndarray`) and JAX-native libraries (`jax`, `jax.numpy`, `jax.lax`, `jaxwt` for transforms).
     - All transforms, wavelets, sparsity/gap computations, certificate checks, and scale unlocks must be implemented using JAX primitives. No NumPy/SciPy or other non-JAX operations are allowed inside the inner loop or in certificate paths.

- At the Boundary (IO, Telemetry, Pre/Post-processing):
     - Use PyArrow, DuckDB, and optionally Polars for fast IO and telemetry aggregation. These tools live outside the runtime loop and must convert to JAX arrays before entering the flow (use `jnp.asarray` / `jax.device_put`).
     - Telemetry must be written in Parquet (PyArrow) and indexed by DuckDB. Polars may be used as an optional convenience layer for offline analysis and dashboarding but is not required.

- Utilities and Tools (Offline Analysis, Plotting):
     - Pandas, NumPy, SciPy, and plotting libraries (matplotlib, plotly) are allowed for offline tooling only and must not be used for any runtime computation or certificate calculation.

Rationale: this prevents accidental mixing of non-JAX code into JIT/differentiable paths, ensures reproducible device placement, and keeps certificate and tuning logic fast and compatible with GPU/TPU.
