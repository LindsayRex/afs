---
description: Project-specific agent instructions. Edit to reflect new project rules.
---

# Agent Rules

## Core Development Methodology
This project follows a strict, verifiable, and incremental development process. Adherence to this methodology is mandatory for all contributions. It is designed to prevent self-deception and ensure correctness.

## Documentation and Knowledge Base
- **New Component Quick Start:** `Design/shim_build/24_new_component_quick_start.md` - Standardized templates and checklists for creating new components
- Before using a function from a third-party library, you MUST consult the local documentation in the `archive/` directory to ensure you are using the correct API and respecting its contract.
- Do not rely on assumed knowledge. Verify the correct usage from the archived documentation.
- Key documentation paths:
  - **JAX:** `archive/jax-docs-archive/`
  - **Optax:** `archive/optax-docs/`
  - **Orbax:** `archive/orbax-docs/`
  - **Jax-Wavelet-Toolbox (jaxwt):** `archive/jaxwt-docs/`
  - **NetworkX:** `archive/networkx-docs/`
  - **DuckDB:** `archive/duckdb-docs/`
  - **PyArrow:** `archive/pyarrow-docs/`
  - **frozendict:** `archive/frozendict-docs/`


**1. Architectural Pattern: Functional Core, Imperative Shell**
   - **Functional Core:** All core logic (primitives, compiler, math) MUST be implemented as pure functions with zero side effects. This is our reusable, testable engine.
   - **Imperative Shell:** All side effects (file I/O, tests, logging) MUST be in the outer shell.
   - Refer to `Design/shim_build/17_design_pattern.md` for the full specification.

**2. Workflow: Test-Driven Development (TDD)**
   - All new functionality MUST be built using the Red-Green-Refactor cycle.
   - **RED:** First, write a failing test that defines the desired functionality.
   - **GREEN:** Write the simplest possible *real* code in the Functional Core to make the test pass. No placeholders.
   - **REFACTOR:** Clean up the code and run tests again to ensure they still pass.

**3. Documentation: QA Logs**
   - After every successful TDD cycle, a new QA log entry MUST be created in the `qa_logs/` directory.
   - The log MUST document the goal, the TDD process, and the outcome, providing a clear audit trail of what was built and why.

**4. Verification Principles**
   - **Design by Contract:** Use tests to enforce the mathematical properties (preconditions, postconditions, invariants) defined in the design documents.
   - **Behavior-Driven:** Frame tests and logs using a "Given-When-Then" structure to describe the observable behavior of the system.
   - **Formal Verification:** Treat the mathematical lemmas in the design documents as proof obligations that are computationally verified by our tests.

# Computable Flow Shim and the TDD/DbC/BDD methodology:

Unit/Contract Tests (Functional Core):
These verify that each module (e.g., FDA estimator, primitive operator) satisfies its mathematical contract in isolation. For example, estimate_gamma must return the algebraic minimum eigenvalue for any operator you give it. This is like testing the radiator with water on a bench.

Behavior/Integration Tests (Imperative Shell):
These verify that, when modules are composed (e.g., compiler → runtime → controller → FDA), the system as a whole behaves as specified. For example, running a full flow and checking that the controller refuses to run unstable systems, or that energy decreases over time. This is like putting the radiator in the car and driving.

Design by Contract:
Each module must declare and enforce its preconditions, postconditions, and invariants. But some contracts only make sense in context (e.g., Lyapunov descent is only meaningful when the flow is running).

Boundaries of Execution:
The pattern encourages you to test modules in isolation where possible, but also to write system-level tests that exercise the full pipeline. Some contracts (like cooling the engine) can only be verified in integration.

What this means for your workflow:

It's normal and correct to have both isolated (unit/contract) tests and integrated (behavior/system) tests.
If a module's contract can only be meaningfully tested in context, write a system-level test for it.
Don't force artificial isolation if the contract is inherently systemic—test it as part of the whole.
If you ever feel like you're "jumping through hoops" to test a module in isolation, ask:

Is this contract meaningful outside the system? If not, write a behavior/integration test instead.
Is the test verifying a real property, or just making the test pass? Only keep tests that enforce real contracts or behaviors.


## Code Guidelines (MANDATORY)

**ALWAYS follow these rules for ALL new code:**

### JAX Usage
- **Use `jax.numpy` functions, NEVER `numpy`** - `jnp.sum()`, `jnp.array()`, etc.
- **All mathematical operations must use JAX arrays** - No raw Python lists or numpy arrays in math functions
- **Reference:** `Design/shim_build/03b_energy_spec_hygiene_spec.md` (JAX Compatibility section)

### Hygiene Requirements
- **Decorate ALL mathematical functions** with `@numerical_stability_check`
- **Use Pydantic models** for ALL data structures (`BaseModel` with field validation)
- **Apply to:** Energy functions, gradient computations, prox operators, runtime primitives
- **Reference:** `Design/shim_build/03b_energy_spec_hygiene_spec.md` (Core Hygiene Requirements)

### Design Pattern Compliance
- **Functional Core:** Pure mathematical functions with zero side effects
- **Imperative Shell:** All I/O, validation, error handling in outer layers
- **Reference:** `Design/shim_build/17_design_pattern.md`

### Verification Checklist (Pre-commit)
- [ ] JAX functions used throughout?
- [ ] `@numerical_stability_check` on all math functions?
- [ ] Pydantic models for data structures?
- [ ] Functional Core/Imperative Shell pattern followed?
- [ ] Tests written and passing?

**These are NOT optional - they prevent catastrophic failures and maintain SDK reliability.**

## Testing guidance
- Prefer small, focused unit tests built via the TDD cycle.
- Use repository `pyproject.toml` pytest settings; run with `pytest -q`.
