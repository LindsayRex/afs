---
description: Project-specific agent instructions. Edit to reflect new project rules.
---

# Agent Rules

## Core Development Methodology
This project follows a strict, verifiable, and incremental development process. Adherence to this methodology is mandatory for all contributions. It is designed to prevent self-deception and ensure correctness.

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

## General Rules
- Branching: `copilot/*` => open PRs to `test`.
- CI must pass on PRs.
- Keep copies of evidence in `qa_logs/`.

## Testing guidance
- Prefer small, focused unit tests built via the TDD cycle.
- Use repository `pyproject.toml` pytest settings; run with `pytest -q`.

## Documentation and Knowledge Base
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

