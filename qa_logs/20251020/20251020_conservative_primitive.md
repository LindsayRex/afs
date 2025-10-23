# QA Log: 2025-10-20 - Conservative Primitive

**Component:** `src/computable_flows_shim/runtime/primitives.py`
**Test:** `tests/test_primitives.py::test_F_Con`

## Goal
To implement and test the `F_Con` (conservative/symplectic) primitive, which handles Hamiltonian dynamics.

## Process (Red-Green-Refactor)

1.  **(RED):** Added a new test, `test_F_Con`, which attempted to import and call `F_Con` from the `primitives.py` module. The test was designed to verify one step of the Leapfrog integrator for a simple harmonic oscillator. It failed as expected with an `ImportError`.

2.  **(GREEN):** Implemented the `F_Con` function in `primitives.py`. The implementation follows the "kick-drift-kick" sequence of the Leapfrog/Stormer-Verlet integrator, using `jax.grad` to compute the gradients of the Hamiltonian at the required intermediate steps. The test then passed.

3.  **(REFACTOR):** The code is simple and required no refactoring. This log entry completes the cycle.

## Outcome
- We have a working, tested implementation of the `F_Con` primitive.
- This completes the implementation of the core deterministic primitives (`F_Dis`, `F_Proj`, `F_Multi`, `F_Con`).
