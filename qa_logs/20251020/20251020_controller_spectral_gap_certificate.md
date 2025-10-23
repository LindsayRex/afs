# QA Log: 2025-10-20 - Controller Spectral Gap Certificate

**Component:** `src/computable_flows_shim/controller.py`
**Test:** `tests/test_controller.py::test_controller_checks_spectral_gap`

## Goal
To enhance the Flight Controller (`run_certified`) to enforce the **Spectral Gap (γ)** certificate. This is a critical "Phase 1" check to ensure the system is stable before running the main loop.

## Process (Red-Green-Refactor)

1.  **(Refactor First):** The `compile_energy` function was refactored to return the core linear operator (`L_apply`) as part of the `CompiledEnergy` tuple. This was necessary to expose the operator to the controller for analysis.

2.  **(RED):** Added a new test, `test_controller_checks_spectral_gap`. This test created an unstable linear operator (with a negative eigenvalue) and asserted that the `run_certified` function should raise a `ValueError` when attempting to run it. The test failed as expected, because the controller had no such check.

3.  **(GREEN):** Modified the `run_certified` function to:
    *   Import the `estimate_gamma` function from the FDA module.
    *   Call `estimate_gamma` on the compiled `L_apply` operator before the main loop.
    *   Check if the returned `gamma` is positive, raising a `ValueError` if it is not.
    *   With this change, all tests passed.

4.  **(REFACTOR):** The code is clean and required no refactoring.

## Outcome
- The Flight Controller now enforces the **Spectral Gap (γ)** certificate, preventing it from running unstable systems.
- We have successfully integrated our FDA module with the controller, demonstrating how the components work together to ensure certified execution.
- This marks a significant step towards a fully-featured, safety-conscious controller.
