# QA Log: 2025-10-20 - Controller Lyapunov Certificate

**Component:** `src/computable_flows_shim/controller.py`
**Test:** `tests/test_controller.py::test_controller_enforces_lyapunov_descent`

## Goal
To enhance the Flight Controller (`run_certified`) to enforce its first FDA certificate: **Lyapunov Descent**. This is the core contract of a dissipative flow, ensuring that the energy of the system does not increase between steps.

## Process (Red-Green-Refactor)

1.  **(RED):** Added a new test, `test_controller_enforces_lyapunov_descent`. This test used dependency injection to pass a "malicious" step function into the controller—one that deliberately increased the energy. The test asserted that the controller should catch this violation and raise a `ValueError`. The test failed as expected, because the controller had no such checking logic.

2.  **(GREEN):** Modified the `run_certified` function to:
    *   Accept an optional `_step_function_for_testing` argument to allow for test-specific dependency injection.
    *   Calculate the energy of the state before and after each step.
    *   Compare the energies and raise a `ValueError` if `new_energy > old_energy`.
    *   After this change, all tests passed.

3.  **(REFACTOR):** The code is clean and required no refactoring. This log entry completes the cycle.

## Outcome
- The Flight Controller now enforces the fundamental **Lyapunov Descent certificate**.
- We have successfully implemented our first piece of the Flow Dynamic Analysis (FDA) framework, making our system more robust and true to its mathematical foundations.
- The use of dependency injection in the test proved to be an effective way to verify the controller's error-handling logic in isolation.
