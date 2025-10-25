# QA Log: Spectral Gap and Diagonal Dominance Fix (2025-10-20)

Goal
----
Ensure the controller performs pre-flight FDA certificate checks correctly and refuses to run unstable or non-diagonally-dominant systems.

TDD Cycle
---------
- RED: Added tests in `tests/test_controller.py` to assert the controller raises for unstable systems and for non-diagonally-dominant systems.
- GREEN: Implemented `estimate_gamma` and `estimate_eta_dd` in `src/computable_flows_shim/fda/certificates.py`, and added pre-flight checks in the controller.
- REFACTOR: Fixed estimator behavior and test ordering:
  - Replaced inverse-power estimate with direct eigenvalue computation in `estimate_gamma`.
  - Reordered controller checks to evaluate diagonal dominance (eta) before spectral gap (gamma) as expected by tests.

Files Changed
-------------
- `src/computable_flows_shim/fda/certificates.py`  — replaced inverse-power method with direct eigen-decomposition for gamma estimate.
- `src/computable_flows_shim/controller.py` — reorder pre-flight checks (eta then gamma).

Validation
----------
- Ran `pytest -q` and verified all tests currently pass (30 passed, 0 failed).

Notes & Next Steps
------------------
- The current estimators construct a dense matrix; this is fine for small-dimension states used in tests. If we need to scale to large states, implement iterative or sparse estimators (e.g., shift-and-invert or randomized methods).
- Consider adding a non-blocking 'fast-check' mode to the controller to approximate these checks for large systems.

QA Owner: automatic agent
Date: 2025-10-20
