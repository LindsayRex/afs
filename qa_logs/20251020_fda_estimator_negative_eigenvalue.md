# QA Log: FDA Estimator Negative Eigenvalue Test (2025-10-20)

Goal
----
Verify that `estimate_gamma` correctly detects and returns the algebraic minimum eigenvalue for a linear operator with a negative eigenvalue.

TDD Cycle
---------
- **RED:** Added a test in `tests/test_fda.py` (`test_estimate_gamma_detects_negative_eigenvalue`) for an operator with matrix [[-2, 0], [0, 3]] (eigenvalues -2, 3). Expected gamma = -2.0.
- **GREEN:** Ran the test; it passed immediately, confirming the estimator is correct.
- **REFACTOR:** No code changes needed; estimator already returns the algebraic minimum eigenvalue.

Files Changed
-------------
- `tests/test_fda.py` — added negative eigenvalue test for `estimate_gamma`.

Validation
----------
- Ran `pytest -q tests/test_fda.py`; all tests passed.

Outcome
-------
- `estimate_gamma` is correct for negative eigenvalues; no further changes required.

QA Owner: automatic agent
Date: 2025-10-20
