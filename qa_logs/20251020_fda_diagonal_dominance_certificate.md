# QA Log: 2025-10-20 - FDA Diagonal Dominance Certificate (η)

**Component:** `src/computable_flows_shim/fda/certificates.py`
**Test:** `tests/test_fda.py::test_estimate_eta_dd`

## Goal
To implement the **Diagonal Dominance (η)** certificate estimator, the second key component of the Flow Dynamic Analysis (FDA) module.

## Process (Red-Green-Refactor)

1.  **(RED):** Added a new test, `test_estimate_eta_dd`, to `tests/test_fda.py`. The test defined a simple diagonal operator (for which η is 0) and attempted to call the non-existent `estimate_eta_dd` function. The test failed as expected with an `ImportError`.

2.  **(GREEN):** Implemented the `estimate_eta_dd` function in `certificates.py`. The implementation follows the mathematical definition:
    *   It first materializes the dense matrix `L` from the `L_apply` function.
    *   It then calculates the ratio of the sum of absolute off-diagonal elements to the absolute diagonal element for each row.
    *   It returns the maximum of these ratios.
    With this implementation, the test passed.

3.  **(REFACTOR):** The code is simple and required no refactoring. Similar to `estimate_gamma`, this implementation is inefficient for large operators but is mathematically correct. Performance can be addressed in a future refactoring cycle.

## Outcome
- We have a working, tested implementation of the `estimate_eta_dd` certificate.
- The core FDA certificate estimators are now in place, providing us with the tools to begin building a more intelligent, certified controller.
