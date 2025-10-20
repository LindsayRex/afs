# Next Steps after FDA Fix (2025-10-20)

Summary
-------
After fixing the spectral-gap estimator and controller pre-flight order, all tests pass. Based on the Shim Build docs and the latest TDD cycle, here are the next recommended tasks prioritized by their impact.

Immediate (High Priority)
-------------------------
1. Add explicit unit tests for `estimate_gamma` and `estimate_eta_dd`:
   - Happy paths: small symmetric SPD matrix.
   - Negative eigenvalue detection.
   - Gershgorin-based fallback for non-symmetric matrices.

2. Add a QA log entry (done) and add an automated check in CI to ensure `estimate_gamma` returns algebraic minimum for small matrices.

3. Add a runtime builder rehearsal test covering `Builder Mode` steps to ensure the compiler writes `CompileReport` with unit normalization table.

Medium Priority
---------------
4. Implement a scalable estimation mode for `estimate_gamma` (Lanczos / ARPACK / shift-and-invert). Add an optional `mode` flag with default `dense` for tests.

5. Implement a randomized probe-based `estimate_eta_dd` (for large dims) and add a `n_probes` argument.

6. Add Flight Recorder telemetry hooks for the new certificate checks and record the results in `telemetry.parquet` and `events.parquet`.

Long-term / Nice-to-have
------------------------
7. Add a GapDial auto-tuner with a simple grid search for `lambda` and `alpha` in the `tuner/` module, guarded by certificates.

8. Improve multiscale transform coverage: more robust frame handling (tight vs general frames) in LW_apply.

CI / QA
------
- Add focused tests to CI running `tests/test_fda.py` with multiple matrix sizes.
- Add a linter pre-commit hook to ensure no non-JAX ops exist in functions used in certificate paths.

Ownership & Timing
------------------
- Owner: core team + AI agent pairing
- Time estimate: 1-2 days for immediate changes (tests and telemetry), 1 week for scalable estimator and tuner prototype.

I'll now mark this plan as completed in the task list. If you'd like, I can start implementing the immediate items (tests and telemetry hooks) next. 

