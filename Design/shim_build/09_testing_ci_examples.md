
# Testing, CI, and Examples

---
**Testing/CI Requirements:**
All tests must use pytest and ruff for linting/QA. CI must validate compatibility with the official package list in the overview doc above.

## Testing Plan
- Unit tests for each primitive (F_Dis, F_Proj, F_Multi, F_Con)
- FDA estimators: golden values for η/γ on toy matrices
- Tuner: synthetic problems with known targets
- Compile→run parity with reference flows

## FDA-specific tests
- Invariants: assert `invariant_drift_max` spikes on deliberate invariant violation and run aborts if beyond tolerance.
- Lens: deterministic selection across seeds for transforms with tied metrics.
- Unit normalization: compile produces `unit_normalization_table` and seeds weights accordingly.
- Flow family: verify `preconditioned` option reduces iterations on ill-conditioned example.
- Multiscale schedule: residual-driven activation triggers `SCALE_ACTIVATED(level)` events.


## Continuous Integration (CI)
- Use GitHub Actions for:
  - Python 3.10/3.11 (CPU JAX)
  - Ruff + mypy for lint/type checks
  - Build wheels and upload to TestPyPI/PyPI

  CI additions for FDA tests:
  - Run FDA unit tests in integration matrix (`tests/test_fda_*.py`)
  - Include `cf readiness-check` to validate compile report and manifest fields

## Example Gallery
- HVAC (15 terms)
- RF acquisition
- iFlow (FFT as F_Multi)
- Hypothesis/UCRC

Each example includes:
- Python DSL spec (examples/*.py)
- Expected certificate snapshot (η, γ) as a small JSON artifact alongside plots
- Output plots

Pareto examples include sidecar artifacts under `runs/<run_id>/pareto/` (trials.parquet, front.parquet, policy.json, front.png) and should be included in integration tests where applicable.

## Engineering Notes

## Dtype and Validation Policy
- All tests and examples must explicitly set and check global dtype (float32 or float64) for all arrays and computations.
- Add tests for dtype consistency, non-finite sentry (NaN/Inf), and shape stability.
- Validation must include spec linter, scale normalization, and certificate checks as per the low-level readiness checklist.



## Telemetry & Controller Integration
- All tests and examples record telemetry and events via Flight Recorder hooks, including certificate checks and phase transitions.
- Controller phases (RED/AMBER/GREEN) enforce build checklist: tests and examples must pass certification before tuning or long runs.
- API: test harnesses and examples call the controller, which manages phase logic and telemetry recording.