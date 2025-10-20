# Contributor Guide: Computable Flows Shim

This guide bridges the naming/layout conventions and the readiness checklist for developers.

## Branch & Commit Naming
- Use feature branches: `feature/<short-description>`
- Use fix branches: `fix/<short-description>`
- Commit messages: start with a verb, reference issue/PR if relevant

## Style Enforcement
- Run `ruff` for linting and auto-formatting
- Run `mypy` for type checks
- Adhere to PEP8 and project-specific style rules

## Test Matrix
- All code must pass tests on CPU (default)
- CUDA tests (if available) should be run for performance-critical modules
- Use `pytest --strict-markers` to enforce marker discipline

## Adding New Adapters
- Implement new adapters in `backends/` or `adapters/` following the ports-and-adapters pattern
- Register new transforms, manifolds, or ops using the registry path constants in `registry_paths.py`
- Add entry points in `pyproject.toml` as described in `11_naming_and_layout.md`

## CI Integration

## Documentation

### JAX-only Runtime Policy

All runtime primitives and certificate/tuner logic must use JAX arrays and JAX-native libraries. This includes transforms (wavelets), sparsity and gap computations, and any diagnostic that affects tuning or certification.

Rules:
- Runtime modules under `primitives/`, `multi/`, `fda/`, and `runtime/` must not import `numpy`, `scipy`, `pandas`, or other non-JAX libraries.
- Convert external inputs at the boundary with `jnp.asarray(...)` and use `jax.device_put` where device placement is required.
- Telemetry and IO modules may use PyArrow, DuckDB, or Polars. They must not be used inside JIT-compiled functions.

Linting & Tests:
- Add a smoke test that imports each runtime module and asserts that no forbidden modules are imported.
- Update `ruff`/`flake8` config to flag `numpy`, `scipy`, and `pandas` imports in runtime folders.

If unsure, open an issue in the repo and tag `runtime` for design review.

### IO Stack Recommendation

- For telemetry and offline analysis, prefer the locked IO stack: PyArrow + DuckDB, optionally Polars for offline ETL/analysis.
- Document any additions to the IO stack and pin versions in `pyproject.toml` or `requirements.txt`.

## Release Process
- Update `environment.lock` or `conda-lock.yml` for new dependencies
- Bump version in `pyproject.toml` and tag release
- Verify optional extras `[dev]` include all developer tools

---

For more details, see `11_naming_and_layout.md` and `cf_low_level_readiness_checklist.md`.