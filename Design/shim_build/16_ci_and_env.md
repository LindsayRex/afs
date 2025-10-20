# CI & Environment Setup for CF Shim

This document describes the recommended environment, CI, and tooling setup for the CF Shim project. Use this as a reference for initializing a new repo or preparing a reproducible development environment.

---

## 1. Python Environment
- **Version:** Python 3.10 or 3.11
- **Virtual Environment:** Use `python -m venv .venv` or Conda
- **Dev dependencies:** pytest, ruff, black, isort, mypy, pre-commit

## 2. Packaging & Lock Files
- **pyproject.toml:** PEP621/Poetry/Setuptools format
- **environment.lock:** Pin all dependencies for CPU
- **environment.cuda.lock:** Pin for CUDA (optional)

## 3. JAX Install (Manual Step)
- For CPU: `pip install --upgrade "jax[cpu]" -f https://storage.googleapis.com/jax-releases/jax_releases.html`
- For GPU: `pip install --upgrade "jax[cuda11_cudnn82]" -f https://storage.googleapis.com/jax-releases/jax_releases.html`
- Do not auto-install JAX in CI; document manual install in this file

## 4. CI Pipeline (GitHub Actions)
- **Lint:** ruff, black, runtime import check
- **Test:** pytest
- **Readiness:** scripts/readiness_check.sh (runs lint, tests, schema-validate)
- **Schema validation:** scripts/schema_validate.py (optional)
- **No build artifacts until readiness passes**

## 5. Pre-commit Hooks
- ruff, black, isort, runtime-imports-check

## 6. Dockerfile (optional)
- Python 3.10-slim, dev dependencies, manual JAX install

## 7. Documentation
- Add this file as `16_ci_and_env.md` in `docs/shim_build/`
- Document all manual steps and environment setup here

---

## Example VS Code Setup
- Use workspace settings to pin Python interpreter
- Add `.vscode/settings.json` for linting, formatting, and test discovery
- Add `.vscode/launch.json` for CLI/test/debug entry points

---

## Next Steps
- Run `scripts/readiness_check.sh` before any build
- Copy only the folders/files you need into your new repo after VS Code config is complete
- Enable GitHub Actions and pre-commit hooks

---

For questions, see this file or the readiness checklist.
