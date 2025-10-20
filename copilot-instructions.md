---
description: Project-specific agent instructions. Edit to reflect new project rules.
---

# Agent rules

- Branching: `copilot/*` => open PRs to `test`.
- CI must pass on PRs.
- Keep copies of evidence in `qa_logs/`.


## Testing guidance
- Prefer small, focused unit tests alongside new modules (happy path + 1 edge)
- Keep quick lane fast; exclude heavy/archived/synthetic content from default discovery
- Use repository `pyproject.toml` pytest settings; run with `pytest -q` unless a specific suite is needed

## Editor and lint configuration
- Single source of truth: `pyproject.toml` (Pylint config under `[tool.pylint.*]`)
- VS Code convenience: set `python.defaultInterpreterPath` to the venv; optionally pass `--rcfile=${workspaceFolder}/pyproject.toml`
- Avoid hiding folders with `files.exclude` unless intentionally removing them from Explorer

## PR checklist (v4.0)
- Add at least one unit test for any new public function
- Update `README.md` if surface changes (commands/paths/workflows)
- Attach lint/test output in Phase QA; save in `qa_logs/`

