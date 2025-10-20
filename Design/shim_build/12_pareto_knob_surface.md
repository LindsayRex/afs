# Pareto Knob Surface

This document adds a Pareto layer on top of the CF Shim to enable structured multi-objective experiments without changing the Shim core.

## Purpose
Introduce a thin Pareto Manager that runs certified sweeps over objective weights and constraints to produce a Pareto front for business users and AI agents.

## 1) Pareto Manager (API & control flow)

Role: orchestrates multi-objective experiments using `run_certified(...)`.

API (concise):

```python
class ParetoManager:
    def __init__(self, controller, telemetry):
        ...

    def register_objectives(self, objectives: dict[str, Callable[[State], float]]): ...
    def register_constraints(self, constraints: dict[str, tuple[Callable[[State], float], float]]):
        """each is (metric_fn, bound) interpreted as metric <= bound"""

    def set_policy(self, name: str, spec: dict):
        """spec may contain: method, ref_point, epsilon, grid, presets"""

    def generate_candidates(self) -> list[dict]:
        """returns list of candidate weight vectors and/or ε-constraints"""

    def evaluate_candidates(self, init_state, base_spec) -> list[dict]:
        """runs controller.run_certified for each candidate with gates on"""

    def pareto_front(self, trials: list[dict]) -> list[dict]:
        """compute non-dominated set in objective space"""
```

Methods supported:
- Weighted-sum scalarization (grid on the simplex for 3–5 objectives).
- ε-constraint (treat some KPIs as hard bounds).
- Reference-point (goal programming).

## 2) Policies & "business knobs"

Examples: HVAC policies (cost_min, carbon_min, comfort_first, balanced). CLI knobs: `--policy` and sliders for weights.

## 3) Certified sweeps

Use RED/AMBER/GREEN gates. Only GREEN runs enter the Pareto set. Use warm-starts and batched short runs for pre-screening.

## 4) Telemetry & artifacts (Pareto sidecar)

Store Pareto results in a sidecar:

```
runs/<run_id>/pareto/
  trials.parquet
  front.parquet
  front.png
  policy.json
```

Add columns to `trials.parquet`: `weights`, `objectives`, `constraints_ok`, `cert_eta_dd`, `cert_gamma`, `lyap_ok`, `status`.

## 5) CLI & notebook UX

Add `cf pareto` commands and a notebook API example.

## 6) Guardrails & tips

- Pareto modifies weights and constraints only; the controller enforces stability.
- Expose at most 3–5 objectives.
- Normalize KPIs.
- Cache and resume per-candidate GREEN states.

## 7) Integration points

- Uses existing energy weights, controller run_certified, tuner, telemetry, and manifest.

---

This thin layer enables business-friendly tradeoff exploration with minimal risk and clear telemetry capture.
