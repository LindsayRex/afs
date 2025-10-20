# Tensor Logic Front-End: Optional Adapter for Mathematicians

This document describes how to add a minimal, declarative Tensor Logic front-end to your Computable Flows Shim. This adapter lets users describe problems as tensor-equation graphs, which are then compiled to your existing energy-based runtime. No changes to the core Shim are required.

## 1. Motivation
- **Audience**: Mathematicians, formal-methods users, and anyone who prefers tensor algebra as a modeling language.
- **Goal**: Accept tensor programs as input, translate them to your EnergySpec, and run them unchanged in your JAX-based Shim.

## 2. Minimal Tensor Program IR
A neutral, in-memory graph for tensor programs:

```python
@dataclass
class TTensor:
    name: str
    shape: tuple[int, ...]
    role: str  # "parameter" | "latent" | "data" | "aux"

@dataclass
class TOp:
    kind: str  # "einsum" | "matmul" | "add" | "reshape" | "concat" | "select" | "project"
    inputs: list[str]
    outputs: list[str]
    spec: dict

@dataclass
class TObjective:
    name: str
    form: str  # "lsq", "cross_entropy", "kl", "l1", "tv", "indicator", "custom"
    on: list[str]
    weight: float | None

@dataclass
class TProgram:
    tensors: dict[str, TTensor]
    ops: list[TOp]
    objectives: list[TObjective]
    constraints: list[TObjective]
```

## 3. Translation to EnergySpec
- **Smooth objectives** (`lsq`, `cross_entropy`, `kl`) → contribute to `f(z)`.
- **Nonsmooth objectives** (`l1`, `tv`, `indicator`) → contribute to `g(z)` (prox terms).
- **Ops** (einsum/matmul/etc.) → select existing Op adapters for JAX compilation.
- **Projection/select** ops → linear selection matrices or indicator constraints.

## 4. Example Mapping
**Tensor-style spec:**
- Ops: `y_pred = einsum('ij,j->i', A, x)`
- Objective 1: `lsq(y_pred, y_obs)`
- Objective 2: `l1(W*x)`

**Translator emits → EnergySpec:**
- `f(x) = 1/2 |A x - y|^2`
- `g(x) = λ |W x|_1`
- Operators: `F_Dis`, `F_Proj` as in Shim

## 5. Runtime and Telemetry
- The runtime is unchanged: you still produce `CompiledEnergy{f_value, f_grad, g_prox, W, L_apply}` and run the same controller/certificates.
- Tag telemetry rows with front-end objective names for business metrics.

## 6. Risks and Payoff
| Item                                   | Effort | Risk | Payoff                                          |
| -------------------------------------- | ------ | ---- | ----------------------------------------------- |
| Define tiny IR & translator            | low    | low  | unlocks declarative input + future interop      |
| Keep JAX Shim intact                   | none   | none | stability & speed you already have              |
| Add optional Pareto knobs at front-end | low    | low  | friendlier UX for business users                |
| Formal theorems later                  | long   | n/a  | academic strength, but not required for utility |

## 7. Recommended Path
1. Prototype the translator for a few op kinds and objective forms.
2. Lower to your current EnergySpec and run the same controller/certificates.
3. Tag telemetry with front-end names.
4. Iterate as needed.

## 8. Cross-References
- See also: `cf_tensor_logic_front.md` (background), `compraison with tensor Logic.md` (review), `tensor logic the langauge of AI.md` (source paper).
- This front-end is optional and does not affect the core Shim.
