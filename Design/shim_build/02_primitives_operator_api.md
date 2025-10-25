
# Primitives & Operator API

---
**Package Requirements:**
All primitives and operator APIs must use the following official packages:
- JAX, jaxlib, NumPy (reference API)
- SciPy (optional)
- Optax (preferred) or Optimistix (optional)
- Registry patterns should support jaxwt for wavelets and jaxlie for manifolds
See the full package list in the overview doc above.

## Core Primitives: Mathematical Definitions & JAX Mappings

### 1. Dissipative Step (F_Dis)
**Math:**
$$
\mathcal{F}_{\text{Dis}}(z;\alpha) := z - \alpha \nabla f(z)
$$
**Properties:** $f$ has $\beta$-Lipschitz gradient; $0 < \alpha < 2/\beta$ for non-expansiveness.

**JAX Code Stub:**
```python
def F_Dis(state, grad_f, step_alpha):
    g = grad_f(state)  # grad_f built by compiler, uses jax.grad
    return {k: state[k] - step_alpha * g[k] for k in state.keys()}
```

### 2. Projective/Proximal Step (F_Proj)
**Math:**
$$
\mathcal{F}_{\text{Proj}}(z;\alpha,\mathcal{W}) := \mathcal{W}^\top\left(\mathrm{prox}_{\alpha g_{\mathcal{W}}}(\mathcal{W}z)\right)
$$
where $g_{\mathcal{W}}(u) := g(\mathcal{W}^\top u)$.

**Properties:** If $g$ is convex, $\mathrm{prox}_{\alpha g}$ is 1-Lipschitz (firmly non-expansive).

**JAX Code Stub:**
```python
def F_Proj(state, prox_in_W, step_alpha, W):
    return prox_in_W(state, step_alpha, W)  # prox_in_W is built by compiler
```

### 3. Multiscale Transform (F_Multi)
**Math:**
$$
\mathcal{F}_{\text{Multi}}^+(z) = \mathcal{W}z,\qquad \mathcal{F}_{\text{Multi}}^-(u) = \mathcal{W}^\top u
$$
**Properties:** $\mathcal{W}$ is unitary or a tight frame; invertible.

**JAX Code Stub:**
```python
@dataclass(frozen=True)
class WaveletOp:
    name: str
    levels: int
    forward: Callable[[Array], Array]   # e.g., jaxwt.dwt_nD
    inverse: Callable[[Array], Array]   # e.g., jaxwt.idwt_nD

def F_Multi_forward(x, W):
    return W.forward(x)

def F_Multi_inverse(u, W):
    return W.inverse(u)
```

### 4. Conservative/Symplectic Step (F_Con, optional)
**Math:** (Leapfrog/Stormer–Verlet)
$$
\begin{aligned}
p_{k+1/2} &= p_k - \tfrac{\Delta t}{2} \partial_q H(q_k, p_k) \\
q_{k+1} &= q_k + \Delta t \partial_p H(q_k, p_{k+1/2}) \\
p_{k+1} &= p_{k+1/2} - \tfrac{\Delta t}{2} \partial_q H(q_{k+1}, p_{k+1/2})
\end{aligned}
$$
**Properties:** Symplectic, time-reversible, energy error $O(\Delta t^2)$.

**Implementation Steps:**
The integrator is implemented as a three-step "kick-drift-kick" sequence which directly translates the mathematical definition:
1.  **Kick (half-step momentum):** First, calculate the momentum at the half-step, $p_{k+1/2}$. This uses the gradient of the Hamiltonian with respect to position, $\partial_q H$, evaluated at the initial state $(q_k, p_k)$.
2.  **Drift (full-step position):** Next, calculate the full-step position, $q_{k+1}$. This uses the gradient of the Hamiltonian with respect to momentum, $\partial_p H$, evaluated at the intermediate state $(q_k, p_{k+1/2})$.
3.  **Kick (full-step momentum):** Finally, calculate the full-step momentum, $p_{k+1}$. This uses the gradient of the Hamiltonian with respect to position, $\partial_q H$, evaluated at the new intermediate state $(q_{k+1}, p_{k+1/2})$.

**JAX Code Stub:**
```python
def F_Con(state, H=None, dt=1.0):
    # Placeholder: implement as needed for Hamiltonian flows
    return state
```

## Operator Interface
```python
class Op:
    def __call__(self, x): ...         # Forward (JAX ops)
    def T(self, x): ...                # Adjoint (optional)
    def lipschitz_hint(self): ...      # Optional β estimate
```

## Extension Points
- Register new ops via Python entry points
- Add new prox maps with decorators
- Plug in new backends (JAX, Torch)

## Example Usage
```python
from computable_flows_shim.ops import FFTChannelOp, WarpSamplerOp
H = FFTChannelOp(...)
S = WarpSamplerOp(...)
```

---

See the energy and runtime docs for how these primitives are composed.

## Config Policy
- User-facing specs must be written in Python DSL (see `11_naming_and_layout.md`). YAML-based specs are deprecated and should not be used for new projects.

## Manifolds: per-slot adapters and how primitives operate

### Manifold adapter interface
Each state slot may declare a manifold. A small adapter interface is used to unify operations:

```python
class Manifold(Protocol):
    def exp(self, x: Array, v: Array) -> Array: ...
    def log(self, x: Array, y: Array) -> Array: ...
    def metric(self, x: Array, v: Array, w: Array) -> Array: ...
    def proj_tangent(self, x: Array, v: Array) -> Array: ...
    def retract(self, x: Array, v: Array) -> Array: ...
    def transport(self, x: Array, v: Array, w: Array) -> Array: ...
```

### How primitives use manifolds
- F_Dis: use `log/exp` or `retract` to perform Riemannian gradient steps (grad = metric^{-1} * Euclidean grad in chart).
- F_Proj: perform prox in tangent (via `log`), then `retract` back to manifold.
- F_Multi: apply multiscale transforms on linear coordinate patches (Euclidean arrays or Lie algebra via `log`).
- F_Con: run symplectic integrators on cotangent slots (q,p) with local charting.

### Example JAX stub for F_Dis with manifolds
```python
def F_Dis(state, grad_f_euclid, step_alpha, manifolds: Dict[str, Manifold]):
    g = grad_f_euclid(state)  # Euclidean gradients (jax.grad)
    new = {}
    for name, x in state.items():
        M = manifolds.get(name)
        if M is None:
            new[name] = x - step_alpha * g[name]
        else:
            # project Euclidean grad to tangent and retract
            grad_R = M.proj_tangent(x, g[name])
            new[name] = M.retract(x, -step_alpha * grad_R)
    return new
```

Add these manifold adapters to your `ops/` or `backends/jax_backend.py` so they are available to the compiler and runtime.

## Telemetry & Controller Integration
- Each primitive step is tracked via Flight Recorder telemetry hooks: `write_iter(row)` records all key metrics per iteration.
- Controller phases (RED/AMBER/GREEN) enforce build checklist: primitives only run in certified order, with phase transitions and certificate checks logged as events.
- API: primitives are called by the controller, which manages phase logic and telemetry recording.
