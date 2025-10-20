OK# Flow Dynamic Analysis (FDA) & Certificates

## Purpose
Implements mathematical and engineering checks for flow stability, convergence, and physical correctness.

## Core Certificates (Explicit Math & Algorithms)

### 1. Diagonal Dominance (η)
$$
\eta_{\mathrm{DD}}(L_{\mathcal{W}}) = \max_i \frac{\sum_{j\neq i} |(L_{\mathcal{W}})_{ij}|}{|(L_{\mathcal{W}})_{ii}| + \varepsilon}
$$
**Go** if $\eta_{\mathrm{DD}} < \eta_{\max}$ (e.g., 0.9).

**JAX Code Sketch:**
```python
def estimate_eta_dd_in_W(L_apply, W, n_probe=16, eps=1e-9):
    # Practical JAX mapping:
    # - Use random Rademacher probes (vmap) to approximate off-diagonal sums or
    #   explicitly form matricized rows on small problems.
    # - L_apply should accept batched v: shape (B, N) -> (B, N)
    # - Use jax.vmap for batching probes, and jnp.abs/jnp.sum to compute row sums.
    # Example (sketch):
    # probes = random.rademacher(key, shape=(n_probe, N))
    # Av = vmap(LW_apply, in_axes=(0, None, None))(probes, L_apply, W)
    # row_sums = jnp.mean(jnp.sum(jnp.abs(Av) - jnp.abs(jnp.diag(Av)), axis=1), axis=0)
    # compute eta_dd from averaged estimates.
    ...
    return float(eta_dd)
```

### 2. Spectral Gap (γ)
**Gershgorin lower bound:**
$$
\gamma_{\mathrm{Gersh}} = \min_i \left( |(L_{\mathcal{W}})_{ii}| - \sum_{j\neq i}|(L_{\mathcal{W}})_{ij}| \right)
$$
**Lanczos estimate:** Use for SPD $L_{\mathcal{W}}$; $\gamma = \lambda_{\min}(L_{\mathcal{W}})$

**JAX Code Sketch:**
```python
def estimate_gamma_in_W(L_apply, W, k=8, iters=64):
    # Practical JAX mapping:
    # - Use Lanczos with jax.scipy.sparse.linalg.eigsh or a small Lanczos implementation.
    # - L_apply should be symmetric/SPD in W-space; wrap with LW_apply.
    # - For large N, run k-step Lanczos with jax.lax.scan for JIT friendliness.
    ...
    return gamma
```

### 3. Lyapunov Descent
$$
E(z^{k+1}) \le E(z^k) - \alpha_0 |\nabla f(z^k)|^2
$$
**JAX Code Sketch:**
```python
def check_lyapunov(E_values, alpha0=None, grad_norms=None):
    # Practical JAX mapping:
    # - E_values: per-iteration energies; grad_norms: per-iteration ||grad||^2
    # - Check E_{k+1} <= E_k - alpha0 * ||grad_k||^2 + tol
    # - Implement as jnp.all(E_values[1:] <= E_values[:-1] - alpha0 * grad_norms[:-1] + tol)
    ...
```

## Frame-aware operator application (LW_apply)
When transforms are not strictly unitary, the core operator in W-space is formed as:

- unitary: L_W = W L W^T
- tight(c): L_W = W L W^T  (norms and thresholds rescaled by c)
- general: L_W = W L \tilde{W}  (analysis W, synthesis \tilde{W})

Code sketch for L_W application:

```python
def LW_apply(v, L_apply, tf: TransformOp):
    # tf.forward = analysis (W), tf.inverse = synthesis (W^T or tilde W)
    u = tf.inverse(v)          # synth -> physical domain
    Lu = L_apply(u)            # apply core operator
    return tf.forward(Lu)     # analysis back to W-space
```

When `tf.frame == 'tight'`, scale norms for certificates by `tf.c` (e.g., divide diag by c).

## Manifolds & core operator selection
The core operator for FDA is usually the dominant quadratic part `L` (or Hessian at x*). For manifold-valued slots:

- Linearize via charts or use tangent/Lie algebra coordinate maps (log at reference point) to form `L`.
- For Riemannian settings, use Riemannian Hessian (if available) or push-forward of Euclidean Hessian into tangent space.

Provide `L_apply(v)` that applies the core operator to a vector `v` respecting manifold charts; the compiler should expose this callable to the FDA routines.

### 4. KKT/Duality Gap, Holdout, Pareto Knobs
KKT and duality gap checks map to residual computations; implement as batched vector-matrix products in JAX.
Example: for a quadratic constraint with linear operator A and slack s,
compute primal-dual residuals r = A x - b + s and ensure norms are below tolerance.

Practical JAX hints:
- Use vmap for batched residual evaluation when testing multiple candidates.
- For holdout experiments, reserve a small validation operator A_val and compute E_val.
- For Pareto tuning, sweep λ candidates via vmap and collect certificates to choose best tradeoff.

## Engineering Implementation
- Use Lanczos/Gershgorin in $\mathcal{W}$-space for spectral checks
- Compute certificates before and during flow execution
- Log and guard runs based on certificate status

## API Example
```python
from computable_flows_shim.fda import estimate_eta_gamma_in_W
eta, gamma = estimate_eta_gamma_in_W(L, W)
```

## Telemetry & Controller Integration
- FDA certificate checks (eta_dd, gamma, Lyapunov) are logged via Flight Recorder events and telemetry rows.
- Controller phases (RED/AMBER/GREEN) enforce build checklist: FDA must pass before tuning; all certificate results and transitions are recorded.
- API: FDA hooks are called by the controller, which manages phase logic and telemetry recording.

## Notes
- All analysis is performed in the multiscale domain
- Certificates drive the auto-tuner and runtime guards

See `11a_fda_hooks.md` for spec fields and runtime hooks that formalize invariants, LensPolicy, FlowPolicy, GapDial, and CertificationProfile.

## Spec hooks (summary)

The Shim supports spec-level hooks that the FDA uses during compilation and runtime:

- `StateSpec.invariants`: `conserved`, `constraints`, `symmetries`. These are validated at Phase 0 and periodically during runs; telemetry records `invariant_drift_max`.
- `LensPolicy`: a set of candidate transforms and probe metrics used by Builder Mode to select the best lens (compressibility or reconstruction-error criteria); `LENS_SELECTED` event is emitted.
- `CertificationProfile`: set of checks and tolerances (Lyapunov, physics residuals, invariants, spectral gap) used to determine GREEN state and final certification.

---

See tuner and runtime docs for how certificates are enforced.