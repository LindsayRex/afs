Love it. Here’s a practical, **composable Atom Library** you can drop into your EF/flow stack. I grouped atoms by role; each entry has the *form*, *solver hook* (grad/prox/proj), and a quick *use* note. It’s meant to be exhaustive enough to cover your RF/HVAC/graph/knapsack/denoise/pathwork.

---

# 0) Notation (minimal)

* Variables (x): vectors/fields; matrices (X); flows (f); paths (p).
* Linear ops (A, D, L, W, \nabla, \operatorname{div}, \operatorname{curl}).
* Graph: nodes (V), edges (E), incidence (A), Laplacian (L_G).
* Prox of (g): (\mathrm{prox}_{\tau g}(z)=\arg\min_x; g(x)+\tfrac1{2\tau}|x-z|^2).

---

# 1) Data fidelity atoms (likelihoods)

* **Quadratic (Gaussian):** (\tfrac12|Ax-b|_2^2). grad (=A^\top(Ax-b)). Use: default LS.
* **Weighted LS:** (\tfrac12|W(Ax-b)|_2^2). grad (=A^\top W^\top W(Ax-b)).
* **Huber:** (\sum_i h_\kappa((Ax-b)_i)). prox (per-entry). Use: robust noise.
* **(\ell_1) (Laplacian noise):** (|Ax-b|_1). prox = soft-threshold on (Ax) via ADMM or variable split.
* **Poisson (GLM):** (\sum_i \big((Ax)_i-b_i\log(Ax)_i\big)). grad (=A^\top(1-b\oslash(Ax))).
* **Logistic / cross-entropy:** standard. Use: classification.
* **KL-divergence:** (D_{\mathrm{KL}}(b|Ax)). Use: nonnegative intensities.

---

# 2) Smoothness / curvature atoms

* **Tikhonov (L2):** (\lambda|Lx|_2^2). grad (=2\lambda L^\top Lx). Use: smooth fields.
* **TV (anisotropic):** (\lambda|Dx|_1). prox: anisotropic shrink on finite diffs.
* **TV2 / trend filtering:** (\lambda|D^2x|_1). Piecewise-linear fits.
* **Sobolev-(H^1):** (\lambda|\nabla x|_2^2). grad (=-2\lambda\Delta x).
* **Biharmonic:** (\lambda|\Delta x|_2^2). Very smooth surfaces.

---

# 3) Sparsity / structure atoms

* **Wavelet (\ell_1):** (\lambda|W x|_1). prox: soft in wavelet domain.
* **Group (\ell_{2,1}):** (\sum_g \lambda_g |x_g|_2). prox: block shrink.
* **Overlapping groups:** via latent splitting.
* **Elastic net:** (\lambda_1|x|_1+\tfrac{\lambda_2}{2}|x|_2^2).
* **Cosparsity (analysis):** (\lambda|C x|_1) (e.g., gradient sparsity).
* **Low-rank (matrix):** (\lambda|X|_*) (nuclear norm). prox: SVT.
* **Rank-1 factorization:** (|X-uv^\top|_F^2) + priors on (u,v).

---

# 4) Graph atoms (signals/paths/flows)

* **Dirichlet energy:** (\lambda,x^\top L_G x). grad (=2\lambda L_G x).
* **Graph TV:** (\lambda\sum_{(i,j)\in E} w_{ij}|x_i-x_j|).
* **Harmonic potential (goal bias):** solve (L_G\psi=0) on (V\setminus{t}), fix (\psi(t)=0); use (\sum_{(u,v)}\gamma(\psi(v)-\psi(u)),f_{uv}).
* **Path length (edge costs):** (\sum_{(u,v)} c_{uv},f_{uv}) with (Af=b) (flow conservation).
* **Curvature on paths:** (\beta,f^\top K f) where (K) encodes turn penalties / second differences.
* **Graph-wavelet sparsity:** (\lambda|W_G x|_1), (W_G) = spectral/vertex-domain wavelets.
* **Cut/partition surrogate:** (\lambda,x^\top L_G x) + simplex/box to approximate min-cut.

---

# 5) Physics / PDE residual atoms

* **Poisson/Heat residual:** (|\Delta x - s|_2^2) or (|\partial_t x - \kappa \Delta x|_2^2).
* **Wave residual:** (|\partial_{tt}x - c^2\Delta x|_2^2).
* **Incompressibility:** (|\operatorname{div} v|*2^2) (or hard constraint (\Pi*{{\operatorname{div}v=0}})).
* **Curl-free / irrotational:** (|\operatorname{curl} v|_2^2).
* **Constitutive (Hooke/Fourier):** (| \sigma - C:\epsilon(u)|_2^2), (|q+\kappa\nabla T|_2^2).
* **Boundary conditions:** penalties (\lambda|x-\bar x|^2_{\partial\Omega}) or projections.

---

# 6) Couplers (multi-EF glue)

* **Quadratic alignment:** (\sum_j \mu_j|W_j u - W_j v|_2^2) (scale-wise coupling).
* **Cross-modality L1:** (\lambda|F u - G v|_1) (robust align).
* **Consensus (ADMM-style):** (\rho|x-z|_2^2) with split variables.
* **Orthogonality / decorrelation:** (\lambda|U^\top U - I|_F^2).
* **Mutual-exclusion (soft):** (\lambda \langle x, y\rangle) or barrier on overlap.

---

# 7) Constraints & projections (feasible sets)

* **Box:** (x\in [\ell,u]). (\Pi)=clip.
* **Simplex:** (x\ge0, \sum x_i=1). (\Pi)=sorting-threshold.
* **(\ell_1)-ball:** (|x|_1\le \tau). (\Pi)=soft+rescale.
* **(\ell_2)-ball:** (|x|_2\le \tau). (\Pi)=scale.
* **Knapsack:** (c^\top x\le B, x\in[0,1]). (\Pi)=weighted shrink+bisection.
* **Matroid / cardinality (relaxed):** (|x|_1\le k) + sparsity.
* **Flow conservation:** (Af=b, f\ge0). (\Pi)=project onto affine + nonneg.
* **Rate limits:** (|x_{t+1}-x_t|\le r). (\Pi)=per-interval clamp.

---

# 8) Robust / information-theoretic atoms

* **Entropy (max-ent):** (-\tau\sum_i x_i\log x_i) on simplex. prox: entropic.
* **Relative entropy:** (\sum_i x_i\log(x_i/p_i)).
* **JS / mutual-info surrogates:** contrastive/NCE-style convex parts.
* **(\chi^2), MMD kernels (quadratic form)** for distribution matching.

---

# 9) Control/actuator & scheduling atoms (HVAC/robotics)

* **Actuator box/rate:** as in §7 (per device).
* **Eff. curves:** (\sum_t \phi(u_t)) convex (piecewise quad) with prox per-time.
* **Energy price / TOU:** (\sum_t p_t \cdot \text{power}(x_t,u_t)).
* **Wear/penalized switches:** (\lambda\sum_t |u_t-u_{t-1}|) (TV on control).
* **Comfort bands:** penalties (\sum_t \psi([T_t - (L_t,U_t)])) with deadbands.

---

# 10) Lenses / bases (choose W to diagonalize)

* **Classical:** Haar, dbN, symN, coifN; DCT/DST; FFT (real/complex).
* **Graph:** diffusion wavelets, spectral GFT, lifting schemes on graphs.
* **Scattering (fixed filters):** energy-preserving multiscale features.
* **Learned dictionary (white-box):** orthogonal (W) with (|W^\top W - I|_F^2) penalty.

---

# 11) Barriers & safety

* **Log-barrier:** (-\mu\sum_i \log(x_i)) for positivity.
* **Obstacle/clearance:** (\sum \phi(d(x,\mathcal{O}))) with hinge/quadratic ramps.

---

# 12) Meta-energy atoms (intent layer)

* **Validation loss:** task metric on hold-out.
* **Certificates:** spectral gap (\gamma(W)) ↑, diagonal dominance (\eta_{dd}(W)) ↓, monotone descent (energy trace).
* **Parsimony:** (\lambda_{\text{mdl}}\times#\text{atoms}) or description length.
* **Latency/ops proxy:** (\lambda_{\text{lat}}\cdot\mathrm{cost}(\text{program})).
* **Invariance gates:** argmin/path-order preserved on calibration sets.

---

# 13) Typical composites (ready-made)

* **Sparse recon (signals/RF):** (\tfrac12|Ax-b|_2^2 + \lambda|W x|_1 + \tau|Dx|_1).
* **Graph shortest path (EF view):** (\min_{f\ge0};\alpha w^\top f + \beta f^\top K f + \gamma \sum (\psi(v)-\psi(u))f_{uv};;s.t.;Af=b).
* **HVAC (module sketch):** plant balance (|Gx - u - d|^2) + comfort bands + TOU energy + wear TV(u) + cross-zone Laplacian + rate/box on (u).
* **Knapsack (relaxed):** (-v^\top x + \tfrac{\beta}{2}x^\top S x) s.t. (c^\top x\le B,;x\in[0,1]).

---

# 14) Certificate hooks (what each atom “reports”)

Each atom in your registry should expose:

* **Type:** smooth / nonsmooth / constraint.
* **Lipschitz or curvature bound** contribution (for step sizes).
* **Sparsity pattern** in a chosen basis (W).
* **Dominance contribution** to (\eta_{dd}(W)).
* **Prox/grad/proj** implementation (and cost model).

---

# 15) Minimal registry schema (so your shim can load it)

```yaml
- name: wavelet_l1
  form: "lambda * || W x ||_1"
  kind: "nonsmooth"
  prox: "soft_threshold_in_W"
  params: { lambda: {range: [1e-4, 1e1], log: true}, W: {choices: [haar, db4, db8]} }
  cert:
    sparsity_band: "diagonal in W"
    affects: ["eta_dd-","gap+"]

- name: graph_dirichlet
  form: "lambda * x^T L_G x"
  kind: "smooth"
  grad: "2 lambda L_G x"
  params: { lambda: {range: [1e-4, 1e2], log: true} }
  cert:
    banded: true
    affects: ["gap+"]

- name: knapsack_proj
  form: "x in [0,1], c^T x <= B"
  kind: "constraint"
  proj: "weighted_shrink_bisection"
  params: { }
  cert:
    stable: true
```

---

## How to use this in discovery

1. Pick a **small atom subset** per domain (3–8 atoms).
2. Let the meta-layer toggle atoms and tune weights; **gate on certificates**.
3. When something helps but hurts (\eta_{dd}), change **basis (W)** (graph-wavelets, DCT, etc.).
4. Log the program: **explicit energy**, **chosen (W)**, **prox/grad/proj**, **certs**.

