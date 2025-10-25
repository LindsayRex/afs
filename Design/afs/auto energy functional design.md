                 ┌─────────────────────────────────────────────────┐
                 │                 PROBLEM INPUTS                  │
                 │  • Data (signals/graphs/arrays)                 │
                 │  • Constraints (budgets, bounds, topology)      │
                 │  • Intent/KPIs (what “good” means)              │
                 └───────────────┬─────────────────────────────────┘
                                 │
                                 ▼
                    ┌───────────────────────────────┐
                    │  ATOM LIBRARY (lego pieces)   │
                    │  • Fidelity terms             │
                    │  • Priors (TV, L2, wavelet L1)│
                    │  • Couplers (multiscale, graph)│
                    │  • Lenses (bases: Haar, db4…) │
                    │  • Projections (feasible sets)│
                    └───────────────┬───────────────┘
                                    │ candidates
                                    ▼
                 ┌───────────────────────────────────────────┐
                 │        META–ENERGY (INTENT LAYER)         │
                 │  E_meta(S) = ValLoss(S)                   │
                 │              + λ_cert · CertPenalty(S)    │
                 │              + λ_mdl  · ModelSize(S)      │
                 │  where S = chosen atoms + weights + lens  │
                 └───────────────┬───────────────────────────┘
                                 │ drives the search
                                 ▼
        ┌──────────────────────────────────────────────────────────┐
        │      STRUCTURE SEARCH (propose → test → accept)          │
        │                                                          │
        │  [1] PROPOSE: add/scale one atom or switch basis         │
        │      S' = S ⊕ {atom, weight, lens}                       │
        │                                                          │
        │  [2] COMPILE ENERGY (MATH):                              │
        │      E_S'(x) = Σ_i λ_i E_i(x)                            │
        │      (symbolic, readable, differentiable)                │
        │                                                          │
        │  [3] COMPILE FLOW (SOLVER):                              │
        │      x_{k+1} = Prox_nonSmooth( x_k - α ∇Smooth )         │
        │      (+ multiscale step W, + projections)                │
        │                                                          │
        │  [4] RUN INNER SOLVE (few iters):                        │
        │      compute:                                            │
        │        • ValLoss (on held-out data)                      │
        │        • Certificates:                                   │
        │            - Lyapunov: E(x_{k+1}) ≤ E(x_k)               │
        │            - Spectral gap γ (in chosen basis)            │
        │            - Diagonal dominance η_dd                      │
        │                                                          │
        │  [5] SCORE: ΔValLoss, Δγ, Δη_dd, ΔModelSize              │
        │                                                          │
        │  [6] ACCEPT / REJECT:                                    │
        │      accept S' iff (ValLoss↓ AND certificates improve)   │
        │      else revert                                         │
        └───────────────────┬──────────────────────────────────────┘
                            │ iterate until no helpful move
                            ▼
               ┌─────────────────────────────────────────┐
               │   DISCOVERED PROGRAM  (WHITE–BOX)       │
               │  • Explicit energy: E*(x)=Σ λ*_i E_i(x) │
               │  • Chosen basis W*                      │
               │  • Minimal flow code (prox/grad steps)  │
               │  • Certificate report (descent, γ, η_dd)│
               └───────────────┬─────────────────────────┘
                               │ deploy
                               ▼
        ┌──────────────────────────────────────────────────────────┐
        │                  EXECUTION / GENERALIZATION              │
        │  New inputs from same regime → run flow → solution x*   │
        │  - Monotone descent holds (Lyapunov)                    │
        │  - Invariance gates (e.g., path order preserved)        │
        │  - Scale knobs (wavelet basis, CSR graphs, NUMBA)       │
        └──────────────────────────────────────────────────────────┘
