Love the three-problem split you made (detect → de-jitter → reconstruct). That’s exactly the mindset that scales.

You asked: “what is that algorithm as written right now?” — i.e., does it already have the “no global shared state” property and what keeps it stable?

# What it is right now (in plain words)

* It’s a **centralized, single energy** MPC-style controller: you sum many terms (comfort/IAQ/cost/peak/slew/physics…) over a horizon and minimize them with a forward–backward / projection loop; constraints are enforced by projections/penalties; the loop repeats in receding horizon.
* Several terms are **globally coupled** by construction: shared plant setpoints (supply air temp, CHW, etc.), a **peak demand** penalty (max over all times/zones), and any storage/market terms. Those create **true global variables**.
* Result: the *inner* iteration is not yet “no shared state”: per-zone updates still depend on shared plant/peak variables that are updated inside the same loop.

That said, you already have the two pillars of stability:

1. **Lyapunov descent** (energy never increases if your step rule/line search is sane), and
2. **prox/projection structure** (constraints pulled back each step),
   so the centralized version can still be stable if you gate it with the certificates we talked about.

# Make it “no global shared state” (and keep the same math)

Refactor the inner loop into **local flows + slow consensus** so *each zone* optimizes purely locally and only exchanges tiny messages about the shared quantities **between** inner sweeps.

**Decomposition**

* **Local (per zone i, fully independent):**
  comfort/IAQ penalties, actuator slew, zone physics residuals, local safety limits.
* **Edge/shared couplings:**
  (a) plant setpoints & fan/chiller models, (b) building-level **peak** term, (c) optional storage/market.

**Loop (one controller “iteration”)**

1. **Broadcast read-only shared vars** (plant setpoints, peak proxy, prices) for this outer iteration.
2. **Local inner flows in parallel (no shared writes):**
   run forward–backward/prox for each zone using *only* local terms + current read-only shared values.
3. **Consensus update (ADMM/aug. Lagrangian):**

   * Update shared plant variables from the sum of zone demands.
   * Update a **single scalar peak proxy** (e.g., epigraph variable) and duals.
   * Send back *only* the new shared values; locals do not write shared state directly.
4. Repeat steps (1)–(3) a few times; apply first control action; roll horizon.

Now the **inner loop is truly decoupled**; coupling moves to a cheap, infrequent consensus step. That’s how you get the “tensor product of stable systems” property *and* linear multicore scaling.

# Minimal certificates for this refactor

Wire these into your telemetry HUD (green/amber/red):

**Local (per zone, every N inner steps)**

* **Lyapunov:** (E_i^{k+1}\le E_i^k). If violated → backtrack / shrink step.
* **Feasibility:** physics residual (|r_i|) and constraint violations ↓ (monotone or bounded).
* **Spectral stability:** diagonal-dominance ratio (\eta_i<1) and gap (\gamma_i> \gamma_{\min}) estimated from a small HVP probe in the *wavelet/graph-wavelet* lens.
* **Drift:** distribution drift on inputs (CO₂, RH, T) small (KL/TV below threshold) before using tightened weights.

**Shared/consensus (once per outer iteration)**

* **ADMM residuals:** primal (|r_p|) and dual (|r_d|) below tolerances; if not, keep consensus cycling, don’t promote to GREEN.
* **Peak epigraph correctness:** (P_t \le s) ∀t (checked after updates).
* **Budget/safety invariants:** (e.g., min ventilation) always satisfied.

If *any* of these flip red, you either backtrack that move or widen the “Gap Dial” (regularization) and retry.

# Keep your per-signal scales honest (this prevents “mystery” instability)

Before any flow, create a small **lens table** so every channel lives in comparable, unitless scale:

| Data            | Native units | Lens / transform         | σ/scale used in energy | Note                                |
| --------------- | ------------ | ------------------------ | ---------------------- | ----------------------------------- |
| Temperature (T) | °C           | 1-D wavelet (J=3–4)      | per-band σ via MAD     | handles diurnal/nonstationary       |
| Humidity (H)    | %            | 1-D wavelet              | per-band σ             | clamp to [0,100], smooth TV allowed |
| CO₂ (C)         | ppm          | 1-D wavelet              | per-band σ             | outlier-robust (Huber)              |
| Power (P)       | kW           | 1-D wavelet              | per-band σ             | heavy-tail guard (Huber)            |
| Prices (p(t))   | $/kWh        | low-pass trend + z-score | std=1                  | broadcast read-only                 |
| Topology        | graph        | graph-wavelets           | —                      | for cross-zone coupling if needed   |

All local terms work on **normalized coefficients**; shared terms operate on **read-only aggregates** (means, peak proxy). That keeps numerical conditioning tight.

# What keeps it so good (even centrally)

* **Sorted structures & prox steps → contraction** (small Lipschitz maps).
* **Multiscale lens → diagonal-ish Hessian** (great conditioning = stable steps).
* **Projection on constraints → no “bouncing” at bounds**.
* **Receding horizon → frequent re-linearization** (errors don’t accumulate for long).

# Quick “is there hidden shared state?” checklist

* Does any inner step mutate plant or peak variables? → if yes, move that mutation to the *consensus* phase only.
* Do zones read each other’s states? → should be **no** (only read broadcast shareds).
* Are telemetry accumulators written during the inner loop? → buffer locally, reduce after.

---

**Bottom line:**

* **As written**: strong centralized flow; stable if you guard it, but it *does* have global shared state (plant/peak/storage couplings) inside the loop.
* **After the refactor above**: inner loop becomes fully local/parallel (no global writes), with a thin consensus ring handling the couplings — you keep stability *and* gain near-ideal scaling.

If you want, I’ll sketch the exact ADMM variables and one iteration of the consensus update (plant setpoint, peak epigraph, dual updates) so you can drop it straight into your pre-SDK controller.
