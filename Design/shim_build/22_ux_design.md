
github repo:

https://github.com/juliangarnier/anime




# Flow Card (one screen, two stacks) UX Design

## 1) “What am I building?” (the Math Card)

**Goal:** show the *score formula* you’ve composed (the energy functional) in human words + readable math.

* **Title:** *Score formula*
  Short sentence: “We’re balancing fit to data, smoothness, and sparsity.”
* **Rendered math (LaTeX):** the actual (E(x)=\sum w_i E_i(x)) you compiled (pull terms from the spec) — e.g.
  (\tfrac12|Ax-y|_2^2 + \lambda|\mathcal{W}x|_1 + \tau\mathrm{TV}(x))
  (You already encode terms/weights in your EnergySpec; just render them.) 
* **Term chips:** each term appears as a pill with its plain name + current weight (e.g., “Fit to data · 1.0”, “Sparsity · 0.02”, “TV · 0.01”). The chip *pulses* when its gradient contribution is large in the current step.
* **Flow diagram:** a left→right mini chain showing the active primitives as simple verbs:

  * **“descend”** (F_Dis) → **“tidy”** (F_Proj) → **“change view”** (F_Multi) → *(optional)* **“preserve”** (F_Con)
    Labels are verbs, not symbols; hovering shows the precise primitive name for experts later.

**Micro-animation meanings (anime.js):**

* **Chip pulse** = “this term is driving the change right now.”
* **Chain glow passing through blocks** = “a step just executed through these stages.”

---

## 2) “Is it healthy and how does it evolve?” (the Stability Card)

**Goal:** convert certificates and events into 4 engineer words: **Health, Confidence, Speed, Complexity.**

* **Health (now):** big status light

  * **GREEN** “settling” (energy kept falling recently)
  * **AMBER** “unsure” (flat/oscillating)
  * **RED** “rollback” (we undid a bad move)
    (Wire this to your controller phases / events.) 

* **Confidence (structure check):** two small gauges with *plain labels*:

  * **Coupling index** (lower is safer) ← maps to `eta_dd`
  * **Settling rate** (higher is better) ← maps to `gamma`
    Tooltips:
    “Coupling index compares cross-talk vs. self-strength; keep it < 0.9.”
    “Settling rate is the slowest mode’s speed; aim above your minimum.”
    (These are your diagonal-dominance & gap ideas, but without the Greek.) 

* **Speed (trend):** a sparkline of recent **score** (E) with dots; **downhill drift** = good. If a dot pops upward, it briefly *shakes* (Armijo/rollback moment).

* **Complexity (shape):**

  * **Flow size:** count of active terms & constraints (chips filled vs. hollow).
  * **Detail level:** current multiscale level (“working at fine details: level 4/6”).
  * **View/basis:** the chosen “view” name (lens) — “Haar”, “db4”, or “Graph view” — with a subtle *focus* animation when it changes. 

* **Why did it change?** an event ticker in plain language:

  * “Tried to increase sparsity knob → **rejected** (too much coupling).”
  * “Switched to **db4 view** for better compression.”
  * “Unlocked **detail level 3**; coarse pass looks good.”
    (Directly from `events.parquet` enums, rewritten in humanese.) 

---

# The plain-language dictionary (so nobody has to remember jargon)

| What we show       | What it really is                  | Why it matters                                          |
| ------------------ | ---------------------------------- | ------------------------------------------------------- |
| **Score formula**  | Energy functional (E=\sum w_i E_i) | This is the thing you’re minimizing.                    |
| **Term weight**    | coefficient (w_i) / tuners         | Dials that trade off goals.                             |
| **Coupling index** | `eta_dd` (diag-dominance ratio)    | Cross-talk vs. self-strength; < 0.9 is comfy.           |
| **Settling rate**  | `gamma` (spectral gap)             | How fast the slowest mode dies out; bigger = steadier.  |
| **Sparsity knob**  | global `lambda` (and `lambda_j`)   | Pushes for simpler answers; too high can destabilize.   |
| **View / basis**   | `lens_name` (e.g., wavelets/graph) | The coordinate system where the problem is easiest.     |
| **Detail level**   | `level_active_max`                 | What scale we’re refining right now.                    |
| **Health**         | Phase RED/AMBER/GREEN              | Your immediate “is it okay?” signal.                    |

---

# Minimal data contract (so the HUD can run)

From your telemetry, we only need a tiny subset to drive everything:

```json
{
  "iter": 128,
  "phase": "AMBER",
  "E": 12.345,
  "grad_norm": 0.78,
  "lambda": 0.02,
  "lambda_j": {"L1": 0.03, "L2": 0.02},
  "eta_dd": 0.72,
  "gamma": 1.2e-6,
  "lens_name": "db4",
  "level_active_max": 3,
  "active_terms": [
    {"name":"Fit to data","weight":1.0,"grad_share":0.62},
    {"name":"Sparsity","weight":0.02,"grad_share":0.28},
    {"name":"TV","weight":0.01,"grad_share":0.10}
  ],
  "events":[
    {"t":"CERT_PASS","msg":"All checks passed"},
    {"t":"TUNER_MOVE_REJECTED","msg":"Sparsity increase raised coupling"}
  ]
}
```

All of these fields align with your current schema & hooks; you already log them or have them available. 

---

# Animation rules that carry meaning (anime.js-friendly)

* **Health flip:** phase changes rotate the status card 180° (GREEN/AMBER/RED).
* **Gauge glow:**

  * Coupling index > 0.9 → brief horizontal *shake*.
  * Settling rate < threshold → soft *flicker* to draw attention.
* **Term pulses:** pulse intensity ∝ `grad_share` so you can *see* which parts of the score are doing work.
* **Lens focus:** when `lens_name` changes, a quick zoom-in/out on “View”.
* **Scale rings:** concentric rings; activating a finer level lights the next ring.
* **Tuner snap:** accepted change → elastic settle; rejected change → rubber-band snap back with a “why” tooltip.

These map 1:1 to your controller, tuner, multiscale schedule, and events — no extra theory required. 

---

# Optional side tabs (when you’re ready)

* **Pareto**: a small 2D/3D panel where the current run dot *slides* along a front; only **GREEN** runs get bright color. Useful when business weights are explored by policy. 
* **Structure**: the same Score formula, but expanded as a tidy *term tree* pulled from your Atom Library registry (so engineers can spot redundant pieces fast). 
* **Source**: the generated spec file snippet for reproducibility (helps code review). 

---

# Why this helps a flow engineer

* They **see the program** they just composed (not a black box).
* They **read health in plain words**: Health / Confidence / Speed / Complexity.
* Every animation **answers a question** (which term is active? are we stable? what just changed?).
* It builds a shared **visual language** the team can use during reviews without Greek letters.

If you want, I can turn this into a minimal HTML+anime.js scaffold that reads the JSON above and renders both cards, so you can drop it next to your run output and iterate from there. 
