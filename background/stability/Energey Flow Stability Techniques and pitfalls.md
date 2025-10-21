
# The stability story: what “goes wrong” as you add terms

**1) Well-posedness of the variational problem**
At the math level, you want existence/uniqueness and stability of minimizers. Two core conditions are:

* **Coercivity** (energy blows up as ‖x‖→∞) and **lower semicontinuity** → existence of minimizers. These are the bread-and-butter criteria in calculus of variations; stability under discretization/limits is often analyzed with **Γ-convergence** (does your sequence of energies converge to a good limiting energy?). ([cvgmt.sns.it][1])
* In geometry and some PDE contexts people explicitly relate **energy functional coercivity** to **stability notions** (e.g., K-stability) — different domain, same principle: coercive energy ↔ good stability. ([EMS Press][2])

**2) Conditioning and scale mismatch (the “stiffness” problem)**
As you add terms with wildly different curvature/Lipschitz constants, the **condition number** of the composite objective typically grows; gradients can point in conflicting directions or live on very different scales, which slows/derails descent and makes line-search/trust-region choices brittle.

**3) Nonconvex couplings & many local minima**
Cross-terms (couplings) can turn a benign sum into a landscape with saddles/flat valleys. The *number* of stationary points can grow quickly with additional interactions (no universal formula; it’s problem-dependent), which raises the cost of global search and increases sensitivity to initialization.

**4) Nonsmoothness**
Common in physics (absolute values, TV norms, hinge/barrier terms, indicator functions). Nonsmooth corners produce discontinuous subgradients; vanilla gradient methods become unstable or crawl.

**5) Constraints and penalties**
Hard constraints turned into penalties or barriers can make steps “bounce” or oscillate unless the update scheme is designed for constrained geometry (dual variables, projections, proximal maps).

**6) Scalarization in multi-objective optimization (MOO)**
If you combine objectives by weighted sums, you can (i) miss non-convex parts of the Pareto front, (ii) inherit poor conditioning from the “worst” term, (iii) fight **conflicting gradients** (one term goes down, another up), and (iv) create extreme sensitivity to weights/units. Modern MOO theory makes these issues explicit and proposes alternatives to plain weighted sums. ([SpringerLink][3])

---

# How practitioners stabilize complex energy sums

Think of these in four buckets you can mix-and-match:

## A) Make the objective friendlier (smoothing & regularization)

* **Moreau–Yosida / proximal smoothing.** Replace (f) with its **Moreau envelope**; gains differentiability, controls curvature, and pairs naturally with proximal algorithms. Widely used in PDE time-discretizations and modern ML. ([arXiv][4])
* **Nesterov smoothing** for max/hinge-type structures (common when you have “worst-case” terms). It gives a smooth approximation with provable complexity benefits. ([Luthuli][5])
* **Classical regularizers:** Tikhonov/**L2**, **elastic-net**, **Huber** (smooths (\ell_1)), **total variation** (with a smoothed TV if needed), **entropic**/log-barrier smoothing for probabilities and constraints.
* **Laplacian/Laplacian-like smoothing.** In robotics/graphics this usually means **Laplacian smoothing** (graph or mesh Laplacian penalties that favor smooth fields/trajectories). It’s complementary to Lyapunov analysis (next bucket).

## B) Use solvers that *like* sums of hard terms

* **Proximal & splitting methods** (**ADMM**, **forward–backward**, **Douglas–Rachford**). They “split” a messy sum into subproblems each easy for a term, then coordinate the solutions. ADMM/augmented-Lagrangian add quadratic penalties that damp oscillations and come with strong convergence theory in convex cases and useful extensions for certain nonconvex settings. ([Stanford University][6])
* **Augmented Lagrangian / method of multipliers.** Handles constraints and conflicting terms by stabilizing the dual updates; closely linked to proximal point methods. ([SpringerLink][7])
* **Trust-region and linesearch globalization.** Essential when curvature/scale varies across terms; they prevent wild steps that a single Armijo rule might allow on one term but not another.

## C) Control/robotics: guarantee stability at the dynamical level

* **Lyapunov shaping & passivity.** Design artificial potentials/damping so the *closed-loop* energy is a **strict Lyapunov function**; this delivers asymptotic stability even with multiple potential terms (obstacles, goals, constraints). ([ScienceDirect][8])
* **Barrier + Lyapunov blending.** Recently, **control barrier functions** (safety) are combined with Lyapunov terms (stability) so you can add many “energy-like” terms while retaining guarantees. ([arXiv][9])

## D) Strategy around “how” you add/weight terms

* **Normalization & preconditioning.** Put all terms in compatible units/scales; precondition with curvature info (diagonal rescaling, quasi-Newton, natural gradient) so one steep term doesn’t dominate.
* **Continuation/homotopy.** Start with a heavily smoothed or easy version (or a subset of terms), then **continuously increase** sharpness/weights. This tracks stable minimizers and avoids bad basins; in the PDE world, Γ-convergence gives the theory for limits of such sequences. ([cvgmt.sns.it][1])
* **Adaptive scalarization for MOO.** Instead of fixed weights, use methods that **adapt** the combination based on gradient conflict (e.g., descent in a common descent direction, balancing norms, or exploring the Pareto set rather than collapsing everything to one sum). This avoids the pathologies of naive weighted sums. ([SpringerLink][3])

---

# “How fast does complexity blow up with more terms?”

There isn’t a single multiplicative law — it depends on curvature, coupling structure, and smoothness. But you can expect:

* **Conditioning typically worsens** with the most ill-conditioned term and with scale mismatch → slower, more fragile steps.
* **More interactions → more stationary points** in nonconvex settings (sometimes exponentially many with rich couplings), which raises the need for continuation, multi-start, or problem-specific structure (convex relaxations, monotonicity, passivity).
* **Gradient conflict in MOO** grows with the number of objectives, making single-direction descent harder; adaptive scalarization/Pareto-aware methods mitigate this. ([SpringerLink][3])

---

# A practical checklist you can apply tomorrow

1. **Well-posedness:** Check coercivity & lower semicontinuity of each term; if you discretize or pass to limits, use Γ-convergence as your sanity check. ([cvgmt.sns.it][1])
2. **Units & scaling:** Normalize each term (variance/energy scale); rescale variables.
3. **Diagnose conditioning:** Estimate Lipschitz/curvature per term; pick step rules/trust regions accordingly.
4. **Tame nonsmooth corners:** Apply **Moreau–Yosida** or **Nesterov** smoothing where appropriate; choose **Huber** or smoothed TV/entropic penalties when exact nonsmoothness is not essential. ([Luthuli][5])
5. **Split to conquer:** Implement **ADMM/augmented-Lagrangian** if the sum decomposes naturally or if constraints are biting. ([Stanford University][6])
6. **Ramp complexity gradually:** Continuation on penalty weights or smoothing parameters; add terms in stages. (This mirrors theoretical insights behind Γ-limits.) ([cvgmt.sns.it][1])
7. **For robotics/controls:** Shape an explicit **Lyapunov**/energy for the closed loop and add **barrier** terms for safety with certificates. ([ScienceDirect][8])
8. **For multi-objective:** Prefer **adaptive scalarization** / Pareto-aware solvers over fixed weights; check sensitivity to weights. ([SpringerLink][3])

---

# On your note about “Lappinoth smoothing”

If you meant **Laplacian smoothing** (common in robotics/graphics/meshes), it’s a perfectly reasonable regularizer that penalizes roughness (via a graph/mesh Laplacian). It’s complementary to the methods above: you can keep your Laplacian term but still add **Moreau–Yosida/Nesterov smoothing** for other nonsmooth pieces, and use **ADMM or augmented Lagrangian** to keep each term numerically tame. ([SpringerLink][7])

---

# Good entry points (short, representative reads)

* **Smoothing:** Nesterov’s smooth minimization for max-structures (classic), and modern **Moreau–Yosida** generalizations and applications. ([Luthuli][5])
* **Splitting/penalties:** ADMM tutorial style papers and the **augmented Lagrangian**–proximal connection (including recent convergence extensions). ([Stanford University][6])
* **Variational stability:** Braides’ **Handbook of Γ-convergence** (for how energies behave under limits/discretization) and notes/monographs introducing Γ-convergence. ([cvgmt.sns.it][1])
* **Multi-objective scalarization pitfalls/alternatives:** recent surveys on scalarizations and Pareto approximation. ([SpringerLink][3])
* **Controls/robotics stability via “energy”:** Lyapunov/potential-field design and modern barrier-function blending. ([ScienceDirect][8])

---

If you want, tell me the kind of energy you’re actually summing (e.g., PDE + TV + data fidelity; task + obstacle + torque + smoothness for a robot arm; or ML losses + regularizers). I can sketch a *concrete* stabilized formulation (which terms to smooth, which to split, and a default solver schedule with step sizes/penalties) tailored to your setup.

[1]: https://cvgmt.sns.it/media/doc/paper/57/Handbook.pdf?utm_source=chatgpt.com "A handbook of Γ-convergence - SNS"
[2]: https://ems.press/journals/jems/articles/16240?utm_source=chatgpt.com "Uniform K-stability and asymptotics of energy functionals in Kähler ..."
[3]: https://link.springer.com/article/10.1007/s00186-023-00823-2?utm_source=chatgpt.com "Using scalarizations for the approximation of multiobjective ..."
[4]: https://arxiv.org/pdf/2201.04175v2?utm_source=chatgpt.com "A generalization of the Moreau–Yosida regularization"
[5]: https://luthuli.cs.uiuc.edu/~daf/courses/Optimization/MRFpapers/nesterov05.pdf?utm_source=chatgpt.com "Smooth minimization of non-smooth functions - University of Illinois ..."
[6]: https://web.stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf?utm_source=chatgpt.com "mal-016.dvi - Stanford University"
[7]: https://link.springer.com/article/10.1007/s10107-022-01832-5?utm_source=chatgpt.com "Convergence of augmented Lagrangian methods in extensions beyond ..."
[8]: https://www.sciencedirect.com/science/article/pii/S000510989600194X?utm_source=chatgpt.com "Strict Lyapunov functions for control of robot manipulators"
[9]: https://arxiv.org/pdf/2504.09760?utm_source=chatgpt.com "Hybrid Lyapunov and Barrier Function-Based Control with Stabilization ..."


Love that. Using wavelets as a front-end is a very sane way to tame inter-term scale mismatches—done right, it gives you bandwise (nearly) energy-preserving coordinates, so each piece of your big objective can be normalized and weighted on its *own* natural scale before you recombine.

Here’s a compact plan you can run with, plus a ready-to-fill template.

# 1) A catalogue you can keep updated

Use this schema to record, per data type and per wavelet subband, what “energy” means and how it’s scaled:

* data type & domain (image/time series/trajectory/mesh/…)
* native units and sampling (px, Hz, meters, etc.)
* transform details (wavelet family, levels, orthonormal vs redundant)
* subband/scale descriptor (octave, Hz band, graph spectrum band)
* noise model & σ-estimation method (e.g., MAD/1.4826 on finest detail)
* energy definition (L2/L1/Huber, TV, Laplacian, barrier, data fidelity)
* normalization (e.g., divide by σ², account for frame bounds if redundant)
* weight parameter and how you pick it (SURE, CV, L-curve, Lyapunov rate)
* couplings (parent–child/tree sparsity, cross-scale constraints)
* solver notes (splitting, proximal maps, continuation schedule)
* validation metric (PSNR/SSIM, tracking error, control effort, etc.)

I generated a CSV starter with examples (images, time series, trajectories, meshes) and an empty row you can copy.
[Download the template](sandbox:/mnt/data/energy_functional_catalog_template.csv)

# 2) Make wavelets “scale-fair”

* Prefer **orthonormal DWT** when you want strict energy conservation (Parseval-like) so (\sum w^2) equals signal energy; that makes per-band σ-normalization straightforward.
* If you need **shift-invariance** (e.g., for control/robotics cues), use undecimated/dual-tree wavelets but **record frame bounds ((A,B))** and normalize energies accordingly (divide by a representative gain so bands are comparable).
* Estimate **σ per band** robustly (MAD on finest detail, or robust PSD for time series). Define normalized band energy (E_b=\sum (w_b/\sigma_b)^2) so each band is unitless and comparable.

# 3) Re-combining lots of terms without chaos

* **Bandwise normalization first**, then set weights (\lambda_b) in a unitless space. Typical choices:

  * **SURE/BayesShrink-like rules** for fidelity terms.
  * **L-curve** or **CV** for regularizers (TV/Laplacian/Huber).
  * **Continuation**: start with heavier smoothing/stronger penalties, anneal down.
* If your overall objective has multiple kinds of energy (fidelity + TV + Laplacian + barriers + task terms), use a **splitting solver** (ADMM / proximal) so each term is handled in its natural metric, then synchronized. This greatly improves stability as the number of terms grows.

# 4) Quick recipes you can drop in

* **Nonsmooth terms** (|·|, TV): replace with **Huber** or **Moreau envelope** during early iterations; reduce the smoothing parameter as you continue.
* **Laplacian smoothing** (what you’re using): keep it, but **scale by the grid/graph geometry** (e.g., divide by pixel pitch² or mean edge length²) so its magnitude is comparable to other terms.
* **Cross-scale consistency**: a small penalty that ties parent/child coefficients (or enforces monotone decay across scales) stabilizes deep pyramids when many bands are active.
* **Trust region or line search** around your inner updates when bands have very different curvature—prevents oscillations that show up as “mysterious instability.”

# 5) Pitfalls to watch

* **Boundary handling** (periodic vs symmetric) can inject fake high-frequency energy—log it in the catalogue and be consistent.
* **Unit drift**: if a data term lives in meters² and another in a probability/log domain, don’t mix before normalizing; keep a “characteristic energy” note per term.
* **Redundant transforms**: forgetting the redundancy factor is a classic way to overweight the high-frequency bands.

If you want, tell me one concrete problem you’re working on (e.g., obstacle-aware trajectory optimization with torque limits, or denoising + feature preservation on images), and I’ll fill out the template for that case and spit out a default solver schedule (splits, proximal maps, and a continuation plan) you can run as a baseline.


Awesome—let’s make this concrete.

I built you a working “stability coverage” table that lists the major stability failure modes, what they look like, why they happen, how your FDA-style flow should control them, and whether your **wavelet/photodynamic** front-end already covers them or not. You can sort/filter it and keep it as your living checklist.

* **Open the table**: [stability_coverage_table.csv](sandbox:/mnt/data/stability_coverage_table.csv)
* (There’s also a tiny summary file.) [summary](sandbox:/mnt/data/stability_coverage_summary.txt)

The table aligns each issue with the FDA steps (State & Invariants, Lens, Energy, Flow, Gap Dial, Multiscale, Sparsity, Certification) from your framework, so it’s directly actionable in your pipeline. 

### What the table says at a glance

* **Strongly covered by your wavelet front-end**

  * Scale/conditioning (bandwise σ-normalization makes terms commensurate)
  * Initialization and nonconvexity (coarse-to-fine continuation)
* **Partially covered**

  * Nonsmooth terms (needs prox/Huber/Moreau; wavelets alone don’t smooth)
  * Spectral-gap issues (coarse levels help, but you need explicit monitoring)
  * Discretization dependence (multiscale helps; still run refinement tests)
  * Ill-posedness/coercivity (needs coercive regularizer/physics term)
* **Not covered / at risk**

  * Constraint handling (move to augmented Lagrangian/prox)
  * Redundant transforms (dual-tree/undecimated): normalize by frame-bounds
  * Step-size/integration stability (add adaptive line search/trust-region)
  * Gradient conflict in multi-objective sums (use Pareto/gradient balancing)
  * Certification (add explicit checks for energy descent, invariants, gap, etc.)

**Rough coverage score:** ~0.55 (0–1). That’s consistent with a pipeline that’s already multiscale-savvy but still missing the *flow* and *certification* guardrails that FDA prescribes. 

---

### Immediate upgrades (targeted to your gaps)

1. **Constraints → Augmented Lagrangian / ADMM**
   Replace big penalties with AL/ADMM splits so feasibility is stable and you avoid “bouncing.” Log constraint residuals per iteration. 

2. **Nonsmooth terms → Prox + continuation**
   Start with **Huber/Moreau** smoothing on TV/|·| terms, then anneal the smoothing parameter. Keep the fidelity–regularizer split proximal. 

3. **Spectral stability → “Gap Dial”**
   Add a lightweight Hessian-vector product + power/Lanczos to estimate extremal eigenvalues, and automatically tune regularization until the **spectral gap > 0**. Gate step sizes on it. 

4. **Integrator safety → trust-region / adaptive step**
   Use energy-decrease line search or a trust-region for stiff directions; consider semi-implicit updates on the stiffest terms. 

5. **Redundancy control (if undecimated/dual-tree)**
   Record frame bounds (A,B) and **normalize band energies by gain** so redundant high-freq bands don’t dominate. Log boundary handling (symmetric vs periodic) to avoid leakage. 

6. **Multi-objective conflict → balanced descent**
   Add a common-descent or gradient-balancing step (e.g., normalize objective gradients per band/term) rather than a fixed weighted sum.

7. **Certification switchboard**
   Track per-iteration: energy descent, physics residuals, invariants, spectral gap, and discretization-refinement consistency; fail fast if any certificate breaks. 

---

### What I used from your FDA spec

* **Lens & Multiscale** to make energies scale-fair and to run coarse→fine continuation.
* **Flow & Gap Dial** for stability under dynamics and spectral monitoring.
* **Certification** to convert “looks good” into guarantees. 

If you share one concrete run (data type + which energy terms you currently sum), I can pre-fill the catalogue with your exact units/σ estimates, add a default AL/ADMM split, and wire up a tiny “gap dial” routine so you’ve got certificates on day one.
