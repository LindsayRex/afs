This is the most critical question you've asked: **How do we formalize the $\mathcal{A}\mathcal{F}\mathcal{S}$ input and process so it's a repeatable science, not a black box?**

You are moving from **Synthesis of Parameters** to **Synthesis of the Program Structure itself** (the primitives and the atoms).

Here is the formalization of the $\mathcal{A}\mathcal{F}\mathcal{S}$ process, using your matrix multiplication problem as the template:

---

## The $\mathcal{A}\mathcal{F}\mathcal{S}$ Synthesis Protocol (The Repeatable Process)

The process is a **Gated Structural Search** that operates at two levels: **Structural Search** (what atoms/primitives) and **Parameter Search** (what weights).

### 1. The $\mathcal{F}\mathcal{P}\mathcal{E}$ Template (The Starting Gun)

An $\mathcal{A}\mathcal{F}\mathcal{S}$ run does not start with a blank slate. It starts with a **Flow Program Engineering ($\mathcal{F}\mathcal{P}\mathcal{E}$) Template** that defines the *type* of solution required:

*   **Goal:** A **Dissipative Flow** with a **Projection** step.
*   **Initial Structure:** $\mathbf{x}_{k+1} = \mathcal{F}_{\text{Proj}} \circ \mathcal{F}_{\text{Dis}}(\mathbf{x}_k)$
*   **The Matrix Multiplication Template (Base $\mathcal{E}$):** A flow that minimizes the total *work* required to compute $C = A \cdot B$.

### 2. The Atom Library and Primitive Choice

This is where the human expertise defines the search space.

| Primitive / Structure | Atom/Tool | Use Case in Matrix Multiplication |
| :--- | :--- | :--- |
| **Dissipative ($\mathcal{F}_{\text{Dis}}$)** | **Computational Intensity ($\alpha$)** | Rewards $O(N^3)$ work (the core algorithm). |
| **Dissipative ($\mathcal{F}_{\text{Dis}}$)** | **SIMD Utilization ($\epsilon$)** | Rewards vectorization efficiency (smooth gradient term). |
| **Projection ($\mathcal{F}_{\text{Proj}}$)** | **Tiling/Loop-Order ($\gamma, \delta$)** | The prox-step here is the actual $C_{ij} \leftarrow A_{ik} B_{kj}$ computation itself, framed as a projection onto the solution manifold. |
| **Multiscale ($\mathcal{F}_{\text{Multi}}$)** | **Wavelet/Fourier Lens ($\mathcal{W}$) ** | *Not necessary for dense matrix multiplication.* This is the critical $\mathbf{\times}$ in the atom library. It must be proven necessary by the $\mathcal{A}\mathcal{F}\mathcal{S}$ search before inclusion. |
| **Certificate** | **Spectral Gap ($\gamma$)** | The mandatory constraint that forces the flow to be fast and stable. |

**Repeatable Process Rule #1:** You only include an Atom/Primitive in the search if it is a plausible physics component. For matrix multiplication, **wavelets are not an initial primitive** because the problem is dense and translational-invariant.

### 3. The $\mathcal{A}\mathcal{F}\mathcal{S}$ Search Protocol (The Repeatable Process)

The $\mathcal{A}\mathcal{F}\mathcal{S}$ engine now performs a **Gated Greedy Search** over the simplified structure:

| Search Layer | Action | The "Gate" |
| :--- | :--- | :--- |
| **Initial Parameter Synthesis** | Search over $\alpha, \beta, \gamma, \dots$ weights for the Base $\mathcal{E}$ (Flow Contraction Objective). | $\mathbf{\gamma > \gamma_{\text{min}}}$ (The Flow Must Be Stable). |
| **Structural Augmentation (Optional)** | Propose adding a new Atom (e.g., the $\mathcal{F}_{\text{Multi}}$ Wavelet primitive). | $\text{IF } \text{Speedup}_{\text{new}} > \text{Speedup}_{\text{old}} \cdot 1.5 \text{ AND } \gamma_{\text{new}} \text{ remains stable}$. |
| **Final CFP Synthesis** | Search for the $\text{IF/THEN}$ logic across problem sizes. | $\mathbf{100\% \ \text{Correctness on the Generalization Pool.}}$ |

---

## Summary and Immediate Engineering Fix

The entire process is anchored by two critical points:

1.  **The Primitives are fixed.** You never need to "train" a primitive.
2.  **The Atoms are domain-specific physics.** (No wavelets for dense MM unless $\mathcal{A}\mathcal{F}\mathcal{S}$ proves it's necessary).

**The Immediate Engineering Fix is the only next step:**

*   **Goal:** Re-synthesize a single $\mathcal{F}\mathcal{P}\mathcal{E}$ for $200 \times 200$ that is **$200\times$ faster** and **Certified Stable.**
*   **Action:** Retune the $\mathcal{A}\mathcal{F}\mathcal{S}$ objective to heavily reward Wall-Clock Speed while enforcing the Spectral Gap.

The repeatable process is now clear: **Define the Template $\to$ Define the Atoms $\to$ Synthesize under the Spectral Gap Constraint.**