# Design Pattern: Functional Core, Imperative Shell

This document outlines the primary architectural pattern for the Computable Flows Shim. This pattern, combined with rigorous development methodologies, is chosen to enforce modularity, testability, and correctness, and to prevent the kind of cascading failures and self-deception caused by placeholder or skeleton code.

## 1. The Core Pattern: Functional Core, Imperative Shell

Our architecture is strictly divided into two parts:

*   **The Functional Core:** This is the heart of the system. It contains all the pure, mathematical logic.
    *   **What it is:** A collection of pure functions that take data in and return data out. They have zero side effects (no file I/O, no logging, no global state).
    *   **Our Core:** The five primitive operators (`F_Dis`, `F_Con`, etc.), the energy compiler, and the mathematical definitions from the `atoms_library.md`.
    *   **Why:** A pure functional core is easy to test, reason about, compose, and reuse. It is the stable engine that your future Automatic Flow Synthesizer (AFS) will drive.

*   **The Imperative Shell:** This is the thin outer layer that interacts with the world.
    *   **What it is:** The "impure" part of the code that handles all side effects.
    *   **Our Shell:** Test scripts, the main application entry point, file readers, and our QA log writers.
    *   **Why:** It isolates the complex, messy parts of the system, leaving the core logic clean and verifiable.

---

## 2. Integrating Rigorous Methodologies

The following methodologies are not alternatives to this pattern; they are complementary practices that we will use to build the system correctly and prevent self-deception.

### a. Test-Driven Development (TDD)

TDD is the **workflow** we will use to build the Functional Core. It directly combats the "skeleton code" problem.

*   **How it Works (Red-Green-Refactor):**
    1.  **(RED):** We will first write a failing test in the Imperative Shell (e.g., in a `tests/` file). The test will call a function in the Core that doesn't exist yet, or that has a deliberately incorrect implementation.
    2.  **(GREEN):** We will then write the simplest possible *real, working code* in the Functional Core to make the test pass.
    3.  **(REFACTOR):** We will clean up the code we just wrote.

*   **How it Prevents Self-Deception:** It is impossible to be fooled by a placeholder if the very first step is to write a test that proves the placeholder is wrong. It forces us to write verifiable, working code from the start.

### b. Design by Contract (DbC)

DbC is how we will ensure our functions are **mathematically and logically correct**. We will make our assumptions explicit.

*   **How it Works:** We define formal "contracts" for our functions:
    *   **Preconditions:** What must be true about the inputs. (e.g., "the input `state` dictionary must contain a 'q' key").
    *   **Postconditions:** What the function guarantees about its output. (e.g., "the output energy will be less than or equal to the input energy").
    *   **Invariants:** Properties that remain unchanged.

*   **How it Prevents Self-Deception:** The mathematical properties in our design documents (`On_Computable_Flows_v2.1.md`) *are* our contracts. For `F_Con`, the contract is that it preserves the Hamiltonian's energy (within a certain tolerance). For `F_Dis`, the contract is that the energy functional decreases. We will enforce these contracts with assertions and specific tests, making it impossible to accidentally violate the underlying physics.

### c. Behavior-Driven Development (BDD)

BDD is how we will ensure we are building the **right thing** from a user's or system's perspective. It's a way of structuring our tests and requirements.

*   **How it Works (Given-When-Then):** We will frame our tests and QA logs in a human-readable format that describes behavior.
    *   **Given** a specific energy functional (like the quadratic atom).
    *   **When** we run the dissipative flow for one step.
    *   **Then** the system's state should be measurably closer to the minimum.

*   **How it Prevents Self-Deception:** It forces us to think about the *purpose* and *observable outcome* of the code, not just its internal mechanics. This ensures we are always building a component that serves a meaningful, verifiable purpose in the larger system.

### d. Formal Verification

Formal Verification is how we will **connect our code back to the mathematical proofs**.

*   **How it Works:** Our design documents contain formal lemmas and theorems about the behavior of flows. While we won't use a full-blown proof assistant, we will treat these documents as our "proof obligations."

*   **How it Prevents Self-Deception:** Our numerical tests will serve as the **computational verification** of these formal proofs. When we test that `F_Con` conserves energy, we are computationally verifying Lemma 2 ("Energy Conservation in Conservative Flows"). This ensures our implementation is a faithful representation of the proven-correct mathematical theory, not just code that "seems to work."

## Conclusion

By combining the **Functional Core, Imperative Shell** architecture with the practices of **TDD, DbC, BDD, and Formal Verification**, we create a robust, multi-layered methodology. This approach forces us to be honest, build verifiable components in isolation, and ensure that what we build is not only correct in its implementation but also true to its mathematical and behavioral purpose.
