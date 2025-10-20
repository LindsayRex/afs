# AFS: Automatic Flow Synthesizer

This repository contains the engine for a novel computational paradigm: casting computation as a physical process of energy minimization over a state space. This is a "physics-first" approach, where the physics of a problem directly defines a transparent, stable, and performant algorithm.

The ultimate goal is to create an **Automatic Flow Synthesizer (AFS)**, an AI that can automatically discover the optimal computational flow for a given business problem.

## The Core Engine: The Computable Flow Shim

The heart of this project is the **Computable Flow Shim**, a small, reliable, and reusable runtime engine built on JAX. It is not a complete application, but a "Functional Core" that the AFS will drive.

The Shim is built on the composition of five primitive, continuous-time dynamical flows:
1.  **Dissipative (`F_Dis`):** Gradient descent, for minimizing energy.
2.  **Projective (`F_Proj`):** Proximal operators, for enforcing constraints.
3.  **Conservative (`F_Con`):** Symplectic integrators, for preserving physical quantities.
4.  **Multiscale (`F_Multi`):** Wavelet transforms, for representing information efficiently.
5.  **Stochastic (`F_Ann`):** Langevin dynamics, for exploration.

By composing these primitives, we can construct algorithms to solve a vast class of optimization problems.

## Development Methodology

This project is built with a strict, verifiable methodology designed for correctness and collaboration with AI agents. Any contributor (human or AI) **must** adhere to these principles.

1.  **Architectural Pattern:** We use a **Functional Core, Imperative Shell** pattern. All mathematical logic is pure and testable; all side effects are isolated.
    *   **See:** `Design/shim_build/17_design_pattern.md`

2.  **Workflow:** We use **Test-Driven Development (TDD)**. We write a failing test first, then write the code to make it pass.
    *   **See:** `copilot-instructions.md`

3.  **Documentation:** We maintain a **QA Log** for every component built, providing a clear audit trail.
    *   **See:** `qa_logs/`

## Getting Started

1.  **Understand the Math:** The foundational mathematics are described in `background/On_Compitable_Flows_v2.1.md`.
2.  **Understand the Architecture:** The Shim's architecture and naming conventions are in `Design/shim_build/`.
3.  **Follow the Rules:** Read and adhere to the `copilot-instructions.md`.

To set up the environment:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest -q
```
