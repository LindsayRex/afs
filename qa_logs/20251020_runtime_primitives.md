# QA Log: Runtime Primitives

**Date:** 2025-10-20

**Author:** GitHub Copilot

**Goal:** To implement and test the core runtime primitives (F_Dis, F_Proj, F_Multi, F_Con) using a strict Test-Driven Development (TDD) and Design by Contract (DbC) methodology.

## TDD Cycle Summary

### F_Dis (Dissipative Step)

- **Contract:** The dissipative step must correctly descend the gradient of a given energy function. This was tested using a simple quadratic energy function.
- **RED:** A failing test, `test_F_Dis_quadratic`, was created to define this contract.
- **GREEN:** The `F_Dis` function was implemented as a simple gradient descent step, which made the test pass.
- **REFACTOR:** The implementation and test were reviewed for clarity and efficiency.

### F_Proj (Projective/Proximal Step)

- **Contract:** The projective step must correctly apply a proximal operator. This was tested using an L1 regularization term (soft-thresholding).
- **RED:** A failing test, `test_F_Proj_l1_contract`, was created to define this contract.
- **GREEN:** The `F_Proj` function was implemented as a simple wrapper around a provided proximal operator function, which made the test pass.
- **REFACTOR:** The implementation and test were reviewed and found to be clean and efficient.

### F_Multi (Multiscale Transform)

- **Contract:** The multiscale transform must be invertible, meaning that the forward and inverse transforms must be consistent.
- **RED:** A failing test, `test_F_Multi_contract_jaxwt`, was created using a real wavelet transform from the `jaxwt` library to define this contract.
- **GREEN:** The `F_Multi_forward` and `F_Multi_inverse` functions were implemented as simple wrappers around the provided transform object, which made the test pass.
- **REFACTOR:** The implementation and test were reviewed and found to be clean and efficient.

### F_Con (Conservative/Symplectic Step)

- **Contract:** The conservative step must conserve the energy of a Hamiltonian system over time.
- **RED:** A failing test, `test_F_Con_energy_conservation`, was created to verify energy conservation over a short trajectory of a simple harmonic oscillator.
- **GREEN:** The `F_Con` function was implemented using the Leapfrog/Stormer–Verlet integrator, which made the test pass.
- **REFACTOR:** The implementation and test were reviewed and found to be clean and efficient.

## Outcome

The core runtime primitives have been successfully implemented and tested in accordance with the project's TDD and DbC methodology. The tests in `tests/test_primitives.py` serve as the executable contracts for these primitives. The next step will be to integrate these primitives into the main runtime engine and connect them to the telemetry system.
