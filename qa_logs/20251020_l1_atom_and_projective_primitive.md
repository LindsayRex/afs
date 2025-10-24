# QA Log: 2025-10-20 - L1 Atom and Projective Primitive

**Components:** 
- `src/computable_flows_shim/energy/compile.py`
- `src/computable_flows_shim/runtime/primitives.py`

**Test:** `tests/test_primitives.py::test_F_Proj_l1`

## Goal
To implement the second major pathway of the shim: handling non-smooth energy terms. This required implementing the **L1 Sparsity Atom** in the compiler and the **`F_Proj` (projective/proximal) primitive** in the runtime.

## Process (Red-Green-Refactor)

1.  **(Refactor First):** Modified the `compile_energy` function to return a `NamedTuple` (`CompiledEnergy`) containing `f_value`, `f_grad`, and `g_prox` functions. This was necessary to support both smooth and non-smooth terms. The existing `test_F_Dis_quadratic` was updated to use this new structure.

2.  **(RED):** Added a new test, `test_F_Proj_l1`. This test defined an `EnergySpec` with an `l1` term and attempted to call the non-existent `F_Proj` primitive. The test failed as expected with an `ImportError`.

3.  **(GREEN):**
    *   Implemented the `F_Proj` function in `primitives.py`.
    *   Added logic to `compile_energy` to recognize the `l1` term and generate a `g_prox` function that performs soft-thresholding.
    *   Fixed a regression in `test_F_Dis_quadratic` that was caused by the compiler refactoring.
    *   After these changes, all tests passed.

4.  **(REFACTOR):** The code is clean and required no further refactoring. This log entry completes the cycle.

## Outcome
- We have a working implementation of the `F_Proj` primitive.
- The compiler can now handle both a smooth (`quadratic`) and a non-smooth (`l1`) atom.
- We have successfully built and verified the second critical pathway of the flow engine:
  **Spec -> Compiler -> Proximal Operator -> Projective Primitive -> Verified State Update**
- This demonstrates the system's ability to handle composite energy functionals, which is central to the entire design.
