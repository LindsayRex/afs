# QA Log: 2025-10-20 - Command-Line Interface (Imperative Shell)

**Component:** `src/computable_flows_shim/cli/run.py`

## Goal
To build the first piece of the "Imperative Shell": a simple command-line interface (CLI) that uses the "Functional Core" (compiler, primitives, step function) to run a complete flow from end to end.

## Process

1.  **Create:** The file `src/computable_flows_shim/cli/run.py` was created.
2.  **Implement:** The script was implemented to perform the full, end-to-end sequence:
    *   Define a hardcoded `EnergySpec` for a composite problem (quadratic + L1).
    *   Call `compile_energy` to get the compiled functions.
    *   Initialize a starting state.
    *   Loop for a fixed number of iterations, calling `run_flow_step` in each iteration.
    *   Print the state at each step to the console.
3.  **Verify:** The script was executed from the command line. The output was manually inspected to confirm that the flow ran as expected and the state converged towards the correct minimum.

## Outcome
- We have successfully built our first **Imperative Shell**.
- We have demonstrated that the entire "Functional Core" can be driven by an external script, proving the effectiveness of our chosen architectural pattern.
- This provides a complete, end-to-end "vertical slice" of the entire application, from problem definition to final result.
- We now have a simple, powerful tool for manually running and debugging our flows as we continue to expand the system's capabilities.
