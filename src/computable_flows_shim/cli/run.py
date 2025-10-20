"""
A simple CLI to run a computable flow.
"""
import sys
from pathlib import Path
import jax
import jax.numpy as jnp

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from computable_flows_shim.energy.specs import EnergySpec, TermSpec, StateSpec
from computable_flows_shim.energy.compile import compile_energy
from computable_flows_shim.api import Op
from computable_flows_shim.runtime.step import run_flow_step

class IdentityOp(Op):
    """A simple identity operator for testing."""
    def __call__(self, x):
        return x

def main():
    """Main function for the CLI."""
    print("--- Starting Computable Flow Shim CLI ---")

    # 1. GIVEN a composite energy functional (quadratic + L1)
    spec = EnergySpec(
        terms=[
            TermSpec(type='quadratic', op='I', weight=1.0, variable='x', target='y'),
            TermSpec(type='l1', op='I', weight=0.5, variable='x')
        ],
        state=StateSpec(shapes={'x': [1], 'y': [1]})
    )
    op_registry = {'I': IdentityOp()}
    
    print(f"Problem Spec: {spec.terms}")

    # 2. Compile the energy functional
    compiled = compile_energy(spec, op_registry)
    print("Energy functional compiled successfully.")

    # 3. Set up the initial state and parameters
    state = {'x': jnp.array([2.0]), 'y': jnp.array([1.0])}
    step_alpha = 0.1
    num_iterations = 10

    print(f"Initial state: x = {state['x'][0]:.4f}")
    print(f"Running for {num_iterations} iterations with alpha = {step_alpha}")
    print("-" * 20)

    # 4. WHEN we run the flow loop
    for i in range(num_iterations):
        state = run_flow_step(state, compiled, step_alpha)
        print(f"Iteration {i+1:02d}: x = {state['x'][0]:.4f}")

    print("-" * 20)
    print(f"Final state: x = {state['x'][0]:.4f}")
    print("--- CLI Run Finished ---")

if __name__ == "__main__":
    main()
