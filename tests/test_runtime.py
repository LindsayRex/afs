"""
Tests for the runtime engine and step execution.
"""
import sys
from pathlib import Path
import jax
import jax.numpy as jnp
# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from computable_flows_shim.energy.specs import EnergySpec, TermSpec, StateSpec
from computable_flows_shim.energy.compile import compile_energy
from computable_flows_shim.api import Op
from computable_flows_shim.runtime.step import run_flow_step

class IdentityOp(Op):
    def __call__(self, x):
        return x

def test_forward_backward_step():
    """
    Tests one full step of Forward-Backward Splitting (F_Dis -> F_Proj).
    """
    # GIVEN a composite energy functional (quadratic + L1)
    spec = EnergySpec(
        terms=[
            TermSpec(type='quadratic', op='I', weight=1.0, variable='x', target='y'),
            TermSpec(type='l1', op='I', weight=0.5, variable='x')
        ],
        state=StateSpec(shapes={'x': [1], 'y': [1]})
    )
    op_registry = {'I': IdentityOp()}
    compiled = compile_energy(spec, op_registry)

    # WHEN we run one step of the flow
    initial_state = {'x': jnp.array([2.0]), 'y': jnp.array([1.0])}
    step_alpha = 0.1
    
    # This will fail because run_flow_step doesn't exist yet
    final_state = run_flow_step(initial_state, compiled, step_alpha)

    # THEN the state should be the result of F_Dis followed by F_Proj
    # 1. F_Dis step:
    #    grad = (x-y) = 2.0 - 1.0 = 1.0
    #    x_after_dis = 2.0 - 0.1 * 1.0 = 1.9
    # 2. F_Proj step:
    #    threshold = alpha * weight = 0.1 * 0.5 = 0.05
    #    x_after_proj = 1.9 - 0.05 = 1.85
    expected_x = 1.85
    assert jnp.isclose(final_state['x'], expected_x)
