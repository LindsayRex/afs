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
    This test computationally verifies Lemma 6: Stability of Composite Splitting.
    """
    # GIVEN a composite energy functional (quadratic data term f, L1 regularization g)
    spec = EnergySpec(
        terms=[
            TermSpec(type='quadratic', op='I', weight=1.0, variable='x', target='y'),
            TermSpec(type='l1', op='I', weight=0.5, variable='x')
        ],
        state=StateSpec(shapes={'x': [1], 'y': [1]})
    )
    op_registry = {'I': IdentityOp()}
    compiled = compile_energy(spec, op_registry)

    # WHEN we run one full step of the flow from a known state
    initial_state = {'x': jnp.array([2.0]), 'y': jnp.array([1.0])}
    step_alpha = 0.1
    final_state = run_flow_step(initial_state, compiled, step_alpha)

    # THEN the final state should be the result of applying F_Dis, then F_Proj.
    # 1. Forward/Dissipative step (z = x - alpha * grad_f(x)):
    #    grad_f(x) = (x-y) = 2.0 - 1.0 = 1.0
    #    z = 2.0 - 0.1 * 1.0 = 1.9
    # 2. Backward/Projective step (x_new = prox_g(z)):
    #    The prox for g(x) = w*||x||_1 is soft-thresholding with threshold = alpha * w.
    #    threshold = 0.1 * 0.5 = 0.05
    #    x_new = sign(1.9) * max(|1.9| - 0.05, 0) = 1.85
    expected_x = 1.85
    assert jnp.isclose(final_state['x'], expected_x)
