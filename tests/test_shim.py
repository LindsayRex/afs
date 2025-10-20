import sys
from pathlib import Path
# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'afs_v2'))

import pytest
import jax.numpy as jnp
from computable_flows_shim.energy.specs import EnergySpec, TermSpec, StateSpec
from computable_flows_shim.controller import run_certified
from computable_flows_shim.ops import Op
from typing import Dict

class IdentityOp(Op):
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return x
    def T(self, x: jnp.ndarray) -> jnp.ndarray:
        return x
    def lipschitz_hint(self) -> float:
        return 1.0

def test_simple_integration():
    """
    A simple integration test for the computable flows shim.
    """
    # 1. Define a simple EnergySpec
    spec = EnergySpec(
        terms=[
            TermSpec(type='quadratic', op='A', weight=1.0, variable='x', target='y'),
            TermSpec(type='l1', op='W', weight=0.1, variable='x')
        ],
        state=StateSpec(shapes={'x': [10], 'y': [10]}),
    )

    op_registry: Dict[str, Op] = {"identity": IdentityOp()}

    initial_state = {
        'x': jnp.ones(10),
        'y': jnp.zeros(10)
    }

    final_state = run_certified(spec, op_registry, initial_state, num_iterations=10)

    assert not jnp.allclose(initial_state['x'], final_state['x'])

    # Clean up the __main__ block
    if __name__ == "__main__":
        pytest.main([__file__])
