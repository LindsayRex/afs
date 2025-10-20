"""
Tests for the energy compiler.
"""
import sys
from pathlib import Path
import jax
import jax.numpy as jnp
# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'afs_v2'))

from computable_flows_shim.energy.specs import EnergySpec, TermSpec
from computable_flows_shim.energy.compile import compile_energy

def test_compile_quadratic():
    """
    Tests compiling a simple quadratic energy function.
    """
    spec = EnergySpec(
        terms=[
            TermSpec(type='quadratic', op='I', weight=1.0, target='y')
        ],
        transforms={},
        state={'x': {'shape': (1,)}}
    )

    def identity_op(x):
        return x

    op_registry = {'I': identity_op}

    compiled = compile_energy(spec, op_registry)

    state = {'x': jnp.array([2.0]), 'y': jnp.array([1.0])}
    
    # f(x) = 0.5 * ||I*x - y||^2 = 0.5 * (2 - 1)^2 = 0.5
    assert jnp.isclose(compiled['f_value'](state), 0.5)

    # grad_f(x) = I^T * (I*x - y) = 2 - 1 = 1
    grad = compiled['f_grad'](state)
    assert jnp.isclose(grad['x'], 1.0)
