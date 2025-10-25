"""
Tests for the energy compiler.
"""
import sys
from pathlib import Path
import jax.numpy as jnp
# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from computable_flows_shim.energy.specs import EnergySpec, TermSpec, StateSpec
from computable_flows_shim.energy.compile import compile_energy
from computable_flows_shim.api import Op

class IdentityOp(Op):
    def __call__(self, x):
        return x

class FiniteDifferenceOp(Op):
    """A simple 1D finite difference operator."""
    def __call__(self, x):
        return x[1:] - x[:-1]
    
    def T(self, y):
        """Adjoint of the finite difference operator."""
        result = jnp.zeros(y.shape[0] + 1)
        result = result.at[0].set(-y[0])
        result = result.at[1:-1].set(y[:-1] - y[1:])
        result = result.at[-1].set(y[-1])
        return result

def test_spec_creation():
    """
    Tests that an EnergySpec object can be created.
    """
    spec = EnergySpec(
        terms=[
            TermSpec(type='quadratic', op='I', weight=0.5, variable='x', target='y')
        ],
        state=StateSpec(shapes={'x': [1], 'y': [1]})
    )
    assert spec is not None

def test_compile_quadratic_term():
    """
    Tests that the compiler can compile a quadratic energy function.
    """
    # GIVEN an energy specification for a simple quadratic term.
    # The energy is E(x) = w * 0.5 * ||Op(x) - y||^2
    spec = EnergySpec(
        terms=[
            TermSpec(type='quadratic', op='I', weight=1.0, variable='x', target='y')
        ],
        state=StateSpec(shapes={'x': [1], 'y': [1]})
    )
    op_registry = {'I': IdentityOp()}
    
    # WHEN the energy functional is compiled
    compiled = compile_energy(spec, op_registry)
    
    # AND evaluated at a specific state
    state = {'x': jnp.array([2.0]), 'y': jnp.array([1.0])}
    
    # THEN the resulting energy should be mathematically correct.
    # E(x) = 1.0 * 0.5 * ||1.0 * 2.0 - 1.0||^2 = 0.5 * 1.0^2 = 0.5
    expected_energy = 0.5
    actual_energy = compiled.f_value(state)
    
    assert jnp.isclose(actual_energy, expected_energy)

def test_compile_tikhonov_term():
    """
    Tests that the compiler can compile a Tikhonov regularization term.
    """
    # GIVEN an energy specification for a Tikhonov regularization term.
    # The energy is E(x) = w * 0.5 * ||Op(x)||^2
    spec = EnergySpec(
        terms=[
            TermSpec(type='tikhonov', op='D', weight=0.5, variable='x', target=None)
        ],
        state=StateSpec(shapes={'x': [3]})
    )
    op_registry = {'D': FiniteDifferenceOp()}

    # WHEN the energy functional is compiled
    compiled = compile_energy(spec, op_registry)

    # AND evaluated at a specific state
    state = {'x': jnp.array([1.0, 3.0, 2.0])}
    
    # THEN the energy and gradient should be mathematically correct.
    # Dx = [3.0 - 1.0, 2.0 - 3.0] = [2.0, -1.0]
    # E(x) = 0.5 * 0.5 * ||[2.0, -1.0]||^2 = 0.25 * (4.0 + 1.0) = 1.25
    expected_energy = 1.25
    assert jnp.isclose(compiled.f_value(state), expected_energy)

    # The gradient of w * 0.5 * ||Op(x)||^2 is w * Op.T(Op(x)).
    # grad_E = 0.5 * D.T([2.0, -1.0])
    #        = 0.5 * [-2.0, 3.0, -1.0]
    #        = [-1.0, 1.5, -0.5]
    expected_grad = jnp.array([-1.0, 1.5, -0.5])
    assert jnp.allclose(compiled.f_grad(state)['x'], expected_grad)
