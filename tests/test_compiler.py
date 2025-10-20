"""
Tests for the energy compiler.
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
    spec = EnergySpec(
        terms=[
            TermSpec(type='quadratic', op='I', weight=1.0, variable='x', target='y')
        ],
        state=StateSpec(shapes={'x': [1], 'y': [1]})
    )

    op_registry = {'I': IdentityOp()}
    
    compiled = compile_energy(spec, op_registry)
    
    state = {'x': jnp.array([2.0]), 'y': jnp.array([1.0])}
    
    # E(x) = 0.5 * ||I*x - y||^2 = 0.5 * (2.0 - 1.0)^2 = 0.5
    # The weight is 1.0, so the final energy is 1.0 * 0.5 = 0.5
    expected_energy = 0.5
    actual_energy = compiled.f_value(state)
    
    assert jnp.isclose(actual_energy, expected_energy)

def test_compile_tikhonov_term():
    """
    Tests that the compiler can compile a Tikhonov regularization term.
    """
    # GIVEN an energy functional with a Tikhonov term
    spec = EnergySpec(
        terms=[
            TermSpec(type='tikhonov', op='D', weight=0.5, variable='x')
        ],
        state=StateSpec(shapes={'x': [3]})
    )
    op_registry = {'D': FiniteDifferenceOp()}
    compiled = compile_energy(spec, op_registry)

    # WHEN we evaluate the energy and gradient
    state = {'x': jnp.array([1.0, 3.0, 2.0])}
    
    # THEN the energy and gradient should be correct
    # E(x) = 0.5 * ||Dx||^2
    # Dx = [3-1, 2-3] = [2, -1]
    # E = 0.5 * (2^2 + (-1)^2) = 0.5 * 5 = 2.5
    # The weight is 0.5, so the final energy is 0.5 * 2.5 = 1.25
    expected_energy = 1.25
    assert jnp.isclose(compiled.f_value(state), expected_energy)

    # grad_E = D^T(Dx)
    # grad_E = D^T([2, -1]) = [-2, 2-(-1), 1] = [-2, 3, -1]
    # The weight is 0.5, and the gradient of 0.5*||f(x)||^2 is f'(x)^T f(x).
    # So, grad_E = 0.5 * D^T(Dx) = 0.5 * [-2, 3, -1] = [-1.0, 1.5, -0.5]
    expected_grad = jnp.array([-1.0, 1.5, -0.5])
    assert jnp.allclose(compiled.f_grad(state)['x'], expected_grad)
