"""
Tests for the primitive operators.
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
from computable_flows_shim.runtime.primitives import F_Dis, F_Proj, F_Multi_forward, F_Multi_inverse, F_Con, F_Ann

class IdentityOp(Op):
    def __call__(self, x):
        return x

def test_F_Dis_quadratic():
    """
    Tests that the dissipative step correctly descends the gradient of a quadratic energy function.
    """
    # GIVEN an energy functional (our quadratic atom)
    spec = EnergySpec(
        terms=[
            TermSpec(type='quadratic', op='I', weight=1.0, variable='x', target='y')
        ],
        state=StateSpec(shapes={'x': [1], 'y': [1]})
    )
    op_registry = {'I': IdentityOp()}
    compiled = compile_energy(spec, op_registry)
    grad_f = compiled.f_grad

    # WHEN we apply the dissipative flow
    initial_state = {'x': jnp.array([2.0]), 'y': jnp.array([1.0])}
    step_alpha = 0.1
    
    # This will fail because F_Dis doesn't exist yet
    final_state = F_Dis(initial_state, grad_f, step_alpha, manifolds={})

    # THEN the state should move closer to the minimum
    # For E(x) = 0.5 * (x-y)^2, the minimum is at x=y=1.0.
    # The gradient at x=2.0 is (x-y) = 1.0.
    # The update is x_new = x_old - alpha * grad = 2.0 - 0.1 * 1.0 = 1.9.
    assert jnp.isclose(final_state['x'], 1.9)

def test_F_Proj_l1():
    """
    Tests that the projective step correctly applies a soft-thresholding operator for an L1 term.
    """
    # GIVEN an energy functional with an L1 term
    spec = EnergySpec(
        terms=[
            TermSpec(type='l1', op='I', weight=0.5, variable='x')
        ],
        state=StateSpec(shapes={'x': [3]})
    )
    op_registry = {'I': IdentityOp()}
    compiled = compile_energy(spec, op_registry)

    # WHEN we apply the projective flow
    initial_state = {'x': jnp.array([1.5, -0.2, 0.8])}
    step_alpha = 0.1 # This is the step size, not the weight
    
    # This will fail because F_Proj doesn't exist yet
    final_state = F_Proj(initial_state, compiled.g_prox, step_alpha)

    # THEN the state should be soft-thresholded
    # The threshold is alpha * weight = 0.1 * 0.5 = 0.05
    # x_new = sign(x) * max(|x| - threshold, 0)
    # x_new[0] = 1.5 - 0.05 = 1.45
    # x_new[1] = -0.2 + 0.05 = -0.15
    # x_new[2] = 0.8 - 0.05 = 0.75
    expected_x = jnp.array([1.45, -0.15, 0.75])
    assert jnp.allclose(final_state['x'], expected_x)

def test_F_Multi():
    """
    Tests the multiscale transform primitive.
    """
    # GIVEN a simple identity transform
    class IdentityTransform:
        def forward(self, x):
            return x
        def inverse(self, x):
            return x

    W = IdentityTransform()
    x = jnp.array([1.0, 2.0, 3.0])

    # WHEN we apply the forward and inverse transforms
    # This will fail because the functions don't exist yet
    u = F_Multi_forward(x, W)
    x_reconstructed = F_Multi_inverse(u, W)

    # THEN the reconstruction should be perfect
    assert jnp.allclose(x, x_reconstructed)

def test_F_Con():
    """
    Tests the conservative (symplectic) step.
    """
    # GIVEN a simple harmonic oscillator Hamiltonian
    def H(state):
        return 0.5 * (state['p']**2 + state['q']**2)

    initial_state = {'q': jnp.array(1.0), 'p': jnp.array(0.0)}
    dt = 0.1

    # WHEN we apply the conservative flow
    # This will fail because F_Con is not implemented
    final_state = F_Con(initial_state, H, dt)

    # THEN the state should follow the exact solution for one step, within tolerance
    # The exact solution is q(t) = cos(t), p(t) = -sin(t)
    expected_q = jnp.cos(dt)
    expected_p = -jnp.sin(dt)

    assert jnp.allclose(final_state['q'], expected_q, atol=1e-3)
    assert jnp.allclose(final_state['p'], expected_p, atol=1e-3)

def test_F_Ann():
    """
    Tests the annealing/stochastic step.
    """
    # GIVEN an initial state and a random key
    initial_state = {'x': jnp.zeros(100)}
    key = jax.random.PRNGKey(0)
    
    # WHEN we apply the annealing flow
    # This will fail because F_Ann is not implemented
    final_state = F_Ann(initial_state, key, temperature=0.1, dt=1.0)

    # THEN the final state should no longer be all zeros
    assert not jnp.allclose(final_state['x'], 0.0)
    
    # AND the variance should be related to the temperature and step size
    # For Langevin dynamics, Var(x) = 2 * T * dt
    assert jnp.allclose(jnp.var(final_state['x']), 2 * 0.1 * 1.0, atol=1e-1)
