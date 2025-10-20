import sys
from pathlib import Path
import jax
import jax.numpy as jnp
# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'afs_v2'))

from computable_flows_shim.runtime.primitives import F_Dis, F_Multi_forward, F_Multi_inverse, F_Proj, F_Con, F_Ann
from computable_flows_shim.multi.wavelets import TransformOp

def test_F_Dis():
    """
    Tests the dissipative step.
    """
    # A simple quadratic energy function
    def f(state):
        return 0.5 * state['x']**2

    grad_f = jax.grad(f)

    initial_state = {'x': jnp.array(1.0)}
    step_alpha = 0.1
    manifolds = {}

    final_state = F_Dis(initial_state, grad_f, step_alpha, manifolds)

    # The gradient is x, so the update is x - alpha * x
    expected_state = {'x': jnp.array(1.0 - 0.1 * 1.0)}

    assert jnp.isclose(final_state['x'], expected_state['x'])

def test_F_Multi():
    """
    Tests the multiscale transform.
    """
    # A simple identity transform
    W = TransformOp(
        name="identity",
        forward=lambda x: x,
        inverse=lambda x: x
    )

    initial_state = {'x': jnp.array([1.0, 2.0, 3.0])}

    # Apply the forward transform
    u = F_Multi_forward(initial_state['x'], W)

    # Apply the inverse transform
    final_state_x = F_Multi_inverse(u, W)

    assert jnp.allclose(initial_state['x'], final_state_x)

def test_F_Proj():
    """
    Tests the projective step.
    """
    # A simple soft-thresholding proximal operator
    def prox_in_W(state, step_alpha, W):
        transformed_state = W.forward(state['x'])
        thresholded_state = jnp.sign(transformed_state) * jnp.maximum(jnp.abs(transformed_state) - step_alpha, 0)
        return {'x': W.inverse(thresholded_state)}

    # A simple identity transform
    W = TransformOp(
        name="identity",
        forward=lambda x: x,
        inverse=lambda x: x
    )

    initial_state = {'x': jnp.array([1.5, -0.5, 0.2])}
    step_alpha = 1.0

    final_state = F_Proj(initial_state, prox_in_W, step_alpha, W)

    expected_state = {'x': jnp.array([0.5, 0.0, 0.0])}

    assert jnp.allclose(final_state['x'], expected_state['x'])

def test_F_Con():
    """
    Tests the conservative step.
    """
    # A simple harmonic oscillator Hamiltonian
    def H(state):
        return 0.5 * (state['p']**2 + state['q']**2)

    initial_state = {'q': jnp.array(1.0), 'p': jnp.array(0.0)}
    dt = 0.1

    final_state = F_Con(initial_state, H, dt)

    # The exact solution is q(t) = cos(t), p(t) = -sin(t)
    expected_state = {'q': jnp.cos(dt), 'p': -jnp.sin(dt)}

    assert jnp.allclose(final_state['q'], expected_state['q'], atol=1e-4)
    assert jnp.allclose(final_state['p'], expected_state['p'], atol=1e-4)

def test_F_Ann():
    """
    Tests the annealing/stochastic step.
    """
    key = jax.random.PRNGKey(0)

    # A simple quadratic energy function
    def f(state):
        return 0.5 * jnp.sum(state['x']**2)

    grad_f = jax.grad(f)

    initial_state = {'x': jnp.ones(10)}
    step_alpha = 0.1
    temperature = 0.01

    # A purely dissipative step for comparison
    dissipative_state = F_Dis(initial_state, grad_f, step_alpha, {})

    # The stochastic step
    stochastic_state = F_Ann(initial_state, grad_f, step_alpha, temperature, key)

    # Check that the output has the correct shape
    assert stochastic_state['x'].shape == initial_state['x'].shape

    # Check that the stochastic step is not identical to the dissipative step
    # (it's technically possible but astronomically unlikely for this seed)
    assert not jnp.allclose(stochastic_state['x'], dissipative_state['x'])

    # Check that for zero temperature, it's just a dissipative step
    zero_temp_state = F_Ann(initial_state, grad_f, step_alpha, 0.0, key)
    assert jnp.allclose(zero_temp_state['x'], dissipative_state['x'])
