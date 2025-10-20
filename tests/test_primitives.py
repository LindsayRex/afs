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
    This test computationally verifies Lemma 1: Monotonic Energy Decay.
    """
    # GIVEN a compiled quadratic energy functional and its gradient
    spec = EnergySpec(
        terms=[
            TermSpec(type='quadratic', op='I', weight=1.0, variable='x', target='y')
        ],
        state=StateSpec(shapes={'x': [1], 'y': [1]})
    )
    op_registry = {'I': IdentityOp()}
    compiled = compile_energy(spec, op_registry)
    grad_f = compiled.f_grad

    # WHEN we apply one step of the dissipative flow from a known state
    initial_state = {'x': jnp.array([2.0]), 'y': jnp.array([1.0])}
    step_alpha = 0.1
    final_state = F_Dis(initial_state, grad_f, step_alpha, manifolds={})

    # THEN the new state should be closer to the minimum (x=y=1.0).
    # The gradient at x=2.0 is grad_E = (x-y) = 1.0.
    # The update rule is x_new = x_old - alpha * grad.
    # x_new = 2.0 - 0.1 * 1.0 = 1.9.
    assert jnp.isclose(final_state['x'], 1.9)

def test_F_Proj_l1():
    """
    Tests that the projective step correctly applies a soft-thresholding operator for an L1 term.
    This test computationally verifies Lemma 3: Constraint Enforcement via Proximal Operators.
    """
    # GIVEN a compiled energy functional with only an L1 term
    spec = EnergySpec(
        terms=[
            TermSpec(type='l1', op='I', weight=0.5, variable='x')
        ],
        state=StateSpec(shapes={'x': [3]})
    )
    op_registry = {'I': IdentityOp()}
    compiled = compile_energy(spec, op_registry)

    # WHEN we apply the projective flow to a known state
    initial_state = {'x': jnp.array([1.5, -0.2, 0.8])}
    step_alpha = 0.1
    final_state = F_Proj(initial_state, compiled.g_prox, step_alpha)

    # THEN the new state should be the result of soft-thresholding.
    # The proximal operator for w*||x||_1 is the soft-thresholding function.
    # The threshold is alpha * weight = 0.1 * 0.5 = 0.05.
    # x_new = sign(x) * max(|x| - threshold, 0).
    # x_new[0] = 1.5 - 0.05 = 1.45
    # x_new[1] = -0.2 + 0.05 = -0.15
    # x_new[2] = 0.8 - 0.05 = 0.75
    expected_x = jnp.array([1.45, -0.15, 0.75])
    assert jnp.allclose(final_state['x'], expected_x)

def test_F_Proj_l1_contract():
    """
    Given: A state and a proximal operator for an L1 term.
    When: We apply the projective primitive F_Proj.
    Then: The new state should be the result of applying the proximal operator.
    """
    # GIVEN a state and a proximal operator for w*||x||_1
    initial_state = {'x': jnp.array([1.5, -0.2, 0.8])}
    weight = 0.5
    step_alpha = 0.1
    threshold = step_alpha * weight

    def prox_g(state, alpha):
        x = state['x']
        return {'x': jnp.sign(x) * jnp.maximum(jnp.abs(x) - alpha * weight, 0)}

    # WHEN we apply the projective flow
    final_state = F_Proj(initial_state, prox_g, step_alpha)

    # THEN the new state should be the result of soft-thresholding
    expected_x = jnp.array([1.45, -0.15, 0.75])
    assert jnp.allclose(final_state['x'], expected_x)

def test_F_Multi():
    """
    Tests the multiscale transform primitive's forward and inverse operations.
    """
    # GIVEN a simple identity transform object
    class IdentityTransform:
        def forward(self, x):
            return x
        def inverse(self, x):
            return x

    W = IdentityTransform()
    x = jnp.array([1.0, 2.0, 3.0])

    # WHEN we apply the forward and then the inverse transform
    u = F_Multi_forward(x, W)
    x_reconstructed = F_Multi_inverse(u, W)

    # THEN the reconstructed vector should be identical to the original.
    assert jnp.allclose(x, x_reconstructed)

def test_F_Multi_contract():
    """
    Given: A state and a multiscale transform operator.
    When: We apply the forward and inverse multiscale transforms.
    Then: The reconstructed state should be identical to the original.
    """
    # GIVEN a simple identity transform object
    class IdentityTransform:
        def forward(self, x):
            return x
        def inverse(self, x):
            return x

    W = IdentityTransform()
    x = jnp.array([1.0, 2.0, 3.0])

    # WHEN we apply the forward and then the inverse transform
    u = F_Multi_forward(x, W)
    x_reconstructed = F_Multi_inverse(u, W)

    # THEN the reconstructed vector should be identical to the original.
    assert jnp.allclose(x, x_reconstructed)

def test_F_Multi_contract_jaxwt():
    """
    Given: A state and a jaxwt wavelet transform.
    When: We apply the forward and inverse multiscale transforms.
    Then: The reconstructed state should be identical to the original.
    """
    import jaxwt
    # GIVEN a jaxwt wavelet transform
    W = jaxwt.dwt_nD
    W_inv = jaxwt.idwt_nD
    
    class JaxwtTransform:
        def forward(self, x):
            return W(x, 'db1', level=1)
        def inverse(self, x):
            return W_inv(x, 'db1')

    W_op = JaxwtTransform()
    x = jnp.ones((4, 4))

    # WHEN we apply the forward and then the inverse transform
    u = F_Multi_forward(x, W_op)
    x_reconstructed = F_Multi_inverse(u, W_op)

    # THEN the reconstructed vector should be identical to the original.
    assert jnp.allclose(x, x_reconstructed)

def test_F_Con():
    """
    Tests the conservative (symplectic) step.
    This test computationally verifies Lemma 2: Energy Conservation in Conservative Flows.
    """

    # GIVEN a simple harmonic oscillator Hamiltonian
    def H(state):
        return 0.5 * (state['p']**2 + state['q']**2)

    initial_state = {'q': jnp.array(1.0), 'p': jnp.array(0.0)}
    dt = 0.1

    # WHEN we apply one step of the conservative flow
    final_state = F_Con(initial_state, H, dt)

    # THEN the new state should approximate the exact analytical solution.
    # The Leapfrog integrator has a known error, so we use a tolerance.
    expected_q = jnp.cos(dt)
    expected_p = -jnp.sin(dt)

    assert jnp.allclose(final_state['q'], expected_q, atol=1e-3)
    assert jnp.allclose(final_state['p'], expected_p, atol=1e-3)

def test_F_Con_energy_conservation():
    """
    Tests that the conservative step conserves energy over a short trajectory.
    """
    # GIVEN a simple harmonic oscillator Hamiltonian
    def H(state):
        return 0.5 * (state['p']**2 + state['q']**2)

    initial_state = {'q': jnp.array(1.0), 'p': jnp.array(0.0)}
    dt = 0.1
    num_steps = 100

    # WHEN we apply the conservative flow for multiple steps
    state = initial_state
    initial_energy = H(state)
    for _ in range(num_steps):
        state = F_Con(state, H, dt)
    final_energy = H(state)

    # THEN the energy should be conserved within a small tolerance.
    assert jnp.allclose(initial_energy, final_energy, atol=1e-3)

def test_F_Ann():
    """
    Tests the annealing/stochastic step.
    This test computationally verifies Lemma 5: Global Exploration via Stochastic Flows.
    """

    # GIVEN an initial state and a random key
    initial_state = {'x': jnp.zeros(100)}
    key = jax.random.PRNGKey(0)
    temperature = 0.1
    dt = 1.0
    
    # WHEN we apply the annealing flow
    final_state = F_Ann(initial_state, key, temperature, dt)

    # THEN the final state should have been perturbed by noise.
    assert not jnp.allclose(final_state['x'], 0.0)
    
    # AND the variance of the noise should match the theoretical value from Langevin dynamics.
    # For dx = sqrt(2*T*dt)*dW, the variance of the state is Var(x) = 2*T*dt.
    expected_variance = 2 * temperature * dt
    assert jnp.allclose(jnp.var(final_state['x']), expected_variance, atol=1e-1)
