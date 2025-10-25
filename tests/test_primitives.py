"""
Tests for the primitive operators.
"""

import sys
from pathlib import Path
from typing import cast

import jax
import jax.numpy as jnp
import pytest

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from computable_flows_shim.api import Op
from computable_flows_shim.energy.compile import compile_energy
from computable_flows_shim.energy.specs import EnergySpec, StateSpec, TermSpec
from computable_flows_shim.runtime.primitives import (
    F_Ann,
    F_Con,
    F_Dis,
    F_Multi,
    F_Proj,
)


class IdentityOp(Op):
    def __call__(self, x):
        return x


@pytest.mark.dtype_parametrized
def test_f_dis_quadratic(float_dtype):
    """
    Tests that the dissipative step correctly descends the gradient of a quadratic energy function.
    This test computationally verifies Lemma 1: Monotonic Energy Decay.
    """
    # GIVEN a compiled quadratic energy functional and its gradient
    spec = EnergySpec(
        terms=[
            TermSpec(type="quadratic", op="I", weight=1.0, variable="x", target="y")
        ],
        state=StateSpec(shapes={"x": [1], "y": [1]}),
    )
    op_registry = {"I": IdentityOp()}
    compiled = compile_energy(spec, op_registry)
    grad_f = compiled.f_grad

    # WHEN we apply one step of the dissipative flow from a known state
    initial_state = {
        "x": jnp.array([2.0], dtype=float_dtype),
        "y": jnp.array([1.0], dtype=float_dtype),
    }
    step_alpha = 0.1
    final_state = F_Dis(initial_state, grad_f, step_alpha, manifolds={})

    # THEN the new state should be closer to the minimum (x=y=1.0).
    # The gradient at x=2.0 is grad_E = (x-y) = 1.0.
    # The update rule is x_new = x_old - alpha * grad.
    # x_new = 2.0 - 0.1 * 1.0 = 1.9.
    tolerance = 1e-5 if float_dtype == jnp.float32 else 1e-12
    assert jnp.isclose(final_state["x"], 1.9, atol=tolerance)


def test_f_proj_l1():
    """
    Tests that the projective step correctly applies a soft-thresholding operator for an L1 term.
    This test computationally verifies Lemma 3: Constraint Enforcement via Proximal Operators.
    """
    # GIVEN a compiled energy functional with only an L1 term
    spec = EnergySpec(
        terms=[TermSpec(type="l1", op="I", weight=0.5, variable="x")],
        state=StateSpec(shapes={"x": [3]}),
    )
    op_registry = {"I": IdentityOp()}
    compiled = compile_energy(spec, op_registry)

    # WHEN we apply the projective flow to a known state
    initial_state = {"x": jnp.array([1.5, -0.2, 0.8])}
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
    assert jnp.allclose(final_state["x"], expected_x)


def test_f_proj_l1_contract():
    """
    Given: A state and a proximal operator for an L1 term.
    When: We apply the projective primitive F_Proj.
    Then: The new state should be the result of applying the proximal operator.
    """
    # GIVEN a state and a proximal operator for w*||x||_1
    initial_state = {"x": jnp.array([1.5, -0.2, 0.8])}
    weight = 0.5
    step_alpha = 0.1
    # threshold = step_alpha * weight

    def prox_g(state, alpha):
        x = state["x"]
        return {"x": jnp.sign(x) * jnp.maximum(jnp.abs(x) - alpha * weight, 0)}

    # WHEN we apply the projective flow
    final_state = F_Proj(initial_state, prox_g, step_alpha)

    # THEN the new state should be the result of soft-thresholding
    expected_x = jnp.array([1.45, -0.15, 0.75])
    assert jnp.allclose(final_state["x"], expected_x)


@pytest.mark.complex_operations
def test_f_multi(complex_dtype):
    """
    Tests the multiscale transform primitive's forward and inverse operations.
    """
    import jaxwt

    # Create a wavelet transform object
    class WaveletTransform:
        def __init__(self, wavelet="haar", level=1):
            self.wavelet = wavelet
            self.level = level

        def forward(self, x):
            return jaxwt.wavedec(x, self.wavelet, level=self.level)

        def inverse(self, x):
            return jaxwt.waverec(x, self.wavelet)

    w = WaveletTransform()
    # Use real input for wavelet transforms (they can produce complex coefficients internally)
    x = jnp.array(
        [1.0, 2.0, 3.0, 4.0],
        dtype=jnp.float64 if complex_dtype == jnp.complex128 else jnp.float32,
    )

    # WHEN we apply the forward and then the inverse transform
    u = F_Multi(x, w, "forward")
    x_reconstructed = cast(jnp.ndarray, F_Multi(u, w, "inverse"))

    # THEN the reconstructed vector should be identical to the original.
    tolerance = 1e-5 if complex_dtype == jnp.complex64 else 1e-12
    assert jnp.allclose(x, x_reconstructed, atol=tolerance)


@pytest.mark.complex_operations
def test_f_multi_contract_jaxwt(complex_dtype):
    """
    Given: A state and a jaxwt wavelet transform.
    When: We apply the forward and inverse multiscale transforms.
    Then: The reconstructed state should be identical to the original.
    """
    import jaxwt

    # Create a wavelet transform object
    class JaxwtTransform:
        def __init__(self, wavelet="db1", level=1):
            self.wavelet = wavelet
            self.level = level

        def forward(self, x):
            return jaxwt.wavedec(x, self.wavelet, level=self.level)

        def inverse(self, x):
            return jaxwt.waverec(x, self.wavelet)

    w_op = JaxwtTransform()
    # Use real input for wavelet transforms
    x = jnp.ones(
        (4,), dtype=jnp.float64 if complex_dtype == jnp.complex128 else jnp.float32
    )

    # WHEN we apply the forward and then the inverse transform
    coeffs = F_Multi(x, w_op, "forward")
    x_reconstructed = cast(jnp.ndarray, F_Multi(coeffs, w_op, "inverse"))

    # THEN the reconstructed vector should be identical to the original.
    tolerance = 1e-5 if complex_dtype == jnp.complex64 else 1e-12
    assert jnp.allclose(x, x_reconstructed, atol=tolerance)


@pytest.mark.complex_operations
def test_f_multi_wavelet_roundtrip(complex_dtype):
    """
    GREEN: Test for proper F_Multi primitive with wavelet roundtrip.
    Given: A 1D signal and wavelet transform object.
    When: We apply forward then inverse wavelet transform.
    Then: The signal should be perfectly reconstructed.
    """
    import jaxwt

    # Create a wavelet transform object
    class HaarTransform:
        def __init__(self, level=1):
            self.level = level

        def forward(self, x):
            return jaxwt.wavedec(x, "haar", level=self.level)

        def inverse(self, x):
            return jaxwt.waverec(x, "haar")

    w = HaarTransform()

    # GIVEN a 1D signal
    x = jnp.array(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        dtype=jnp.float64 if complex_dtype == jnp.complex128 else jnp.float32,
    )

    # WHEN we apply F_Multi forward and inverse
    coeffs = F_Multi(x, w, "forward")
    reconstructed = cast(jnp.ndarray, F_Multi(coeffs, w, "inverse"))

    # THEN the signal should be perfectly reconstructed
    tolerance = 1e-5 if complex_dtype == jnp.complex64 else 1e-12
    assert jnp.allclose(x, reconstructed, atol=tolerance)


def test_f_con():
    """
    Tests the conservative (symplectic) step.
    This test computationally verifies Lemma 2: Energy Conservation in Conservative Flows.
    """

    # GIVEN a simple harmonic oscillator Hamiltonian
    def h(state):
        return 0.5 * (state["p"] ** 2 + state["q"] ** 2)

    initial_state = {"q": jnp.array(1.0), "p": jnp.array(0.0)}
    dt = 0.1

    # WHEN we apply one step of the conservative flow
    final_state = F_Con(initial_state, h, dt)

    # THEN the new state should approximate the exact analytical solution.
    # The Leapfrog integrator has a known error, so we use a tolerance.
    expected_q = jnp.cos(dt)
    expected_p = -jnp.sin(dt)

    assert jnp.allclose(final_state["q"], expected_q, atol=1e-3)
    assert jnp.allclose(final_state["p"], expected_p, atol=1e-3)


def test_f_con_energy_conservation():
    """
    Tests that the conservative step conserves energy over a short trajectory.
    """

    # GIVEN a simple harmonic oscillator Hamiltonian
    def h(state):
        return 0.5 * (state["p"] ** 2 + state["q"] ** 2)

    initial_state = {"q": jnp.array(1.0), "p": jnp.array(0.0)}
    dt = 0.1
    num_steps = 100

    # WHEN we apply the conservative flow for multiple steps
    state = initial_state
    initial_energy = h(state)
    for _ in range(num_steps):
        state = F_Con(state, h, dt)
    final_energy = h(state)

    # THEN the energy should be conserved within a small tolerance.
    assert jnp.allclose(initial_energy, final_energy, atol=1e-3)


def test_f_ann():
    """
    Tests the annealing/stochastic step.
    This test computationally verifies Lemma 5: Global Exploration via Stochastic Flows.
    """

    # GIVEN an initial state and a random key
    initial_state = {"x": jnp.zeros(100)}
    key = jax.random.PRNGKey(0)
    temperature = 0.1
    dt = 1.0

    # WHEN we apply the annealing flow
    final_state = F_Ann(initial_state, key, temperature, dt)

    # THEN the final state should have been perturbed by noise.
    assert not jnp.allclose(final_state["x"], 0.0)

    # AND the variance of the noise should match the theoretical value from Langevin dynamics.
    # For dx = sqrt(2*T*dt)*dW, the variance of the state is Var(x) = 2*T*dt.
    expected_variance = 2 * temperature * dt
    assert jnp.allclose(jnp.var(final_state["x"]), expected_variance, atol=1e-1)
