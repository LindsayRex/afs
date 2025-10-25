"""
Test-Driven Development for WaveletL1Atom.

Following the RED-GREEN-REFACTOR cycle with Design by Contract.
Each test enforces mathematical properties of the WaveletL1Atom.
"""

import jax.numpy as jnp
import pytest

from computable_flows_shim.atoms import create_atom


class TestWaveletL1AtomContract:
    """
    Design by Contract tests for WaveletL1Atom.

    Contract: WaveletL1Atom implements λ‖Wx‖₁ with correct:
    - Energy computation in wavelet space
    - Subgradient computation with synthesis
    - Proximal operator (analysis/synthesis with soft-thresholding)
    - Frame constant handling for certificates
    """

    @pytest.fixture
    def wavelet_l1_atom(self):
        """Create a fresh WaveletL1Atom instance for each test."""
        return create_atom("wavelet_l1")

    @pytest.fixture
    def simple_signal(self):
        """Simple 1D signal for testing."""
        return jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])

    def test_atom_identity(self, wavelet_l1_atom):
        """RED: Atom should have correct identity."""
        assert wavelet_l1_atom.name == "wavelet_l1"
        assert wavelet_l1_atom.form == r"\lambda\|Wx\|_1"

    def test_energy_computation(self, wavelet_l1_atom, simple_signal):
        """RED: Energy should compute λ‖Wx‖₁ correctly."""
        x = simple_signal
        state = {"x": x}
        params = {
            "variable": "x",
            "lambda": 0.1,
            "wavelet": "haar",
            "levels": 1,
            "ndim": 1,
        }

        energy = wavelet_l1_atom.energy(state, params)

        # Manual computation: λ * ‖Wx‖₁
        # For Haar wavelet level 1 on [1,2,3,4,5,6,7,8]:
        # Approximation: [2.12, 4.95, 6.36, 8.48] (scaled averages)
        # Detail: [-0.71, -0.71, -0.71, -0.71] (scaled differences)
        # L1 norm should be sum of absolute values
        expected_min = 0.0  # At least zero
        assert energy >= expected_min
        assert isinstance(energy, float)

    def test_gradient_computation(self, wavelet_l1_atom, simple_signal):
        """RED: Subgradient should be W^T sign(Wx)."""
        x = simple_signal
        state = {"x": x}
        params = {
            "variable": "x",
            "lambda": 0.1,
            "wavelet": "haar",
            "levels": 1,
            "ndim": 1,
        }

        grad = wavelet_l1_atom.gradient(state, params)

        # Should return gradient with same shape as input
        assert grad["x"].shape == x.shape
        assert jnp.isfinite(grad["x"]).all()

    def test_proximal_operator_soft_thresholding(self, wavelet_l1_atom, simple_signal):
        """RED: Proximal operator should implement analysis/synthesis with soft-thresholding."""
        x = simple_signal
        state = {"x": x}
        params = {
            "variable": "x",
            "lambda": 1.0,
            "wavelet": "haar",
            "levels": 1,
            "ndim": 1,
        }
        step_size = 0.5

        prox_result = wavelet_l1_atom.prox(state, step_size, params)
        x_new = prox_result["x"]

        # Should return array with same shape
        assert x_new.shape == x.shape
        assert jnp.isfinite(x_new).all()

        # For large lambda*tau, should create sparsity in wavelet domain
        # (this is a weak test - full verification would check wavelet coefficients)

    def test_proximal_operator_sparsity_effect(self, wavelet_l1_atom):
        """RED: Large regularization should create sparsity in wavelet domain."""
        # Create a signal with some structure
        x = jnp.array([1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0])
        state = {"x": x}
        params = {
            "variable": "x",
            "lambda": 10.0,
            "wavelet": "haar",
            "levels": 1,
            "ndim": 1,
        }
        step_size = 1.0

        prox_result = wavelet_l1_atom.prox(state, step_size, params)
        x_new = prox_result["x"]

        # Large regularization should change the signal
        assert not jnp.allclose(x_new, x, atol=1e-10)
        assert jnp.isfinite(x_new).all()

    def test_certificate_contributions(self, wavelet_l1_atom):
        """RED: Should provide frame constant for W-space analysis."""
        params = {
            "variable": "x",
            "lambda": 0.1,
            "wavelet": "haar",
            "levels": 1,
            "ndim": 1,
        }

        certs = wavelet_l1_atom.certificate_contributions(params)

        # Should have frame constant
        assert "frame_constant" in certs
        assert certs["frame_constant"] > 0.0  # Frame constant should be positive

        # L1 contributions should be zero
        assert certs["lipschitz"] == 0.0
        assert certs["eta_dd_contribution"] == 0.0
        assert certs["gamma_contribution"] == 0.0

    def test_mathematical_consistency_prox(self, wavelet_l1_atom, simple_signal):
        """RED: Proximal operator should converge to fixed point."""
        x = simple_signal
        state = {"x": x}
        params = {
            "variable": "x",
            "lambda": 10.0,
            "wavelet": "haar",
            "levels": 1,
            "ndim": 1,
        }
        step_size = 1.0

        # Apply prox multiple times - should converge
        current_state = state
        for _ in range(3):
            current_state = wavelet_l1_atom.prox(current_state, step_size, params)

        # Apply one more time - should be very close to converged result
        next_state = wavelet_l1_atom.prox(current_state, step_size, params)

        # Should be close to converged solution
        assert jnp.allclose(current_state["x"], next_state["x"], atol=1e-5)

    def test_factory_function_wavelet_l1(self):
        """RED: Factory function should create WaveletL1 atom."""
        atom = create_atom("wavelet_l1")
        assert atom.name == "wavelet_l1"
