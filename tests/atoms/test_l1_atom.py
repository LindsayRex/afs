"""
Test-Driven Development for L1Atom.

Following the RED-GREEN-REFACTOR cycle with Design by Contract.
Each test enforces mathematical properties of the L1Atom.
"""

import jax.numpy as jnp
import pytest

from computable_flows_shim.atoms import create_atom


class TestL1AtomContract:
    """
    Design by Contract tests for L1Atom.

    Contract: L1Atom implements λ‖x‖₁ with correct:
    - Energy computation (L1 norm)
    - Subgradient computation (sign function)
    - Proximal operator (soft-thresholding)
    - Certificate contributions (nonsmooth regularization)
    """

    @pytest.fixture
    def l1_atom(self):
        """Create a fresh L1Atom instance for each test."""
        return create_atom("l1")

    @pytest.fixture
    def simple_vector(self):
        """Simple test vector with positive, negative, and zero elements."""
        return jnp.array([2.0, -1.5, 0.0, 0.5])

    def test_atom_identity(self, l1_atom):
        """RED: Atom should have correct identity."""
        assert l1_atom.name == "l1"
        assert l1_atom.form == r"\lambda\|x\|_1"

    def test_energy_computation(self, l1_atom, simple_vector):
        """RED: Energy should compute λ‖x‖₁ correctly."""
        x = simple_vector
        state = {"x": x}
        params = {"variable": "x", "lambda": 0.1}

        energy = l1_atom.energy(state, params)

        # Manual computation: λ‖x‖₁
        expected = 0.1 * float(jnp.sum(jnp.abs(x)))

        assert abs(energy - expected) < 1e-10

    def test_subgradient_computation(self, l1_atom, simple_vector):
        """RED: Subgradient should be λ*sign(x)."""
        x = simple_vector
        state = {"x": x}
        params = {"variable": "x", "lambda": 0.1}

        subgrad = l1_atom.gradient(state, params)

        # Manual computation: λ*sign(x)
        expected_subgrad = 0.1 * jnp.sign(x)

        assert jnp.allclose(subgrad["x"], expected_subgrad, atol=1e-10)

    def test_proximal_operator_soft_thresholding(self, l1_atom, simple_vector):
        """RED: Proximal operator should implement soft-thresholding."""
        x = simple_vector
        state = {"x": x}
        params = {"variable": "x", "lambda": 0.1}
        step_size = 0.5

        prox_result = l1_atom.prox(state, step_size, params)

        # Soft-thresholding: S_λτ(x) = sign(x) * max(|x| - λτ, 0)
        threshold = 0.1 * step_size  # λ * τ
        expected_x = jnp.sign(x) * jnp.maximum(jnp.abs(x) - threshold, 0)

        assert jnp.allclose(prox_result["x"], expected_x, atol=1e-10)

    def test_proximal_operator_sparsity(self, l1_atom):
        """RED: Soft-thresholding should create sparsity."""
        # Create a vector with small and large elements
        x = jnp.array([0.1, 2.0, -0.05, -3.0])
        state = {"x": x}
        params = {"variable": "x", "lambda": 1.0}
        step_size = 1.0

        prox_result = l1_atom.prox(state, step_size, params)
        x_new = prox_result["x"]

        # Elements smaller than λτ = 1.0 should be thresholded to zero
        assert abs(x_new[0]) < 1e-10  # 0.1 < 1.0, should be zero
        assert abs(x_new[2]) < 1e-10  # -0.05 < 1.0, should be zero

        # Larger elements should be shrunk but not zeroed
        assert abs(x_new[1]) > 0.0  # 2.0 > 1.0, should be 1.0
        assert abs(x_new[3]) > 0.0  # -3.0 > 1.0, should be -2.0

    def test_certificate_contributions(self, l1_atom):
        """RED: L1 should have zero contributions to smooth certificates."""
        params = {"variable": "x", "lambda": 0.1}

        certs = l1_atom.certificate_contributions(params)

        # L1 regularization doesn't contribute to Lipschitz constants
        assert certs["lipschitz"] == 0.0
        assert certs["eta_dd_contribution"] == 0.0
        assert certs["gamma_contribution"] == 0.0

    def test_factory_function_l1(self):
        """RED: Factory function should create L1 atom."""
        atom = create_atom("l1")
        assert atom.name == "l1"
