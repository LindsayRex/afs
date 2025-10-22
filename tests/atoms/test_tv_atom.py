"""
Test-Driven Development for TVAtom.

Following the RED-GREEN-REFACTOR cycle with Design by Contract.
Each test enforces mathematical properties of the TVAtom.
"""

import pytest
import jax.numpy as jnp
import jax
from computable_flows_shim.atoms import create_atom


class TestTVAtomContract:
    """
    Design by Contract tests for TVAtom.

    Contract: TVAtom implements λ‖Dx‖₁ with correct:
    - Energy computation (anisotropic TV norm)
    - Subgradient computation (finite difference signs)
    - Proximal operator (shrinkage on differences)
    - Certificate contributions (nonsmooth regularization)
    """

    @pytest.fixture
    def tv_atom(self):
        """Create a fresh TVAtom instance for each test."""
        return create_atom('tv')

    @pytest.fixture
    def simple_signal(self):
        """Simple 1D signal with varying differences."""
        return jnp.array([1.0, 3.0, 2.0, 4.0, 3.0])

    def test_atom_identity(self, tv_atom):
        """RED: Atom should have correct identity."""
        assert tv_atom.name == "tv"
        assert tv_atom.form == r"\lambda\|Dx\|_1"

    def test_energy_computation_1d(self, tv_atom, simple_signal):
        """RED: Energy should compute λ‖Dx‖₁ correctly for 1D."""
        x = simple_signal
        state = {'x': x}
        params = {'variable': 'x', 'lambda': 0.5}

        energy = tv_atom.energy(state, params)

        # Manual computation: λ * ‖[2, -1, 2, -1]‖₁ = 0.5 * (2 + 1 + 2 + 1) = 3.0
        expected = 0.5 * (2.0 + 1.0 + 2.0 + 1.0)  # Differences: 3-1=2, 2-3=-1, 4-2=2, 3-4=-1
        assert abs(energy - expected) < 1e-10

    def test_gradient_computation_1d(self, tv_atom, simple_signal):
        """RED: Subgradient should be λ * D^T sign(Dx) for 1D."""
        x = simple_signal
        state = {'x': x}
        params = {'variable': 'x', 'lambda': 0.5}

        grad = tv_atom.gradient(state, params)

        # For x = [1,3,2,4,3], differences = [2, -1, 2, -1]
        # sign(differences) = [1, -1, 1, -1]
        # D^T sign(Dx) = [-1, 1-(-1), -1-1, 1-(-1), -1] = [-1, 2, -2, 2, -1]
        # Times λ = 0.5: [-0.5, 1.0, -1.0, 1.0, -0.5]
        expected = jnp.array([-0.5, 1.0, -1.0, 1.0, -0.5])
        assert jnp.allclose(grad['x'], expected, atol=1e-10)

    def test_proximal_operator_1d(self, tv_atom, simple_signal):
        """RED: Proximal operator should shrink differences."""
        x = simple_signal
        state = {'x': x}
        params = {'variable': 'x', 'lambda': 1.0}
        step_size = 0.1

        prox_result = tv_atom.prox(state, step_size, params)
        x_new = prox_result['x']

        # Should return array with same shape
        assert x_new.shape == x.shape
        assert jnp.isfinite(x_new).all()

        # Should be different from original (shrinkage effect)
        assert not jnp.allclose(x_new, x, atol=1e-10)

    def test_proximal_operator_constant_signal(self, tv_atom):
        """RED: Constant signal should be unchanged by TV prox."""
        x = jnp.array([2.0, 2.0, 2.0, 2.0])
        state = {'x': x}
        params = {'variable': 'x', 'lambda': 1.0}
        step_size = 0.1

        prox_result = tv_atom.prox(state, step_size, params)
        x_new = prox_result['x']

        # Constant signal has zero TV norm, so prox should not change it
        assert jnp.allclose(x_new, x, atol=1e-6)

    def test_certificate_contributions(self, tv_atom):
        """RED: TV should have zero contributions to smooth certificates."""
        params = {'variable': 'x', 'lambda': 0.1}

        certs = tv_atom.certificate_contributions(params)

        # TV is nonsmooth, so no contributions to smooth certificates
        assert certs['lipschitz'] == 0.0
        assert certs['eta_dd_contribution'] == 0.0
        assert certs['gamma_contribution'] == 0.0

    def test_factory_function_tv(self):
        """RED: Factory function should create TV atom."""
        atom = create_atom('tv')
        assert atom.name == "tv"