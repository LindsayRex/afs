"""
Test-Driven Development for TikhonovAtom.

Following the RED-GREEN-REFACTOR cycle with Design by Contract.
Each test enforces mathematical properties of the TikhonovAtom.
"""

import pytest
import jax.numpy as jnp
import jax
from computable_flows_shim.atoms import create_atom


class TestTikhonovAtomContract:
    """
    Design by Contract tests for TikhonovAtom.

    Contract: TikhonovAtom implements (1/2)‖Ax - b‖² + (λ/2)‖x‖² with correct:
    - Energy computation with regularization
    - Gradient computation with regularization term
    - Proximal operator for regularized system
    - Certificate contributions with improved conditioning
    """

    @pytest.fixture
    def tikhonov_atom(self):
        """Create a fresh TikhonovAtom instance for each test."""
        return create_atom('tikhonov')

    @pytest.fixture
    def simple_problem(self):
        """Simple 2x2 linear system for testing."""
        A = jnp.array([[2.0, 1.0], [1.0, 3.0]])
        b = jnp.array([1.0, 2.0])
        x = jnp.array([0.5, 1.0])
        return A, b, x

    def test_atom_identity(self, tikhonov_atom):
        """RED: Atom should have correct identity."""
        assert tikhonov_atom.name == "tikhonov"
        assert tikhonov_atom.form == r"\frac{1}{2}\|Ax - b\|_2^2 + \frac{\lambda}{2}\|x\|_2^2"

    def test_energy_computation(self, tikhonov_atom, simple_problem):
        """RED: Energy should compute (1/2)‖Ax - b‖² + (λ/2)‖x‖² correctly."""
        A, b, x = simple_problem
        state = {'x': x}
        params = {'A': A, 'b': b, 'variable': 'x', 'lambda': 0.1}

        energy = tikhonov_atom.energy(state, params)

        # Manual computation
        residual = A @ x - b
        data_fidelity = 0.5 * float(jnp.sum(residual**2))
        regularization = 0.5 * 0.1 * float(jnp.sum(x**2))
        expected = data_fidelity + regularization

        assert abs(energy - expected) < 1e-10

    def test_gradient_computation(self, tikhonov_atom, simple_problem):
        """RED: Gradient should be A^T(Ax - b) + λx."""
        A, b, x = simple_problem
        state = {'x': x}
        params = {'A': A, 'b': b, 'variable': 'x', 'lambda': 0.1}

        grad = tikhonov_atom.gradient(state, params)

        # Manual computation: A^T(Ax - b) + λx
        residual = A @ x - b
        expected_grad = A.T @ residual + 0.1 * x

        assert jnp.allclose(grad['x'], expected_grad, atol=1e-10)

    def test_proximal_operator(self, tikhonov_atom, simple_problem):
        """RED: Proximal operator should solve the regularized system."""
        A, b, x = simple_problem
        state = {'x': x}
        params = {'A': A, 'b': b, 'variable': 'x', 'lambda': 0.1}
        step_size = 0.1

        prox_result = tikhonov_atom.prox(state, step_size, params)

        # The prox should satisfy: x_new = argmin_x (1/2)‖Ax - b‖² + (λ/2)‖x‖² + (1/(2τ))‖x - x_old‖²
        # Solution: (A^T A + λI + I/τ) x = A^T b + x/τ
        ATA = A.T @ A
        ATb = A.T @ b
        lhs = ATA + 0.1 * jnp.eye(2) + jnp.eye(2) / step_size
        rhs = ATb + x / step_size
        expected_x = jnp.linalg.solve(lhs, rhs)

        assert jnp.allclose(prox_result['x'], expected_x, atol=1e-6)

    def test_certificate_contributions(self, tikhonov_atom, simple_problem):
        """RED: Should provide improved Lipschitz constant and regularization benefits."""
        A, b, x = simple_problem
        params = {'A': A, 'b': b, 'variable': 'x', 'lambda': 0.1}

        certs = tikhonov_atom.certificate_contributions(params)

        # Should have effective Lipschitz constant (larger than unregularized)
        assert 'lipschitz' in certs
        expected_min_lipschitz = float(jnp.linalg.norm(A.T @ A, ord=2))  # Unregularized
        assert certs['lipschitz'] >= expected_min_lipschitz

        # Should have positive diagonal dominance contribution
        assert 'eta_dd_contribution' in certs
        assert certs['eta_dd_contribution'] == 0.1  # The regularization parameter

        # Should have certificate contributions
        assert 'gamma_contribution' in certs

    def test_regularization_reduces_conditioning(self, tikhonov_atom, simple_problem):
        """RED: Regularization should improve conditioning for solving but affect certificates differently."""
        A, b, x = simple_problem
        params_regularized = {'A': A, 'b': b, 'variable': 'x', 'lambda': 0.1}
        params_unregularized = {'A': A, 'b': b, 'variable': 'x'}

        quad_atom = create_atom('quadratic')
        quad_certs = quad_atom.certificate_contributions(params_unregularized)
        tikh_certs = tikhonov_atom.certificate_contributions(params_regularized)

        # Regularized version has larger Lipschitz constant (worse for certificates)
        assert tikh_certs['lipschitz'] > quad_certs['lipschitz']

        # But provides diagonal dominance improvement
        assert tikh_certs['eta_dd_contribution'] > quad_certs['eta_dd_contribution']

    def test_factory_function_tikhonov(self):
        """RED: Factory function should create Tikhonov atom."""
        atom = create_atom('tikhonov')
        assert atom.name == "tikhonov"