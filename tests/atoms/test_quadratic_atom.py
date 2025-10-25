"""
Test-Driven Development for QuadraticAtom.

Following the RED-GREEN-REFACTOR cycle with Design by Contract.
Each test enforces mathematical properties of the QuadraticAtom.
"""

import jax.numpy as jnp
import pytest

from computable_flows_shim.atoms import create_atom


class TestQuadraticAtomContract:
    """
    Design by Contract tests for QuadraticAtom.

    Contract: QuadraticAtom implements (1/2)‖Ax - b‖² with correct:
    - Energy computation
    - Gradient computation
    - Proximal operator
    - Certificate contributions
    """

    @pytest.fixture
    def quadratic_atom(self):
        """Create a fresh QuadraticAtom instance for each test."""
        return create_atom("quadratic")

    @pytest.fixture
    def simple_problem(self):
        """Simple 2x2 linear system for testing."""
        A = jnp.array([[2.0, 1.0], [1.0, 3.0]])
        b = jnp.array([1.0, 2.0])
        x = jnp.array([0.5, 1.0])
        return A, b, x

    def test_atom_identity(self, quadratic_atom):
        """RED: Atom should have correct identity."""
        assert quadratic_atom.name == "quadratic"
        assert quadratic_atom.form == r"\frac{1}{2}\|Ax - b\|_2^2"

    def test_energy_computation(self, quadratic_atom, simple_problem):
        """RED: Energy should compute (1/2)‖Ax - b‖² correctly."""
        A, b, x = simple_problem
        state = {"x": x}
        params = {"A": A, "b": b, "variable": "x"}

        energy = quadratic_atom.energy(state, params)

        # Manual computation: (1/2)‖Ax - b‖²
        residual = A @ x - b
        expected = 0.5 * float(jnp.sum(residual**2))

        assert abs(energy - expected) < 1e-10

    def test_gradient_computation(self, quadratic_atom, simple_problem):
        """RED: Gradient should be A^T(Ax - b)."""
        A, b, x = simple_problem
        state = {"x": x}
        params = {"A": A, "b": b, "variable": "x"}

        grad = quadratic_atom.gradient(state, params)

        # Manual computation: A^T(Ax - b)
        residual = A @ x - b
        expected_grad = A.T @ residual

        assert jnp.allclose(grad["x"], expected_grad, atol=1e-10)

    def test_proximal_operator(self, quadratic_atom, simple_problem):
        """RED: Proximal operator should solve the regularized system."""
        A, b, x = simple_problem
        state = {"x": x}
        params = {"A": A, "b": b, "variable": "x"}
        step_size = 0.1

        prox_result = quadratic_atom.prox(state, step_size, params)

        # The prox should satisfy: x_new = argmin_x (1/2)‖Ax - b‖² + (1/(2τ))‖x - x_old‖²
        # This gives: (A^T A + I/τ) x_new = A^T b + x_old/τ
        ATA = A.T @ A
        ATb = A.T @ b
        lhs = ATA + jnp.eye(2) / step_size
        rhs = ATb + x / step_size
        expected_x = jnp.linalg.solve(lhs, rhs)

        assert jnp.allclose(prox_result["x"], expected_x, atol=1e-6)

    def test_certificate_contributions(self, quadratic_atom, simple_problem):
        """RED: Should provide Lipschitz constant and certificate contributions."""
        A, b, x = simple_problem
        params = {"A": A, "b": b, "variable": "x"}

        certs = quadratic_atom.certificate_contributions(params)

        # Should have Lipschitz constant (spectral norm of A^T A)
        assert "lipschitz" in certs
        expected_lipschitz = float(jnp.linalg.norm(A.T @ A, ord=2))
        assert abs(certs["lipschitz"] - expected_lipschitz) < 1e-10

        # Should have certificate contributions
        assert "eta_dd_contribution" in certs
        assert "gamma_contribution" in certs

    def test_mathematical_consistency(self, quadratic_atom, simple_problem):
        """RED: Energy should decrease under gradient descent."""
        A, b, x = simple_problem
        state = {"x": x}
        params = {"A": A, "b": b, "variable": "x"}

        # Compute initial energy
        energy_before = quadratic_atom.energy(state, params)

        # Take a gradient step
        grad = quadratic_atom.gradient(state, params)
        step_size = 0.01
        new_state = {"x": x - step_size * grad["x"]}

        # Compute new energy
        energy_after = quadratic_atom.energy(new_state, params)

        # Energy should decrease (sufficient decrease condition)
        assert energy_after < energy_before

    def test_factory_function(self):
        """RED: Factory function should create correct atom types."""
        atom = create_atom("quadratic")
        assert atom.name == "quadratic"

        # Should raise for unknown types
        with pytest.raises(ValueError, match="Unknown atom type"):
            create_atom("unknown_atom")
