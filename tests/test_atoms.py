"""
Test-Driven Development for Atoms Library.

Following the RED-GREEN-REFACTOR cycle with Design by Contract.
Each test enforces mathematical properties of the atoms.
"""

import pytest
import jax.numpy as jnp
import jax
from computable_flows_shim.atoms import QuadraticAtom, TikhonovAtom, L1Atom, create_atom


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
        return QuadraticAtom()
    
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
        state = {'x': x}
        params = {'A': A, 'b': b, 'variable': 'x'}
        
        energy = quadratic_atom.energy(state, params)
        
        # Manual computation: (1/2)‖Ax - b‖²
        residual = A @ x - b
        expected = 0.5 * float(jnp.sum(residual**2))
        
        assert abs(energy - expected) < 1e-10
    
    def test_gradient_computation(self, quadratic_atom, simple_problem):
        """RED: Gradient should be A^T(Ax - b)."""
        A, b, x = simple_problem
        state = {'x': x}
        params = {'A': A, 'b': b, 'variable': 'x'}
        
        grad = quadratic_atom.gradient(state, params)
        
        # Manual computation: A^T(Ax - b)
        residual = A @ x - b
        expected_grad = A.T @ residual
        
        assert jnp.allclose(grad['x'], expected_grad, atol=1e-10)
    
    def test_proximal_operator(self, quadratic_atom, simple_problem):
        """RED: Proximal operator should solve the regularized system."""
        A, b, x = simple_problem
        state = {'x': x}
        params = {'A': A, 'b': b, 'variable': 'x'}
        step_size = 0.1
        
        prox_result = quadratic_atom.prox(state, step_size, params)
        
        # The prox should satisfy: x_new = argmin_x (1/2)‖Ax - b‖² + (1/(2τ))‖x - x_old‖²
        # This gives: (A^T A + I/τ) x_new = A^T b + x_old/τ
        ATA = A.T @ A
        ATb = A.T @ b
        lhs = ATA + jnp.eye(2) / step_size
        rhs = ATb + x / step_size
        expected_x = jnp.linalg.solve(lhs, rhs)
        
        assert jnp.allclose(prox_result['x'], expected_x, atol=1e-6)
    
    def test_certificate_contributions(self, quadratic_atom, simple_problem):
        """RED: Should provide Lipschitz constant and certificate contributions."""
        A, b, x = simple_problem
        params = {'A': A, 'b': b, 'variable': 'x'}
        
        certs = quadratic_atom.certificate_contributions(params)
        
        # Should have Lipschitz constant (spectral norm of A^T A)
        assert 'lipschitz' in certs
        expected_lipschitz = float(jnp.linalg.norm(A.T @ A, ord=2))
        assert abs(certs['lipschitz'] - expected_lipschitz) < 1e-10
        
        # Should have certificate contributions
        assert 'eta_dd_contribution' in certs
        assert 'gamma_contribution' in certs
    
    def test_mathematical_consistency(self, quadratic_atom, simple_problem):
        """RED: Energy should decrease under gradient descent."""
        A, b, x = simple_problem
        state = {'x': x}
        params = {'A': A, 'b': b, 'variable': 'x'}
        
        # Compute initial energy
        energy_before = quadratic_atom.energy(state, params)
        
        # Take a gradient step
        grad = quadratic_atom.gradient(state, params)
        step_size = 0.01
        new_state = {'x': x - step_size * grad['x']}
        
        # Compute new energy
        energy_after = quadratic_atom.energy(new_state, params)
        
        # Energy should decrease (sufficient decrease condition)
        assert energy_after < energy_before
    
    def test_factory_function(self):
        """RED: Factory function should create correct atom types."""
        atom = create_atom('quadratic')
        assert isinstance(atom, QuadraticAtom)
        assert atom.name == "quadratic"
        
        # Should raise for unknown types
        with pytest.raises(ValueError, match="Unknown atom type"):
            create_atom('unknown_atom')


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
        return TikhonovAtom()
    
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
        
        quad_atom = QuadraticAtom()
        quad_certs = quad_atom.certificate_contributions(params_unregularized)
        tikh_certs = tikhonov_atom.certificate_contributions(params_regularized)
        
        # Regularized version has larger Lipschitz constant (worse for certificates)
        assert tikh_certs['lipschitz'] > quad_certs['lipschitz']
        
        # But provides diagonal dominance improvement
        assert tikh_certs['eta_dd_contribution'] > quad_certs['eta_dd_contribution']
    
    def test_factory_function_tikhonov(self):
        """RED: Factory function should create Tikhonov atom."""
        atom = create_atom('tikhonov')
        assert isinstance(atom, TikhonovAtom)
        assert atom.name == "tikhonov"


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
        return L1Atom()
    
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
        state = {'x': x}
        params = {'variable': 'x', 'lambda': 0.1}
        
        energy = l1_atom.energy(state, params)
        
        # Manual computation: λ‖x‖₁
        expected = 0.1 * float(jnp.sum(jnp.abs(x)))
        
        assert abs(energy - expected) < 1e-10
    
    def test_subgradient_computation(self, l1_atom, simple_vector):
        """RED: Subgradient should be λ*sign(x)."""
        x = simple_vector
        state = {'x': x}
        params = {'variable': 'x', 'lambda': 0.1}
        
        subgrad = l1_atom.gradient(state, params)
        
        # Manual computation: λ*sign(x)
        expected_subgrad = 0.1 * jnp.sign(x)
        
        assert jnp.allclose(subgrad['x'], expected_subgrad, atol=1e-10)
    
    def test_proximal_operator_soft_thresholding(self, l1_atom, simple_vector):
        """RED: Proximal operator should implement soft-thresholding."""
        x = simple_vector
        state = {'x': x}
        params = {'variable': 'x', 'lambda': 0.1}
        step_size = 0.5
        
        prox_result = l1_atom.prox(state, step_size, params)
        
        # Soft-thresholding: S_λτ(x) = sign(x) * max(|x| - λτ, 0)
        threshold = 0.1 * step_size  # λ * τ
        expected_x = jnp.sign(x) * jnp.maximum(jnp.abs(x) - threshold, 0)
        
        assert jnp.allclose(prox_result['x'], expected_x, atol=1e-10)
    
    def test_proximal_operator_sparsity(self, l1_atom):
        """RED: Soft-thresholding should create sparsity."""
        # Create a vector with small and large elements
        x = jnp.array([0.1, 2.0, -0.05, -3.0])
        state = {'x': x}
        params = {'variable': 'x', 'lambda': 1.0}
        step_size = 1.0
        
        prox_result = l1_atom.prox(state, step_size, params)
        x_new = prox_result['x']
        
        # Elements smaller than λτ = 1.0 should be thresholded to zero
        assert abs(x_new[0]) < 1e-10  # 0.1 < 1.0, should be zero
        assert abs(x_new[2]) < 1e-10  # -0.05 < 1.0, should be zero
        
        # Larger elements should be shrunk but not zeroed
        assert abs(x_new[1]) > 0.0  # 2.0 > 1.0, should be 1.0
        assert abs(x_new[3]) > 0.0  # -3.0 > 1.0, should be -2.0
    
    def test_certificate_contributions(self, l1_atom):
        """RED: L1 should have zero contributions to smooth certificates."""
        params = {'variable': 'x', 'lambda': 0.1}
        
        certs = l1_atom.certificate_contributions(params)
        
        # L1 regularization doesn't contribute to Lipschitz constants
        assert certs['lipschitz'] == 0.0
        assert certs['eta_dd_contribution'] == 0.0
        assert certs['gamma_contribution'] == 0.0
    
    def test_factory_function_l1(self):
        """RED: Factory function should create L1 atom."""
        atom = create_atom('l1')
        assert isinstance(atom, L1Atom)
        assert atom.name == "l1"