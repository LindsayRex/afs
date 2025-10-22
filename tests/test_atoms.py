"""
Test-Driven Development for Atoms Library.

Following the RED-GREEN-REFACTOR cycle with Design by Contract.
Each test enforces mathematical properties of the atoms.
"""

import pytest
import jax.numpy as jnp
import jax
from computable_flows_shim.atoms.library import QuadraticAtom, TikhonovAtom, L1Atom, WaveletL1Atom, TVAtom, create_atom


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
        return WaveletL1Atom()
    
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
        state = {'x': x}
        params = {'variable': 'x', 'lambda': 0.1, 'wavelet': 'haar', 'levels': 1, 'ndim': 1}
        
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
        state = {'x': x}
        params = {'variable': 'x', 'lambda': 0.1, 'wavelet': 'haar', 'levels': 1, 'ndim': 1}
        
        grad = wavelet_l1_atom.gradient(state, params)
        
        # Should return gradient with same shape as input
        assert grad['x'].shape == x.shape
        assert jnp.isfinite(grad['x']).all()
    
    def test_proximal_operator_soft_thresholding(self, wavelet_l1_atom, simple_signal):
        """RED: Proximal operator should implement analysis/synthesis with soft-thresholding."""
        x = simple_signal
        state = {'x': x}
        params = {'variable': 'x', 'lambda': 1.0, 'wavelet': 'haar', 'levels': 1, 'ndim': 1}
        step_size = 0.5
        
        prox_result = wavelet_l1_atom.prox(state, step_size, params)
        x_new = prox_result['x']
        
        # Should return array with same shape
        assert x_new.shape == x.shape
        assert jnp.isfinite(x_new).all()
        
        # For large lambda*tau, should create sparsity in wavelet domain
        # (this is a weak test - full verification would check wavelet coefficients)
    
    def test_proximal_operator_sparsity_effect(self, wavelet_l1_atom):
        """RED: Large regularization should create sparsity in wavelet domain."""
        # Create a signal with some structure
        x = jnp.array([1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0])
        state = {'x': x}
        params = {'variable': 'x', 'lambda': 10.0, 'wavelet': 'haar', 'levels': 1, 'ndim': 1}
        step_size = 1.0
        
        prox_result = wavelet_l1_atom.prox(state, step_size, params)
        x_new = prox_result['x']
        
        # Large regularization should change the signal
        assert not jnp.allclose(x_new, x, atol=1e-10)
        assert jnp.isfinite(x_new).all()
    
    def test_certificate_contributions(self, wavelet_l1_atom):
        """RED: Should provide frame constant for W-space analysis."""
        params = {'variable': 'x', 'lambda': 0.1, 'wavelet': 'haar', 'levels': 1, 'ndim': 1}
        
        certs = wavelet_l1_atom.certificate_contributions(params)
        
        # Should have frame constant
        assert 'frame_constant' in certs
        assert certs['frame_constant'] > 0.0  # Frame constant should be positive
        
        # L1 contributions should be zero
        assert certs['lipschitz'] == 0.0
        assert certs['eta_dd_contribution'] == 0.0
        assert certs['gamma_contribution'] == 0.0
    
    def test_mathematical_consistency_prox(self, wavelet_l1_atom, simple_signal):
        """RED: Proximal operator should converge to fixed point."""
        x = simple_signal
        state = {'x': x}
        params = {'variable': 'x', 'lambda': 10.0, 'wavelet': 'haar', 'levels': 1, 'ndim': 1}
        step_size = 1.0
        
        # Apply prox multiple times - should converge
        current_state = state
        for _ in range(3):
            current_state = wavelet_l1_atom.prox(current_state, step_size, params)
        
        # Apply one more time - should be very close to converged result
        next_state = wavelet_l1_atom.prox(current_state, step_size, params)
        
        # Should be close to converged solution
        assert jnp.allclose(current_state['x'], next_state['x'], atol=1e-5)
    
    def test_factory_function_wavelet_l1(self):
        """RED: Factory function should create WaveletL1 atom."""
        atom = create_atom('wavelet_l1')
        assert isinstance(atom, WaveletL1Atom)
        assert atom.name == "wavelet_l1"


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
        return TVAtom()
    
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
        assert isinstance(atom, TVAtom)
        assert atom.name == "tv"