"""
Test-Driven Development for Energy Compiler.

Following the RED-GREEN-REFACTOR cycle with Design by Contract.
Tests ensure energy specifications compile to correct JAX functions using Atoms Library.
"""

import pytest
import jax.numpy as jnp
from computable_flows_shim.energy.specs import EnergySpec, TermSpec, StateSpec
from computable_flows_shim.energy.compile import compile_energy
from computable_flows_shim.atoms import QuadraticAtom, TikhonovAtom, L1Atom


class TestEnergyCompilerContract:
    """
    Design by Contract tests for Energy Compiler.
    
    Contract: Energy Compiler takes declarative specs and produces JAX functions
    that correctly compute energy, gradients, and proximal operators using Atoms Library.
    """
    
    @pytest.fixture
    def simple_quadratic_spec(self):
        """Simple quadratic energy spec for testing."""
        return EnergySpec(
            terms=[
                TermSpec(
                    type='quadratic',
                    op='A',
                    weight=1.0,
                    variable='x',
                    target='y'
                )
            ],
            state=StateSpec(shapes={'x': [3], 'y': [3]})
        )
    
    @pytest.fixture
    def op_registry(self):
        """Simple operator registry for testing."""
        A = jnp.array([[2.0, 0.0, 0.0], [0.0, 1.5, 0.0], [0.0, 0.0, 1.0]])
        return {'A': lambda x: A @ x}
    
    def test_compile_quadratic_term(self, simple_quadratic_spec, op_registry):
        """RED: Compiler should handle quadratic terms correctly."""
        compiled = compile_energy(simple_quadratic_spec, op_registry)
        
        # Test with simple state
        state = {'x': jnp.array([1.0, 2.0, 3.0]), 'y': jnp.array([2.0, 3.0, 3.0])}
        
        # Energy should be (1/2)‖Ax - y‖²
        A = jnp.array([[2.0, 0.0, 0.0], [0.0, 1.5, 0.0], [0.0, 0.0, 1.0]])
        Ax = A @ state['x']
        residual = Ax - state['y']
        expected_energy = 0.5 * float(jnp.sum(residual**2))
        
        actual_energy = compiled.f_value(state)
        assert abs(actual_energy - expected_energy) < 1e-10
    
    def test_compile_gradient(self, simple_quadratic_spec, op_registry):
        """RED: Compiler should produce correct gradients."""
        compiled = compile_energy(simple_quadratic_spec, op_registry)
        
        state = {'x': jnp.array([1.0, 2.0, 3.0]), 'y': jnp.array([2.0, 3.0, 3.0])}
        
        grad = compiled.f_grad(state)
        
        # Manual gradient computation: A^T(Ax - y)
        A = jnp.array([[2.0, 0.0, 0.0], [0.0, 1.5, 0.0], [0.0, 0.0, 1.0]])
        Ax = A @ state['x']
        residual = Ax - state['y']
        expected_grad_x = A.T @ residual
        
        assert jnp.allclose(grad['x'], expected_grad_x, atol=1e-10)
    
    def test_compile_multiple_terms(self):
        """RED: Compiler should handle multiple terms."""
        spec = EnergySpec(
            terms=[
                TermSpec(type='quadratic', op='A', weight=1.0, variable='x', target='y'),
                TermSpec(type='tikhonov', op='I', weight=0.1, variable='x')
            ],
            state=StateSpec(shapes={'x': [3], 'y': [3]})
        )
        
        op_registry = {
            'A': lambda x: jnp.array([[2.0, 0.0, 0.0], [0.0, 1.5, 0.0], [0.0, 0.0, 1.0]]) @ x,
            'I': lambda x: x  # Identity for Tikhonov
        }
        
        compiled = compile_energy(spec, op_registry)
        
        state = {'x': jnp.array([1.0, 2.0, 3.0]), 'y': jnp.array([2.0, 3.0, 3.0])}
        
        # Energy should be (1/2)‖Ax - y‖² + 0.1 * (1/2)‖x‖²
        A = jnp.array([[2.0, 0.0, 0.0], [0.0, 1.5, 0.0], [0.0, 0.0, 1.0]])
        Ax = A @ state['x']
        residual = Ax - state['y']
        quad_energy = 0.5 * float(jnp.sum(residual**2))
        tikh_energy = 0.1 * 0.5 * float(jnp.sum(state['x']**2))
        expected_energy = quad_energy + tikh_energy
        
        actual_energy = compiled.f_value(state)
        assert abs(actual_energy - expected_energy) < 1e-10
    
    def test_unknown_atom_type_error(self):
        """RED: Compiler should raise error for unknown atom types."""
        spec = EnergySpec(
            terms=[TermSpec(type='unknown_atom', op='A', weight=1.0, variable='x')],
            state=StateSpec(shapes={'x': [3]})
        )
        
        op_registry = {'A': lambda x: x}
        
        with pytest.raises(ValueError, match="Unknown atom type"):
            compile_energy(spec, op_registry)
    
    def test_jit_compilation(self, simple_quadratic_spec, op_registry):
        """RED: Compiled functions should be JIT-compiled."""
        compiled = compile_energy(simple_quadratic_spec, op_registry)
        
        # JAX functions should have jit compilation info
        assert hasattr(compiled.f_value, 'lower')
        assert hasattr(compiled.f_grad, 'lower')
        assert hasattr(compiled.g_prox, 'lower')