"""
Tests for the Flight Controller.
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
from computable_flows_shim.controller import run_certified
from computable_flows_shim.runtime.step import run_flow_step
import pytest

class IdentityOp(Op):
    def __call__(self, x):
        return x

def test_controller_runs_loop():
    """
    Tests that the controller can run a simple flow to convergence.
    """
    # GIVEN a simple quadratic energy functional
    spec = EnergySpec(
        terms=[
            TermSpec(type='quadratic', op='I', weight=1.0, variable='x', target='y')
        ],
        state=StateSpec(shapes={'x': [1], 'y': [1]})
    )
    op_registry = {'I': IdentityOp()}
    compiled = compile_energy(spec, op_registry)

    # WHEN we run the flow using the controller for a single step
    initial_state = {'x': jnp.array([10.0]), 'y': jnp.array([0.0])}
    
    final_state = run_certified(initial_state, compiled, num_iterations=1, step_alpha=0.1)

    # THEN the final state should be the result of one Forward-Backward step
    # 1. F_Dis: z = x - alpha * grad(f(x)) = 10.0 - 0.1 * (10.0 - 0.0) = 9.0
    # 2. F_Proj: prox_g is an identity op since there are no non-smooth terms.
    assert jnp.allclose(final_state['x'], 9.0, atol=1e-3)

class NonDiagonallyDominantOp(Op):
    def __call__(self, x):
        # L = [[1, 2], [2, 1]]. Row 0: |2|/|1| = 2 > 1. Not DD.
        return jnp.array([1.0 * x[0] + 2.0 * x[1], 2.0 * x[0] + 1.0 * x[1]])

def test_controller_checks_diagonal_dominance():
    """
    Tests that the controller refuses to run a system that is not diagonally dominant.
    """
    # GIVEN an energy functional with a non-diagonally dominant operator
    spec = EnergySpec(
        terms=[
            TermSpec(type='tikhonov', op='L', weight=1.0, variable='x')
        ],
        state=StateSpec(shapes={'x': [2]})
    )
    op_registry = {'L': NonDiagonallyDominantOp()}
    compiled = compile_energy(spec, op_registry)

    # WHEN we try to run the flow
    # THEN the controller should detect the high eta_dd and raise an error
    with pytest.raises(ValueError, match="System failed certification"):
        run_certified(
            initial_state={'x': jnp.array([1.0, 1.0])},
            compiled=compiled,
            num_iterations=10,
            step_alpha=0.1
        )

def test_controller_enforces_lyapunov_descent():
    """
    Tests that the controller aborts if energy increases (violating Lyapunov descent).
    This test computationally verifies our Design by Contract for F_Dis.
    """
    # GIVEN a simple quadratic energy functional
    spec = EnergySpec(
        terms=[
            TermSpec(type='quadratic', op='I', weight=1.0, variable='x', target='y')
        ],
        state=StateSpec(shapes={'x': [1], 'y': [1]})
    )
    op_registry = {'I': IdentityOp()}
    compiled = compile_energy(spec, op_registry)

    # AND a malicious step function that increases energy on the second step
    def malicious_step_function(state, compiled, step_alpha):
        # First step is normal
        if state['x'][0] == 10.0:
            return run_flow_step(state, compiled, step_alpha)
        # Second step, deliberately increase the energy by moving away from the minimum
        else:
            return {'x': jnp.array([99.0]), 'y': state['y']}

    # WHEN we run the flow with the malicious step function
    initial_state = {'x': jnp.array([10.0]), 'y': jnp.array([0.0])}
    
    # THEN the controller should detect the energy increase and raise an error
    with pytest.raises(ValueError, match="Step failed to decrease energy"):
        run_certified(
            initial_state, 
            compiled, 
            num_iterations=5, 
            step_alpha=0.1,
            _step_function_for_testing=malicious_step_function # Inject our malicious function
        )

def test_controller_amber_step_remediation():
    """
    Tests that the controller can remediate step failures by reducing alpha.
    """
    # GIVEN a simple quadratic energy functional
    spec = EnergySpec(
        terms=[
            TermSpec(type='quadratic', op='I', weight=1.0, variable='x', target='y')
        ],
        state=StateSpec(shapes={'x': [1], 'y': [1]})
    )
    op_registry = {'I': IdentityOp()}
    compiled = compile_energy(spec, op_registry)

    # AND a step function that fails with large alpha but succeeds with small alpha
    def picky_step_function(state, compiled, step_alpha):
        if step_alpha > 0.05:  # Large alpha causes energy increase
            return {'x': jnp.array([state['x'][0] + 10.0]), 'y': state['y']}  # Bad step
        else:  # Small alpha works
            return run_flow_step(state, compiled, step_alpha)

    # WHEN we run the flow with a large initial alpha that needs remediation
    initial_state = {'x': jnp.array([10.0]), 'y': jnp.array([0.0])}
    
    final_state = run_certified(
        initial_state, 
        compiled, 
        num_iterations=1, 
        step_alpha=0.1,  # Large alpha that will fail first attempt
        _step_function_for_testing=picky_step_function
    )

    # THEN the controller should have remediated and succeeded
    # With small alpha, it should converge closer to 0
    assert jnp.allclose(final_state['x'], 9.5, atol=1e-3)

def test_controller_checks_spectral_gap():
    """
    Tests that the controller refuses to run an unstable system (non-positive spectral gap).
    """
    # GIVEN an energy functional with an unstable linear operator (negative eigenvalue)
    class UnstableOp(Op):
        def __call__(self, x):
            # This operator has eigenvalues -1 and 2. The spectral gap is -1.
            return jnp.array([-1.0 * x[0], 2.0 * x[1]])

    spec = EnergySpec(
        terms=[
            TermSpec(type='tikhonov', op='L', weight=1.0, variable='x')
        ],
        state=StateSpec(shapes={'x': [2]})
    )
    op_registry = {'L': UnstableOp()}
    compiled = compile_energy(spec, op_registry)
    
    # WHEN we try to run the flow
    # THEN the controller should detect the negative spectral gap and raise an error
    with pytest.raises(ValueError, match="System failed certification"):
        run_certified(
            initial_state={'x': jnp.array([1.0, 1.0])},
            compiled=compiled,
            num_iterations=10,
            step_alpha=0.1
        )
