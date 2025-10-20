"""
Sample quadratic flow specification for testing the CFS CLI.

This defines a simple quadratic minimization problem: min_x ||x - y||²
"""
import jax
import jax.numpy as jnp
from computable_flows_shim.energy.specs import EnergySpec, TermSpec, StateSpec
from computable_flows_shim.api import Op

class IdentityOp(Op):
    """Identity operator for testing."""
    def __call__(self, x):
        return x

# Flow specification
spec = EnergySpec(
    terms=[
        TermSpec(type='quadratic', op='I', weight=1.0, variable='x', target='y')
    ],
    state=StateSpec(shapes={'x': [10], 'y': [10]})
)

# Operator registry
op_registry = {'I': IdentityOp()}

# Initial state - start with random noise around the target
key = jax.random.PRNGKey(42)
target_y = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
noise = jax.random.normal(key, (10,)) * 2.0
initial_state = {
    'x': target_y + noise,  # Start away from the target
    'y': target_y
}

# Flow parameters
step_alpha = 0.1
num_iterations = 50
flow_name = "quadratic_demo"