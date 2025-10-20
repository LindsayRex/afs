"""
Tests for the Flow Dynamic Analysis (FDA) certificate estimators.
"""
import sys
from pathlib import Path
import jax
import jax.numpy as jnp
import pytest

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from computable_flows_shim.fda.certificates import estimate_gamma
from computable_flows_shim.api import Op

class IdentityOp(Op):
    def __call__(self, x):
        return x

def test_estimate_gamma():
    """
    Tests the spectral gap (gamma) estimation for a simple linear operator.
    """
    # GIVEN a simple, well-conditioned linear operator (a diagonal matrix)
    # L = [[3, 0], [0, 5]]. The eigenvalues are 3 and 5.
    # The spectral gap is the smallest eigenvalue, which is 3.
    def L_apply(v):
        return jnp.array([3.0 * v[0], 5.0 * v[1]])

    input_shape = (2,)
    key = jax.random.PRNGKey(0)

    # WHEN we estimate the spectral gap (smallest eigenvalue)
    gamma = estimate_gamma(L_apply, key, input_shape)

    # THEN the estimated gamma should be close to the true smallest eigenvalue.
    assert jnp.isclose(gamma, 3.0, atol=1e-3)
