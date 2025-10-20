import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
import pytest
import jax
import jax.numpy as jnp
from computable_flows_shim.fda.certificates import estimate_gamma


def test_estimate_gamma_lanczos_mode():
    """
    RED: estimate_gamma should accept mode='lanczos' and return a finite positive gamma
    for a simple SPD operator.
    """
    key = jax.random.PRNGKey(0)

    # Simple SPD matrix operator: A = diag(1,2,3,4)
    def L_apply(v):
        return jnp.array([1.0*v[0], 2.0*v[1], 3.0*v[2], 4.0*v[3]])

    gamma = estimate_gamma(L_apply, key, input_shape=(4,), mode='lanczos')

    assert gamma is not None
    assert gamma > 0
