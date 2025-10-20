import sys
from pathlib import Path
# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'afs_v2'))

import pytest
import jax
import jax.numpy as jnp
from computable_flows_shim.fda.certificates import estimate_eta_dd_in_W, estimate_gamma_in_W
from computable_flows_shim.multi.wavelets import TransformOp

def test_estimate_eta_dd_in_W():
    """
    Tests the diagonal dominance estimation.
    """
    key = jax.random.PRNGKey(0)
    
    # A simple diagonal operator
    def L_apply(v):
        return 2.0 * v

    # A simple identity transform
    W = TransformOp(
        name="identity",
        forward=lambda x: x,
        inverse=lambda x: x
    )

    eta_dd = estimate_eta_dd_in_W(L_apply, W, key, input_shape=(10,))

    assert eta_dd == 0.0

def test_estimate_gamma_in_W():
    """
    Tests the spectral gap estimation.
    """
    key = jax.random.PRNGKey(0)
    
    # A simple diagonal operator
    def L_apply(v):
        return 2.0 * v

    # A simple identity transform
    W = TransformOp(
        name="identity",
        forward=lambda x: x,
        inverse=lambda x: x
    )

    gamma = estimate_gamma_in_W(L_apply, W, key, input_shape=(10,))

    assert jnp.isclose(gamma, 2.0, atol=1e-5)
