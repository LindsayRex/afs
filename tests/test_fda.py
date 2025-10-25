def test_estimate_gamma_gershgorin_lower_bound():
    """
    Given a non-symmetric operator,
    When estimate_gamma is called,
    Then it should return the minimum eigenvalue or Gershgorin lower bound if not symmetric.
    """
    import jax
    import jax.numpy as jnp

    from computable_flows_shim.fda.certificates import estimate_gamma

    class NonSymOp:
        def __call__(self, x):
            # Matrix [[2, 3], [1, 4]]
            return jnp.array([2.0 * x[0] + 3.0 * x[1], 1.0 * x[0] + 4.0 * x[1]])

    op = NonSymOp()
    key = jax.random.PRNGKey(0)
    gamma = estimate_gamma(op, key, (2,))
    basis = jnp.eye(2)
    l_matrix = jnp.stack([op(basis[i]) for i in range(2)])
    diag = jnp.diag(l_matrix)
    off_diag_sum = jnp.array(
        [jnp.sum(jnp.abs(l_matrix[i, :])) - jnp.abs(l_matrix[i, i]) for i in range(2)]
    )
    gershgorin_bounds = diag - off_diag_sum
    print(f"l_matrix: {l_matrix}")
    print(f"Gershgorin bounds: {gershgorin_bounds}")
    print(f"gamma returned: {gamma}")
    # Eigenvalues are approx 1.382, 4.618; Gershgorin bounds: row 0: -1, row 1: 3
    assert (
        gamma == -1.0 or abs(gamma - 1.382) < 1e-2
    ), f"Expected gamma=-1.0 (Gershgorin) or 1.382 (eig), got {gamma}"


def test_estimate_eta_dd_contract():
    """
    Given a linear operator with known diagonal dominance,
    When estimate_eta_dd is called,
    Then it should return the correct eta_dd value.
    """
    import jax.numpy as jnp

    from computable_flows_shim.fda.certificates import estimate_eta_dd

    class DDOp:
        def __call__(self, x):
            # Matrix [[3, 1], [1, 3]]
            return jnp.array([3.0 * x[0] + 1.0 * x[1], 1.0 * x[0] + 3.0 * x[1]])

    op = DDOp()
    eta = estimate_eta_dd(op, (2,))
    # For each row: sum off-diagonal / diagonal = 1/3
    assert abs(eta - (1.0 / 3.0)) < 1e-6, f"Expected eta_dd=1/3, got {eta}"


def test_estimate_gamma_detects_negative_eigenvalue():
    """
    Given a linear operator with a negative eigenvalue,
    When estimate_gamma is called,
    Then it should return the algebraic minimum eigenvalue (negative).
    """
    import jax
    import jax.numpy as jnp

    from computable_flows_shim.fda.certificates import estimate_gamma

    class NegEigOp:
        def __call__(self, x):
            # Matrix [[-2, 0], [0, 3]] has eigenvalues -2, 3
            return jnp.array([-2.0 * x[0], 3.0 * x[1]])

    op = NegEigOp()
    key = jax.random.PRNGKey(0)
    gamma = estimate_gamma(op, key, (2,))
    # Should return -2.0
    assert gamma == -2.0, f"Expected -2.0, got {gamma}"


import sys
from pathlib import Path

import jax
import jax.numpy as jnp

from computable_flows_shim.api import Op
from computable_flows_shim.fda.certificates import estimate_eta_dd, estimate_gamma

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

"""
Tests for the Flow Dynamic Analysis (FDA) certificate estimators.
"""


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
    def l_apply(v):
        return jnp.array([3.0 * v[0], 5.0 * v[1]])

    input_shape = (2,)
    key = jax.random.PRNGKey(0)

    # WHEN we estimate the spectral gap (smallest eigenvalue)
    gamma = estimate_gamma(l_apply, key, input_shape)

    # THEN the estimated gamma should be close to the true smallest eigenvalue.
    assert jnp.isclose(gamma, 3.0, atol=1e-3)


def test_estimate_eta_dd():
    """
    Tests the diagonal dominance (eta) estimation for a simple linear operator.
    """

    # GIVEN a simple, diagonally dominant linear operator (a diagonal matrix)
    def l_apply(v):
        return jnp.array([3.0 * v[0], 5.0 * v[1]])

    input_shape = (2,)

    # WHEN we estimate the diagonal dominance
    # This will fail because the function doesn't exist yet
    eta = estimate_eta_dd(l_apply, input_shape)

    # THEN the estimated eta should be 0 for a diagonal matrix.
    assert jnp.isclose(eta, 0.0)
