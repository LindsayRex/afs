import sys
from pathlib import Path

import jax
import jax.numpy as jnp

from computable_flows_shim.api import Op
from computable_flows_shim.fda.certificates import (
    estimate_eta_dd,
    estimate_gamma,
    validate_invariants,
)

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

"""
Tests for the Flow Dynamic Analysis (FDA) certificate estimators.
"""


class IdentityOp(Op):
    def __call__(self, x):
        return x


def test_estimate_gamma_gershgorin_lower_bound():
    """
    Given a non-symmetric operator,
    When estimate_gamma is called,
    Then it should return the minimum eigenvalue or Gershgorin lower bound if not symmetric.
    """

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
    assert gamma == -1.0 or abs(gamma - 1.382) < 1e-2, (
        f"Expected gamma=-1.0 (Gershgorin) or 1.382 (eig), got {gamma}"
    )


def test_estimate_eta_dd_contract():
    """
    Given a linear operator with known diagonal dominance,
    When estimate_eta_dd is called,
    Then it should return the correct eta_dd value.
    """

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

    class NegEigOp:
        def __call__(self, x):
            # Matrix [[-2, 0], [0, 3]] has eigenvalues -2, 3
            return jnp.array([-2.0 * x[0], 3.0 * x[1]])

    op = NegEigOp()
    key = jax.random.PRNGKey(0)
    gamma = estimate_gamma(op, key, (2,))
    # Should return -2.0
    assert gamma == -2.0, f"Expected -2.0, got {gamma}"


def test_validate_invariants_none_spec():
    """
    Given no invariants specification,
    When validate_invariants is called,
    Then it should return 0.0 (no drift).
    """
    state = {"x": jnp.array([1.0, 2.0])}
    drift = validate_invariants(state, None)
    assert drift == 0.0


def test_validate_invariants_conserved_quantity():
    """
    Given a conserved quantity checker,
    When validate_invariants is called,
    Then it should return drift from reference value.
    """

    def mass_checker(state):
        return float(jnp.sum(state["x"]))

    invariants_spec = {"conserved": {"mass": mass_checker}}

    # Use the same reference_values dict across calls
    reference_values = {}

    # First call establishes reference
    state1 = {"x": jnp.array([1.0, 2.0])}  # sum = 3.0
    drift1 = validate_invariants(state1, invariants_spec, reference_values)
    assert drift1 == 0.0  # First call, no drift

    # Second call with same state
    drift2 = validate_invariants(state1, invariants_spec, reference_values)
    assert drift2 == 0.0  # Same state, no drift

    # Third call with different state
    state2 = {"x": jnp.array([1.5, 2.5])}  # sum = 4.0, drift = 1.0
    drift3 = validate_invariants(state2, invariants_spec, reference_values)
    assert abs(drift3 - 1.0) < 1e-6


def test_validate_invariants_constraints():
    """
    Given constraint checkers,
    When validate_invariants is called,
    Then it should return maximum constraint violation.
    """

    def constraint1(state):
        return float(jnp.sum(state["x"]))  # Should be 0

    def constraint2(state):
        return float(jnp.mean(state["x"]) - 1.0)  # Should be 0

    invariants_spec = {
        "constraints": {"sum_zero": constraint1, "mean_one": constraint2}
    }

    # State where constraints are satisfied
    state_good = {
        "x": jnp.array([1.0, 1.0])
    }  # sum=2 (not 0), mean=1.0 (constraint2 = 1.0-1.0=0)
    drift_good = validate_invariants(state_good, invariants_spec)
    assert abs(drift_good - 2.0) < 1e-6  # constraint1 violation = 2.0

    # State where constraints are violated
    state_bad = {
        "x": jnp.array([2.0, 1.0])
    }  # sum=3 (constraint1 violation=3), mean=1.5 (constraint2 violation=0.5)
    drift_bad = validate_invariants(state_bad, invariants_spec)
    assert abs(drift_bad - 3.0) < 1e-6  # Maximum violation is 3.0


def test_validate_invariants_mixed():
    """
    Given both conserved quantities and constraints,
    When validate_invariants is called,
    Then it should return maximum drift across all invariants.
    """

    def conserved_checker(state):
        return float(jnp.sum(state["x"]))

    def constraint_checker(state):
        return float(jnp.mean(state["x"]))

    invariants_spec = {
        "conserved": {"total": conserved_checker},
        "constraints": {"mean_zero": constraint_checker},
    }

    # Use the same reference_values dict across calls
    reference_values = {}

    # Establish reference with initial state
    state1 = {"x": jnp.array([1.0, -1.0])}  # sum=0, mean=0
    drift1 = validate_invariants(state1, invariants_spec, reference_values)
    assert drift1 == 0.0

    # Test with violations in both
    state2 = {"x": jnp.array([2.0, 1.0])}  # sum=3 (drift=3), mean=1.5 (drift=1.5)
    drift2 = validate_invariants(state2, invariants_spec, reference_values)
    assert abs(drift2 - 3.0) < 1e-6  # Maximum is 3.0 from conserved quantity


def test_validate_invariants_invalid_checker():
    """
    Given invalid (non-callable) checkers,
    When validate_invariants is called,
    Then it should skip them and return finite drift.
    """
    invariants_spec = {
        "conserved": {"bad": "not_callable"},
        "constraints": {"also_bad": 123},
    }

    state = {"x": jnp.array([1.0, 2.0])}
    drift = validate_invariants(state, invariants_spec)
    assert drift == 0.0  # Invalid checkers are skipped


def test_validate_invariants_checker_failure():
    """
    Given checkers that raise exceptions,
    When validate_invariants is called,
    Then it should return infinite drift.
    """

    def failing_checker(state):
        raise ValueError("Checker failed")

    invariants_spec = {"conserved": {"failing": failing_checker}}

    state = {"x": jnp.array([1.0, 2.0])}
    drift = validate_invariants(state, invariants_spec)
    assert drift == float("inf")
