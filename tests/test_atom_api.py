"""
Tests for the atom-based API.

This module tests the high-level atom-based optimization API that allows users
to specify optimization problems using atoms directly, rather than constructing
EnergySpec manually.
"""

import jax.numpy as jnp
import pytest

from computable_flows_shim.api import atom_spec_to_energy_spec, run_certified


def test_run_certified_with_atom_spec_green():
    """
    GREEN: Test that run_certified can accept an AtomSpec and run optimization.

    This test verifies the atom-based API works end-to-end:
    - Users can specify atoms directly
    - The API compiles atoms to EnergySpec internally
    - Optimization runs and produces results
    """
    # GIVEN an AtomSpec with L1 regularization (simpler test case)
    atom_spec = {
        "atoms": [
            {"type": "l1", "params": {"lambda": 0.1}, "weight": 1.0, "variable": "x"},
        ],
        "state": {"x": {"shape": [2]}},
        "initial_state": {"x": jnp.array([1.0, 1.0])},
        "num_iterations": 5,
        "step_alpha": 0.1,
    }

    # WHEN we call run_certified with the atom spec
    final_state, telemetry = run_certified(atom_spec)

    # THEN it should compile and run the optimization
    assert final_state is not None
    assert "x" in final_state
    assert final_state["x"].shape == (2,)
    assert telemetry is not None


def test_atom_spec_validation_green():
    """
    GREEN: Test that AtomSpec validation works correctly.

    Invalid atom specs should be rejected with clear error messages.
    """
    # GIVEN an invalid atom spec (unknown atom type)
    invalid_spec = {
        "atoms": [
            {
                "type": "unknown_atom",  # This atom doesn't exist
                "params": {},
                "weight": 1.0,
                "variable": "x",
            }
        ],
        "state": {"x": {"shape": [2]}},
        "initial_state": {"x": jnp.array([1.0, 2.0])},
        "num_iterations": 10,
        "step_alpha": 0.1,
    }

    # WHEN we try to run it
    # THEN it should fail with a clear ValidationError
    from pydantic_core import ValidationError

    with pytest.raises(ValidationError) as exc_info:
        run_certified(invalid_spec)

    # Verify the error message mentions unknown atom type
    assert "Unknown atom type" in str(exc_info.value)


def test_atom_spec_to_energy_spec_conversion_green():
    """
    GREEN: Test that AtomSpec can be converted to EnergySpec.

    The atom-based API should internally convert AtomSpec to EnergySpec
    for compatibility with the existing compilation pipeline.
    """
    # GIVEN a simple atom spec
    atom_spec_dict = {
        "atoms": [
            {
                "type": "quadratic",
                "params": {"A": jnp.eye(2), "b": jnp.array([1.0, 2.0])},
                "weight": 1.0,
                "variable": "x",
            }
        ],
        "state": {"x": {"shape": [2]}},
        "initial_state": {"x": jnp.array([0.0, 0.0])},
        "num_iterations": 5,
        "step_alpha": 0.1,
    }

    # Convert to AtomBasedSpec first
    from computable_flows_shim.api import AtomBasedSpec

    atom_spec = AtomBasedSpec(**atom_spec_dict)

    # WHEN we convert to EnergySpec
    energy_spec, op_registry = atom_spec_to_energy_spec(atom_spec)

    # THEN it should produce a valid EnergySpec
    assert energy_spec is not None
    assert op_registry is not None
    assert len(energy_spec.terms) == 1
    assert energy_spec.terms[0].type == "quadratic"
