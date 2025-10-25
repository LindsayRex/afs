"""
Test-Driven Development for Energy Specification Validation.

Tests Pydantic validation of EnergySpec, TermSpec, and StateSpec models.
Ensures type safety and prevents invalid specifications.
"""

import pytest
from pydantic import ValidationError

from computable_flows_shim.energy.specs import EnergySpec, StateSpec, TermSpec


class TestTermSpecValidation:
    """Test TermSpec Pydantic validation."""

    def test_valid_term_spec(self):
        """Test valid TermSpec creation."""
        term = TermSpec(type="quadratic", op="A", weight=1.0, variable="x", target="y")
        assert term.type == "quadratic"
        assert term.weight == 1.0

    def test_invalid_term_type(self):
        """Test rejection of unknown term types."""
        with pytest.raises(ValidationError) as exc_info:
            TermSpec(type="invalid_type", op="A", weight=1.0, variable="x")
        assert "Unknown term type" in str(exc_info.value)

    def test_negative_weight(self):
        """Test rejection of negative weights."""
        with pytest.raises(ValidationError) as exc_info:
            TermSpec(type="quadratic", op="A", weight=-1.0, variable="x")
        assert "greater_than" in str(exc_info.value)

    def test_zero_weight(self):
        """Test rejection of zero weights."""
        with pytest.raises(ValidationError) as exc_info:
            TermSpec(type="quadratic", op="A", weight=0.0, variable="x")
        assert "greater_than" in str(exc_info.value)

    def test_extreme_weight(self):
        """Test rejection of extremely large weights."""
        with pytest.raises(ValidationError) as exc_info:
            TermSpec(
                type="quadratic",
                op="A",
                weight=1e7,  # Above 1e6 limit
                variable="x",
            )
        assert "less_than_equal" in str(exc_info.value)

    def test_wavelet_term_spec(self):
        """Test wavelet-specific parameters."""
        term = TermSpec(
            type="wavelet_l1",
            op="W",
            weight=0.1,
            variable="x",
            wavelet="db4",
            levels=3,
            ndim=2,
        )
        assert term.wavelet == "db4"
        assert term.levels == 3
        assert term.ndim == 2

    def test_invalid_wavelet_levels(self):
        """Test rejection of invalid wavelet levels."""
        with pytest.raises(ValidationError) as exc_info:
            TermSpec(
                type="wavelet",
                op="W",
                weight=1.0,
                variable="x",
                levels=0,  # Must be > 0
            )
        assert "levels" in str(exc_info.value)


class TestStateSpecInvariantsValidation:
    """Test StateSpec invariants contract validation."""

    def test_valid_invariants_spec(self):
        """Test valid invariants specification."""

        def mass_checker(state):
            return state.sum()

        def balance_checker(state):
            return state.mean()

        invariants = {
            "conserved": {"mass": mass_checker},
            "constraints": {"balance": balance_checker},
            "symmetries": ["translation", "rotation"],
        }

        state = StateSpec(shapes={"x": [10, 20]}, invariants=invariants)
        assert state.invariants == invariants

    def test_invariants_none_allowed(self):
        """Test that invariants can be None (default)."""
        state = StateSpec(shapes={"x": [10, 20]})
        assert state.invariants is None

    def test_invariants_wrong_type(self):
        """Test rejection of non-dict invariants."""
        with pytest.raises(ValidationError) as exc_info:
            StateSpec(shapes={"x": [10, 20]}, invariants="invalid")
        assert "Input should be a valid dictionary" in str(exc_info.value)

    def test_invariants_unknown_key(self):
        """Test rejection of unknown invariant types."""
        with pytest.raises(ValidationError) as exc_info:
            StateSpec(shapes={"x": [10, 20]}, invariants={"unknown_type": {}})
        assert "Unknown invariant type" in str(exc_info.value)

    def test_conserved_not_dict(self):
        """Test rejection when conserved is not a dict."""
        with pytest.raises(ValidationError) as exc_info:
            StateSpec(shapes={"x": [10, 20]}, invariants={"conserved": "not_a_dict"})
        assert "invariants.conserved must be a dictionary" in str(exc_info.value)

    def test_conserved_non_callable(self):
        """Test rejection when conserved value is not callable."""
        with pytest.raises(ValidationError) as exc_info:
            StateSpec(
                shapes={"x": [10, 20]},
                invariants={"conserved": {"mass": "not_callable"}},
            )
        assert "must be callable" in str(exc_info.value)

    def test_constraints_not_dict(self):
        """Test rejection when constraints is not a dict."""
        with pytest.raises(ValidationError) as exc_info:
            StateSpec(shapes={"x": [10, 20]}, invariants={"constraints": "not_a_dict"})
        assert "invariants.constraints must be a dictionary" in str(exc_info.value)

    def test_constraints_non_callable(self):
        """Test rejection when constraints value is not callable."""
        with pytest.raises(ValidationError) as exc_info:
            StateSpec(
                shapes={"x": [10, 20]},
                invariants={"constraints": {"balance": "not_callable"}},
            )
        assert "must be callable" in str(exc_info.value)

    def test_symmetries_not_list(self):
        """Test rejection when symmetries is not a list."""
        with pytest.raises(ValidationError) as exc_info:
            StateSpec(shapes={"x": [10, 20]}, invariants={"symmetries": "not_a_list"})
        assert "invariants.symmetries must be a list" in str(exc_info.value)

    def test_symmetries_non_string(self):
        """Test rejection when symmetries contains non-strings."""
        with pytest.raises(ValidationError) as exc_info:
            StateSpec(
                shapes={"x": [10, 20]}, invariants={"symmetries": ["translation", 123]}
            )
        assert "invariants.symmetries must contain only strings" in str(exc_info.value)


class TestEnergySpecValidation:
    """Test EnergySpec Pydantic validation."""

    def test_valid_energy_spec(self):
        """Test valid EnergySpec creation."""
        spec = EnergySpec(
            terms=[
                TermSpec(type="quadratic", op="A", weight=1.0, variable="x", target="y")
            ],
            state=StateSpec(shapes={"x": [3], "y": [3]}),
        )
        assert len(spec.terms) == 1
        assert "x" in spec.state.shapes  # pylint: disable=no-member

    def test_empty_terms(self):
        """Test rejection of specs with no terms."""
        with pytest.raises(ValidationError) as exc_info:
            EnergySpec(terms=[], state=StateSpec(shapes={"x": [3]}))
        assert "too_short" in str(exc_info.value)

    def test_too_many_terms(self):
        """Test rejection of specs with too many terms."""
        terms = [
            TermSpec(type="quadratic", op=f"A{i}", weight=1.0, variable="x", target="y")
            for i in range(51)  # 51 terms > 50 max
        ]
        with pytest.raises(ValidationError) as exc_info:
            EnergySpec(terms=terms, state=StateSpec(shapes={"x": [3], "y": [3]}))
        assert "too_long" in str(exc_info.value)

    def test_validate_against_state_success(self):
        """Test successful state validation."""
        spec = EnergySpec(
            terms=[
                TermSpec(type="quadratic", op="A", weight=1.0, variable="x", target="y")
            ],
            state=StateSpec(shapes={"x": [3], "y": [3]}),
        )
        # Should not raise
        spec.validate_against_state()

    def test_validate_against_state_undefined_variable(self):
        """Test rejection of undefined variables."""
        spec = EnergySpec(
            terms=[
                TermSpec(
                    type="quadratic", op="A", weight=1.0, variable="x", target="z"
                )  # z not in state
            ],
            state=StateSpec(shapes={"x": [3], "y": [3]}),
        )
        with pytest.raises(ValueError) as exc_info:
            spec.validate_against_state()
        assert "undefined variables" in str(exc_info.value)

    def test_validate_against_state_unused_variable(self):
        """Test warning for unused variables."""
        spec = EnergySpec(
            terms=[
                TermSpec(type="quadratic", op="A", weight=1.0, variable="x", target="y")
            ],
            state=StateSpec(shapes={"x": [3], "y": [3], "z": [5]}),  # z unused
        )
        with pytest.warns(UserWarning, match="State variables defined but not used"):
            spec.validate_against_state()


class TestBackwardCompatibility:
    """Test that legacy dataclasses still work."""

    def test_dataclass_import(self):
        """Test that legacy dataclasses can still be imported."""
        from computable_flows_shim.energy.specs import (
            EnergySpecDataclass,
            StateSpecDataclass,
            TermSpecDataclass,
        )

        # Should be able to create instances
        term = TermSpecDataclass(type="quadratic", op="A", weight=1.0, variable="x")
        assert term.type == "quadratic"

        state = StateSpecDataclass(shapes={"x": [3]})
        assert state.shapes["x"] == [3]

        spec = EnergySpecDataclass(terms=[term], state=state)
        assert len(spec.terms) == 1
