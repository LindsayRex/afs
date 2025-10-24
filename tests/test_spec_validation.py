"""
Test-Driven Development for Energy Specification Validation.

Tests Pydantic validation of EnergySpec, TermSpec, and StateSpec models.
Ensures type safety and prevents invalid specifications.
"""

import pytest
from pydantic import ValidationError
from computable_flows_shim.energy.specs import EnergySpec, TermSpec, StateSpec


class TestTermSpecValidation:
    """Test TermSpec Pydantic validation."""

    def test_valid_term_spec(self):
        """Test valid TermSpec creation."""
        term = TermSpec(
            type="quadratic",
            op="A",
            weight=1.0,
            variable="x",
            target="y"
        )
        assert term.type == "quadratic"
        assert term.weight == 1.0

    def test_invalid_term_type(self):
        """Test rejection of unknown term types."""
        with pytest.raises(ValidationError) as exc_info:
            TermSpec(
                type="invalid_type",
                op="A",
                weight=1.0,
                variable="x"
            )
        assert "Unknown term type" in str(exc_info.value)

    def test_negative_weight(self):
        """Test rejection of negative weights."""
        with pytest.raises(ValidationError) as exc_info:
            TermSpec(
                type="quadratic",
                op="A",
                weight=-1.0,
                variable="x"
            )
        assert "greater_than" in str(exc_info.value)

    def test_zero_weight(self):
        """Test rejection of zero weights."""
        with pytest.raises(ValidationError) as exc_info:
            TermSpec(
                type="quadratic",
                op="A",
                weight=0.0,
                variable="x"
            )
        assert "greater_than" in str(exc_info.value)

    def test_extreme_weight(self):
        """Test rejection of extremely large weights."""
        with pytest.raises(ValidationError) as exc_info:
            TermSpec(
                type="quadratic",
                op="A",
                weight=1e7,  # Above 1e6 limit
                variable="x"
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
            ndim=2
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
                levels=0  # Must be > 0
            )
        assert "levels" in str(exc_info.value)


class TestStateSpecValidation:
    """Test StateSpec Pydantic validation."""

    def test_valid_state_spec(self):
        """Test valid StateSpec creation."""
        state = StateSpec(
            shapes={
                "x": [10, 20],
                "y": [5]
            }
        )
        assert state.shapes["x"] == [10, 20]
        assert state.shapes["y"] == [5]

    def test_empty_shapes(self):
        """Test rejection of empty shapes dictionary."""
        with pytest.raises(ValidationError) as exc_info:
            StateSpec(shapes={})
        assert "too_short" in str(exc_info.value)

    def test_invalid_shape_type(self):
        """Test rejection of non-list shapes."""
        with pytest.raises(ValidationError) as exc_info:
            StateSpec(shapes={"x": "invalid"})
        assert "list_type" in str(exc_info.value)

    def test_empty_shape_list(self):
        """Test rejection of empty shape lists."""
        with pytest.raises(ValidationError) as exc_info:
            StateSpec(shapes={"x": []})
        assert "Shape for variable 'x' must be non-empty list" in str(exc_info.value)

    def test_negative_dimensions(self):
        """Test rejection of negative dimensions."""
        with pytest.raises(ValidationError) as exc_info:
            StateSpec(shapes={"x": [-1, 10]})
        assert "All dimensions for variable 'x' must be positive integers" in str(exc_info.value)

    def test_zero_dimensions(self):
        """Test rejection of zero dimensions."""
        with pytest.raises(ValidationError) as exc_info:
            StateSpec(shapes={"x": [0, 10]})
        assert "All dimensions for variable 'x' must be positive integers" in str(exc_info.value)

    def test_too_many_dimensions(self):
        """Test rejection of shapes with too many dimensions."""
        with pytest.raises(ValidationError) as exc_info:
            StateSpec(shapes={"x": [1, 2, 3, 4, 5]})  # 5 dimensions > 4 max
        assert "Shape for variable 'x' has too many dimensions" in str(exc_info.value)


class TestEnergySpecValidation:
    """Test EnergySpec Pydantic validation."""

    def test_valid_energy_spec(self):
        """Test valid EnergySpec creation."""
        spec = EnergySpec(
            terms=[
                TermSpec(type="quadratic", op="A", weight=1.0, variable="x", target="y")
            ],
            state=StateSpec(shapes={"x": [3], "y": [3]})
        )
        assert len(spec.terms) == 1
        assert "x" in spec.state.shapes

    def test_empty_terms(self):
        """Test rejection of specs with no terms."""
        with pytest.raises(ValidationError) as exc_info:
            EnergySpec(
                terms=[],
                state=StateSpec(shapes={"x": [3]})
            )
        assert "too_short" in str(exc_info.value)

    def test_too_many_terms(self):
        """Test rejection of specs with too many terms."""
        terms = [
            TermSpec(type="quadratic", op=f"A{i}", weight=1.0, variable="x", target="y")
            for i in range(51)  # 51 terms > 50 max
        ]
        with pytest.raises(ValidationError) as exc_info:
            EnergySpec(
                terms=terms,
                state=StateSpec(shapes={"x": [3], "y": [3]})
            )
        assert "too_long" in str(exc_info.value)

    def test_validate_against_state_success(self):
        """Test successful state validation."""
        spec = EnergySpec(
            terms=[
                TermSpec(type="quadratic", op="A", weight=1.0, variable="x", target="y")
            ],
            state=StateSpec(shapes={"x": [3], "y": [3]})
        )
        # Should not raise
        spec.validate_against_state()

    def test_validate_against_state_undefined_variable(self):
        """Test rejection of undefined variables."""
        spec = EnergySpec(
            terms=[
                TermSpec(type="quadratic", op="A", weight=1.0, variable="x", target="z")  # z not in state
            ],
            state=StateSpec(shapes={"x": [3], "y": [3]})
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
            state=StateSpec(shapes={"x": [3], "y": [3], "z": [5]})  # z unused
        )
        with pytest.warns(UserWarning, match="State variables defined but not used"):
            spec.validate_against_state()


class TestBackwardCompatibility:
    """Test that legacy dataclasses still work."""

    def test_dataclass_import(self):
        """Test that legacy dataclasses can still be imported."""
        from computable_flows_shim.energy.specs import TermSpecDataclass, StateSpecDataclass, EnergySpecDataclass

        # Should be able to create instances
        term = TermSpecDataclass(
            type="quadratic",
            op="A",
            weight=1.0,
            variable="x"
        )
        assert term.type == "quadratic"

        state = StateSpecDataclass(shapes={"x": [3]})
        assert state.shapes["x"] == [3]

        spec = EnergySpecDataclass(
            terms=[term],
            state=state
        )
        assert len(spec.terms) == 1