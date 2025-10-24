"""
Energy specification models with Pydantic validation.

Provides type-safe specifications for computable flow problems with automatic validation.
"""
from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field, field_validator, ValidationError


class TermSpec(BaseModel):
    """Specification for a single term in the energy functional with validation."""

    type: str = Field(..., min_length=1, max_length=50, description="Term type (e.g., 'quadratic', 'wavelet')")
    op: str = Field(..., min_length=1, max_length=50, description="Operator name")
    weight: float = Field(..., gt=0, le=1e6, description="Positive weight coefficient")
    variable: str = Field(..., min_length=1, max_length=50, description="Variable name")
    target: Optional[str] = Field(None, min_length=1, max_length=50, description="Target variable name")

    # Wavelet-specific parameters
    wavelet: Optional[str] = None
    levels: Optional[int] = None
    ndim: Optional[int] = None

    @field_validator('type')
    @classmethod
    def validate_term_type(cls, v):
        """Validate term type is one of the supported atom types."""
        known_types = {'quadratic', 'tikhonov', 'l1', 'wavelet_l1'}
        if v not in known_types:
            raise ValueError(f"Unknown term type '{v}'. Supported types: {sorted(known_types)}")
        return v

    @field_validator('wavelet')
    @classmethod
    def validate_wavelet(cls, v):
        """Validate wavelet parameter if provided."""
        if v is not None and (len(v) < 1 or len(v) > 20):
            raise ValueError("wavelet must be between 1 and 20 characters")
        return v

    @field_validator('levels')
    @classmethod
    def validate_levels(cls, v):
        """Validate levels parameter if provided."""
        if v is not None and (v <= 0 or v > 10):
            raise ValueError("levels must be between 1 and 10")
        return v

    @field_validator('ndim')
    @classmethod
    def validate_ndim(cls, v):
        """Validate ndim parameter if provided."""
        if v is not None and (v <= 0 or v > 3):
            raise ValueError("ndim must be between 1 and 3")
        return v


class StateSpec(BaseModel):
    """Specification for the state variables with validation."""

    shapes: Dict[str, List[int]] = Field(..., min_length=1, description="Variable shapes dictionary")

    @field_validator('shapes')
    @classmethod
    def validate_shapes(cls, v):
        """Validate shape specifications."""
        for var_name, shape in v.items():
            if not isinstance(shape, list) or not shape:
                raise ValueError(f"Shape for variable '{var_name}' must be non-empty list")
            if not all(isinstance(dim, int) and dim > 0 for dim in shape):
                raise ValueError(f"All dimensions for variable '{var_name}' must be positive integers")
            if len(shape) > 4:
                raise ValueError(f"Shape for variable '{var_name}' has too many dimensions (max 4)")
        return v


class EnergySpec(BaseModel):
    """The complete energy specification with comprehensive validation."""

    terms: List[TermSpec] = Field(..., min_length=1, max_length=50, description="Energy terms")
    state: StateSpec = Field(..., description="State variable specifications")

    def validate_against_state(self) -> None:
        """Runtime validation that terms are consistent with state spec."""
        state_vars = set(self.state.shapes.keys())
        term_vars = set()

        for term in self.terms:
            term_vars.add(term.variable)
            if term.target:
                term_vars.add(term.target)

        undefined_vars = term_vars - state_vars
        if undefined_vars:
            raise ValueError(f"Terms reference undefined variables: {undefined_vars}")

        unused_vars = state_vars - term_vars
        if unused_vars:
            # Warning, not error - variables might be used elsewhere
            import warnings
            warnings.warn(f"State variables defined but not used in terms: {unused_vars}")


# Backward compatibility: keep dataclass versions for existing code
# These will be deprecated in favor of Pydantic models
from dataclasses import dataclass

@dataclass(frozen=True)
class TermSpecDataclass:
    """Legacy dataclass version - deprecated, use TermSpec instead."""
    type: str
    op: str
    weight: float
    variable: str
    target: Optional[str] = None
    wavelet: Optional[str] = None
    levels: Optional[int] = None
    ndim: Optional[int] = None

@dataclass(frozen=True)
class StateSpecDataclass:
    """Legacy dataclass version - deprecated, use StateSpec instead."""
    shapes: Dict[str, List[int]]

@dataclass(frozen=True)
class EnergySpecDataclass:
    """Legacy dataclass version - deprecated, use EnergySpec instead."""
    terms: List[TermSpecDataclass]
    state: StateSpecDataclass
