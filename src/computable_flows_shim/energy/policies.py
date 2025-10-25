"""
FlowPolicy and MultiscaleSchedule specifications with Pydantic validation.

Provides type-safe policy specifications for controlling flow execution strategies
and multiscale behavior in the Computable Flow Shim.
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class FlowPolicy(BaseModel):
    """
    Policy specification for controlling flow execution strategies.

    Defines how primitives are composed and executed during optimization.
    Used by the runtime engine to select appropriate primitive variants.

    Contract:
    - Pre: Valid family, discretization, and preconditioner (if provided)
    - Post: Policy object ready for runtime integration
    - Invariant: Policy parameters are immutable and validated
    """

    model_config = {"frozen": True}  # Make immutable

    family: Literal["basic", "preconditioned", "accelerated"] = Field(
        "basic",
        description="Flow family: basic (standard primitives), preconditioned (with preconditioning), accelerated (with momentum/acceleration)",
    )

    discretization: Literal["explicit", "implicit", "symplectic"] = Field(
        "explicit",
        description="Discretization method: explicit (forward Euler), implicit (backward Euler), symplectic (energy-preserving)",
    )

    preconditioner: str | None = Field(
        None,
        min_length=1,
        max_length=50,
        description="Preconditioner operator name (required when family='preconditioned')",
    )

    @field_validator("preconditioner")
    @classmethod
    def validate_preconditioner(cls, v, values):
        """Validate preconditioner is provided when family requires it."""
        if values.data.get("family") == "preconditioned" and v is None:
            raise ValueError("preconditioner is required when family='preconditioned'")
        return v

    @field_validator("preconditioner")
    @classmethod
    def validate_preconditioner_name(cls, v):
        """Validate preconditioner name format."""
        if v is not None and not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError(
                "preconditioner name must contain only alphanumeric characters, underscores, and hyphens"
            )
        return v


class MultiscaleSchedule(BaseModel):
    """
    Schedule specification for multiscale flow execution.

    Controls how multiscale levels are activated and managed during optimization.
    Used by the runtime engine to determine scale transitions.

    Contract:
    - Pre: Valid mode, positive levels, and activation rule
    - Post: Schedule object ready for runtime integration
    - Invariant: Schedule parameters are immutable and validated
    """

    model_config = {"frozen": True}  # Make immutable

    mode: Literal["fixed_schedule", "residual_driven", "energy_driven"] = Field(
        "fixed_schedule",
        description="Multiscale mode: fixed_schedule (sequential), residual_driven (by residual magnitude), energy_driven (by energy decrease)",
    )

    levels: int = Field(
        ..., ge=1, le=20, description="Number of multiscale levels (1-20)"
    )

    activate_rule: str = Field(
        "level_complete",
        min_length=1,
        max_length=100,
        description="Rule for activating finer scales (e.g., 'residual>tau', 'energy_decrease>0.01')",
    )

    @field_validator("activate_rule")
    @classmethod
    def validate_activate_rule(cls, v):
        """Validate activation rule format."""
        if not v or not v.strip():
            raise ValueError("activate_rule cannot be empty")

        # Basic syntax validation - should contain comparison operators
        valid_operators = [">", "<", ">=", "<=", "==", "!="]
        has_operator = any(op in v for op in valid_operators)

        if not has_operator and v != "level_complete":
            raise ValueError(
                "activate_rule must contain a comparison operator or be 'level_complete'"
            )

        return v.strip()


# Backward compatibility dataclasses (deprecated)
from dataclasses import dataclass


@dataclass(frozen=True)
class FlowPolicyDataclass:
    """Legacy dataclass version - deprecated, use FlowPolicy instead."""

    family: str
    discretization: str
    preconditioner: str | None = None


@dataclass(frozen=True)
class MultiscaleScheduleDataclass:
    """Legacy dataclass version - deprecated, use MultiscaleSchedule instead."""

    mode: str
    levels: int
    activate_rule: str = "level_complete"
