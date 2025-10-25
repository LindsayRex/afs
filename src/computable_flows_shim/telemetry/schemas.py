"""
Telemetry Schema Validation System

This module provides Pydantic models for validating telemetry and events schemas
according to the AFS telemetry specification (schema_version: 3).

Infrastructure component using TDD methodology - provides schema validation
for telemetry data structures and ensures data integrity.
"""

import json
from datetime import UTC, datetime
from typing import Any, ClassVar

from pydantic import BaseModel, Field, field_validator


class TelemetrySample(BaseModel):
    """
    Pydantic model for a single telemetry sample in telemetry.parquet.

    Validates all required and optional fields according to schema_version: 3.
    """

    # Core/Minimal fields (must always be recorded)
    run_id: str = Field(
        ..., min_length=1, max_length=100, description="Unique run identifier"
    )
    flow_name: str = Field(
        ..., min_length=1, max_length=100, description="User-visible flow/spec name"
    )
    phase: str = Field(
        ..., pattern="^(RED|AMBER|GREEN)$", description="Optimization phase"
    )
    iter: int = Field(..., ge=0, description="Iteration number")
    t_wall_ms: float = Field(
        ..., ge=0, description="Wall-clock time in milliseconds since run start"
    )
    E: float = Field(..., description="Energy value")
    grad_norm: float = Field(..., ge=0, description="Norm of ∇f (for Lyapunov checks)")
    eta_dd: float = Field(..., ge=0, description="Diagonal dominance metric (FDA)")
    gamma: float = Field(..., ge=0, description="Spectral gap (FDA)")
    alpha: float = Field(
        ..., gt=0, description="Step-size / learning rate used for the iteration"
    )

    # Promoted core fields (essential for certification)
    phi_residual: float = Field(
        ...,
        ge=0,
        description="Physics residual norm - measures PDE/constraint violation",
    )
    invariant_drift_max: float = Field(
        ..., ge=0, description="Maximum absolute drift across declared invariants"
    )

    # Recommended fields (useful for tuning and QA)
    trial_id: str | None = Field(
        None, min_length=1, max_length=100, description="For tuner trials and groupings"
    )
    lambda_: float | None = Field(
        None, alias="lambda", description="Global sparsity parameter"
    )
    lambda_j: dict[str, Any] | str | None = Field(
        None, description="Per-scale sparsity parameters (dict or JSON string)"
    )
    sparsity_wx: float | None = Field(
        None, ge=0, le=1, description="Sparsity fraction in W-space"
    )
    level_active_max: int | None = Field(
        None, ge=0, description="Current finest active scale"
    )
    sparsity_mode: str | None = Field(
        None, pattern="^(l1|group_l1|tree)$", description="Sparsity regularization mode"
    )
    flow_family: str | None = Field(
        None,
        pattern="^(gradient|preconditioned|proximal|hamiltonian_damped)$",
        description="Flow family type",
    )
    lens_name: str | None = Field(
        None, min_length=1, max_length=100, description="Selected transform name"
    )

    # Diagnostic/Optional fields (high-cardinality)
    metric_ber: float | None = Field(
        None, description="Domain-specific metric (unitless)"
    )
    warnings: str | None = Field(None, description="Comma-separated warnings")
    notes: str | None = Field(None, description="Free-form notes captured on events")

    class Config:
        """Pydantic configuration."""

        allow_population_by_field_name = True
        json_encoders: ClassVar[dict] = {datetime: lambda v: v.isoformat()}

    @field_validator("lambda_j", mode="before")
    @classmethod
    def validate_lambda_j(cls, v):
        """Convert lambda_j to JSON string if it's a dict."""
        if isinstance(v, dict):
            return json.dumps(v)
        return v

    @field_validator("t_wall_ms")
    @classmethod
    def validate_t_wall_ms(cls, v):
        """Ensure t_wall_ms is reasonable (not in the future, not too old)."""
        now_ms = datetime.now(UTC).timestamp() * 1000
        # Allow timestamps up to 1 hour in the future (for clock skew)
        # and up to 1 year in the past
        if v > now_ms + 3600000:  # 1 hour in milliseconds
            raise ValueError("t_wall_ms appears to be in the future")
        if v < now_ms - 31536000000:  # 1 year in milliseconds
            raise ValueError("t_wall_ms appears to be too old")
        return v


class TelemetryEvent(BaseModel):
    """
    Pydantic model for a single event in events.parquet.

    Validates event structure according to schema_version: 3.
    """

    run_id: str = Field(
        ..., min_length=1, max_length=100, description="Unique run identifier"
    )
    t_wall_ms: float = Field(
        ..., ge=0, description="Wall-clock time in milliseconds since run start"
    )
    event: str = Field(..., description="Event type from predefined enum")
    payload: dict[str, Any] | str = Field(
        ..., description="Arbitrary event payload as dict or JSON string"
    )

    # Valid event types according to schema
    VALID_EVENTS: ClassVar[set[str]] = {
        "SPEC_LINT_FAIL",
        "CERT_FAIL",
        "CERT_PASS",
        "TUNER_MOVE_REJECTED",
        "ROLLBACK",
        "TIMEOUT",
        "CANCELLED",
        "RUN_STARTED",
        "RUN_FINISHED",
        "LENS_SELECTED",
        "SCALE_ACTIVATED",
    }

    @field_validator("event")
    @classmethod
    def validate_event_type(cls, v):
        """Ensure event is from the predefined enum."""
        if v not in cls.VALID_EVENTS:
            raise ValueError(
                f"Invalid event type '{v}'. Must be one of: {', '.join(sorted(cls.VALID_EVENTS))}"
            )
        return v

    @field_validator("payload", mode="before")
    @classmethod
    def validate_payload(cls, v):
        """Convert payload to JSON string if it's a dict."""
        if isinstance(v, dict):
            return json.dumps(v)
        return v

    @field_validator("t_wall_ms")
    @classmethod
    def validate_t_wall_ms(cls, v):
        """Ensure t_wall_ms is reasonable."""
        now_ms = datetime.now(UTC).timestamp() * 1000
        if v > now_ms + 3600000:  # 1 hour in milliseconds
            raise ValueError("t_wall_ms appears to be in the future")
        if v < now_ms - 31536000000:  # 1 year in milliseconds
            raise ValueError("t_wall_ms appears to be too old")
        return v


class TelemetrySchemaValidator:
    """
    Validator for telemetry data structures.

    Provides methods to validate individual samples/events and batches.
    """

    @staticmethod
    def validate_sample(data: dict[str, Any]) -> TelemetrySample:
        """
        Validate a single telemetry sample.

        Args:
            data: Raw telemetry data dictionary

        Returns:
            Validated TelemetrySample instance

        Raises:
            ValidationError: If data doesn't match schema
        """
        return TelemetrySample(**data)

    @staticmethod
    def validate_event(data: dict[str, Any]) -> TelemetryEvent:
        """
        Validate a single telemetry event.

        Args:
            data: Raw event data dictionary

        Returns:
            Validated TelemetryEvent instance

        Raises:
            ValidationError: If data doesn't match schema
        """
        return TelemetryEvent(**data)

    @staticmethod
    def validate_samples_batch(
        data_list: list[dict[str, Any]],
    ) -> list[TelemetrySample]:
        """
        Validate a batch of telemetry samples.

        Args:
            data_list: List of raw telemetry data dictionaries

        Returns:
            List of validated TelemetrySample instances

        Raises:
            ValidationError: If any sample doesn't match schema
        """
        return [TelemetrySample(**data) for data in data_list]

    @staticmethod
    def validate_events_batch(data_list: list[dict[str, Any]]) -> list[TelemetryEvent]:
        """
        Validate a batch of telemetry events.

        Args:
            data_list: List of raw event data dictionaries

        Returns:
            List of validated TelemetryEvent instances

        Raises:
            ValidationError: If any event doesn't match schema
        """
        return [TelemetryEvent(**data) for data in data_list]

    @staticmethod
    def get_schema_version() -> int:
        """Get the current schema version."""
        return 3

    @staticmethod
    def get_required_fields() -> list[str]:
        """Get list of required fields for telemetry samples."""
        return [
            field
            for field, field_info in TelemetrySample.__fields__.items()
            if field_info.is_required()
        ]

    @staticmethod
    def get_optional_fields() -> list[str]:
        """Get list of optional fields for telemetry samples."""
        return [
            field
            for field, field_info in TelemetrySample.__fields__.items()
            if not field_info.is_required()
        ]
