"""
Tests for telemetry schema validation.
"""

import time

import pytest
from pydantic import ValidationError

from computable_flows_shim.telemetry.schemas import (
    TelemetryEvent,
    TelemetrySample,
    TelemetrySchemaValidator,
)


class TestTelemetrySample:
    """Tests for TelemetrySample model validation."""

    def test_valid_sample_creation(self):
        """Test creating a valid telemetry sample."""
        sample_data = {
            "run_id": "test_run_123",
            "flow_name": "test_flow",
            "phase": "GREEN",
            "iter": 1,
            "t_wall_ms": int(time.time() * 1000),
            "E": 1.5,
            "grad_norm": 0.1,
            "eta_dd": 0.01,
            "gamma": 0.9,
            "alpha": 0.1,
            "phi_residual": 0.001,
            "invariant_drift_max": 0.0001,
        }

        sample = TelemetrySample(**sample_data)

        assert sample.run_id == "test_run_123"
        assert sample.flow_name == "test_flow"
        assert sample.phase == "GREEN"
        assert sample.iter == 1
        assert sample.E == 1.5
        assert sample.grad_norm == 0.1
        assert sample.eta_dd == 0.01
        assert sample.gamma == 0.9
        assert sample.alpha == 0.1
        assert sample.phi_residual == 0.001
        assert sample.invariant_drift_max == 0.0001

    def test_invalid_phase(self):
        """Test that invalid phase values are rejected."""
        sample_data = {
            "run_id": "test_run_123",
            "flow_name": "test_flow",
            "phase": "INVALID_PHASE",  # Invalid phase
            "iter": 1,
            "t_wall_ms": int(time.time() * 1000),
            "E": 1.5,
            "grad_norm": 0.1,
            "eta_dd": 0.01,
            "gamma": 0.9,
            "alpha": 0.1,
            "phi_residual": 0.001,
            "invariant_drift_max": 0.0001,
        }

        with pytest.raises(ValidationError) as exc_info:
            TelemetrySample(**sample_data)

        assert "string_pattern_mismatch" in str(exc_info.value)
        assert "phase" in str(exc_info.value)

    def test_missing_required_field(self):
        """Test that missing required fields are rejected."""
        sample_data = {
            # Missing flow_name
            "run_id": "test_run_123",
            "phase": "GREEN",
            "iter": 1,
            "t_wall_ms": int(time.time() * 1000),
            "E": 1.5,
            "grad_norm": 0.1,
            "eta_dd": 0.01,
            "gamma": 0.9,
            "alpha": 0.1,
            "phi_residual": 0.001,
            "invariant_drift_max": 0.0001,
        }

        with pytest.raises(ValidationError) as exc_info:
            TelemetrySample(**sample_data)

        assert "missing" in str(exc_info.value)
        assert "flow_name" in str(exc_info.value)

    def test_invalid_timestamp_future(self):
        """Test that future timestamps are rejected."""
        future_time = int(time.time() * 1000) + 7200000  # 2 hours in future

        sample_data = {
            "run_id": "test_run_123",
            "flow_name": "test_flow",
            "phase": "GREEN",
            "iter": 1,
            "t_wall_ms": future_time,  # Future timestamp
            "E": 1.5,
            "grad_norm": 0.1,
            "eta_dd": 0.01,
            "gamma": 0.9,
            "alpha": 0.1,
            "phi_residual": 0.001,
            "invariant_drift_max": 0.0001,
        }

        with pytest.raises(ValidationError) as exc_info:
            TelemetrySample(**sample_data)

        assert "t_wall_ms appears to be in the future" in str(exc_info.value)

    def test_invalid_timestamp_too_old(self):
        """Test that timestamps that are too old are rejected."""
        old_time = int(time.time() * 1000) - 94608000000  # 3 years ago

        sample_data = {
            "run_id": "test_run_123",
            "flow_name": "test_flow",
            "phase": "GREEN",
            "iter": 1,
            "t_wall_ms": old_time,  # Too old timestamp
            "E": 1.5,
            "grad_norm": 0.1,
            "eta_dd": 0.01,
            "gamma": 0.9,
            "alpha": 0.1,
            "phi_residual": 0.001,
            "invariant_drift_max": 0.0001,
        }

        with pytest.raises(ValidationError) as exc_info:
            TelemetrySample(**sample_data)

        assert "t_wall_ms appears to be too old" in str(exc_info.value)

    def test_optional_fields(self):
        """Test that optional fields work correctly."""
        sample_data = {
            "run_id": "test_run_123",
            "flow_name": "test_flow",
            "phase": "GREEN",
            "iter": 1,
            "t_wall_ms": int(time.time() * 1000),
            "E": 1.5,
            "grad_norm": 0.1,
            "eta_dd": 0.01,
            "gamma": 0.9,
            "alpha": 0.1,
            "phi_residual": 0.001,
            "invariant_drift_max": 0.0001,
            "trial_id": "trial_001",
            "lambda": 0.5,
            "sparsity_wx": 0.1,
            "flow_family": "gradient",
            "metric_ber": 0.95,
            "warnings": "Some warning",
            "notes": "Some notes",
        }

        sample = TelemetrySample(**sample_data)

        assert sample.trial_id == "trial_001"
        assert sample.lambda_ == 0.5
        assert sample.sparsity_wx == 0.1
        assert sample.flow_family == "gradient"
        assert sample.metric_ber == 0.95
        assert sample.warnings == "Some warning"
        assert sample.notes == "Some notes"


class TestTelemetryEvent:
    """Tests for TelemetryEvent model validation."""

    def test_valid_event_creation(self):
        """Test creating a valid telemetry event."""
        event_data = {
            "run_id": "test_run_123",
            "t_wall_ms": int(time.time() * 1000),
            "event": "RUN_STARTED",
            "payload": {"message": "Run started successfully"},
        }

        event = TelemetryEvent(**event_data)

        assert event.run_id == "test_run_123"
        assert event.event == "RUN_STARTED"
        assert event.payload == '{"message": "Run started successfully"}'

    def test_invalid_event_type(self):
        """Test that invalid event types are rejected."""
        event_data = {
            "run_id": "test_run_123",
            "t_wall_ms": int(time.time() * 1000),
            "event": "INVALID_EVENT",  # Invalid event type
            "payload": "test payload",
        }

        with pytest.raises(ValidationError) as exc_info:
            TelemetryEvent(**event_data)

        assert "Invalid event type" in str(exc_info.value)
        assert "INVALID_EVENT" in str(exc_info.value)

    def test_payload_dict_conversion(self):
        """Test that dict payloads are converted to JSON strings."""
        event_data = {
            "run_id": "test_run_123",
            "t_wall_ms": int(time.time() * 1000),
            "event": "RUN_STARTED",
            "payload": {"key": "value", "number": 42},  # Dict payload
        }

        event = TelemetryEvent(**event_data)

        assert event.payload == '{"key": "value", "number": 42}'

    def test_payload_string_passthrough(self):
        """Test that string payloads are passed through unchanged."""
        event_data = {
            "run_id": "test_run_123",
            "t_wall_ms": int(time.time() * 1000),
            "event": "RUN_STARTED",
            "payload": "plain string payload",  # String payload
        }

        event = TelemetryEvent(**event_data)

        assert event.payload == "plain string payload"


class TestTelemetrySchemaValidator:
    """Tests for TelemetrySchemaValidator utility methods."""

    def test_validate_sample(self):
        """Test single sample validation."""
        sample_data = {
            "run_id": "test_run_123",
            "flow_name": "test_flow",
            "phase": "GREEN",
            "iter": 1,
            "t_wall_ms": int(time.time() * 1000),
            "E": 1.5,
            "grad_norm": 0.1,
            "eta_dd": 0.01,
            "gamma": 0.9,
            "alpha": 0.1,
            "phi_residual": 0.001,
            "invariant_drift_max": 0.0001,
        }

        sample = TelemetrySchemaValidator.validate_sample(sample_data)

        assert isinstance(sample, TelemetrySample)
        assert sample.run_id == "test_run_123"

    def test_validate_event(self):
        """Test single event validation."""
        event_data = {
            "run_id": "test_run_123",
            "t_wall_ms": int(time.time() * 1000),
            "event": "RUN_STARTED",
            "payload": "test payload",
        }

        event = TelemetrySchemaValidator.validate_event(event_data)

        assert isinstance(event, TelemetryEvent)
        assert event.event == "RUN_STARTED"

    def test_validate_samples_batch(self):
        """Test batch sample validation."""
        samples_data = [
            {
                "run_id": "test_run_123",
                "flow_name": "test_flow",
                "phase": "GREEN",
                "iter": i,
                "t_wall_ms": int(time.time() * 1000) + i * 1000,
                "E": 1.5,
                "grad_norm": 0.1,
                "eta_dd": 0.01,
                "gamma": 0.9,
                "alpha": 0.1,
                "phi_residual": 0.001,
                "invariant_drift_max": 0.0001,
            }
            for i in range(3)
        ]

        samples = TelemetrySchemaValidator.validate_samples_batch(samples_data)

        assert len(samples) == 3
        assert all(isinstance(s, TelemetrySample) for s in samples)
        assert samples[0].iter == 0
        assert samples[1].iter == 1
        assert samples[2].iter == 2

    def test_validate_events_batch(self):
        """Test batch event validation."""
        events_data = [
            {
                "run_id": "test_run_123",
                "t_wall_ms": int(time.time() * 1000) + i * 1000,
                "event": "RUN_STARTED",
                "payload": f"payload_{i}",
            }
            for i in range(2)
        ]

        events = TelemetrySchemaValidator.validate_events_batch(events_data)

        assert len(events) == 2
        assert all(isinstance(e, TelemetryEvent) for e in events)
        assert events[0].payload == "payload_0"
        assert events[1].payload == "payload_1"

    def test_schema_version(self):
        """Test schema version reporting."""
        assert TelemetrySchemaValidator.get_schema_version() == 3

    def test_required_fields(self):
        """Test required fields listing."""
        required = TelemetrySchemaValidator.get_required_fields()

        expected_required = [
            "run_id",
            "flow_name",
            "phase",
            "iter",
            "t_wall_ms",
            "E",
            "grad_norm",
            "eta_dd",
            "gamma",
            "alpha",
            "phi_residual",
            "invariant_drift_max",
        ]

        assert set(required) == set(expected_required)

    def test_optional_fields(self):
        """Test optional fields listing."""
        optional = TelemetrySchemaValidator.get_optional_fields()

        # Should include all fields not in required
        required = TelemetrySchemaValidator.get_required_fields()
        all_fields = set(TelemetrySample.__fields__.keys())
        expected_optional = all_fields - set(required)

        assert set(optional) == expected_optional

    def test_batch_validation_with_invalid_data(self):
        """Test that batch validation fails with invalid data."""
        samples_data = [
            {
                "run_id": "test_run_123",
                "flow_name": "test_flow",
                "phase": "GREEN",
                "iter": 0,
                "t_wall_ms": int(time.time() * 1000),
                "E": 1.5,
                "grad_norm": 0.1,
                "eta_dd": 0.01,
                "gamma": 0.9,
                "alpha": 0.1,
                "phi_residual": 0.001,
                "invariant_drift_max": 0.0001,
            },
            {
                "run_id": "test_run_123",
                "flow_name": "test_flow",
                "phase": "INVALID_PHASE",  # Invalid phase
                "iter": 1,
                "t_wall_ms": int(time.time() * 1000),
                "E": 1.5,
                "grad_norm": 0.1,
                "eta_dd": 0.01,
                "gamma": 0.9,
                "alpha": 0.1,
                "phi_residual": 0.001,
                "invariant_drift_max": 0.0001,
            },
        ]

        with pytest.raises(ValidationError):
            TelemetrySchemaValidator.validate_samples_batch(samples_data)
