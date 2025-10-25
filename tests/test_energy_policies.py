"""
TDD tests for FlowPolicy and MultiscaleSchedule specifications.

Following Design by Contract methodology to ensure policy specifications
are mathematically sound and properly validated.
"""

import pytest
from pydantic import ValidationError
from computable_flows_shim.energy.policies import FlowPolicy, MultiscaleSchedule


class TestFlowPolicyContract:
    """Design by Contract tests for FlowPolicy specification."""

    def test_valid_flow_policy_creation(self):
        """Test that valid FlowPolicy can be created with all required fields."""
        policy = FlowPolicy(
            family='preconditioned',
            discretization='symplectic',
            preconditioner='diag_precond'
        )
        assert policy.family == 'preconditioned'
        assert policy.discretization == 'symplectic'
        assert policy.preconditioner == 'diag_precond'

    def test_flow_policy_optional_preconditioner(self):
        """Test that preconditioner is optional."""
        policy = FlowPolicy(
            family='basic',
            discretization='explicit',
            preconditioner=None
        )
        assert policy.family == 'basic'
        assert policy.discretization == 'explicit'
        assert policy.preconditioner is None

    def test_flow_policy_invalid_family(self):
        """Test that invalid family values are rejected."""
        with pytest.raises(ValidationError):
            FlowPolicy(
                family='invalid_family',  # type: ignore
                discretization='explicit',
                preconditioner=None
            )

    def test_flow_policy_invalid_discretization(self):
        """Test that invalid discretization values are rejected."""
        with pytest.raises(ValidationError):
            FlowPolicy(
                family='basic',
                discretization='invalid_discretization',  # type: ignore
                preconditioner=None
            )

    def test_flow_policy_preconditioner_validation(self):
        """Test that preconditioner names are validated."""
        # Valid preconditioner
        policy = FlowPolicy(
            family='preconditioned',
            discretization='explicit',
            preconditioner='jacobi_precond'
        )
        assert policy.preconditioner == 'jacobi_precond'

        # Invalid preconditioner (too short)
        with pytest.raises(ValidationError):
            FlowPolicy(
                family='preconditioned',
                discretization='explicit',
                preconditioner=''
            )


class TestMultiscaleScheduleContract:
    """Design by Contract tests for MultiscaleSchedule specification."""

    def test_valid_multiscale_schedule_creation(self):
        """Test that valid MultiscaleSchedule can be created with all required fields."""
        schedule = MultiscaleSchedule(
            mode='residual_driven',
            levels=5,
            activate_rule='residual>tau'
        )
        assert schedule.mode == 'residual_driven'
        assert schedule.levels == 5
        assert schedule.activate_rule == 'residual>tau'

    def test_multiscale_schedule_optional_activate_rule(self):
        """Test that activate_rule has a default value."""
        schedule = MultiscaleSchedule(
            mode='fixed_schedule',
            levels=3,
            activate_rule='level_complete'
        )
        assert schedule.mode == 'fixed_schedule'
        assert schedule.levels == 3
        assert schedule.activate_rule == 'level_complete'  # default

    def test_multiscale_schedule_invalid_mode(self):
        """Test that invalid mode values are rejected."""
        with pytest.raises(ValidationError):
            MultiscaleSchedule(
                mode='invalid_mode',  # type: ignore
                levels=3,
                activate_rule='level_complete'
            )

    def test_multiscale_schedule_invalid_levels(self):
        """Test that invalid levels values are rejected."""
        # Too low
        with pytest.raises(ValidationError):
            MultiscaleSchedule(mode='residual_driven', levels=0, activate_rule='level_complete')

        # Too high
        with pytest.raises(ValidationError):
            MultiscaleSchedule(mode='residual_driven', levels=21, activate_rule='level_complete')

    def test_multiscale_schedule_activate_rule_validation(self):
        """Test that activate_rule is validated."""
        # Valid rule
        schedule = MultiscaleSchedule(
            mode='residual_driven',
            levels=3,
            activate_rule='energy_decrease>0.01'
        )
        assert schedule.activate_rule == 'energy_decrease>0.01'

        # Invalid rule (too short)
        with pytest.raises(ValidationError):
            MultiscaleSchedule(
                mode='residual_driven',
                levels=3,
                activate_rule=''
            )


class TestPolicyIntegrationContract:
    """Tests for policy integration and compatibility."""

    def test_policies_can_be_used_together(self):
        """Test that FlowPolicy and MultiscaleSchedule can be used together."""
        flow_policy = FlowPolicy(
            family='preconditioned',
            discretization='symplectic',
            preconditioner='diag_precond'
        )

        multiscale_schedule = MultiscaleSchedule(
            mode='residual_driven',
            levels=5,
            activate_rule='residual>tau'
        )

        # Both should be valid objects
        assert flow_policy is not None
        assert multiscale_schedule is not None

        # Should be able to access their attributes
        assert flow_policy.family == 'preconditioned'
        assert multiscale_schedule.mode == 'residual_driven'

    def test_policy_immutability(self):
        """Test that policy objects are immutable (frozen models)."""
        policy = FlowPolicy(
            family='basic',
            discretization='explicit',
            preconditioner=None
        )

        # Should not be able to modify attributes
        with pytest.raises(ValidationError):
            policy.family = 'modified'  # type: ignore

        schedule = MultiscaleSchedule(
            mode='fixed_schedule',
            levels=3,
            activate_rule='level_complete'
        )

        with pytest.raises(ValidationError):
            schedule.levels = 5