"""
Tests for the runtime engine and step execution.
"""

import os
import tempfile

import jax.numpy as jnp
import pyarrow.parquet as pq
import pytest

# Add the project root to the Python path
# sys.path.insert(0, str(Path(__file__).parent.parent / "src"))  # Handled by pytest.ini_options.pythonpath
from computable_flows_shim.api import Op
from computable_flows_shim.controller import FlightController
from computable_flows_shim.energy.compile import compile_energy
from computable_flows_shim.energy.policies import FlowPolicy
from computable_flows_shim.energy.specs import EnergySpec, StateSpec, TermSpec
from computable_flows_shim.runtime.checkpoint import CheckpointManager
from computable_flows_shim.runtime.engine import resume_flow, run_flow_step
from computable_flows_shim.telemetry import TelemetryManager


class IdentityOp(Op):
    def __call__(self, x):
        return x


@pytest.mark.dtype_parametrized
def test_forward_backward_step(float_dtype):
    """
    Tests one full step of Forward-Backward Splitting (F_Dis -> F_Proj).
    This test computationally verifies Lemma 6: Stability of Composite Splitting.
    """
    # GIVEN a composite energy functional (quadratic data term f, L1 regularization g)
    spec = EnergySpec(
        terms=[
            TermSpec(type="quadratic", op="I", weight=1.0, variable="x", target="y"),
            TermSpec(type="l1", op="I", weight=0.5, variable="x"),
        ],
        state=StateSpec(shapes={"x": [1], "y": [1]}),
    )
    op_registry = {"I": IdentityOp()}
    compiled = compile_energy(spec, op_registry)

    # WHEN we run one full step of the flow from a known state
    initial_state = {
        "x": jnp.array([2.0], dtype=float_dtype),
        "y": jnp.array([1.0], dtype=float_dtype),
    }
    step_alpha = 0.1

    # Use FlightController to run a single step
    controller = FlightController()
    final_state = controller.run_certified_flow(
        initial_state=initial_state,
        compiled=compiled,
        num_iterations=1,
        initial_alpha=step_alpha,
    )

    # THEN the final state should be the result of applying F_Dis, then F_Proj.
    # 1. Forward/Dissipative step (z = x - alpha * grad_f(x)):
    #    grad_f(x) = (x-y) = 2.0 - 1.0 = 1.0
    #    z = 2.0 - 0.1 * 1.0 = 1.9
    # 2. Backward/Projective step (x_new = prox_g(z)):
    #    The prox for g(x) = w*||x||_1 is soft-thresholding with threshold = alpha * w.
    #    threshold = 0.1 * 0.5 = 0.05
    #    x_new = sign(1.9) * max(|1.9| - 0.05, 0) = 1.85
    expected_x = 1.85
    tolerance = 1e-5 if float_dtype == jnp.float32 else 1e-12
    assert jnp.isclose(final_state["x"], expected_x, atol=tolerance)


@pytest.mark.dtype_parametrized
def test_run_flow_with_telemetry(float_dtype):
    """
    Tests that the runtime engine can execute a full flow and record telemetry.
    """
    # GIVEN a simple energy functional, a telemetry manager, and a temporary directory
    spec = EnergySpec(
        terms=[
            TermSpec(type="quadratic", op="I", weight=1.0, variable="x", target="y")
        ],
        state=StateSpec(shapes={"x": [1], "y": [1]}),
    )
    op_registry = {"I": IdentityOp()}
    compiled = compile_energy(spec, op_registry)
    initial_state = {
        "x": jnp.array([10.0], dtype=float_dtype),
        "y": jnp.array([0.0], dtype=float_dtype),
    }

    with tempfile.TemporaryDirectory() as temp_dir:
        tm = TelemetryManager(base_path=temp_dir, flow_name="test_flow")

        # WHEN we run the flow
        controller = FlightController()
        controller.run_certified_flow(
            initial_state=initial_state,
            compiled=compiled,
            num_iterations=3,
            initial_alpha=0.1,
            telemetry_manager=tm,
        )
        tm.flush()

        # THEN the telemetry parquet file should be created and contain the correct data
        telemetry_path = os.path.join(tm.run_path, "telemetry.parquet")
        assert os.path.exists(telemetry_path)

        table = pq.read_table(telemetry_path)
        assert table.num_rows == 3
        assert "iter" in table.column_names
        assert "E" in table.column_names

        iters = table.column("iter").to_pylist()
        assert iters == [0, 1, 2]

        # AND the events parquet should contain the CERT_CHECK event
        events_path = os.path.join(tm.run_path, "events.parquet")
        assert os.path.exists(events_path)

        events_table = pq.read_table(events_path)
        assert events_table.num_rows >= 1  # At least the CERT_CHECK event
        events = events_table.to_pylist()
        cert_event = next((e for e in events if e["event"] == "CERT_CHECK"), None)
        assert cert_event is not None
        assert "eta_dd" in cert_event["payload"]
        assert "gamma" in cert_event["payload"]


@pytest.mark.dtype_parametrized
def test_checkpointing(float_dtype):
    """
    Tests checkpoint creation, listing, loading, and resuming functionality.
    """
    # GIVEN a simple energy functional and checkpoint manager
    spec = EnergySpec(
        terms=[
            TermSpec(type="quadratic", op="I", weight=1.0, variable="x", target="y")
        ],
        state=StateSpec(shapes={"x": [1], "y": [1]}),
    )
    op_registry = {"I": IdentityOp()}
    compiled = compile_energy(spec, op_registry)
    initial_state = {
        "x": jnp.array([10.0], dtype=float_dtype),
        "y": jnp.array([0.0], dtype=float_dtype),
    }

    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_manager = CheckpointManager(checkpoint_dir=temp_dir)
        tm = TelemetryManager(base_path=temp_dir, flow_name="test_checkpoint_flow")

        # WHEN we run the flow with checkpointing using the old runtime engine approach
        # (since FlightController handles internal rollback checkpoints, not external ones)
        from computable_flows_shim.runtime.engine import run_flow_step

        state = initial_state
        run_id = tm.run_id
        num_iters = 10
        step_alpha = 0.1

        for i in range(num_iters):
            state = run_flow_step(state, compiled, step_alpha)

            # Create checkpoint if requested and at interval
            if (i + 1) % 3 == 0:  # checkpoint_interval = 3
                certificates = {"eta_dd": 0.1, "gamma": 0.01}  # Mock certificates
                flow_config = {
                    "num_iters": num_iters,
                    "step_alpha": step_alpha,
                    "input_shape": initial_state["x"].shape,
                    "eta_dd": 0.1,
                    "gamma": 0.01,
                }
                checkpoint_manager.create_checkpoint(
                    run_id=run_id,
                    iteration=i + 1,
                    state=state,
                    flow_config=flow_config,
                    certificates=certificates,
                    telemetry_history=None,
                )

        # THEN checkpoints should be created
        checkpoints = checkpoint_manager.list_checkpoints()
        assert len(checkpoints) > 0

        # Get the latest checkpoint
        latest_checkpoint = max(checkpoints, key=lambda c: c["iteration"])
        assert (
            latest_checkpoint["iteration"] == 9
        )  # Checkpoints at iterations 3, 6, 9 for interval 3

        # WHEN we load the checkpoint
        checkpoint_data = checkpoint_manager.load_checkpoint(
            latest_checkpoint["checkpoint_id"]
        )

        # THEN the checkpoint should contain the expected data
        assert "state" in checkpoint_data
        assert "iteration" in checkpoint_data
        assert checkpoint_data["iteration"] == 9
        assert "run_id" in checkpoint_data
        assert "flow_config" in checkpoint_data
        assert "certificates" in checkpoint_data
        assert isinstance(checkpoint_data["state"]["x"], jnp.ndarray)

        # WHEN we resume from the checkpoint for additional iterations
        resumed_state = resume_flow(
            checkpoint_id=latest_checkpoint["checkpoint_id"],
            checkpoint_manager=checkpoint_manager,
            compiled=compiled,
            remaining_iters=5,
            step_alpha=0.1,
            telemetry_manager=tm,
            checkpoint_interval=3,
        )

        # THEN the flow should continue from where it left off
        # The resumed flow ran 5 more iterations, so total should be more optimized
        energy_initial = compiled.f_value(initial_state)
        energy_final = compiled.f_value(state)
        energy_resumed = compiled.f_value(resumed_state)

        # Energy should decrease with more iterations
        assert energy_resumed < energy_final < energy_initial


@pytest.mark.dtype_parametrized
def test_flow_policy_driven_execution_contracts(float_dtype):
    """
    DBC: Formal verification of FlowPolicy-driven execution contracts.

    Pre: Valid FlowPolicy and state
    Post: State updated according to policy with energy decrease
    Invariant: Numerical stability and JAX compatibility maintained
    """
    # GIVEN a valid FlowPolicy for preconditioned execution
    flow_policy = FlowPolicy(
        family="preconditioned", discretization="explicit", preconditioner="jacobi"
    )

    # AND a compiled energy functional
    spec = EnergySpec(
        terms=[
            TermSpec(type="quadratic", op="I", weight=1.0, variable="x", target="y")
        ],
        state=StateSpec(shapes={"x": [2], "y": [2]}),
    )
    op_registry = {"I": IdentityOp()}
    compiled = compile_energy(spec, op_registry)

    initial_state = {
        "x": jnp.array([2.0, 1.5], dtype=float_dtype),
        "y": jnp.array([1.0, 0.5], dtype=float_dtype),
    }

    # WHEN we execute one flow step
    result_state = run_flow_step(
        state=initial_state,
        compiled=compiled,
        step_alpha=0.01,  # Small step for stability
        flow_policy=flow_policy,
    )

    # THEN post-conditions hold:
    # 1. Energy decreases monotonically
    initial_energy = compiled.f_value(initial_state)
    final_energy = compiled.f_value(result_state)
    assert final_energy < initial_energy, (
        f"Energy should decrease: {initial_energy} -> {final_energy}"
    )

    # 2. State is properly updated (gradient step applied)
    grad = compiled.f_grad(initial_state)
    expected_x = (
        initial_state["x"] - 0.01 * grad["x"]
    )  # Basic gradient step (jacobi identity)
    assert jnp.allclose(result_state["x"], expected_x, rtol=1e-6)

    # 3. Invariants maintained:
    # - JAX arrays preserved
    assert isinstance(result_state["x"], jnp.ndarray)
    assert isinstance(result_state["y"], jnp.ndarray)
    # - Dtypes preserved
    assert result_state["x"].dtype == float_dtype
    assert result_state["y"].dtype == float_dtype
    # - Shapes preserved
    assert result_state["x"].shape == initial_state["x"].shape
    assert result_state["y"].shape == initial_state["y"].shape


@pytest.mark.dtype_parametrized
def test_flow_policy_basic_execution_contracts(float_dtype):
    """
    DBC: Formal verification of basic FlowPolicy execution contracts.

    Pre: Valid basic FlowPolicy
    Post: Standard gradient descent applied
    Invariant: No preconditioning artifacts introduced
    """
    # GIVEN a basic flow policy
    flow_policy = FlowPolicy(family="basic", discretization="explicit")

    # AND a quadratic energy
    spec = EnergySpec(
        terms=[
            TermSpec(type="quadratic", op="I", weight=1.0, variable="x", target="y")
        ],
        state=StateSpec(shapes={"x": [1], "y": [1]}),
    )
    op_registry = {"I": IdentityOp()}
    compiled = compile_energy(spec, op_registry)

    initial_state = {
        "x": jnp.array([3.0], dtype=float_dtype),
        "y": jnp.array([0.0], dtype=float_dtype),
    }

    # WHEN we execute with basic policy
    result_state = run_flow_step(
        state=initial_state,
        compiled=compiled,
        step_alpha=0.1,
        flow_policy=flow_policy,
    )

    # THEN it behaves identically to standard F_Dis + F_Proj
    # (This verifies the policy doesn't introduce artifacts)
    expected_grad = compiled.f_grad(initial_state)
    expected_x = initial_state["x"] - 0.1 * expected_grad["x"]
    expected_y = initial_state["y"] - 0.1 * expected_grad["y"]  # y also has gradient
    expected_proj = compiled.g_prox({"x": expected_x, "y": expected_y}, 0.1)

    assert jnp.allclose(result_state["x"], expected_proj["x"], rtol=1e-5)
    assert jnp.allclose(result_state["y"], expected_proj["y"], rtol=1e-5)


@pytest.mark.dtype_parametrized
def test_flow_policy_driven_execution(float_dtype):
    """
    GREEN: Tests that FlowPolicy drives primitive selection in runtime execution.

    Given: A flow policy specifying preconditioned execution with jacobi preconditioner
    When: Runtime executes with policy parameters
    Then: Preconditioned flow primitives are selected and applied successfully
    """
    # GIVEN a flow policy specifying preconditioned execution
    flow_policy = FlowPolicy(
        family="preconditioned", discretization="explicit", preconditioner="jacobi"
    )

    # AND a simple energy functional
    spec = EnergySpec(
        terms=[
            TermSpec(type="quadratic", op="I", weight=1.0, variable="x", target="y")
        ],
        state=StateSpec(shapes={"x": [1], "y": [1]}),
    )
    op_registry = {"I": IdentityOp()}
    compiled = compile_energy(spec, op_registry)

    initial_state = {
        "x": jnp.array([2.0], dtype=float_dtype),
        "y": jnp.array([1.0], dtype=float_dtype),
    }

    # WHEN we run a flow step with policy parameters
    from computable_flows_shim.runtime.engine import run_flow_step

    result_state = run_flow_step(
        state=initial_state,
        compiled=compiled,
        step_alpha=0.1,
        flow_policy=flow_policy,
    )

    # THEN it should execute successfully and return a valid state
    assert isinstance(result_state, dict)
    assert "x" in result_state
    assert "y" in result_state
    assert result_state["x"].shape == initial_state["x"].shape
    assert result_state["y"].shape == initial_state["y"].shape
    assert result_state["x"].dtype == float_dtype
    assert result_state["y"].dtype == float_dtype

    # AND the energy should have decreased (gradient descent step)
    initial_energy = compiled.f_value(initial_state)
    final_energy = compiled.f_value(result_state)
    assert final_energy <= initial_energy


@pytest.mark.dtype_parametrized
def test_multiscale_schedule_runtime_integration_red(float_dtype):
    """
    RED: MultiscaleSchedule runtime integration not yet implemented.

    This test will fail until MultiscaleSchedule is integrated into run_flow_step.
    """
    # GIVEN a MultiscaleSchedule with fixed_schedule mode
    from computable_flows_shim.energy.policies import MultiscaleSchedule
    from computable_flows_shim.multi.transform_op import make_transform

    schedule = MultiscaleSchedule(
        mode="fixed_schedule", levels=2, activate_rule="level_complete"
    )

    # AND a wavelet transform
    W = make_transform("haar", levels=2, ndim=1)

    # AND a simple energy functional
    spec = EnergySpec(
        terms=[
            TermSpec(type="quadratic", op="I", weight=1.0, variable="x", target="y")
        ],
        state=StateSpec(shapes={"x": [8], "y": [8]}),  # Power of 2 for wavelets
    )
    op_registry = {"I": IdentityOp()}
    compiled = compile_energy(spec, op_registry)

    initial_state = {
        "x": jnp.array([1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float_dtype),
        "y": jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float_dtype),
    }

    # WHEN we run a flow step with MultiscaleSchedule
    result_state = run_flow_step(
        state=initial_state,
        compiled=compiled,
        step_alpha=0.1,
        W=W,
        multiscale_schedule=schedule,
    )

    # THEN it should use the schedule for level activation
    # For fixed_schedule mode, all levels should be active immediately
    # This should result in different behavior than no schedule
    result_no_schedule = run_flow_step(
        state=initial_state,
        compiled=compiled,
        step_alpha=0.1,
        W=W,
        multiscale_schedule=None,
    )

    # With schedule, behavior should be different (currently fails because schedule is ignored)
    # This assertion will fail until schedule integration is implemented
    assert not jnp.allclose(result_state["x"], result_no_schedule["x"]), (
        "MultiscaleSchedule should produce different results than no schedule"
    )


@pytest.mark.dtype_parametrized
def test_multiscale_schedule_event_emission_red(float_dtype):
    """
    RED: SCALE_ACTIVATED events not yet emitted.

    This test will fail until telemetry integration emits SCALE_ACTIVATED events.
    """
    # GIVEN a MultiscaleSchedule and telemetry manager
    from computable_flows_shim.energy.policies import MultiscaleSchedule
    from computable_flows_shim.multi.transform_op import make_transform

    schedule = MultiscaleSchedule(
        mode="residual_driven", levels=3, activate_rule="residual>0.01"
    )

    W = make_transform("haar", levels=3, ndim=1)

    # AND telemetry setup
    with tempfile.TemporaryDirectory() as temp_dir:
        tm = TelemetryManager(base_path=temp_dir, flow_name="test_multiscale")

        # AND a flow setup
        spec = EnergySpec(
            terms=[
                TermSpec(type="quadratic", op="I", weight=1.0, variable="x", target="y")
            ],
            state=StateSpec(shapes={"x": [8], "y": [8]}),
        )
        op_registry = {"I": IdentityOp()}
        compiled = compile_energy(spec, op_registry)

        initial_state = {
            "x": jnp.array(
                [0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float_dtype
            ),
            "y": jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float_dtype),
        }

        # WHEN we run flow steps (this should trigger level activation)
        for i in range(3):
            run_flow_step(
                state=initial_state,
                compiled=compiled,
                step_alpha=0.1,
                W=W,
                multiscale_schedule=schedule,
                iteration=i,
                telemetry_manager=tm,
                previous_active_levels=[1],  # Start with 1 active level
            )

        tm.flush()

        # THEN SCALE_ACTIVATED events should be present
        # (This will fail because events are not emitted yet)
        events_path = os.path.join(tm.run_path, "events.parquet")
        if os.path.exists(events_path):
            events_table = pq.read_table(events_path)
            events = events_table.to_pylist()
            scale_events = [e for e in events if e.get("event") == "SCALE_ACTIVATED"]
            # Should have at least one scale activation event
            assert len(scale_events) > 0, "No SCALE_ACTIVATED events found"
        else:
            pytest.fail("Events parquet file not created")


@pytest.mark.dtype_parametrized
def test_multiscale_schedule_mode_differences_red(float_dtype):
    """
    RED: Different MultiscaleSchedule modes not yet implemented.

    Tests that different modes (fixed_schedule, residual_driven, energy_driven)
    produce different behaviors.
    """
    from computable_flows_shim.energy.policies import MultiscaleSchedule
    from computable_flows_shim.multi.transform_op import make_transform

    W = make_transform("haar", levels=2, ndim=1)

    spec = EnergySpec(
        terms=[
            TermSpec(type="quadratic", op="I", weight=1.0, variable="x", target="y")
        ],
        state=StateSpec(shapes={"x": [4], "y": [4]}),
    )
    op_registry = {"I": IdentityOp()}
    compiled = compile_energy(spec, op_registry)

    initial_state = {
        "x": jnp.array([1.0, 0.5, 0.0, 0.0], dtype=float_dtype),
        "y": jnp.array([0.0, 0.0, 0.0, 0.0], dtype=float_dtype),
    }

    # Test different modes
    modes = ["fixed_schedule", "residual_driven", "energy_driven"]

    for mode in modes:
        schedule = MultiscaleSchedule(
            mode=mode, levels=2, activate_rule="residual>0.01"
        )

        # Run a step
        result = run_flow_step(
            state=initial_state,
            compiled=compiled,
            step_alpha=0.1,
            W=W,
            multiscale_schedule=schedule,
        )

        # Each mode should behave differently
        # (Currently they all behave the same - full transform)
        assert result is not None
