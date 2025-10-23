"""
Tests for the runtime engine and step execution.
"""
import sys
from pathlib import Path
import jax
import jax.numpy as jnp
import tempfile
import os
import pyarrow.parquet as pq
import pytest
# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from computable_flows_shim.energy.specs import EnergySpec, TermSpec, StateSpec
from computable_flows_shim.energy.compile import compile_energy
from computable_flows_shim.api import Op
from computable_flows_shim.controller import FlightController
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
            TermSpec(type='quadratic', op='I', weight=1.0, variable='x', target='y'),
            TermSpec(type='l1', op='I', weight=0.5, variable='x')
        ],
        state=StateSpec(shapes={'x': [1], 'y': [1]})
    )
    op_registry = {'I': IdentityOp()}
    compiled = compile_energy(spec, op_registry)

    # WHEN we run one full step of the flow from a known state
    initial_state = {'x': jnp.array([2.0], dtype=float_dtype), 'y': jnp.array([1.0], dtype=float_dtype)}
    step_alpha = 0.1

    # Use FlightController to run a single step
    controller = FlightController()
    final_state = controller.run_certified_flow(
        initial_state=initial_state,
        compiled=compiled,
        num_iterations=1,
        initial_alpha=step_alpha
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
    assert jnp.isclose(final_state['x'], expected_x, atol=tolerance)

@pytest.mark.dtype_parametrized
def test_run_flow_with_telemetry(float_dtype):
    """
    Tests that the runtime engine can execute a full flow and record telemetry.
    """
    # GIVEN a simple energy functional, a telemetry manager, and a temporary directory
    spec = EnergySpec(
        terms=[
            TermSpec(type='quadratic', op='I', weight=1.0, variable='x', target='y')
        ],
        state=StateSpec(shapes={'x': [1], 'y': [1]})
    )
    op_registry = {'I': IdentityOp()}
    compiled = compile_energy(spec, op_registry)
    initial_state = {'x': jnp.array([10.0], dtype=float_dtype), 'y': jnp.array([0.0], dtype=float_dtype)}
    
    with tempfile.TemporaryDirectory() as temp_dir:
        tm = TelemetryManager(base_path=temp_dir, flow_name="test_flow")
        
        # WHEN we run the flow
        controller = FlightController()
        final_state = controller.run_certified_flow(
            initial_state=initial_state,
            compiled=compiled,
            num_iterations=3,
            initial_alpha=0.1,
            telemetry_manager=tm
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
        cert_event = next((e for e in events if e['event'] == 'CERT_CHECK'), None)
        assert cert_event is not None
        assert 'eta_dd' in cert_event['payload']
        assert 'gamma' in cert_event['payload']

from computable_flows_shim.runtime.checkpoint import CheckpointManager
from computable_flows_shim.runtime.engine import resume_flow

@pytest.mark.dtype_parametrized
def test_checkpointing(float_dtype):
    """
    Tests checkpoint creation, listing, loading, and resuming functionality.
    """
    # GIVEN a simple energy functional and checkpoint manager
    spec = EnergySpec(
        terms=[
            TermSpec(type='quadratic', op='I', weight=1.0, variable='x', target='y')
        ],
        state=StateSpec(shapes={'x': [1], 'y': [1]})
    )
    op_registry = {'I': IdentityOp()}
    compiled = compile_energy(spec, op_registry)
    initial_state = {'x': jnp.array([10.0], dtype=float_dtype), 'y': jnp.array([0.0], dtype=float_dtype)}
    
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
                    "input_shape": initial_state['x'].shape,
                    "eta_dd": 0.1,
                    "gamma": 0.01
                }
                checkpoint_manager.create_checkpoint(
                    run_id=run_id,
                    iteration=i + 1,
                    state=state,
                    flow_config=flow_config,
                    certificates=certificates,
                    telemetry_history=None
                )
        
        # THEN checkpoints should be created
        checkpoints = checkpoint_manager.list_checkpoints()
        assert len(checkpoints) > 0
        
        # Get the latest checkpoint
        latest_checkpoint = max(checkpoints, key=lambda c: c['iteration'])
        assert latest_checkpoint['iteration'] == 9  # Checkpoints at iterations 3, 6, 9 for interval 3
        
        # WHEN we load the checkpoint
        checkpoint_data = checkpoint_manager.load_checkpoint(latest_checkpoint['checkpoint_id'])
        
        # THEN the checkpoint should contain the expected data
        assert 'state' in checkpoint_data
        assert 'iteration' in checkpoint_data
        assert checkpoint_data['iteration'] == 9
        assert 'run_id' in checkpoint_data
        assert 'flow_config' in checkpoint_data
        assert 'certificates' in checkpoint_data
        assert isinstance(checkpoint_data['state']['x'], jnp.ndarray)
        
        # WHEN we resume from the checkpoint for additional iterations
        resumed_state = resume_flow(
            checkpoint_id=latest_checkpoint['checkpoint_id'],
            checkpoint_manager=checkpoint_manager,
            compiled=compiled,
            remaining_iters=5,
            step_alpha=0.1,
            telemetry_manager=tm,
            checkpoint_interval=3
        )
        
        # THEN the flow should continue from where it left off
        # The resumed flow ran 5 more iterations, so total should be more optimized
        energy_initial = compiled.f_value(initial_state)
        energy_final = compiled.f_value(state)
        energy_resumed = compiled.f_value(resumed_state)
        
        # Energy should decrease with more iterations
        assert energy_resumed < energy_final < energy_initial
