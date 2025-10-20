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
# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from computable_flows_shim.energy.specs import EnergySpec, TermSpec, StateSpec
from computable_flows_shim.energy.compile import compile_energy
from computable_flows_shim.api import Op
from computable_flows_shim.runtime.engine import run_flow_step, run_flow
from computable_flows_shim.telemetry import TelemetryManager

class IdentityOp(Op):
    def __call__(self, x):
        return x

def test_forward_backward_step():
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
    initial_state = {'x': jnp.array([2.0]), 'y': jnp.array([1.0])}
    step_alpha = 0.1
    final_state = run_flow_step(initial_state, compiled, step_alpha)

    # THEN the final state should be the result of applying F_Dis, then F_Proj.
    # 1. Forward/Dissipative step (z = x - alpha * grad_f(x)):
    #    grad_f(x) = (x-y) = 2.0 - 1.0 = 1.0
    #    z = 2.0 - 0.1 * 1.0 = 1.9
    # 2. Backward/Projective step (x_new = prox_g(z)):
    #    The prox for g(x) = w*||x||_1 is soft-thresholding with threshold = alpha * w.
    #    threshold = 0.1 * 0.5 = 0.05
    #    x_new = sign(1.9) * max(|1.9| - 0.05, 0) = 1.85
    expected_x = 1.85
    assert jnp.isclose(final_state['x'], expected_x)

def test_run_flow_with_telemetry():
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
    initial_state = {'x': jnp.array([10.0]), 'y': jnp.array([0.0])}
    
    with tempfile.TemporaryDirectory() as temp_dir:
        tm = TelemetryManager(base_path=temp_dir, flow_name="test_flow")
        
        # WHEN we run the flow
        run_flow(
            initial_state=initial_state,
            compiled=compiled,
            num_iters=3,
            step_alpha=0.1,
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

def test_run_flow_step_multiscale():
    """
    Tests run_flow_step with multiscale transforms.
    """
    # GIVEN a simple energy functional and an identity transform
    spec = EnergySpec(
        terms=[
            TermSpec(type='quadratic', op='I', weight=1.0, variable='x', target='y'),
            TermSpec(type='l1', op='I', weight=0.5, variable='x')
        ],
        state=StateSpec(shapes={'x': [4], 'y': [4]})
    )
    op_registry = {'I': IdentityOp()}
    compiled = compile_energy(spec, op_registry)
    
    # Identity transform
    class IdentityTransform:
        def forward(self, x):
            return x
        def inverse(self, x):
            return x
    
    W = IdentityTransform()
    initial_state = {'x': jnp.array([2.0, -1.0, 0.5, -0.2]), 'y': jnp.array([1.0, 1.0, 1.0, 1.0])}
    step_alpha = 0.1
    
    # WHEN we run one step with multiscale
    final_state = run_flow_step(initial_state, compiled, step_alpha, W=W)
    
    # THEN the state should be updated (basic check)
    assert 'x' in final_state
    assert 'y' in final_state
    assert final_state['x'].shape == initial_state['x'].shape
