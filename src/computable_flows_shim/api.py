"""
Core API for the Computable Flows Shim.
"""
from typing import Any, Protocol

class Op(Protocol):
    """Protocol for a linear operator."""
    def __call__(self, x: Any) -> Any: ...

from computable_flows_shim.controller import FlightController
from computable_flows_shim.telemetry import FlightRecorder, write_manifest
import os

def run_certified_with_telemetry(
    initial_state,
    compiled,
    num_iterations,
    step_alpha,
    flow_name,
    run_id,
    out_dir,
    schema_version=3,
    residual_details=None,
    extra_manifest=None
):
    """
    Runs a certified flow and records telemetry, events, and manifest to out_dir.
    Returns (final_state, recorder)
    """
    import os
    os.makedirs(out_dir, exist_ok=True)
    telemetry_path = os.path.join(out_dir, "telemetry.parquet")
    events_path = os.path.join(out_dir, "events.parquet")
    recorder = FlightRecorder(path=telemetry_path, events_path=events_path)
    controller = FlightController()
    final_state = controller.run_certified_flow(
        initial_state=initial_state,
        compiled=compiled,
        num_iterations=num_iterations,
        initial_alpha=step_alpha,
        telemetry_manager=recorder,
        flow_name=flow_name,
        run_id=run_id
    )
    recorder.flush()
    # Write manifest
    if residual_details is None:
        residual_details = {"method": "unknown", "norm": "L2", "notes": "not provided"}
    write_manifest(
        out_dir=out_dir,
        schema_version=schema_version,
        flow_name=flow_name,
        run_id=run_id,
        residual_details=residual_details,
        extra=extra_manifest or {}
    )
    return final_state, recorder
