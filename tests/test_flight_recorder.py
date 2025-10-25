import pyarrow.parquet as pq

from computable_flows_shim.telemetry import FlightRecorder


class DummyController:
    def __init__(self, recorder):
        self.recorder = recorder

    def run(self, n=3):
        for i in range(n):
            self.recorder.log(
                run_id="test_run_001",
                flow_name="dummy_flow",
                phase="GREEN",
                iter=i,
                t_wall_ms=10.0 * i,
                E=1.0 / (i + 1),
                grad_norm=0.1 * (i + 1),
                eta_dd=0.5 + 0.1 * i,
                gamma=0.01 * (i + 1),
                alpha=0.2,
                phi_residual=0.05 * i,
                invariant_drift_max=0.001 * i,
            )


def test_flight_recorder_logs_core_fields(tmp_path):
    telemetry_file = tmp_path / "telemetry.parquet"
    rec = FlightRecorder(str(telemetry_file))
    ctrl = DummyController(rec)
    ctrl.run(n=5)
    rec.flush()

    # Read the parquet file and check its contents
    table = pq.read_table(telemetry_file)
    df = table.to_pydict()

    # Check that all required fields are present and non-null
    required = [
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
    for col in required:
        assert col in df, f"Missing column: {col}"
        assert all(x is not None for x in df[col]), f"Nulls in required column: {col}"
    # Check row count
    assert len(df["iter"]) == 5
