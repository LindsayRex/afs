import pyarrow.parquet as pq

from computable_flows_shim.telemetry import FlightRecorder
from computable_flows_shim.telemetry.manifest_writer import write_manifest

try:
    import toml

    HAS_TOML = True
except ImportError:
    HAS_TOML = False
    toml = None


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


def test_manifest_writer_basic_fields(tmp_path):
    """Test manifest writer with basic required fields."""
    out_dir = str(tmp_path)
    write_manifest(
        out_dir=out_dir,
        schema_version=3,
        flow_name="test_flow",
        run_id="test_run_001",
        dtype="float32",
    )

    manifest_path = tmp_path / "manifest.toml"
    assert manifest_path.exists()

    # Read and verify content
    with open(manifest_path, encoding="utf-8") as f:
        content = f.read()

    assert "schema_version = 3" in content
    assert 'flow_name = "test_flow"' in content
    assert 'run_id = "test_run_001"' in content
    assert 'dtype = "float32"' in content
    assert "invariants_present = false" in content
    assert "redact_artifacts = false" in content


def test_manifest_writer_complete_fields(tmp_path):
    """Test manifest writer with all optional fields."""
    if not HAS_TOML:
        import pytest

        pytest.skip("toml module not available")

    out_dir = str(tmp_path)
    write_manifest(
        out_dir=out_dir,
        schema_version=3,
        flow_name="test_flow",
        run_id="test_run_001",
        dtype="float64",
        lens_name="db4",
        unit_normalization_table={"term1": 1.5, "term2": 2.0},
        invariants_present=True,
        redact_artifacts=True,
        versions={"jax": "0.4.0", "pyarrow": "12.0.0"},
        shapes={"x": [100, 100], "y": [50]},
        frame_type="unitary",
        gates={"eta_dd_min": 0.1, "gamma_max": 0.01},
        budgets={"max_iter": 1000, "max_time_ms": 60000},
        seeds={"rng": 42, "tuner": 123},
        residual_details={"norm_type": "L2", "units": "normalized"},
        extra={"custom_field": "value"},
    )

    manifest_path = tmp_path / "manifest.toml"
    assert manifest_path.exists()

    # Parse TOML and verify all fields
    with open(manifest_path, encoding="utf-8") as f:
        manifest = toml.load(f)

    assert manifest["schema_version"] == 3
    assert manifest["flow_name"] == "test_flow"
    assert manifest["run_id"] == "test_run_001"
    assert manifest["dtype"] == "float64"
    assert manifest["lens_name"] == "db4"
    assert manifest["unit_normalization_table"] == {"term1": 1.5, "term2": 2.0}
    assert manifest["invariants_present"] is True
    assert manifest["redact_artifacts"] is True
    assert manifest["versions"] == {"jax": "0.4.0", "pyarrow": "12.0.0"}
    assert manifest["shapes"] == {"x": [100, 100], "y": [50]}
    assert manifest["frame_type"] == "unitary"
    assert manifest["gates"] == {"eta_dd_min": 0.1, "gamma_max": 0.01}
    assert manifest["budgets"] == {"max_iter": 1000, "max_time_ms": 60000}
    assert manifest["seeds"] == {"rng": 42, "tuner": 123}
    assert manifest["residual"] == {"norm_type": "L2", "units": "normalized"}
    assert manifest["custom_field"] == "value"


def test_manifest_writer_fallback_serializer(tmp_path, monkeypatch):
    """Test manifest writer fallback serializer when toml is not available."""
    # Mock toml as None to force fallback serializer
    monkeypatch.setattr("computable_flows_shim.telemetry.manifest_writer.toml", None)

    out_dir = str(tmp_path)
    write_manifest(
        out_dir=out_dir,
        schema_version=3,
        flow_name="test_flow",
        run_id="test_run_001",
        dtype="float32",
        lens_name="haar",
        invariants_present=True,
    )

    manifest_path = tmp_path / "manifest.toml"
    assert manifest_path.exists()

    # Read and verify fallback format
    with open(manifest_path, encoding="utf-8") as f:
        content = f.read()

    assert "schema_version = 3" in content
    assert 'flow_name = "test_flow"' in content
    assert 'run_id = "test_run_001"' in content
    assert 'dtype = "float32"' in content
    assert 'lens_name = "haar"' in content
    assert "invariants_present = true" in content
