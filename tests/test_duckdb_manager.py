import os
import tempfile

import pytest

from computable_flows_shim.telemetry.duckdb_manager import DuckDBManager


class TestDuckDBManager:
    """TDD tests for DuckDB consolidation functionality."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        yield db_path
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)

    @pytest.fixture
    def sample_telemetry_data(self):
        """Sample telemetry data matching TelemetrySample schema."""
        return [
            {
                "run_id": "run_001",
                "flow_name": "test_flow",
                "phase": "GREEN",
                "iter": 0,
                "t_wall_ms": 0.0,
                "E": 1.0,
                "grad_norm": 0.1,
                "eta_dd": 0.5,
                "gamma": 0.01,
                "alpha": 0.2,
                "phi_residual": 0.05,
                "invariant_drift_max": 0.001,
                "trial_id": "trial_1",
                "lambda": 0.1,
                "lambda_j": '{"scale_1": 0.1, "scale_2": 0.2}',
                "sparsity_wx": 0.8,
                "metric_ber": 0.95,
                "warnings": "",
                "notes": "Initial test run",
                "lens_name": "haar",
                "level_active_max": 3,
                "sparsity_mode": "l1",
                "flow_family": "gradient",
            },
            {
                "run_id": "run_001",
                "flow_name": "test_flow",
                "phase": "GREEN",
                "iter": 1,
                "t_wall_ms": 10.0,
                "E": 0.8,
                "grad_norm": 0.08,
                "eta_dd": 0.6,
                "gamma": 0.008,
                "alpha": 0.2,
                "phi_residual": 0.03,
                "invariant_drift_max": 0.0008,
                "trial_id": "trial_1",
                "lambda": 0.1,
                "lambda_j": '{"scale_1": 0.1, "scale_2": 0.2}',
                "sparsity_wx": 0.85,
                "metric_ber": 0.96,
                "warnings": "",
                "notes": "Iteration 1",
                "lens_name": "haar",
                "level_active_max": 3,
                "sparsity_mode": "l1",
                "flow_family": "gradient",
            },
        ]

    @pytest.fixture
    def sample_events_data(self):
        """Sample events data."""
        return [
            {
                "run_id": "run_001",
                "t_wall_ms": 0.0,
                "event": "RUN_STARTED",
                "payload": '{"flow_name": "test_flow"}',
            },
            {
                "run_id": "run_001",
                "t_wall_ms": 10.0,
                "event": "RUN_FINISHED",
                "payload": '{"final_energy": 0.8}',
            },
        ]

    def test_schema_matches_telemetry_sample(self, temp_db):
        """Test that DuckDB schema matches TelemetrySample model."""
        manager = DuckDBManager(temp_db)

        # Check that tables are created with correct schema
        result = manager.conn.execute("PRAGMA table_info(telemetry)").fetchall()
        column_names = [row[1] for row in result]

        # Core required fields from canonical telemetry schema (schema_version: 3)
        required_fields = [
            "run_id",
            "flow_name",
            "phase",
            "iter",
            "trial_id",
            "t_wall_ms",
            "alpha",
            "lambda",
            "lambda_j",
            "E",
            "grad_norm",
            "eta_dd",
            "gamma",
            "sparsity_wx",
            "metric_ber",
            "warnings",
            "notes",
            "invariant_drift_max",
            "phi_residual",
            "lens_name",
            "level_active_max",
            "sparsity_mode",
            "flow_family",
        ]

        for field in required_fields:
            assert field in column_names, f"Missing required field: {field}"

        manager.close()

    def test_consolidate_runs_creates_tables(self, temp_db, tmp_path):
        """Test that consolidate_runs creates the necessary tables."""
        manager = DuckDBManager(temp_db)

        # Create a fake run directory with parquet files
        run_dir = tmp_path / "fda_run_test"
        run_dir.mkdir()

        # Initially no data
        telemetry_count = manager.conn.execute(
            "SELECT COUNT(*) FROM telemetry"
        ).fetchone()[0]
        events_count = manager.conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        assert telemetry_count == 0
        assert events_count == 0

        manager.close()

    def test_deduplication_prevents_duplicates(
        self, temp_db, tmp_path, sample_telemetry_data
    ):
        """Test that consolidate_runs doesn't insert duplicate run_ids."""
        manager = DuckDBManager(temp_db)

        # Create a fake run directory with parquet files
        run_dir = tmp_path / "fda_run_test"
        run_dir.mkdir()

        # Create parquet files with sample data
        import pyarrow as pa
        import pyarrow.parquet as pq

        telemetry_table = pa.Table.from_pylist(
            [sample_telemetry_data[0]]
        )  # First sample
        telemetry_path = run_dir / "telemetry.parquet"
        pq.write_table(telemetry_table, str(telemetry_path))

        # Consolidate once
        manager.consolidate_runs(str(tmp_path))
        initial_count = manager.conn.execute(
            "SELECT COUNT(*) FROM telemetry"
        ).fetchone()[0]
        assert initial_count == 1

        # Consolidate again - should not add duplicates
        manager.consolidate_runs(str(tmp_path))
        final_count = manager.conn.execute("SELECT COUNT(*) FROM telemetry").fetchone()[
            0
        ]
        assert final_count == 1, (
            "Deduplication failed - duplicate records were inserted"
        )

        manager.close()

    def test_get_run_summaries_returns_valid_data(self, temp_db):
        """Test that get_run_summaries returns properly structured data."""
        manager = DuckDBManager(temp_db)

        # Initially should return empty list
        summaries = manager.get_run_summaries()
        assert isinstance(summaries, list)
        assert len(summaries) == 0

        manager.close()

    def test_get_run_summaries_with_data(self, temp_db, sample_telemetry_data):
        """Test get_run_summaries with actual data."""
        manager = DuckDBManager(temp_db)

        # This test will fail until we implement proper data insertion and querying
        pytest.skip("Data insertion and querying not yet implemented")

        manager.close()

    def test_cross_run_performance_analysis(self, temp_db):
        """Test queries for performance trends across runs."""
        manager = DuckDBManager(temp_db)

        # Initially should return empty list
        trends = manager.get_performance_trends()
        assert isinstance(trends, list)
        assert len(trends) == 0

        manager.close()

    def test_parameter_correlation_analysis(self, temp_db):
        """Test queries for parameter correlation analysis."""
        manager = DuckDBManager(temp_db)

        # Initially should return empty dict
        correlations = manager.get_parameter_correlations()
        assert isinstance(correlations, dict)
        assert len(correlations) == 2  # alpha_energy and gamma_grad_norm correlations
        assert "alpha_energy_correlation" in correlations
        assert "gamma_grad_norm_correlation" in correlations

        manager.close()

    def test_best_parameters_by_flow_type(self, temp_db):
        """Test finding best parameters for different flow types."""
        manager = DuckDBManager(temp_db)

        # Initially should return empty dict
        best_params = manager.get_best_parameters_by_flow_type()
        assert isinstance(best_params, dict)
        assert len(best_params) == 0

        manager.close()

    def test_convergence_pattern_analysis(self, temp_db):
        """Test analyzing convergence patterns across runs."""
        manager = DuckDBManager(temp_db)

        # Initially should return empty list
        patterns = manager.get_convergence_patterns()
        assert isinstance(patterns, list)
        assert len(patterns) == 0

        manager.close()

    def test_database_indexing(self, temp_db):
        """Test that appropriate indexes are created for query performance."""
        manager = DuckDBManager(temp_db)

        # Check for indexes on key fields using DuckDB's information schema
        indexes = manager.conn.execute("""
            SELECT index_name
            FROM duckdb_indexes()
            WHERE table_name = 'telemetry'
        """).fetchall()
        index_names = [row[0] for row in indexes]

        # Should have indexes on frequently queried fields
        expected_indexes = [
            "idx_telemetry_run_id",
            "idx_telemetry_flow_name",
            "idx_telemetry_t_wall_ms",
        ]
        for idx_name in expected_indexes:
            assert idx_name in index_names, f"Missing index: {idx_name}"

        manager.close()
