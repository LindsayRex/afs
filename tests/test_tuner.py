"""
Tests for the Tuner module.
"""

import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from computable_flows_shim.telemetry.duckdb_manager import DuckDBManager
from computable_flows_shim.tuner import suggest_parameters


def test_tuner_suggests_lower_alpha_on_high_remediation(tmp_path):
    """
    Tests that the tuner suggests lower alpha when historical runs show high remediation frequency.
    This enforces the contract: tuner should optimize parameters based on observed performance.
    """
    # Create a test database
    db_path = str(tmp_path / "test.db")
    db_manager = DuckDBManager(db_path)

    # Insert test data directly
    db_manager.conn.execute("""
        CREATE TABLE telemetry (
            run_id VARCHAR,
            alpha DOUBLE,
            flow_name VARCHAR
        );
    """)
    db_manager.conn.execute("""
        CREATE TABLE events (
            run_id VARCHAR,
            event VARCHAR
        );
    """)

    # Insert test runs
    test_data = [
        ("run1", 0.1, "STEP_REMEDIATION"),
        ("run1", 0.1, "STEP_REMEDIATION"),
        ("run1", 0.1, "STEP_REMEDIATION"),
        ("run1", 0.1, "STEP_REMEDIATION"),
        ("run1", 0.1, "STEP_REMEDIATION"),
        ("run2", 0.1, "STEP_REMEDIATION"),
        ("run2", 0.1, "STEP_REMEDIATION"),
        ("run2", 0.1, "STEP_REMEDIATION"),
        ("run2", 0.1, "STEP_REMEDIATION"),
        ("run2", 0.1, "STEP_REMEDIATION"),
        ("run2", 0.1, "STEP_REMEDIATION"),
        ("run2", 0.1, "STEP_REMEDIATION"),
        ("run3", 0.05, "STEP_REMEDIATION"),
    ]

    for run_id, alpha, event in test_data:
        db_manager.conn.execute(
            "INSERT INTO telemetry VALUES (?, ?, 'test_flow')", [run_id, alpha]
        )
        if event:
            db_manager.conn.execute("INSERT INTO events VALUES (?, ?)", [run_id, event])

    # WHEN we ask the tuner for parameter suggestions
    suggestions = suggest_parameters(db_manager, flow_name="test_flow")

    # THEN it should suggest a lower alpha to reduce remediations
    assert "alpha" in suggestions
    assert suggestions["alpha"] < 0.1  # Lower than the problematic value
    assert suggestions["alpha"] > 0.01  # But not ridiculously low

    db_manager.close()


def test_tuner_handles_empty_database(tmp_path):
    """
    Tests that the tuner provides a reasonable default when no historical data is available.
    """
    # Create an empty database
    db_path = str(tmp_path / "empty.db")
    db_manager = DuckDBManager(db_path)

    # WHEN we ask for suggestions
    suggestions = suggest_parameters(db_manager)

    # THEN it should return a default alpha
    assert "alpha" in suggestions
    assert suggestions["alpha"] == 0.1

    db_manager.close()


def test_tuner_suggests_best_alpha_from_history(tmp_path):
    """
    Tests that the tuner selects the alpha with the best performance from history.
    """
    # Create a test database
    db_path = str(tmp_path / "best.db")
    db_manager = DuckDBManager(db_path)

    # Setup tables
    db_manager.conn.execute("""
        CREATE TABLE telemetry (
            run_id VARCHAR,
            alpha DOUBLE,
            flow_name VARCHAR
        );
    """)
    db_manager.conn.execute("""
        CREATE TABLE events (
            run_id VARCHAR,
            event VARCHAR
        );
    """)

    # Insert runs with different alphas, where 0.01 performs best (no remediations)
    test_data = [
        ("run1", 0.2, "STEP_REMEDIATION"),  # High remediation
        ("run1", 0.2, "STEP_REMEDIATION"),
        ("run2", 0.1, "STEP_REMEDIATION"),  # Medium remediation
        ("run2", 0.1, "STEP_REMEDIATION"),
        ("run3", 0.05, "STEP_REMEDIATION"),  # Low remediation
        ("run4", 0.01, None),  # No remediation - best
    ]

    for run_id, alpha, event in test_data:
        db_manager.conn.execute(
            "INSERT INTO telemetry VALUES (?, ?, 'test_flow')", [run_id, alpha]
        )
        if event:
            db_manager.conn.execute("INSERT INTO events VALUES (?, ?)", [run_id, event])

    # WHEN we ask for suggestions
    suggestions = suggest_parameters(db_manager, flow_name="test_flow")

    # THEN it should suggest the best performing alpha (0.01 in this case)
    assert suggestions["alpha"] == 0.01

    db_manager.close()
