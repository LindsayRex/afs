quota, yeah import duckdb
import os

class DuckDBManager:
    """
    Manages the DuckDB database for querying telemetry data across runs.
    """
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = duckdb.connect(database=self.db_path, read_only=False)

    def consolidate_runs(self, base_path: str):
        """
        Scans for Parquet files in the run directories and consolidates them into DuckDB tables.
        """
        # Create tables if they don't exist
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS telemetry (
                run_id VARCHAR,
                phase VARCHAR,
                iter INTEGER,
                trial_id VARCHAR,
                t_wall_ms DOUBLE,
                alpha DOUBLE,
                lambda DOUBLE,
                lambda_j VARCHAR,
                E DOUBLE,
                grad_norm DOUBLE,
                eta_dd DOUBLE,
                gamma DOUBLE,
                sparsity_wx DOUBLE,
                metric_ber DOUBLE,
                warnings VARCHAR,
                notes VARCHAR,
                invariant_drift_max DOUBLE,
                phi_residual DOUBLE,
                lens_name VARCHAR,
                level_active_max INTEGER,
                sparsity_mode VARCHAR,
                flow_family VARCHAR
            );
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                run_id VARCHAR,
                t_wall_ms DOUBLE,
                event VARCHAR,
                payload VARCHAR
            );
        """)

        # Find all telemetry.parquet and events.parquet files
        for root, _, files in os.walk(base_path):
            for file in files:
                if file == "telemetry.parquet":
                    path = os.path.join(root, file).replace('\\', '/')
                    self.conn.execute(f"INSERT INTO telemetry SELECT * FROM read_parquet('{path}');")
                elif file == "events.parquet":
                    path = os.path.join(root, file).replace('\\', '/')
                    self.conn.execute(f"INSERT INTO events SELECT * FROM read_parquet('{path}');")

    def close(self):
        """
        Closes the database connection.
        """
        self.conn.close()
