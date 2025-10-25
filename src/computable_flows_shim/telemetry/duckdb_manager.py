import os

import duckdb


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
                    path = os.path.join(root, file).replace("\\", "/")
                    self.conn.execute(
                        f"INSERT INTO telemetry SELECT * FROM read_parquet('{path}');"
                    )
                elif file == "events.parquet":
                    path = os.path.join(root, file).replace("\\", "/")
                    self.conn.execute(
                        f"INSERT INTO events SELECT * FROM read_parquet('{path}');"
                    )

    def get_run_summaries(self, flow_name: str | None = None, limit: int = 100) -> list:
        """
        Get summary statistics for historical runs.

        Returns list of dicts with keys: run_id, alpha, avg_remediations, final_energy, iterations
        """
        # Check if tables exist
        try:
            self.conn.execute("SELECT 1 FROM telemetry LIMIT 1")
        except (RuntimeError, ValueError):
            return []  # No data available

        query = """
        SELECT
            t.run_id,
            AVG(t.alpha) as alpha,
            COUNT(CASE WHEN e.event = 'STEP_REMEDIATION' THEN 1 END) as num_remediations
        FROM telemetry t
        LEFT JOIN events e ON t.run_id = e.run_id
        """

        if flow_name:
            query += f" WHERE t.flow_name = '{flow_name}'"

        query += """
        GROUP BY t.run_id
        ORDER BY t.run_id DESC
        """

        if limit:
            query += f" LIMIT {limit}"

        result = self.conn.execute(query).fetchall()

        summaries = [
            {
                "run_id": row[0],
                "alpha": float(row[1]) if row[1] else 0.1,
                "num_remediations": int(row[2]) if row[2] else 0,
                "final_energy": 0.0,  # Placeholder
                "iterations": 0,  # Placeholder
            }
            for row in result
        ]

        return summaries

    def close(self):
        """
        Closes the database connection.
        """
        self.conn.close()
