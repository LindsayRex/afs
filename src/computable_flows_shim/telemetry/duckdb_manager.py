import os

import duckdb


class DuckDBManager:
    """
    Manages the DuckDB database for querying telemetry data across runs.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        # Ensure the database file doesn't exist as an invalid file
        if os.path.exists(self.db_path):
            # Check if it's a valid DuckDB file by trying to read it
            try:
                test_conn = duckdb.connect(database=self.db_path, read_only=True)
                test_conn.close()
            except (duckdb.IOException, RuntimeError):
                # File exists but is not valid, remove it
                os.remove(self.db_path)

        self.conn = duckdb.connect(database=self.db_path, read_only=False)
        self._initialize_schema()

    def _initialize_schema(self):
        """Initialize the database schema with proper tables and indexes."""
        # Create telemetry table with complete schema matching TelemetrySample
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS telemetry (
                run_id VARCHAR,
                flow_name VARCHAR,
                phase VARCHAR,
                iter INTEGER,
                trial_id VARCHAR,
                t_wall_ms DOUBLE,
                alpha DOUBLE,
                "lambda" DOUBLE,
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

        # Create events table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                run_id VARCHAR,
                t_wall_ms DOUBLE,
                event VARCHAR,
                payload VARCHAR
            );
        """)

        # Create indexes for query performance
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_telemetry_run_id ON telemetry(run_id);"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_telemetry_flow_name ON telemetry(flow_name);"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_telemetry_t_wall_ms ON telemetry(t_wall_ms);"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_run_id ON events(run_id);"
        )

    def consolidate_runs(self, base_path: str):
        """
        Scans for Parquet files in the run directories and consolidates them into DuckDB tables.
        Implements deduplication to prevent duplicate run_ids from being inserted.
        """
        # Find all telemetry.parquet and events.parquet files
        for root, _, files in os.walk(base_path):
            for file in files:
                if file == "telemetry.parquet":
                    path = os.path.join(root, file).replace("\\", "/")
                    # Insert only if run_id doesn't already exist
                    self.conn.execute(f"""
                        INSERT INTO telemetry (
                            run_id, flow_name, phase, iter, trial_id, t_wall_ms,
                            alpha, "lambda", lambda_j, E, grad_norm, eta_dd, gamma,
                            sparsity_wx, metric_ber, warnings, notes,
                            invariant_drift_max, phi_residual, lens_name,
                            level_active_max, sparsity_mode, flow_family
                        )
                        SELECT
                            run_id, flow_name, phase, iter, trial_id, t_wall_ms,
                            alpha, "lambda", lambda_j, E, grad_norm, eta_dd, gamma,
                            sparsity_wx, metric_ber, warnings, notes,
                            invariant_drift_max, phi_residual, lens_name,
                            level_active_max, sparsity_mode, flow_family
                        FROM read_parquet('{path}')
                        WHERE run_id NOT IN (SELECT DISTINCT run_id FROM telemetry);
                    """)
                elif file == "events.parquet":
                    path = os.path.join(root, file).replace("\\", "/")
                    # Insert only if run_id doesn't already exist
                    self.conn.execute(f"""
                        INSERT INTO events
                        SELECT * FROM read_parquet('{path}')
                        WHERE run_id NOT IN (SELECT DISTINCT run_id FROM events);
                    """)

    def get_run_summaries(self, flow_name: str | None = None, limit: int = 100) -> list:
        """
        Get summary statistics for historical runs.

        Returns list of dicts with keys: run_id, flow_name, final_energy, iterations, duration_ms, convergence_rate
        """
        # Check if tables exist and have data
        try:
            result = self.conn.execute("SELECT COUNT(*) FROM telemetry").fetchone()
            if result[0] == 0:
                return []
        except (RuntimeError, ValueError):
            return []  # No data available

        query = """
        SELECT
            t.run_id,
            t.flow_name,
            MIN(t.E) as final_energy,
            MAX(t.iter) as iterations,
            MAX(t.t_wall_ms) - MIN(t.t_wall_ms) as duration_ms,
            AVG(t.grad_norm) as avg_grad_norm,
            COUNT(CASE WHEN t.phase = 'GREEN' THEN 1 END) as green_iterations
        FROM telemetry t
        """

        if flow_name:
            query += f" WHERE t.flow_name = '{flow_name}'"

        query += """
        GROUP BY t.run_id, t.flow_name
        ORDER BY t.run_id DESC
        """

        if limit:
            query += f" LIMIT {limit}"

        result = self.conn.execute(query).fetchall()

        summaries = []
        for row in result:
            total_iterations = row[3] if row[3] else 0
            green_iterations = row[6] if row[6] else 0
            convergence_rate = (
                green_iterations / total_iterations if total_iterations > 0 else 0.0
            )

            summaries.append(
                {
                    "run_id": row[0],
                    "flow_name": row[1],
                    "final_energy": float(row[2]) if row[2] else 0.0,
                    "iterations": int(total_iterations),
                    "duration_ms": float(row[4]) if row[4] else 0.0,
                    "avg_grad_norm": float(row[5]) if row[5] else 0.0,
                    "convergence_rate": convergence_rate,
                }
            )

        return summaries

    def get_performance_trends(
        self, flow_name: str | None = None, limit: int = 50
    ) -> list:
        """
        Analyze performance trends across runs.

        Returns list of dicts with convergence metrics over time.
        """
        try:
            result = self.conn.execute("SELECT COUNT(*) FROM telemetry").fetchone()
            if result[0] == 0:
                return []
        except (RuntimeError, ValueError):
            return []

        query = """
        SELECT
            t.run_id,
            t.flow_name,
            AVG(t.E) as avg_energy,
            MIN(t.E) as min_energy,
            MAX(t.iter) as max_iterations,
            AVG(t.grad_norm) as avg_grad_norm,
            COUNT(*) as total_samples
        FROM telemetry t
        """

        if flow_name:
            query += f" WHERE t.flow_name = '{flow_name}'"

        query += """
        GROUP BY t.run_id, t.flow_name
        ORDER BY t.run_id DESC
        """

        if limit:
            query += f" LIMIT {limit}"

        result = self.conn.execute(query).fetchall()

        trends = [
            {
                "run_id": row[0],
                "flow_name": row[1],
                "avg_energy": float(row[2]) if row[2] else 0.0,
                "min_energy": float(row[3]) if row[3] else 0.0,
                "max_iterations": int(row[4]) if row[4] else 0,
                "avg_grad_norm": float(row[5]) if row[5] else 0.0,
                "total_samples": int(row[6]) if row[6] else 0,
            }
            for row in result
        ]

        return trends

    def get_parameter_correlations(self, flow_name: str | None = None) -> dict:
        """
        Analyze correlations between parameters and performance metrics.
        """
        try:
            result = self.conn.execute("SELECT COUNT(*) FROM telemetry").fetchone()
            if result[0] == 0:
                return {
                    "alpha_energy_correlation": 0.0,
                    "gamma_grad_norm_correlation": 0.0,
                }
        except (RuntimeError, ValueError):
            return {
                "alpha_energy_correlation": 0.0,
                "gamma_grad_norm_correlation": 0.0,
            }

        where_clause = f"WHERE flow_name = '{flow_name}'" if flow_name else ""

        # Correlation between alpha and final energy
        alpha_energy_corr = self.conn.execute(f"""
            SELECT CORR(alpha, E) as correlation
            FROM telemetry
            {where_clause}
        """).fetchone()[0]

        # Correlation between gamma and convergence speed (grad_norm)
        gamma_grad_corr = self.conn.execute(f"""
            SELECT CORR(gamma, grad_norm) as correlation
            FROM telemetry
            {where_clause}
        """).fetchone()[0]

        return {
            "alpha_energy_correlation": float(alpha_energy_corr)
            if alpha_energy_corr
            else 0.0,
            "gamma_grad_norm_correlation": float(gamma_grad_corr)
            if gamma_grad_corr
            else 0.0,
        }

    def get_best_parameters_by_flow_type(self, metric: str = "min_energy") -> dict:
        """
        Find best parameters for different flow types based on specified metric.
        """
        try:
            result = self.conn.execute("SELECT COUNT(*) FROM telemetry").fetchone()
            if result[0] == 0:
                return {}
        except (RuntimeError, ValueError):
            return {}

        # Group by flow_name and find best parameters
        query = """
        SELECT
            flow_name,
            AVG(alpha) as best_alpha,
            AVG(gamma) as best_gamma,
            AVG(eta_dd) as best_eta_dd,
            MIN(E) as min_energy,
            AVG(grad_norm) as avg_grad_norm
        FROM telemetry
        GROUP BY flow_name
        ORDER BY min_energy ASC
        """

        result = self.conn.execute(query).fetchall()

        best_params = {}
        for row in result:
            best_params[row[0]] = {
                "alpha": float(row[1]) if row[1] else 0.1,
                "gamma": float(row[2]) if row[2] else 0.01,
                "eta_dd": float(row[3]) if row[3] else 0.5,
                "min_energy": float(row[4]) if row[4] else 0.0,
                "avg_grad_norm": float(row[5]) if row[5] else 0.0,
            }

        return best_params

    def get_convergence_patterns(self, flow_name: str | None = None) -> list:
        """
        Analyze convergence patterns across runs.
        """
        try:
            result = self.conn.execute("SELECT COUNT(*) FROM telemetry").fetchone()
            if result[0] == 0:
                return []
        except (RuntimeError, ValueError):
            return []

        query = """
        SELECT
            run_id,
            flow_name,
            COUNT(CASE WHEN phase = 'GREEN' THEN 1 END) as green_phases,
            COUNT(CASE WHEN phase = 'YELLOW' THEN 1 END) as yellow_phases,
            COUNT(CASE WHEN phase = 'RED' THEN 1 END) as red_phases,
            MAX(iter) as total_iterations,
            AVG(E) as avg_energy,
            MIN(E) as final_energy
        FROM telemetry
        """

        if flow_name:
            query += f" WHERE flow_name = '{flow_name}'"

        query += """
        GROUP BY run_id, flow_name
        ORDER BY run_id DESC
        """

        result = self.conn.execute(query).fetchall()

        patterns = []
        for row in result:
            total_phases = (row[2] or 0) + (row[3] or 0) + (row[4] or 0)
            convergence_ratio = (
                (row[2] or 0) / total_phases if total_phases > 0 else 0.0
            )

            patterns.append(
                {
                    "run_id": row[0],
                    "flow_name": row[1],
                    "green_phases": int(row[2] or 0),
                    "yellow_phases": int(row[3] or 0),
                    "red_phases": int(row[4] or 0),
                    "total_iterations": int(row[5] or 0),
                    "avg_energy": float(row[6] or 0.0),
                    "final_energy": float(row[7] or 0.0),
                    "convergence_ratio": convergence_ratio,
                }
            )

        return patterns

    def close(self):
        """
        Closes the database connection.
        """
        self.conn.close()
