"""
Parameter Suggester for offline parameter optimization.

Analyzes historical telemetry data to suggest optimal parameters for future runs.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def suggest_parameters(db_manager, flow_name: str | None = None) -> dict[str, Any]:
    """
    Suggest optimal parameters based on historical telemetry data.

    Analyzes past runs to identify patterns and suggest improvements.
    Focuses on alpha parameter optimization based on remediation frequency.

    Args:
        db_manager: Database manager with historical telemetry data
        flow_name: Optional flow name to filter analysis

    Returns:
        Dictionary of suggested parameters
    """
    try:
        # Query historical runs and their remediation patterns
        query = """
            SELECT
                t.run_id,
                t.alpha,
                COUNT(CASE WHEN e.event = 'STEP_REMEDIATION' THEN 1 END) as remediation_count,
                COUNT(e.event) as total_events
            FROM telemetry t
            LEFT JOIN events e ON t.run_id = e.run_id
        """

        params = []
        if flow_name:
            query += " WHERE t.flow_name = ?"
            params.append(flow_name)

        query += """
            GROUP BY t.run_id, t.alpha
            ORDER BY t.run_id
        """

        results = db_manager.conn.execute(query, params).fetchall()

        if not results:
            # No historical data - return defaults
            logger.info("No historical data found, returning default parameters")
            return {"alpha": 0.1}

        # Analyze remediation patterns
        alpha_performance = {}

        for _, alpha, remediation_count, total_events in results:
            if alpha not in alpha_performance:
                alpha_performance[alpha] = {
                    "runs": 0,
                    "total_remediations": 0,
                    "total_events": 0,
                }

            alpha_performance[alpha]["runs"] += 1
            alpha_performance[alpha]["total_remediations"] += remediation_count
            alpha_performance[alpha]["total_events"] += total_events

        # Calculate average remediation rate per alpha
        best_alpha = None
        best_score = float("inf")  # Lower remediation rate is better

        for alpha, stats in alpha_performance.items():
            if stats["runs"] > 0:
                avg_remediation_rate = stats["total_remediations"] / stats["runs"]

                # Prefer alphas with lower remediation rates
                if avg_remediation_rate < best_score:
                    best_score = avg_remediation_rate
                    best_alpha = alpha

        # If we found a good alpha, suggest it
        if best_alpha is not None:
            logger.info(
                "Suggesting alpha=%s based on historical performance "
                "(remediation rate: %.2f)",
                best_alpha,
                best_score,
            )
            return {"alpha": best_alpha}

        # Fallback to conservative default
        logger.info(
            "Could not determine optimal alpha from history, using conservative default"
        )
        return {"alpha": 0.05}  # More conservative than the default 0.1

    except Exception as e:
        logger.warning(
            "Error analyzing historical data: %s, falling back to defaults", e
        )
        return {"alpha": 0.1}
