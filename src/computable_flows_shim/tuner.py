"""
Tuner module for cross-run parameter optimization.

This module analyzes historical telemetry data to suggest improved parameters
for future flow runs, implementing proactive auto-tuning.
"""

from typing import Dict, Any, Optional
from statistics import mean
from .telemetry.duckdb_manager import DuckDBManager


def suggest_parameters(db_manager: DuckDBManager, flow_name: Optional[str] = None, limit: int = 50) -> Dict[str, float]:
    """
    Analyzes historical run data from telemetry database and suggests parameter improvements.
    
    Contract:
    - Precondition: db_manager is connected to a populated telemetry database
    - Postcondition: Returns dict with suggested parameters, including 'alpha'
    - Invariant: Suggested alpha is positive and reasonable (0.001 < alpha < 1.0)
    """
    history = db_manager.get_run_summaries(flow_name=flow_name, limit=limit)
    
    if not history:
        return {'alpha': 0.1}  # Default fallback
    
    # Group runs by alpha and compute average remediations
    alpha_stats = {}
    for run in history:
        alpha = run.get('alpha', 0.1)
        rems = run.get('num_remediations', 0)
        if alpha not in alpha_stats:
            alpha_stats[alpha] = []
        alpha_stats[alpha].append(rems)
    
    # Find alpha with lowest average remediations
    if not alpha_stats:
        return {'alpha': 0.1}
    
    best_alpha = min(alpha_stats.keys(), 
                     key=lambda a: mean(alpha_stats[a]))
    
    # If best alpha still has high remediations, suggest more conservative value
    avg_rems = mean(alpha_stats[best_alpha])
    if avg_rems > 2.0:
        suggested_alpha = best_alpha * 0.7  # Reduce by 30%
    else:
        suggested_alpha = best_alpha
    
    # Clamp to reasonable bounds to prevent instability
    suggested_alpha = max(0.001, min(1.0, suggested_alpha))
    
    return {'alpha': suggested_alpha}