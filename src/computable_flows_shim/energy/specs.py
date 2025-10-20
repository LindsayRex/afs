"""
Energy specification dataclasses, defining the structure of a computable flow problem.
"""
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

@dataclass(frozen=True)
class TermSpec:
    """Specification for a single term in the energy functional."""
    type: str
    op: str
    weight: float
    variable: str
    target: Optional[str] = None

@dataclass(frozen=True)
class StateSpec:
    """Specification for the state variables."""
    shapes: Dict[str, List[int]]

@dataclass(frozen=True)
class EnergySpec:
    """The complete energy specification."""
    terms: List[TermSpec]
    state: StateSpec
