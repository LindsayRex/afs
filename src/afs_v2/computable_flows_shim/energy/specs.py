from dataclasses import dataclass, field
from typing import List, Dict, Any, Callable, Optional

@dataclass(frozen=True)
class TermSpec:
    type: str
    op: str
    weight: float
    variable: str # The state variable this term applies to
    target: str = ""
    # Add other relevant fields from your design

@dataclass(frozen=True)
class TransformRef:
    name: str
    # Add other relevant fields

@dataclass(frozen=True)
class LensPolicy:
    candidates: List[TransformRef]
    probe_metrics: List[str]
    selection_rule: str

@dataclass(frozen=True)
class StateSpec:
    shapes: Dict[str, List[int]]
    invariants: Dict[str, Dict[str, Callable]] = field(default_factory=dict)

@dataclass(frozen=True)
class FlowPolicy:
    family: str
    discretization: str
    preconditioner: Optional[str] = None

@dataclass(frozen=True)
class GapDial:
    eta_max: float
    gamma_min: float
    beta_estimator: str
    step_rule: str
    per_scale_init: Dict[str, bool]

@dataclass(frozen=True)
class MultiscaleSchedule:
    mode: str
    levels: int
    activate_rule: str

@dataclass(frozen=True)
class SparsityPolicy:
    penalty: str
    thresholding: str
    adaptive_rule: str

@dataclass(frozen=True)
class CertificationProfile:
    checks: List[str]
    tolerances: Dict[str, float]
    refinement_test: Dict[str, Any]

@dataclass(frozen=True)
class EnergySpec:
    terms: List[TermSpec]
    state: StateSpec
    transforms: Dict[str, Any] = field(default_factory=dict)
    lens_policy: Optional[LensPolicy] = None
    flow_policy: Optional[FlowPolicy] = None
    gap_dial: Optional[GapDial] = None
    multiscale_schedule: Optional[MultiscaleSchedule] = None
    sparsity_policy: Optional[SparsityPolicy] = None
    certification_profile: Optional[CertificationProfile] = None
    # Add other top-level spec fields from your design documents
