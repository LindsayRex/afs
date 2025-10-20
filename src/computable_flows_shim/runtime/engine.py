from typing import Any, Callable, Dict, Optional
from .primitives import F_Dis, F_Proj, F_Multi_forward, F_Multi_inverse
from ..telemetry import TelemetryManager
from ..energy.compile import CompiledEnergy # Import canonical CompiledEnergy

# JAX types for clarity
Array = Any
State = Dict[str, Array]

def run_flow_step(
    state: Dict[str, Array],
    compiled: CompiledEnergy,
    step_alpha: float,
    manifolds: Dict[str, Any] = {}
) -> Dict[str, Array]:
    """
    Runs one full step of a Forward-Backward Splitting flow.
    
    This corresponds to: z_{k+1} = F_Proj(F_Dis(z_k))
    """
    if manifolds is None:
        manifolds = {}

    # Forward step (dissipative)
    state_after_dis = F_Dis(state, compiled.f_grad, step_alpha, manifolds)
    
    # Backward step (projective/proximal)
    state_after_proj = F_Proj(state_after_dis, compiled.g_prox, step_alpha)
    
    return state_after_proj

def run_flow(
    initial_state: State,
    compiled: CompiledEnergy,
    num_iters: int,
    step_alpha: float,
    telemetry_manager: Optional[TelemetryManager] = None
) -> State:
    """
    Runs the full computable flow for a given number of iterations.
    """
    state = initial_state
    for i in range(num_iters):
        state = run_flow_step(state, compiled, step_alpha)
        if telemetry_manager:
            # This is a placeholder for the actual telemetry logging
            telemetry_manager.flight_recorder.log(
                iter=i,
                E=float(compiled.f_value(state))
            )
    return state
