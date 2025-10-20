from typing import Any, Callable, Dict, Optional
from .primitives import F_Dis, F_Proj, F_Multi_forward, F_Multi_inverse
from ..telemetry import TelemetryManager
from ..energy.compile import CompiledEnergy # Import canonical CompiledEnergy
from ..fda.certificates import estimate_eta_dd, estimate_gamma

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
    import jax
    key = jax.random.PRNGKey(42)  # For FDA estimates
    
    # Assume single variable 'x' for now
    input_shape = initial_state['x'].shape
    
    # Compute FDA certificates
    eta_dd = estimate_eta_dd(compiled.L_apply, input_shape)
    gamma = estimate_gamma(compiled.L_apply, key, input_shape)
    
    # Log certificates
    if telemetry_manager:
        telemetry_manager.flight_recorder.log_event(
            run_id=telemetry_manager.run_id,
            event="CERT_CHECK",
            payload={"eta_dd": eta_dd, "gamma": gamma}
        )
    
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
