from typing import Any, Callable, Dict, Optional
import jax.numpy as jnp
from .primitives import F_Dis, F_Proj, F_Multi
from ..telemetry import TelemetryManager
from ..energy.compile import CompiledEnergy # Import canonical CompiledEnergy
from ..fda.certificates import estimate_eta_dd, estimate_gamma

# JAX types for clarity
Array = jnp.ndarray
State = Dict[str, Array]

def run_flow_step(
    state: Dict[str, jnp.ndarray],
    compiled: CompiledEnergy,
    step_alpha: float,
    manifolds: Dict[str, Any] = {},
    W: Optional[Any] = None
) -> Dict[str, jnp.ndarray]:
    """
    Runs one full step of a Forward-Backward Splitting flow.
    
    If W is provided, includes multiscale transforms:
    F_Dis → F_Multi_forward → F_Proj → F_Multi_inverse
    Otherwise, simple: F_Dis → F_Proj
    """
    if manifolds is None:
        manifolds = {}

    # Forward step (dissipative) - always in physical domain
    state_after_dis = F_Dis(state, compiled.f_grad, step_alpha, manifolds)
    
    if W is not None:
        # Multiscale: transform to W-space
        u = F_Multi(state_after_dis['x'], W, 'forward')
        # Projective step in W-space
        # For now, assume prox is for the transformed space
        # TODO: Update compiler to handle W-space prox
        u_proj = compiled.g_prox({'x': u}, step_alpha)['x']
        # Transform back to physical domain
        x_new = F_Multi(u_proj, W, 'inverse')
        state_after_proj = {'x': x_new, 'y': state['y']}  # Keep y unchanged
    else:
        # Simple flow: projective in physical domain
        state_after_proj = F_Proj(state_after_dis, compiled.g_prox, step_alpha)
    
    return state_after_proj

def run_flow(
    initial_state: Dict[str, jnp.ndarray],
    compiled: CompiledEnergy,
    num_iters: int,
    step_alpha: float,
    telemetry_manager: Optional[TelemetryManager] = None
) -> Dict[str, jnp.ndarray]:
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
