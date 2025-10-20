"""
The Flight Controller for the Computable Flows Shim.
"""
from typing import Dict, Any, Callable, Optional
import jax
import jax.numpy as jnp
from computable_flows_shim.energy.compile import CompiledEnergy
from computable_flows_shim.runtime.step import run_flow_step
from computable_flows_shim.fda.certificates import estimate_gamma, estimate_eta_dd

def run_certified(
    initial_state: Dict[str, jnp.ndarray],
    compiled: CompiledEnergy,
    num_iterations: int,
    step_alpha: float,
    recorder=None,
    flow_name: str = "",
    run_id: str = "",
    _step_function_for_testing: Optional[Callable] = None
) -> Dict[str, jnp.ndarray]:
    """
    Runs a full flow for a fixed number of iterations, enforcing certificates.
    """
    # --- Phase 1: Certificate Checks ---
    key = jax.random.PRNGKey(0)  # A dummy key for now.
    input_shape = initial_state['x'].shape
    eta = estimate_eta_dd(compiled.L_apply, input_shape)
    if eta >= 1.0:
        if recorder:
            recorder.log_event(run_id=run_id, event="CERT_FAIL", payload={"eta": float(eta)})
        raise ValueError(f"System is not diagonally dominant: eta ({eta:.4f}) >= 1.0.")
    gamma = estimate_gamma(compiled.L_apply, key, input_shape)
    if gamma <= 0:
        if recorder:
            recorder.log_event(run_id=run_id, event="CERT_FAIL", payload={"gamma": float(gamma)})
        raise ValueError(f"System is unstable: Spectral gap (gamma) is non-positive ({gamma:.4f}).")

    # --- Phase 2: Main Loop ---
    state = initial_state
    energy = compiled.f_value(state)
    step_func = _step_function_for_testing or run_flow_step

    import time
    t0 = time.time()
    for i in range(num_iterations):
        t_wall_ms = (time.time() - t0) * 1000.0
        # Compute all required telemetry fields
        grad = compiled.f_grad(state)
        grad_norm = float(jnp.linalg.norm(grad['x']))
        eta = estimate_eta_dd(compiled.L_apply, input_shape)
        gamma = estimate_gamma(compiled.L_apply, key, input_shape)
        # Placeholder for residual/invariant metrics
        phi_residual = float(jnp.nan)
        invariant_drift_max = float(jnp.nan)
        # Log telemetry
        if recorder:
            recorder.log(
                run_id=run_id,
                flow_name=flow_name,
                phase="GREEN",
                iter=i,
                t_wall_ms=t_wall_ms,
                E=float(energy),
                grad_norm=grad_norm,
                eta_dd=float(eta),
                gamma=float(gamma),
                alpha=float(step_alpha),
                phi_residual=phi_residual,
                invariant_drift_max=invariant_drift_max
            )
        state = step_func(state, compiled, step_alpha)
        new_energy = compiled.f_value(state)
        if new_energy > energy:
            if recorder:
                recorder.log_event(run_id=run_id, event="LYAP_FAIL", payload={"E_prev": float(energy), "E_new": float(new_energy)})
            raise ValueError("Lyapunov descent violated: Energy increased.")
        energy = new_energy
    if recorder:
        recorder.log_event(run_id=run_id, event="RUN_FINISHED", payload={})
    return state
