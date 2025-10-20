"""
The Flight Controller for the Computable Flows Shim.
"""
from typing import Dict, Any, Callable, Optional
import jax
import jax.numpy as jnp
from computable_flows_shim.energy.compile import CompiledEnergy
from computable_flows_shim.runtime.step import run_flow_step
from computable_flows_shim.fda.certificates import estimate_gamma, estimate_eta_dd
from .telemetry import TelemetryManager

def run_certified(
    initial_state: Dict[str, jnp.ndarray],
    compiled: CompiledEnergy,
    num_iterations: int,
    step_alpha: float,
    telemetry_manager: Optional[TelemetryManager] = None,
    flow_name: str = "",
    run_id: str = "",
    _step_function_for_testing: Optional[Callable] = None,
    max_remediation_attempts: int = 3
) -> Dict[str, jnp.ndarray]:
    """
    Runs a full flow for a fixed number of iterations, with RED/AMBER/GREEN phases and auto-remediation.
    """
    # --- Phase 0: RED - Certificate Checks with Remediation ---
    key = jax.random.PRNGKey(0)  # A dummy key for now.
    input_shape = initial_state['x'].shape
    current_alpha = step_alpha
    
    for attempt in range(max_remediation_attempts + 1):
        eta = estimate_eta_dd(compiled.L_apply, input_shape)
        gamma = estimate_gamma(compiled.L_apply, key, input_shape)
        
        if eta < 1.0 and gamma > 0:
            # GREEN: Certificates pass
            phase = "GREEN"
            if telemetry_manager:
                telemetry_manager.flight_recorder.log_event(run_id=run_id, event="PHASE_TRANSITION", payload={"from": "AMBER", "to": "GREEN", "attempt": attempt})
            break
        elif attempt < max_remediation_attempts:
            # AMBER: Try remediation
            phase = "AMBER"
            if telemetry_manager:
                telemetry_manager.flight_recorder.log_event(run_id=run_id, event="PHASE_TRANSITION", payload={"from": "RED", "to": "AMBER", "attempt": attempt, "eta": float(eta), "gamma": float(gamma)})
            # Remediation: reduce alpha by half
            current_alpha *= 0.5
        else:
            # RED: Failed all attempts
            phase = "RED"
            if telemetry_manager:
                telemetry_manager.flight_recorder.log_event(run_id=run_id, event="CERT_FAIL", payload={"eta": float(eta), "gamma": float(gamma), "attempts": attempt})
            raise ValueError(f"System failed certification after {max_remediation_attempts} remediation attempts. Final eta={eta:.4f}, gamma={gamma:.4f}.")
    
    # --- Phase 2: GREEN - Main Loop ---
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
        # Compute sparsity from W-space (multiscale representation)
        # Sparsity ratio: ||x||₁ / (||x||₂ * √n) - measures concentration of energy
        x = state['x']
        l1_norm = float(jnp.linalg.norm(x, ord=1))
        l2_norm = float(jnp.linalg.norm(x, ord=2))
        n = float(jnp.prod(jnp.array(x.shape)))
        sparsity_wx = l1_norm / (l2_norm * jnp.sqrt(n)) if l2_norm > 0 else 0.0
        # Log telemetry
        if telemetry_manager:
            telemetry_manager.flight_recorder.log(
                run_id=run_id,
                flow_name=flow_name,
                phase=phase,
                iter=i,
                t_wall_ms=t_wall_ms,
                E=float(energy),
                grad_norm=grad_norm,
                eta_dd=float(eta),
                gamma=float(gamma),
                alpha=float(current_alpha),
                phi_residual=phi_residual,
                invariant_drift_max=invariant_drift_max,
                sparsity_wx=sparsity_wx
            )
        # Try step with current alpha, with remediation if energy increases
        max_step_attempts = 3
        step_alpha_local = current_alpha
        for step_attempt in range(max_step_attempts):
            candidate_state = step_func(state, compiled, step_alpha_local)
            new_energy = compiled.f_value(candidate_state)
            if new_energy <= energy:
                # Success
                state = candidate_state
                energy = new_energy
                break
            else:
                # AMBER: Energy increased, reduce alpha and retry
                if telemetry_manager:
                    telemetry_manager.flight_recorder.log_event(run_id=run_id, event="STEP_REMEDIATION", payload={"iter": i, "attempt": step_attempt, "E_prev": float(energy), "E_new": float(new_energy), "alpha": float(step_alpha_local)})
                step_alpha_local *= 0.5
                if step_attempt == max_step_attempts - 1:
                    # Failed all attempts
                    if telemetry_manager:
                        telemetry_manager.flight_recorder.log_event(run_id=run_id, event="STEP_FAIL", payload={"iter": i, "attempts": step_attempt + 1, "E_prev": float(energy), "E_new": float(new_energy)})
                    raise ValueError(f"Step failed to decrease energy after {max_step_attempts} remediation attempts at iteration {i}.")
        else:
            # This shouldn't happen due to the raise above
            pass
    if telemetry_manager:
        telemetry_manager.flight_recorder.log_event(run_id=run_id, event="RUN_FINISHED", payload={})
        telemetry_manager.flush()  # Ensure all data is written
    return state
