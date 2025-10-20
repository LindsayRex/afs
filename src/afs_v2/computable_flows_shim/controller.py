from typing import Dict
from frozendict import frozendict
import jax
import jax.numpy as jnp

from computable_flows_shim.energy.compile import compile_energy
from computable_flows_shim.energy.specs import EnergySpec
from computable_flows_shim.ops import Op
from computable_flows_shim.runtime.step import run_flow_step
from computable_flows_shim.fda.certificates import estimate_eta_dd_in_W, estimate_gamma_in_W

def run_certified(spec: EnergySpec, op_registry: Dict[str, Op], initial_state: Dict[str, jnp.ndarray], num_iterations: int = 100):
    """
    The main entry point for running a certified flow.
    """
    # Phase 0: Lint/Normalize (RED→AMBER)
    print("Phase 0: Linting and Normalization...")
    # (Placeholder for actual linting and normalization)
    print("Status: AMBER")

    # Compile the energy specification
    compiled_energy = compile_energy(spec, op_registry)

    # Phase 1: Certificates (AMBER→GREEN)
    print("Phase 1: Certificate Estimation...")
    key = jax.random.PRNGKey(0) # A real implementation would handle keys more robustly
    
    # Estimate certificates
    eta_dd = estimate_eta_dd_in_W(compiled_energy.L_apply, compiled_energy.W, key)
    gamma = estimate_gamma_in_W(compiled_energy.L_apply, compiled_energy.W, key)
    beta = gamma # This is a simplification, beta should be the Lipschitz constant
    
    # Check certificates
    if eta_dd > 0.9 or gamma < 0.1:
        print(f"Certification failed: eta_dd={eta_dd}, gamma={gamma}")
        print("Status: RED")
        return initial_state # Abort

    # JIT-compile the step function for performance
    jitted_step = jax.jit(run_flow_step, static_argnums=(0, 1))

    # Run the flow
    print("Phase 2: Running Certified Flow...")
    state = initial_state
    manifolds = frozendict({}) # Empty manifolds for now
    
    # Determine step size
    step_alpha = 1.0 / beta

    for i in range(num_iterations):
        state = jitted_step(compiled_energy, manifolds, state, step_alpha)
        if i % 10 == 0:
            # A real implementation would log telemetry here
            print(f"Iteration {i}...")


    # Phase 3: Polish / optional F_Con
    # (Placeholder)

    print("Flow finished.")
    return state
