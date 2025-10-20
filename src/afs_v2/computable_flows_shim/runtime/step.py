from typing import Dict, Callable, Union
from dataclasses import dataclass
import jax
from frozendict import frozendict
from .primitives import F_Dis, F_Proj, F_Multi_forward, F_Multi_inverse, F_Con
from computable_flows_shim.energy.specs import EnergySpec
from computable_flows_shim.multi.wavelets import TransformOp
from computable_flows_shim.manifolds.manifolds import Manifold

# This is a simplified representation of what the compiler will produce.
# In the full implementation, this will be a more complex, compiled JAX function.
@dataclass(frozen=True)
class CompiledEnergy:
    f_value: Callable
    f_grad: Callable
    g_prox: Callable
    W: TransformOp
    L_apply: Callable

def run_flow_step(
    compiled: CompiledEnergy,
    manifolds: Union[Dict[str, Manifold], frozendict],
    state: Dict[str, jax.numpy.ndarray],
    step_alpha: float
) -> Dict[str, jax.numpy.ndarray]:
    """
    Executes one step of the compiled flow.
    """
    # F_Dis → F_Multi → F_Proj → F_Multi⁻¹ (± F_Con)
    
    # Dissipative step
    z = F_Dis(state, compiled.f_grad, step_alpha, manifolds)
    
    # Multiscale transform (forward)
    # Note: This is a simplified example. A real implementation would handle
    # multiple state variables and transforms.
    u = F_Multi_forward(z['x'], compiled.W)
    
    # Projective/proximal step
    u_prox = F_Proj({'x': u}, compiled.g_prox, step_alpha, compiled.W)['x']
    
    # Multiscale transform (inverse)
    z_new = F_Multi_inverse(u_prox, compiled.W)
    
    # Optional conservative step
    # final_state = F_Con({'x': z_new}, H=None, dt=1.0)
    
    return {'x': z_new, 'y': state['y']}
