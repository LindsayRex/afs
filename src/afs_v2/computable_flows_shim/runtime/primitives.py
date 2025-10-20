from typing import Callable, Dict, Optional, Union
import jax
import jax.numpy as jnp
from frozendict import frozendict
from computable_flows_shim.multi.wavelets import TransformOp
from computable_flows_shim.manifolds.manifolds import Manifold

def F_Dis(state: Dict[str, jnp.ndarray], grad_f: Callable, step_alpha: float, manifolds: Union[Dict[str, Manifold], frozendict]) -> Dict[str, jnp.ndarray]:
    """Dissipative Step (F_Dis)"""
    g = grad_f(state)  # grad_f built by compiler, uses jax.grad
    new_state = {}
    for name, x in state.items():
        M = manifolds.get(name)
        if M is None:
            new_state[name] = x - step_alpha * g[name]
        else:
            # project Euclidean grad to tangent and retract
            grad_R = M.proj_tangent(x, g[name])
            new_state[name] = M.retract(x, -step_alpha * grad_R)
    return new_state

def F_Proj(state: Dict[str, jnp.ndarray], prox_in_W: Callable, step_alpha: float, W: TransformOp) -> Dict[str, jnp.ndarray]:
    """Projective/Proximal Step (F_Proj)"""
    # This is a placeholder; the actual implementation will be more complex
    # and will be built by the energy compiler.
    return prox_in_W(state, step_alpha, W)

def F_Multi_forward(x: jnp.ndarray, W: TransformOp) -> jnp.ndarray:
    """Multiscale Transform (F_Multi) - Forward"""
    return W.forward(x)

def F_Multi_inverse(u: jnp.ndarray, W: TransformOp) -> jnp.ndarray:
    """Multiscale Transform (F_Multi) - Inverse"""
    return W.inverse(u)

def F_Con(state: Dict[str, jnp.ndarray], H: Optional[Callable] = None, dt: float = 1.0) -> Dict[str, jnp.ndarray]:
    """Conservative/Symplectic Step (F_Con, optional)"""
    if H is None:
        return state

    grad_H = jax.grad(H)

    # Initial state
    q, p = state['q'], state['p']

    # Leapfrog/Stormer–Verlet integrator
    # Step 1: p_half = p_k - (dt/2) * dH/dq(q_k, p_k)
    grad_q_k = grad_H(state)['q']
    p_half = p - 0.5 * dt * grad_q_k

    # Step 2: q_full = q_k + dt * dH/dp(q_k, p_k+1/2)
    state_half_p = {'q': q, 'p': p_half}
    grad_p_half = grad_H(state_half_p)['p']
    q_full = q + dt * grad_p_half

    # Step 3: p_full = p_k+1/2 - (dt/2) * dH/dq(q_k+1, p_k+1/2)
    state_full_q_half_p = {'q': q_full, 'p': p_half}
    grad_q_full = grad_H(state_full_q_half_p)['q']
    p_full = p_half - 0.5 * dt * grad_q_full

    return {'q': q_full, 'p': p_full}

def F_Ann(state: Dict[str, jnp.ndarray], grad_f: Callable, step_alpha: float, temperature: float, key: jax.Array) -> Dict[str, jnp.ndarray]:
    """Annealing/Stochastic Step (F_Ann)"""
    g = grad_f(state)
    new_state = {}
    for name, x in state.items():
        noise = jax.random.normal(key, shape=x.shape, dtype=x.dtype)
        # Discretized Langevin dynamics
        new_state[name] = x - step_alpha * g[name] + jnp.sqrt(2 * step_alpha * temperature) * noise
    return new_state
