"""
Primitive flow operators for the Computable Flows Shim.
"""
from typing import Callable, Dict, Any
import jax
import jax.numpy as jnp
import jax.random as jr

def F_Dis(state: Dict[str, jnp.ndarray], grad_f: Callable, step_alpha: float, manifolds: Dict[str, Any]) -> Dict[str, jnp.ndarray]:
    """
    Dissipative Step (F_Dis): Performs a single gradient descent step.
    
    _Math:_
    .. math::
        z_{k+1} = z_k - \alpha \nabla f(z_k)
    """
    g = grad_f(state)
    
    # For now, we only handle the Euclidean case (no manifolds)
    new_state = {}
    for name, x in state.items():
        if name in g:
            new_state[name] = x - step_alpha * g[name]
        else:
            # This variable is not part of the energy function, so it doesn't change
            new_state[name] = x
            
    return new_state

def F_Proj(state: Dict[str, jnp.ndarray], prox_g: Callable, step_alpha: float) -> Dict[str, jnp.ndarray]:
    """
    Projective/Proximal Step (F_Proj): Applies a proximal operator.
    """
    return prox_g(state, step_alpha)

def F_Multi_forward(x: jnp.ndarray, W: Any) -> jnp.ndarray:
    """Multiscale Transform (F_Multi) - Forward"""
    return W.forward(x)

def F_Multi_inverse(u: jnp.ndarray, W: Any) -> jnp.ndarray:
    """Multiscale Transform (F_Multi) - Inverse"""
    return W.inverse(u)

def F_Con(state: Dict[str, jnp.ndarray], H: Callable, dt: float) -> Dict[str, jnp.ndarray]:
    """
    Conservative/Symplectic Step (F_Con): Performs one step of a symplectic integrator.
    """
    grad_H = jax.grad(H)

    # Initial state
    q, p = state['q'], state['p']

    # Leapfrog/Stormer–Verlet integrator (kick-drift-kick)
    p_half = p - 0.5 * dt * grad_H(state)['q']
    q_full = q + dt * grad_H({'q': q, 'p': p_half})['p']
    p_full = p_half - 0.5 * dt * grad_H({'q': q_full, 'p': p_half})['q']

    return {'q': q_full, 'p': p_full}

def F_Ann(state: Dict[str, jnp.ndarray], key: jnp.ndarray, temperature: float, dt: float) -> Dict[str, jnp.ndarray]:
    """
    Annealing/Stochastic Step (F_Ann): Adds noise for Langevin dynamics.
    """
    new_state = {}
    for name, x in state.items():
        key, subkey = jax.random.split(key)
        noise = jax.random.normal(subkey, shape=x.shape)
        new_state[name] = x + jnp.sqrt(2 * temperature * dt) * noise
    return new_state
