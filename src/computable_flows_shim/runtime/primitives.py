"""
Primitive flow operators for the Computable Flows Shim.
"""
from typing import Callable, Dict, Any, Union, List, Protocol
import jax
import jax.numpy as jnp

# JAX types for clarity
Array = jnp.ndarray
State = Dict[str, Array]

class TransformProtocol(Protocol):
    """Protocol for multiscale transform objects."""
    def forward(self, x: Array) -> List[Array]:
        ...
    
    def inverse(self, x: List[Array]) -> Array:
        ...

def F_Dis(state: State, grad_f: Callable[[State], State], step_alpha: float, manifolds: Dict[str, Any]) -> State:
    """
    Dissipative Step (F_Dis): Performs a single gradient descent step.
    
    Math:
    .. math::
        z_{k+1} = z_k - \alpha \nabla f(z_k)
    """
    g = grad_f(state)
    new_state = {}
    for name, x in state.items():
        M = manifolds.get(name)
        if M is None or isinstance(M, dict) and M.get('type') == 'euclidean':
            # Standard Euclidean gradient step
            if name in g:
                new_state[name] = x - step_alpha * g[name]
            else:
                new_state[name] = x
        else:
            # Riemannian manifold gradient step
            if isinstance(M, dict):
                # Create manifold adapter from config
                from computable_flows_shim.runtime.manifolds import create_manifold
                manifold_adapter = create_manifold(M['type'], **M.get('params', {}))
            else:
                # M is already a manifold adapter
                manifold_adapter = M
            
            if name in g:
                # Compute Riemannian gradient and retract
                riemannian_grad = manifold_adapter.riemannian_gradient(x, g[name])
                new_state[name] = manifold_adapter.retract(x, -step_alpha * riemannian_grad)
            else:
                new_state[name] = x
            
    return new_state

def F_Proj(state: State, prox_g: Callable, step_alpha: float) -> State:
    """
    Projective/Proximal Step (F_Proj): Applies a proximal operator.
    """
    return prox_g(state, step_alpha)

def F_Multi(x: Union[Array, List[Array]], W: TransformProtocol, direction: str) -> Union[Array, List[Array]]:
    """
    Multiscale Transform (F_Multi): Applies wavelet transform forward or inverse.
    
    Args:
        x: Input array (for forward) or coefficient list (for inverse)
        W: Transform object with forward(x) and inverse(x) methods
        direction: 'forward' or 'inverse'
        
    Returns:
        For forward: List of coefficient arrays
        For inverse: Reconstructed array
    """
    if direction == 'forward':
        return W.forward(x)  # type: ignore
    elif direction == 'inverse':
        return W.inverse(x)  # type: ignore
    else:
        raise ValueError(f"Invalid direction: {direction}. Must be 'forward' or 'inverse'.")

def F_Con(state: State, H: Callable[[State], Array], dt: float) -> State:
    """
    Conservative/Symplectic Step (F_Con): Performs one step of a symplectic integrator.
    """
    # We expect the state to have 'q' and 'p' for Hamiltonian dynamics
    q, p = state['q'], state['p']
    
    # The gradient of the Hamiltonian with respect to the state
    grad_H_q = jax.grad(H, argnums=0)

    # Leapfrog/Stormer–Verlet integrator (kick-drift-kick)
    p_half = p - 0.5 * dt * grad_H_q({'q': q, 'p': p})['q']
    q_full = q + dt * p_half  # Assuming H = 0.5*p^2 + V(q)
    p_full = p_half - 0.5 * dt * grad_H_q({'q': q_full, 'p': p_half})['q']

    return {'q': q_full, 'p': p_full}

def F_Ann(state: State, key: Array, temperature: float, dt: float) -> State:
    """
    Annealing/Stochastic Step (F_Ann): Adds noise for Langevin dynamics.
    """
    new_state = {}
    for name, x in state.items():
        key, subkey = jax.random.split(key)
        noise = jax.random.normal(subkey, shape=x.shape, dtype=x.dtype)
        new_state[name] = x + jnp.sqrt(2 * temperature * dt) * noise
    return new_state
