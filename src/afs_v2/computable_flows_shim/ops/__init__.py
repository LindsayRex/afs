from typing import Protocol
import jax.numpy as jnp

class Op(Protocol):
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray: ...         # Forward (JAX ops)
    def T(self, x: jnp.ndarray) -> jnp.ndarray: ...                # Adjoint (optional)
    def lipschitz_hint(self) -> float: ...      # Optional β estimate
