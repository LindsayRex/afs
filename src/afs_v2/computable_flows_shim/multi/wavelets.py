from dataclasses import dataclass
from typing import Callable
import jax.numpy as jnp

@dataclass(frozen=True)
class TransformOp:
    name: str
    forward: Callable[[jnp.ndarray], jnp.ndarray]
    inverse: Callable[[jnp.ndarray], jnp.ndarray]
    frame: str = "unitary"  # "unitary" | "tight" | "general"
    c: float = 1.0            # tight-frame constant if frame == "tight"
