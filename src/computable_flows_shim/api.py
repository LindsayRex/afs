"""
Core API for the Computable Flows Shim.
"""
from typing import Any, Protocol

class Op(Protocol):
    """Protocol for a linear operator."""
    def __call__(self, x: Any) -> Any: ...
