"""
Base classes and interfaces for the Atoms Library.

This module contains the abstract base classes that define the Atom interface.
All atoms must inherit from these classes and implement their abstract methods.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import jax.numpy as jnp

# Type aliases
Array = jnp.ndarray
State = Dict[str, Array]


class Atom(ABC):
    """
    Abstract base class for energy functional atoms.

    Each atom represents a fundamental building block of energy functionals
    with well-defined mathematical properties and computational implementations.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name identifier for this atom type."""
        pass

    @property
    @abstractmethod
    def form(self) -> str:
        """LaTeX mathematical form of this atom."""
        pass

    @abstractmethod
    def energy(self, state: State, params: Dict[str, Any]) -> float:
        """Compute the energy contribution of this atom."""
        pass

    @abstractmethod
    def gradient(self, state: State, params: Dict[str, Any]) -> State:
        """Compute the gradient contribution of this atom."""
        pass

    @abstractmethod
    def prox(self, state: State, step_size: float, params: Dict[str, Any]) -> State:
        """Apply the proximal operator for this atom."""
        pass

    @abstractmethod
    def certificate_contributions(self, params: Dict[str, Any]) -> Dict[str, float]:
        """Return contributions to FDA certificates (eta_dd, gamma, etc.)."""
        pass