"""
Atom Registry System.

This module provides the registration system for atoms, allowing dynamic
discovery and instantiation of atom types.
"""

from typing import Dict, Type
from .base import Atom


# Global registry of available atoms
ATOM_REGISTRY: Dict[str, Type[Atom]] = {}


def register_atom(atom_type: str, atom_class: Type[Atom]):
    """Register a new atom type."""
    ATOM_REGISTRY[atom_type] = atom_class


def get_registered_atoms() -> Dict[str, Type[Atom]]:
    """Get a copy of the current atom registry."""
    return ATOM_REGISTRY.copy()


def is_atom_registered(atom_type: str) -> bool:
    """Check if an atom type is registered."""
    return atom_type in ATOM_REGISTRY


def get_atom_class(atom_type: str) -> Type[Atom]:
    """Get the atom class for a given type."""
    if not is_atom_registered(atom_type):
        available = list(ATOM_REGISTRY.keys())
        raise ValueError(f"Unknown atom type: {atom_type}. Available: {available}")
    return ATOM_REGISTRY[atom_type]