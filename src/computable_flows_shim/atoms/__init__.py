"""
Atoms Library for Computable Flows Shim.

This package provides the fundamental building blocks (atoms) of energy functionals,
each with mathematically rigorous implementations and certificate hooks.
"""

# Import base classes
from .base import Atom

# Import registry system
from .registry import ATOM_REGISTRY, register_atom, get_registered_atoms, is_atom_registered, get_atom_class

# Import all atom implementations (this registers them automatically)
from . import quadratic
from . import tikhonov
from . import l1
from . import wavelet_l1
from . import tv


def create_atom(atom_type: str, **params) -> Atom:
    """
    Factory function to create atom instances.

    Args:
        atom_type: Type of atom to create ('quadratic', etc.)
        **params: Optional parameters to pass to atom constructor

    Returns:
        Configured atom instance

    Raises:
        ValueError: If atom_type is not registered
    """
    atom_class = get_atom_class(atom_type)
    # For now, atoms don't take constructor parameters
    # In future versions, this could be extended
    return atom_class()


__all__ = [
    'Atom',
    'ATOM_REGISTRY',
    'create_atom',
    'register_atom',
    'get_registered_atoms',
    'is_atom_registered',
    'get_atom_class'
]