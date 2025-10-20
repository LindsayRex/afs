"""
Atoms Library for Computable Flows Shim.

This package provides the fundamental building blocks (atoms) of energy functionals,
each with mathematically rigorous implementations and certificate hooks.
"""

from .library import Atom, QuadraticAtom, TikhonovAtom, L1Atom, ATOM_REGISTRY, create_atom, register_atom

__all__ = [
    'Atom', 
    'QuadraticAtom', 
    'TikhonovAtom',
    'L1Atom',
    'ATOM_REGISTRY', 
    'create_atom', 
    'register_atom'
]