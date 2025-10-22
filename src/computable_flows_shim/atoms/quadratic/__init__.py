"""
Quadratic Atom Package.

This package provides the quadratic data fidelity atom.
"""

from .quadratic_atom import QuadraticAtom
from ..registry import register_atom

# Register the quadratic atom
register_atom('quadratic', QuadraticAtom)

__all__ = ['QuadraticAtom']