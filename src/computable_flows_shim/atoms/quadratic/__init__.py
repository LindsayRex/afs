"""
Quadratic Atom Package.

This package provides the quadratic data fidelity atom.
"""

from ..registry import register_atom
from .quadratic_atom import QuadraticAtom

# Register the quadratic atom
register_atom("quadratic", QuadraticAtom)

__all__ = ["QuadraticAtom"]
