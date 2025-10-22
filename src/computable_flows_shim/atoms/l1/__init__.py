"""
L1 Atom Package.

This package provides the L1 regularization atom.
"""

from .l1_atom import L1Atom
from ..registry import register_atom

# Register the l1 atom
register_atom('l1', L1Atom)

__all__ = ['L1Atom']