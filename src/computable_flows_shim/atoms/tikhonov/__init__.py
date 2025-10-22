"""
Tikhonov Atom Package.

This package provides the Tikhonov regularized quadratic atom.
"""

from .tikhonov_atom import TikhonovAtom
from ..registry import register_atom

# Register the tikhonov atom
register_atom('tikhonov', TikhonovAtom)

__all__ = ['TikhonovAtom']