"""
Tikhonov Atom Package.

This package provides the Tikhonov regularized quadratic atom.
"""

from ..registry import register_atom
from .tikhonov_atom import TikhonovAtom

# Register the tikhonov atom
register_atom("tikhonov", TikhonovAtom)

__all__ = ["TikhonovAtom"]
