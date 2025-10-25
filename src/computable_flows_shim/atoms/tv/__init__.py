"""
TV Atom Package.

This package provides the total variation regularization atom.
"""

from ..registry import register_atom
from .tv_atom import TVAtom

# Register the tv atom
register_atom("tv", TVAtom)

__all__ = ["TVAtom"]
