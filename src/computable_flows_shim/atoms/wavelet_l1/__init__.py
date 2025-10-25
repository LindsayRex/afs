"""
Wavelet L1 Atom Package.

This package provides the wavelet L1 regularization atom.
"""

from ..registry import register_atom
from .wavelet_l1_atom import WaveletL1Atom

# Register the wavelet_l1 atom
register_atom("wavelet_l1", WaveletL1Atom)

__all__ = ["WaveletL1Atom"]
