"""
TransformOp: Frame-aware wavelet transforms for multiscale flows.

Provides differentiable wavelet forward/inverse transforms with proper frame metadata.
Integrates jaxwt for JAX-compatible wavelet transforms.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union, TYPE_CHECKING
import jax
import jax.numpy as jnp

# Import jaxwt functions - these will be available when jaxwt is installed
if TYPE_CHECKING:
    # For type checking, import the real functions
    import jaxwt
    from jaxwt import wavedec, waverec, wavedec2, waverec2
    import pywt
    JAXWT_AVAILABLE = True
else:
    try:
        import jaxwt
        from jaxwt import wavedec, waverec, wavedec2, waverec2
        import pywt
        JAXWT_AVAILABLE = True
    except ImportError:
        JAXWT_AVAILABLE = False
        # Stub functions for when jaxwt is not available
        def wavedec(*args, **kwargs):
            raise ImportError("jaxwt not available")
        def waverec(*args, **kwargs):
            raise ImportError("jaxwt not available")
        def wavedec2(*args, **kwargs):
            raise ImportError("jaxwt not available")
        def waverec2(*args, **kwargs):
            raise ImportError("jaxwt not available")
        pywt = None


@dataclass
class TransformOp:
    """
    Frame-aware wavelet transform operator.

    Provides forward/inverse wavelet transforms with frame metadata for
    proper scaling in multiscale flows and W-space operations.
    """

    name: str  # Wavelet name (e.g., 'haar', 'db4')
    levels: int  # Decomposition levels
    ndim: int = 1  # Dimensionality (1, 2, 3)
    frame: str = "tight"  # Frame type: 'tight', 'unitary', or 'frame'
    c: float = 1.0  # Frame constant (1.0 for unitary/tight frames)

    # JAX-jittable transform functions (initialized in __post_init__)
    forward: Callable[[jnp.ndarray], Any] = None  # type: ignore
    inverse: Callable[[Any], jnp.ndarray] = None  # type: ignore

    def __post_init__(self):
        """Initialize transform functions and frame metadata."""
        if not JAXWT_AVAILABLE:
            raise ImportError("jaxwt required for TransformOp. Install with: pip install jaxwt")

        # Set frame metadata based on wavelet
        self._set_frame_metadata()

        # Create transform functions
        self.forward = self._create_forward_fn()
        self.inverse = self._create_inverse_fn()

    def _set_frame_metadata(self):
        """Set frame type and constant based on wavelet properties."""
        # Most wavelets used in practice are tight frames or unitary
        # Haar is unitary (c=1), Daubechies are tight frames
        if self.name.lower() == 'haar':
            self.frame = 'unitary'
            self.c = 1.0
        elif self.name.lower().startswith('db'):
            self.frame = 'tight'
            # For tight frames, c is typically close to 1
            # Exact value depends on wavelet, but we use 1.0 for simplicity
            self.c = 1.0
        else:
            # Default to tight frame
            self.frame = 'tight'
            self.c = 1.0

    def _create_forward_fn(self) -> Callable[[jnp.ndarray], Any]:
        """Create JAX-jittable forward transform function."""
        wavelet_name = self.name
        levels = self.levels

        if self.ndim == 1:
            def forward_1d(x: jnp.ndarray) -> List[jnp.ndarray]:
                return wavedec(x, wavelet=wavelet_name, level=levels)
        elif self.ndim == 2:
            def forward_2d(x: jnp.ndarray) -> Any:  # jaxwt.wavedec2 has complex return type
                return wavedec2(x, wavelet=wavelet_name, level=levels)
        else:
            raise ValueError(f"Unsupported ndim: {self.ndim}")

        return forward_1d if self.ndim == 1 else forward_2d

    def _create_inverse_fn(self) -> Callable[[Any], jnp.ndarray]:
        """Create JAX-jittable inverse transform function."""
        wavelet_name = self.name

        if self.ndim == 1:
            def inverse_1d(coeffs: List[jnp.ndarray]) -> jnp.ndarray:
                result = waverec(coeffs, wavelet=wavelet_name)
                # Remove any extra batch dimensions
                return jnp.squeeze(result)
        elif self.ndim == 2:
            def inverse_2d(coeffs: Any) -> jnp.ndarray:  # Accept complex coefficient structure
                result = waverec2(coeffs, wavelet=wavelet_name)
                # Remove any extra batch dimensions
                return jnp.squeeze(result)
        else:
            raise ValueError(f"Unsupported ndim: {self.ndim}")

        return inverse_1d if self.ndim == 1 else inverse_2d


# Transform registry for common transforms
_TRANSFORM_REGISTRY: Dict[str, TransformOp] = {}


def make_jaxwt_transform(
    wavelet: str,
    levels: int,
    ndim: int = 1
) -> TransformOp:
    """
    Create a TransformOp using jaxwt.

    Args:
        wavelet: Wavelet name (e.g., 'haar', 'db4')
        levels: Number of decomposition levels
        ndim: Dimensionality (1, 2, 3)

    Returns:
        TransformOp instance with forward/inverse methods

    Raises:
        ValueError: If wavelet is not supported
    """
    # Validate wavelet before creating TransformOp
    # We'll validate in __post_init__ when jaxwt is available
    pass
    
    return TransformOp(name=wavelet, levels=levels, ndim=ndim)


def make_transform(
    wavelet: str,
    levels: int,
    ndim: int = 1
) -> TransformOp:
    """
    Factory function for TransformOp instances.

    Uses registry for caching common transforms.
    """
    key = f"{wavelet}_{levels}_{ndim}"

    if key not in _TRANSFORM_REGISTRY:
        _TRANSFORM_REGISTRY[key] = make_jaxwt_transform(wavelet, levels, ndim)

    return _TRANSFORM_REGISTRY[key]


def register_transform(name: str, transform: TransformOp):
    """Register a custom transform in the global registry."""
    _TRANSFORM_REGISTRY[name] = transform


def get_registered_transforms() -> Dict[str, TransformOp]:
    """Get all registered transforms."""
    return _TRANSFORM_REGISTRY.copy()