"""
Atoms Library for Computable Flows Shim.

This module implements the fundamental building blocks (atoms) of energy functionals,
each with proper mathematical formulations, proximal operators, and certificate hooks.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Callable, Protocol, Union
import jax
import jax.numpy as jnp

# Type aliases
Array = jnp.ndarray
State = Dict[str, Array]


class Atom(ABC):
    """
    Abstract base class for energy functional atoms.
    
    Each atom represents a fundamental building block of energy functionals
    with well-defined mathematical properties and computational implementations.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name identifier for this atom type."""
        pass
    
    @property
    @abstractmethod 
    def form(self) -> str:
        """LaTeX mathematical form of this atom."""
        pass
    
    @abstractmethod
    def energy(self, state: State, params: Dict[str, Any]) -> float:
        """Compute the energy contribution of this atom."""
        pass
    
    @abstractmethod
    def gradient(self, state: State, params: Dict[str, Any]) -> State:
        """Compute the gradient contribution of this atom."""
        pass
    
    @abstractmethod
    def prox(self, state: State, step_size: float, params: Dict[str, Any]) -> State:
        """Apply the proximal operator for this atom."""
        pass
    
    @abstractmethod
    def certificate_contributions(self, params: Dict[str, Any]) -> Dict[str, float]:
        """Return contributions to FDA certificates (eta_dd, gamma, etc.)."""
        pass


class QuadraticAtom(Atom):
    """
    Quadratic data fidelity atom: (1/2)‖Ax - b‖²
    
    This is the fundamental atom for least squares and Gaussian likelihoods.
    Mathematical properties:
    - Convex and differentiable
    - Lipschitz gradient with constant σ_max(A^T A)
    - Proximal operator has closed-form solution
    """
    
    @property
    def name(self) -> str:
        return "quadratic"
    
    @property
    def form(self) -> str:
        return r"\frac{1}{2}\|Ax - b\|_2^2"
    
    def energy(self, state: State, params: Dict[str, Any]) -> float:
        """Compute quadratic energy: (1/2)‖Ax - b‖²"""
        A = params['A']
        b = params['b']
        x = state[params['variable']]
        
        residual = A @ x - b
        return 0.5 * float(jnp.sum(residual**2))
    
    def gradient(self, state: State, params: Dict[str, Any]) -> State:
        """Compute gradient: A^T(Ax - b)"""
        A = params['A']
        b = params['b']
        x = state[params['variable']]
        
        residual = A @ x - b
        grad_x = A.T @ residual
        
        return {params['variable']: grad_x}
    
    def prox(self, state: State, step_size: float, params: Dict[str, Any]) -> State:
        """
        Proximal operator for quadratic (exact solution via linear system).
        
        Solves: argmin_x (1/2)‖Ax - b‖² + (1/(2τ))‖x - x₀‖²
        Solution: (A^T A + I/τ) x = A^T b + x₀/τ
        """
        A = params['A']
        b = params['b']
        x = state[params['variable']]
        
        # Form the regularized system: (A^T A + I/step_size)
        ATA = A.T @ A
        ATb = A.T @ b
        
        # Regularized system matrix and RHS
        lhs = ATA + jnp.eye(ATA.shape[0]) / step_size
        rhs = ATb + x / step_size
        
        # Solve the linear system
        x_new = jnp.linalg.solve(lhs, rhs)
        
        return {params['variable']: x_new}
    
    def certificate_contributions(self, params: Dict[str, Any]) -> Dict[str, float]:
        """
        Certificate contributions for quadratic atom.
        
        Returns Lipschitz constant and diagonal dominance contributions.
        For quadratic atoms, the Lipschitz constant is σ_max(A^T A).
        """
        A = params['A']
        
        # Compute spectral norm of A^T A (Lipschitz constant of gradient)
        ATA = A.T @ A
        lipschitz = float(jnp.linalg.norm(ATA, ord=2))
        
        return {
            'lipschitz': lipschitz,
            'eta_dd_contribution': 0.0,  # Quadratic terms don't affect diagonal dominance
            'gamma_contribution': -lipschitz  # Contributes negatively to spectral gap
        }


class TikhonovAtom(Atom):
    """
    Tikhonov regularized quadratic atom: (1/2)‖Ax - b‖² + (λ/2)‖x‖²
    
    This implements Tikhonov regularization for ill-posed inverse problems.
    The regularization parameter λ controls the trade-off between data fidelity and smoothness.
    """
    
    @property
    def name(self) -> str:
        return "tikhonov"
    
    @property
    def form(self) -> str:
        return r"\frac{1}{2}\|Ax - b\|_2^2 + \frac{\lambda}{2}\|x\|_2^2"
    
    def energy(self, state: State, params: Dict[str, Any]) -> float:
        """Compute Tikhonov energy: (1/2)‖Ax - b‖² + (λ/2)‖x‖²"""
        A = params['A']
        b = params['b']
        lam = params.get('lambda', 1.0)  # Default regularization parameter
        x = state[params['variable']]
        
        residual = A @ x - b
        data_fidelity = 0.5 * float(jnp.sum(residual**2))
        regularization = 0.5 * lam * float(jnp.sum(x**2))
        
        return data_fidelity + regularization
    
    def gradient(self, state: State, params: Dict[str, Any]) -> State:
        """Compute gradient: A^T(Ax - b) + λx"""
        A = params['A']
        b = params['b']
        lam = params.get('lambda', 1.0)
        x = state[params['variable']]
        
        residual = A @ x - b
        grad_x = A.T @ residual + lam * x
        
        return {params['variable']: grad_x}
    
    def prox(self, state: State, step_size: float, params: Dict[str, Any]) -> State:
        """
        Proximal operator for Tikhonov regularization.
        
        Solves: argmin_x (1/2)‖Ax - b‖² + (λ/2)‖x‖² + (1/(2τ))‖x - x₀‖²
        Solution: (A^T A + λI + I/τ) x = A^T b + x₀/τ
        """
        A = params['A']
        b = params['b']
        lam = params.get('lambda', 1.0)
        x = state[params['variable']]
        
        # Form the regularized system: (A^T A + λI + I/step_size)
        ATA = A.T @ A
        ATb = A.T @ b
        
        # Add regularization and proximal regularization
        regularization_matrix = lam * jnp.eye(ATA.shape[0])
        proximal_matrix = jnp.eye(ATA.shape[0]) / step_size
        
        lhs = ATA + regularization_matrix + proximal_matrix
        rhs = ATb + x / step_size
        
        x_new = jnp.linalg.solve(lhs, rhs)
        
        return {params['variable']: x_new}
    
    def certificate_contributions(self, params: Dict[str, Any]) -> Dict[str, float]:
        """
        Certificate contributions for Tikhonov atom.
        
        The regularization improves conditioning and provides better certificates.
        """
        A = params['A']
        lam = params.get('lambda', 1.0)
        
        # Effective Lipschitz constant: σ_max(A^T A + λI)
        ATA = A.T @ A
        regularization_matrix = lam * jnp.eye(ATA.shape[0])
        effective_matrix = ATA + regularization_matrix
        lipschitz = float(jnp.linalg.norm(effective_matrix, ord=2))
        
        return {
            'lipschitz': lipschitz,
            'eta_dd_contribution': lam,  # Regularization improves diagonal dominance
            'gamma_contribution': -lipschitz  # Still contributes negatively, but less than unregularized
        }


class L1Atom(Atom):
    """
    L1 regularization atom: λ‖x‖₁
    
    This implements L1 regularization for sparse recovery and compressed sensing.
    The proximal operator is the soft-thresholding function.
    """
    
    @property
    def name(self) -> str:
        return "l1"
    
    @property
    def form(self) -> str:
        return r"\lambda\|x\|_1"
    
    def energy(self, state: State, params: Dict[str, Any]) -> float:
        """Compute L1 energy: λ‖x‖₁"""
        lam = params.get('lambda', 1.0)
        x = state[params['variable']]
        
        return lam * float(jnp.sum(jnp.abs(x)))
    
    def gradient(self, state: State, params: Dict[str, Any]) -> State:
        """L1 regularization is not differentiable, but subgradient exists."""
        # L1 is not differentiable at zero, so we return a subgradient
        # For practical purposes, we can return the sign function
        lam = params.get('lambda', 1.0)
        x = state[params['variable']]
        
        # Subgradient of λ‖x‖₁ is λ*sign(x) where sign(0) can be any value in [-1, 1]
        subgrad_x = lam * jnp.sign(x)
        
        return {params['variable']: subgrad_x}
    
    def prox(self, state: State, step_size: float, params: Dict[str, Any]) -> State:
        """
        Proximal operator for L1 regularization: soft-thresholding.
        
        prox_τ^g(x) where g(y) = λ‖y‖₁ is the soft-thresholding operator:
        S_λτ(x_i) = sign(x_i) * max(|x_i| - λτ, 0)
        """
        lam = params.get('lambda', 1.0)
        x = state[params['variable']]
        
        # Soft-thresholding: S_λτ(x) = sign(x) * max(|x| - λτ, 0)
        threshold = lam * step_size
        x_new = jnp.sign(x) * jnp.maximum(jnp.abs(x) - threshold, 0)
        
        return {params['variable']: x_new}
    
    def certificate_contributions(self, params: Dict[str, Any]) -> Dict[str, float]:
        """
        Certificate contributions for L1 atom.
        
        L1 regularization doesn't contribute to Lipschitz constants but affects convergence.
        """
        lam = params.get('lambda', 1.0)
        
        return {
            'lipschitz': 0.0,  # L1 doesn't contribute to gradient Lipschitz
            'eta_dd_contribution': 0.0,  # No diagonal dominance contribution
            'gamma_contribution': 0.0   # No spectral contribution (nonsmooth)
        }


class WaveletL1Atom(Atom):
    """
    Wavelet L1 regularization atom: λ‖Wx‖₁
    
    This implements L1 regularization in wavelet space for sparse recovery
    with multiscale representations. Uses TransformOp for frame-aware transforms.
    """
    
    @property
    def name(self) -> str:
        return "wavelet_l1"
    
    @property
    def form(self) -> str:
        return r"\lambda\|Wx\|_1"
    
    def energy(self, state: State, params: Dict[str, Any]) -> float:
        """Compute wavelet L1 energy: λ‖Wx‖₁"""
        from computable_flows_shim.multi.transform_op import make_transform
        
        lam = params.get('lambda', 1.0)
        transform = make_transform(
            params.get('wavelet', 'haar'),
            params.get('levels', 2),
            params.get('ndim', 1)
        )
        
        x = state[params['variable']]
        coeffs = transform.forward(x)
        
        # Sum L1 norm over all coefficient arrays
        total_l1 = 0.0
        for coeff in coeffs:
            total_l1 += float(jnp.sum(jnp.abs(coeff)))
        
        return lam * total_l1
    
    def gradient(self, state: State, params: Dict[str, Any]) -> State:
        """Compute subgradient of wavelet L1 regularization."""
        from computable_flows_shim.multi.transform_op import make_transform
        
        lam = params.get('lambda', 1.0)
        transform = make_transform(
            params.get('wavelet', 'haar'),
            params.get('levels', 2),
            params.get('ndim', 1)
        )
        
        x = state[params['variable']]
        coeffs = transform.forward(x)
        
        # Subgradient in wavelet space: λ * sign(Wx)
        subgrad_coeffs = []
        for coeff in coeffs:
            subgrad_coeffs.append(lam * jnp.sign(coeff))
        
        # Transform back to original space
        subgrad_x = transform.inverse(subgrad_coeffs)
        
        return {params['variable']: subgrad_x}
    
    def prox(self, state: State, step_size: float, params: Dict[str, Any]) -> State:
        """
        Proximal operator for wavelet L1 regularization.
        
        Solves: argmin_x λ‖Wx‖₁ + (1/(2τ))‖x - x₀‖²
        Solution: x = W^T prox_λτ( W x₀ )
        """
        from computable_flows_shim.multi.transform_op import make_transform
        
        lam = params.get('lambda', 1.0)
        transform = make_transform(
            params.get('wavelet', 'haar'),
            params.get('levels', 2),
            params.get('ndim', 1)
        )
        
        x = state[params['variable']]
        
        # Analysis: transform to wavelet space
        coeffs = transform.forward(x)
        
        # Soft-thresholding in wavelet space
        threshold = lam * step_size
        thresholded_coeffs = []
        for coeff in coeffs:
            thresholded_coeffs.append(
                jnp.sign(coeff) * jnp.maximum(jnp.abs(coeff) - threshold, 0)
            )
        
        # Synthesis: transform back to original space
        x_new = transform.inverse(thresholded_coeffs)
        
        return {params['variable']: x_new}
    
    def certificate_contributions(self, params: Dict[str, Any]) -> Dict[str, float]:
        """
        Certificate contributions for wavelet L1 atom.
        
        Wavelet transforms are frame operators with frame bounds.
        """
        from computable_flows_shim.multi.transform_op import make_transform
        
        lam = params.get('lambda', 1.0)
        transform = make_transform(
            params.get('wavelet', 'haar'),
            params.get('levels', 2),
            params.get('ndim', 1)
        )
        
        # Frame constant affects conditioning
        frame_constant = transform.c
        
        return {
            'lipschitz': 0.0,  # L1 doesn't contribute to gradient Lipschitz
            'eta_dd_contribution': 0.0,  # No diagonal dominance contribution
            'gamma_contribution': 0.0,   # No spectral contribution (nonsmooth)
            'frame_constant': frame_constant  # For W-space analysis
        }


class TVAtom(Atom):
    """
    Total Variation regularization atom: λ‖Dx‖₁
    
    This implements anisotropic total variation regularization for piecewise-constant
    signals/images. The finite difference operator D creates differences between
    neighboring elements. The proximal operator uses shrinkage on these differences.
    """
    
    @property
    def name(self) -> str:
        return "tv"
    
    @property
    def form(self) -> str:
        return r"\lambda\|Dx\|_1"
    
    def energy(self, state: State, params: Dict[str, Any]) -> float:
        """Compute TV energy: λ‖Dx‖₁"""
        lam = params.get('lambda', 1.0)
        x = state[params['variable']]
        
        # Compute finite differences (anisotropic TV)
        if x.ndim == 1:
            # 1D signal: forward differences
            diff = x[1:] - x[:-1]
        else:
            # Multi-dimensional: anisotropic differences along each axis
            diff = jnp.zeros_like(x)
            for axis in range(x.ndim):
                slices = [slice(None)] * x.ndim
                slices[axis] = slice(1, None)
                diff = diff + (x[tuple(slices)] - x[tuple(slices[:-1] + [slice(None, -1)])])**2
            diff = jnp.sqrt(diff)
        
        return lam * float(jnp.sum(jnp.abs(diff)))
    
    def gradient(self, state: State, params: Dict[str, Any]) -> State:
        """Compute subgradient of TV regularization."""
        lam = params.get('lambda', 1.0)
        x = state[params['variable']]
        
        if x.ndim == 1:
            # 1D TV subgradient
            diff = x[1:] - x[:-1]
            subgrad = jnp.zeros_like(x)
            subgrad = subgrad.at[:-1].add(-lam * jnp.sign(diff))
            subgrad = subgrad.at[1:].add(lam * jnp.sign(diff))
        else:
            # Multi-D anisotropic TV subgradient
            subgrad = jnp.zeros_like(x)
            for axis in range(x.ndim):
                # Forward differences along this axis
                slices_fwd = [slice(None)] * x.ndim
                slices_fwd[axis] = slice(1, None)
                slices_bwd = [slice(None)] * x.ndim
                slices_bwd[axis] = slice(None, -1)
                
                diff = x[tuple(slices_fwd)] - x[tuple(slices_bwd)]
                sign_diff = jnp.sign(diff)
                
                # Add to subgradient
                subgrad = subgrad.at[tuple(slices_bwd)].add(-lam * sign_diff)
                subgrad = subgrad.at[tuple(slices_fwd)].add(lam * sign_diff)
        
        return {params['variable']: subgrad}
    
    def prox(self, state: State, step_size: float, params: Dict[str, Any]) -> State:
        """
        Proximal operator for TV regularization.
        
        For 1D TV, this uses the taut-string algorithm or iterative shrinkage.
        For simplicity, we implement a basic iterative proximal method.
        """
        lam = params.get('lambda', 1.0)
        x = state[params['variable']]
        
        if x.ndim == 1:
            # 1D TV prox using iterative soft-thresholding on differences
            # This is a simplified implementation - full TV prox is more complex
            tau = step_size
            
            # Iterative proximal for TV (simplified)
            for _ in range(5):  # Few iterations for approximation
                # Compute differences
                diff = x[1:] - x[:-1]
                # Soft-threshold differences
                thresholded_diff = jnp.sign(diff) * jnp.maximum(jnp.abs(diff) - lam * tau, 0)
                # Reconstruct signal
                x = jnp.cumsum(jnp.concatenate([x[:1], thresholded_diff]))
                # Project back to maintain mean (simplified TV prox)
                x = x - jnp.mean(x) + jnp.mean(state[params['variable']])
        else:
            # Multi-D: simplified anisotropic TV prox
            # This is a very basic approximation - real TV prox needs more sophisticated methods
            tau = step_size
            for axis in range(x.ndim):
                # Apply 1D TV prox along each axis
                for _ in range(3):  # Few iterations
                    slices = [slice(None)] * x.ndim
                    slices[axis] = slice(1, None)
                    diff = x[tuple(slices)] - x[tuple([slice(None) if i != axis else slice(None, -1) for i in range(x.ndim)])]
                    thresholded_diff = jnp.sign(diff) * jnp.maximum(jnp.abs(diff) - lam * tau, 0)
                    
                    # Reconstruct along this axis (simplified)
                    cumsum_axis = jnp.cumsum(jnp.concatenate([x[tuple([slice(None) if i != axis else slice(1) for i in range(x.ndim)])], thresholded_diff]), axis=axis)
                    x = cumsum_axis
        
        return {params['variable']: x}
    
    def certificate_contributions(self, params: Dict[str, Any]) -> Dict[str, float]:
        """
        Certificate contributions for TV atom.
        
        TV regularization is nonsmooth and doesn't contribute to Lipschitz constants.
        """
        return {
            'lipschitz': 0.0,  # TV doesn't contribute to gradient Lipschitz
            'eta_dd_contribution': 0.0,  # No diagonal dominance contribution
            'gamma_contribution': 0.0   # No spectral contribution (nonsmooth)
        }



# Registry of available atoms
ATOM_REGISTRY: Dict[str, type[Atom]] = {
    'quadratic': QuadraticAtom,
    'tikhonov': TikhonovAtom,
    'l1': L1Atom,
    'wavelet_l1': WaveletL1Atom,
    'tv': TVAtom,
}


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
    if atom_type not in ATOM_REGISTRY:
        available = list(ATOM_REGISTRY.keys())
        raise ValueError(f"Unknown atom type: {atom_type}. Available: {available}")
    
    atom_class = ATOM_REGISTRY[atom_type]
    # For now, atoms don't take constructor parameters
    # In future versions, this could be extended
    return atom_class()


def register_atom(atom_type: str, atom_class: type[Atom]):
    """Register a new atom type."""
    ATOM_REGISTRY[atom_type] = atom_class