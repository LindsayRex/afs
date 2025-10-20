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


# Registry of available atoms
ATOM_REGISTRY: Dict[str, type[Atom]] = {
    'quadratic': QuadraticAtom,
    'tikhonov': TikhonovAtom,
    'l1': L1Atom,
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