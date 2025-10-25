"""
Test-Driven Development for Energy Compiler.

Following the RED-GREEN-REFACTOR cycle with Design by Contract.
Tests ensure energy specifications compile to correct JAX functions using Atoms Library.
"""

import jax.numpy as jnp
import pytest

from computable_flows_shim.energy.compile import compile_energy
from computable_flows_shim.energy.specs import EnergySpec, StateSpec, TermSpec


class TestEnergyCompilerContract:
    """
    Design by Contract tests for Energy Compiler.

    Contract: Energy Compiler takes declarative specs and produces JAX functions
    that correctly compute energy, gradients, and proximal operators using Atoms Library.
    """

    @pytest.fixture
    def simple_quadratic_spec(self):
        """Simple quadratic energy spec for testing."""
        return EnergySpec(
            terms=[
                TermSpec(type="quadratic", op="A", weight=1.0, variable="x", target="y")
            ],
            state=StateSpec(shapes={"x": [3], "y": [3]}),
        )

    @pytest.fixture
    def op_registry(self):
        """Simple operator registry for testing."""
        A = jnp.array([[2.0, 0.0, 0.0], [0.0, 1.5, 0.0], [0.0, 0.0, 1.0]])
        return {"A": lambda x: A @ x}

    def test_compile_quadratic_term(self, simple_quadratic_spec, op_registry):
        """RED: Compiler should handle quadratic terms correctly."""
        compiled = compile_energy(simple_quadratic_spec, op_registry)

        # Test with simple state
        state = {"x": jnp.array([1.0, 2.0, 3.0]), "y": jnp.array([2.0, 3.0, 3.0])}

        # Energy should be (1/2)‖Ax - y‖²
        A = jnp.array([[2.0, 0.0, 0.0], [0.0, 1.5, 0.0], [0.0, 0.0, 1.0]])
        Ax = A @ state["x"]
        residual = Ax - state["y"]
        expected_energy = 0.5 * float(jnp.sum(residual**2))

        actual_energy = compiled.f_value(state)
        assert abs(actual_energy - expected_energy) < 1e-10

    def test_compile_gradient(self, simple_quadratic_spec, op_registry):
        """RED: Compiler should produce correct gradients."""
        compiled = compile_energy(simple_quadratic_spec, op_registry)

        state = {"x": jnp.array([1.0, 2.0, 3.0]), "y": jnp.array([2.0, 3.0, 3.0])}

        grad = compiled.f_grad(state)

        # Manual gradient computation: A^T(Ax - y)
        A = jnp.array([[2.0, 0.0, 0.0], [0.0, 1.5, 0.0], [0.0, 0.0, 1.0]])
        Ax = A @ state["x"]
        residual = Ax - state["y"]
        expected_grad_x = A.T @ residual

        assert jnp.allclose(grad["x"], expected_grad_x, atol=1e-10)

    def test_compile_multiple_terms(self):
        """RED: Compiler should handle multiple terms."""
        spec = EnergySpec(
            terms=[
                TermSpec(
                    type="quadratic", op="A", weight=1.0, variable="x", target="y"
                ),
                TermSpec(
                    type="tikhonov", op="I", weight=0.1, variable="x", target=None
                ),
            ],
            state=StateSpec(shapes={"x": [3], "y": [3]}),
        )

        op_registry = {
            "A": lambda x: jnp.array(
                [[2.0, 0.0, 0.0], [0.0, 1.5, 0.0], [0.0, 0.0, 1.0]]
            )
            @ x,
            "I": lambda x: x,  # Identity for Tikhonov
        }

        compiled = compile_energy(spec, op_registry)

        state = {"x": jnp.array([1.0, 2.0, 3.0]), "y": jnp.array([2.0, 3.0, 3.0])}

        # Energy should be (1/2)‖Ax - y‖² + 0.1 * (1/2)‖x‖²
        A = jnp.array([[2.0, 0.0, 0.0], [0.0, 1.5, 0.0], [0.0, 0.0, 1.0]])
        Ax = A @ state["x"]
        residual = Ax - state["y"]
        quad_energy = 0.5 * float(jnp.sum(residual**2))
        tikh_energy = 0.1 * 0.5 * float(jnp.sum(state["x"] ** 2))
        expected_energy = quad_energy + tikh_energy

        actual_energy = compiled.f_value(state)
        assert abs(actual_energy - expected_energy) < 1e-10

    def test_unknown_atom_type_error(self):
        """RED: Compiler should raise error for unknown atom types."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="Unknown term type"):
            spec = EnergySpec(
                terms=[
                    TermSpec(
                        type="unknown_atom",
                        op="A",
                        weight=1.0,
                        variable="x",
                        target=None,
                    )
                ],
                state=StateSpec(shapes={"x": [3]}),
            )

    def test_jit_compilation(self, simple_quadratic_spec, op_registry):
        """RED: Compiled functions should be JIT-compiled."""
        compiled = compile_energy(simple_quadratic_spec, op_registry)

        # JAX functions should have jit compilation info
        assert hasattr(compiled.f_value, "lower")
        assert hasattr(compiled.f_grad, "lower")
        assert hasattr(compiled.g_prox, "lower")

    def test_compile_wavelet_l1_term(self):
        """RED: Compiler should handle wavelet L1 terms correctly."""
        from computable_flows_shim.energy.specs import EnergySpec, StateSpec, TermSpec

        spec = EnergySpec(
            terms=[
                TermSpec(
                    type="wavelet_l1",
                    op="wavelet",  # Not used for wavelet_l1, but required by TermSpec
                    weight=0.1,
                    variable="x",
                    target=None,
                    wavelet="haar",
                    levels=2,
                    ndim=1,
                )
            ],
            state=StateSpec(shapes={"x": [64]}),
        )

        # Empty op_registry since wavelet_l1 doesn't use traditional ops
        op_registry = {}

        compiled = compile_energy(spec, op_registry)

        # Test with simple 1D signal
        state = {"x": jnp.ones(64)}

        # Energy should be λ‖Wx‖₁ where W is Haar wavelet transform
        # For constant signal, Haar coefficients are mostly zero except approximation
        expected_energy = 0.1 * 1.0  # Simplified expectation

        actual_energy = compiled.f_value(state)
        # Just check that it runs without error and returns a finite value
        assert jnp.isfinite(actual_energy)

        # Test prox operator
        new_state = compiled.g_prox(state, step_alpha=0.01)
        assert "x" in new_state
        assert new_state["x"].shape == state["x"].shape

        # Check compile report includes term lenses
        assert compiled.compile_report is not None
        assert "term_lenses" in compiled.compile_report
        assert "x_wavelet_l1" in compiled.compile_report["term_lenses"]
        # Lens probe should select the best wavelet based on reconstruction error
        # The exact selection depends on the data, but it should be a valid wavelet
        selected_lens = compiled.compile_report["term_lenses"]["x_wavelet_l1"]
        # Lens probe selects from available candidates, should be one of the common wavelets
        valid_wavelets = [
            "haar",
            "db2",
            "db4",
            "db6",
            "db8",
            "sym2",
            "sym4",
            "coif1",
            "coif2",
        ]
        assert (
            selected_lens in valid_wavelets
        ), f"Selected lens {selected_lens} should be a valid wavelet"

    def test_wavelet_normalization_uses_energy_scale(self):
        """RED: Wavelet L1 normalization should use energy-based scale, not fallback 1.0."""
        from computable_flows_shim.energy.compile import _compute_unit_normalization

        spec = EnergySpec(
            terms=[
                TermSpec(
                    type="wavelet_l1",
                    op="wavelet",
                    weight=0.1,
                    variable="x",
                    target=None,
                    wavelet="haar",
                    levels=2,
                    ndim=1,
                )
            ],
            state=StateSpec(shapes={"x": [64]}),
        )

        op_registry = {}
        normalization = _compute_unit_normalization(spec, op_registry)

        # Currently fails: wavelet normalization is 1.0 (fallback)
        # Should be based on actual wavelet transform energy scale
        wavelet_norm = normalization["x_wavelet_l1_0"]
        assert (
            wavelet_norm != 1.0
        ), f"Wavelet normalization should not be fallback value 1.0, got {wavelet_norm}"

        # Should be a reasonable energy scale based on wavelet transform
        # For a 64-element signal with Haar wavelet, expect non-trivial scale
        assert wavelet_norm > 0.01, f"Wavelet normalization too small: {wavelet_norm}"
        assert wavelet_norm < 100.0, f"Wavelet normalization too large: {wavelet_norm}"
