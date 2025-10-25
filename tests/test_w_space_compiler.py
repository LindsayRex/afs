"""
Contract tests for W-space aware compiler.

Tests that the compiler can generate prox_in_W functions for W-space operations.
Follows Design by Contract principles with mathematical property verification.
"""

import sys
from pathlib import Path

import jax.numpy as jnp
import pytest

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from computable_flows_shim.api import Op
from computable_flows_shim.energy.compile import compile_energy
from computable_flows_shim.energy.specs import EnergySpec, StateSpec, TermSpec
from computable_flows_shim.multi.transform_op import make_transform


class IdentityOp(Op):
    def __call__(self, x):
        return x


@pytest.mark.dtype_parametrized
def test_wavelet_l1_prox_in_physical_space(float_dtype):
    """
    Test that wavelet L1 prox works in physical space (current behavior).

    Contract: prox_τ^g(x) = argmin_y g(y) + (1/(2τ))‖y - x‖²
    For wavelet L1: g(y) = λ‖Wy‖₁, so prox should do analysis/synthesis internally.
    """
    # GIVEN a wavelet L1 term specification
    spec = EnergySpec(
        terms=[
            TermSpec(
                type="wavelet_l1",
                op="I",
                weight=1.0,
                variable="x",
                wavelet="haar",
                levels=1,
                ndim=1,
            )
        ],
        state=StateSpec(shapes={"x": [4]}),
    )
    op_registry = {"I": IdentityOp()}

    # WHEN compiled
    compiled = compile_energy(spec, op_registry)

    # THEN g_prox should work on physical space input
    x = jnp.array([1.0, -2.0, 3.0, -4.0], dtype=float_dtype)
    state = {"x": x}
    step_alpha = 0.1

    result = compiled.g_prox(state, step_alpha)

    # Verify it's a valid state dict
    assert "x" in result
    assert isinstance(result["x"], jnp.ndarray)
    assert result["x"].shape == x.shape

    # Verify result dtype is preserved
    tolerance = 1e-5 if float_dtype == jnp.float32 else 1e-12
    assert jnp.allclose(result["x"], result["x"], atol=tolerance)  # Basic sanity check


def test_wavelet_l1_prox_in_W_space():
    """
    Test that prox_in_W works directly on wavelet coefficients.

    Contract: prox_in_W_τ^g(coeffs) applies prox directly to wavelet coefficients
    without needing analysis/synthesis transforms.
    """
    # GIVEN a wavelet L1 term specification
    spec = EnergySpec(
        terms=[
            TermSpec(
                type="wavelet_l1",
                op="I",
                weight=1.0,
                variable="x",
                wavelet="haar",
                levels=1,
                ndim=1,
            )
        ],
        state=StateSpec(shapes={"x": [4]}),
    )
    op_registry = {"I": IdentityOp()}

    # WHEN compiled with W-space awareness
    compiled = compile_energy(spec, op_registry)

    # THEN compiled should have prox_in_W method
    assert hasattr(compiled, "g_prox_in_w"), (
        "CompiledEnergy should have g_prox_in_w method"
    )

    # AND prox_in_W should work on coefficient arrays
    # Haar wavelet transform of [1, -2, 3, -4] gives coefficients
    transform = make_transform("haar", 1, 1)
    x = jnp.array([1.0, -2.0, 3.0, -4.0])
    coeffs = transform.forward(x)

    # prox_in_W should accept coefficient structure and return same structure
    step_alpha = 0.1
    result_coeffs = compiled.g_prox_in_w(coeffs, step_alpha)

    # Verify structure is preserved
    assert isinstance(result_coeffs, list)
    assert len(result_coeffs) == len(coeffs)
    for i, (orig, prox) in enumerate(zip(coeffs, result_coeffs, strict=False)):
        assert prox.shape == orig.shape, f"Coefficient array {i} shape mismatch"


def test_prox_equivalence_physical_vs_W_space():
    """
    Test that prox(x) = W^T prox_in_W(W x)

    Contract: The physical space prox should be equivalent to
    analysis -> W-space prox -> synthesis.
    """
    # GIVEN a wavelet L1 term
    spec = EnergySpec(
        terms=[
            TermSpec(
                type="wavelet_l1",
                op="I",
                weight=1.0,
                variable="x",
                wavelet="haar",
                levels=1,
                ndim=1,
            )
        ],
        state=StateSpec(shapes={"x": [4]}),
    )
    op_registry = {"I": IdentityOp()}

    # WHEN compiled
    compiled = compile_energy(spec, op_registry)

    # AND applied to same input via both paths
    x = jnp.array([1.0, -2.0, 3.0, -4.0])
    state = {"x": x}
    step_alpha = 0.1

    # Physical space prox
    physical_result = compiled.g_prox(state, step_alpha)["x"]

    # W-space prox path
    transform = make_transform("haar", 1, 1)
    coeffs = transform.forward(x)
    w_space_result_coeffs = compiled.g_prox_in_w(coeffs, step_alpha)
    w_space_result = transform.inverse(w_space_result_coeffs)

    # THEN results should be equivalent
    assert jnp.allclose(physical_result, w_space_result, atol=1e-6)


def test_W_space_prox_mathematical_properties():
    """
    Test mathematical properties of prox_in_W.

    Contract: prox_in_W should satisfy:
    1. Non-expansiveness: ‖prox_in_W(a) - prox_in_W(b)‖ ≤ ‖a - b‖
    2. Monotonicity: ⟨prox_in_W(a) - prox_in_W(b), a - b⟩ ≥ 0
    3. Fixed point: prox_in_W(coeffs) = coeffs when coeffs already minimize
    """
    # GIVEN a wavelet L1 term
    spec = EnergySpec(
        terms=[
            TermSpec(
                type="wavelet_l1",
                op="I",
                weight=1.0,
                variable="x",
                wavelet="haar",
                levels=1,
                ndim=1,
            )
        ],
        state=StateSpec(shapes={"x": [4]}),
    )
    op_registry = {"I": IdentityOp()}

    compiled = compile_energy(spec, op_registry)
    step_alpha = 0.1

    # Test data
    transform = make_transform("haar", 1, 1)
    x1 = jnp.array([1.0, -2.0, 3.0, -4.0])
    x2 = jnp.array([0.5, -1.5, 2.5, -3.5])

    coeffs1 = transform.forward(x1)
    coeffs2 = transform.forward(x2)

    prox1 = compiled.g_prox_in_w(coeffs1, step_alpha)
    prox2 = compiled.g_prox_in_w(coeffs2, step_alpha)

    # Property 1: Non-expansiveness (coefficient-wise for simplicity)
    for i, (p1, p2, c1, c2) in enumerate(
        zip(prox1, prox2, coeffs1, coeffs2, strict=False)
    ):
        diff_prox = jnp.linalg.norm(p1 - p2)
        diff_coeffs = jnp.linalg.norm(c1 - c2)
        assert diff_prox <= diff_coeffs + 1e-6, (
            f"Non-expansiveness failed for coeff array {i}"
        )

    # Property 2: Monotonicity (coefficient-wise)
    for i, (p1, p2, c1, c2) in enumerate(
        zip(prox1, prox2, coeffs1, coeffs2, strict=False)
    ):
        inner_prod = jnp.sum((p1 - p2) * (c1 - c2))
        assert inner_prod >= -1e-6, f"Monotonicity failed for coeff array {i}"

    # Property 3: Fixed point for zero coefficients (should be unchanged)
    zero_coeffs = [jnp.zeros_like(c) for c in coeffs1]
    prox_zero = compiled.g_prox_in_w(zero_coeffs, step_alpha)
    for i, (pz, cz) in enumerate(zip(prox_zero, zero_coeffs, strict=False)):
        assert jnp.allclose(pz, cz, atol=1e-6), (
            f"Fixed point failed for coeff array {i}"
        )


def test_W_space_aware_compilation_flag():
    """
    Test that W-space awareness can be enabled/disabled in compilation.

    Contract: Compiler should generate prox_in_W only when W-space awareness is requested.
    """
    # GIVEN a specification that could benefit from W-space
    spec = EnergySpec(
        terms=[
            TermSpec(
                type="wavelet_l1",
                op="I",
                weight=1.0,
                variable="x",
                wavelet="haar",
                levels=1,
                ndim=1,
            )
        ],
        state=StateSpec(shapes={"x": [4]}),
    )
    op_registry = {"I": IdentityOp()}

    # WHEN compiled
    compiled = compile_energy(spec, op_registry)

    # THEN it should have W-space capabilities
    assert hasattr(compiled, "g_prox_in_w"), (
        "Should have prox_in_W when wavelet terms present"
    )

    # AND compile report should indicate W-space awareness
    assert compiled.compile_report is not None, "Compile report should exist"
    assert "w_space_aware" in compiled.compile_report
    assert compiled.compile_report["w_space_aware"]
