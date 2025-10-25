"""
Tests for compiler CompileReport (Builder rehearsal contract).
"""

import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from computable_flows_shim.energy.compile import compile_energy
from computable_flows_shim.energy.specs import EnergySpec, StateSpec, TermSpec


def test_compile_writes_compile_report():
    """
    Builder Rehearsal Contract (RED): The compiler must expose a CompileReport
    containing a `unit_normalization_table` (per-term statistics) and `lens_name`.
    """
    spec = EnergySpec(
        terms=[
            TermSpec(type="quadratic", op="I", weight=1.0, variable="x", target="y"),
            TermSpec(type="l1", op="W", weight=0.1, variable="x", target=None),
        ],
        state=StateSpec(shapes={"x": [4], "y": [4]}),
    )

    class IdentityOp:
        def __call__(self, x):
            return x

    op_registry = {"I": IdentityOp(), "W": IdentityOp()}

    compiled = compile_energy(spec, op_registry)

    # RED: Compiler should produce a compile report. Test fails until implemented.
    assert hasattr(
        compiled, "compile_report"
    ), "Compiler must attach a 'compile_report' to CompiledEnergy"
    report = compiled.compile_report
    assert report is not None, "CompileReport must not be None"
    assert (
        "unit_normalization_table" in report
    ), "CompileReport must include 'unit_normalization_table'"
    assert "lens_name" in report, "CompileReport must include 'lens_name'"
