import os
from typing import Any

# Prefer 'toml' if available, otherwise fall back to a tiny serializer for our
# simple manifest structure so runtime does not require an extra dependency.
try:
    import toml  # type: ignore
except ImportError:
    toml = None


def write_manifest(
    out_dir: str,
    schema_version: int,
    flow_name: str,
    run_id: str,
    dtype: str,
    lens_name: str | None = None,
    unit_normalization_table: dict[str, Any] | None = None,
    invariants_present: bool = False,
    redact_artifacts: bool = False,
    versions: dict[str, str] | None = None,
    shapes: dict[str, Any] | None = None,
    frame_type: str | None = None,
    gates: dict[str, Any] | None = None,
    budgets: dict[str, Any] | None = None,
    seeds: dict[str, Any] | None = None,
    residual_details: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
):
    """
    Write manifest.toml with complete metadata for a run.

    Args:
        out_dir: Directory to write manifest.toml
        schema_version: Telemetry schema version
        flow_name: User-visible flow/spec name
        run_id: Unique run identifier
        dtype: Global dtype used (e.g., 'float32', 'float64')
        lens_name: Selected transform name
        unit_normalization_table: Per-term RMS/MAD normalization values
        invariants_present: Whether invariants are declared and checkers exist
        redact_artifacts: Whether to suppress large/sensitive artifacts
        versions: Version information for components
        shapes: Shape information for arrays/states
        frame_type: Transform frame type ('unitary', 'tight', 'general')
        gates: Certificate gates and thresholds
        budgets: Resource budgets (iterations, time, etc.)
        seeds: Random seeds for reproducibility
        residual_details: Residual computation details
        extra: Additional metadata
    """
    manifest = {
        "schema_version": schema_version,
        "flow_name": flow_name,
        "run_id": run_id,
        "dtype": dtype,
        "invariants_present": invariants_present,
        "redact_artifacts": redact_artifacts,
    }

    # Optional fields
    if lens_name is not None:
        manifest["lens_name"] = lens_name
    if unit_normalization_table is not None:
        manifest["unit_normalization_table"] = unit_normalization_table
    if versions is not None:
        manifest["versions"] = versions
    if shapes is not None:
        manifest["shapes"] = shapes
    if frame_type is not None:
        manifest["frame_type"] = frame_type
    if gates is not None:
        manifest["gates"] = gates
    if budgets is not None:
        manifest["budgets"] = budgets
    if seeds is not None:
        manifest["seeds"] = seeds
    if residual_details is not None:
        manifest["residual"] = residual_details
    if extra:
        manifest.update(extra)

    path = os.path.join(out_dir, "manifest.toml")
    if toml is not None:
        with open(path, "w", encoding="utf-8") as f:
            toml.dump(manifest, f)
    else:
        # Minimal TOML serializer for our known manifest shape
        lines = []
        lines.append(f"schema_version = {int(manifest['schema_version'])}")
        lines.append(f'flow_name = "{manifest["flow_name"]}"')
        lines.append(f'run_id = "{manifest["run_id"]}"')
        lines.append(f'dtype = "{manifest["dtype"]}"')
        lines.append(
            f"invariants_present = {str(manifest['invariants_present']).lower()}"
        )
        lines.append(f"redact_artifacts = {str(manifest['redact_artifacts']).lower()}")

        # Optional fields
        if "lens_name" in manifest:
            lines.append(f'lens_name = "{manifest["lens_name"]}"')
        if "unit_normalization_table" in manifest:
            lines.append("[unit_normalization_table]")
            for k, v in manifest["unit_normalization_table"].items():
                lines.append(f"{k} = {v}")
        if "versions" in manifest:
            lines.append("[versions]")
            for k, v in manifest["versions"].items():
                lines.append(f'{k} = "{v}"')
        if "shapes" in manifest:
            lines.append("[shapes]")
            for k, v in manifest["shapes"].items():
                if isinstance(v, list):
                    lines.append(f"{k} = {v}")
                else:
                    lines.append(f'{k} = "{v}"')
        if "frame_type" in manifest:
            lines.append(f'frame_type = "{manifest["frame_type"]}"')
        if "gates" in manifest:
            lines.append("[gates]")
            for k, v in manifest["gates"].items():
                lines.append(f"{k} = {v}")
        if "budgets" in manifest:
            lines.append("[budgets]")
            for k, v in manifest["budgets"].items():
                lines.append(f"{k} = {v}")
        if "seeds" in manifest:
            lines.append("[seeds]")
            for k, v in manifest["seeds"].items():
                lines.append(f"{k} = {v}")
        if "residual" in manifest:
            lines.append("[residual]")
            for k, v in manifest["residual"].items():
                if isinstance(v, str):
                    lines.append(f'{k} = "{v}"')
                elif isinstance(v, bool):
                    lines.append(f"{k} = {str(v).lower()}")
                else:
                    lines.append(f"{k} = {v}")
        if extra:
            lines.append("[extra]")
            for k, v in extra.items():
                if isinstance(v, str):
                    lines.append(f'{k} = "{v}"')
                elif isinstance(v, bool):
                    lines.append(f"{k} = {str(v).lower()}")
                else:
                    lines.append(f"{k} = {v}")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
