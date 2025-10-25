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
    residual_details: dict[str, Any] | None,
    extra: dict[str, Any] | None = None,
):
    """
    Write manifest.toml with required metadata for a run.
    """
    manifest = {
        "schema_version": schema_version,
        "flow_name": flow_name,
        "run_id": run_id,
        "residual": residual_details,
    }
    if extra:
        manifest.update(extra)
    path = os.path.join(out_dir, "manifest.toml")
    if toml is not None:
        with open(path, "w") as f:
            toml.dump(manifest, f)
    else:
        # Minimal TOML serializer for our known manifest shape
        lines = []
        lines.append(f"schema_version = {int(manifest['schema_version'])}")
        lines.append(f'flow_name = "{manifest["flow_name"]}"')
        lines.append(f'run_id = "{manifest["run_id"]}"')
        lines.append("[residual]")
        residual = manifest.get("residual") or {}
        for k, v in residual.items():
            if isinstance(v, str):
                lines.append(f'{k} = "{v}"')
            elif isinstance(v, bool):
                lines.append(f"{k} = {str(v).lower()}")
            else:
                # Assume numeric
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
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
