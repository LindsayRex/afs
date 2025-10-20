import toml
import os
from typing import Dict, Any

def write_manifest(
    out_dir: str,
    schema_version: int,
    flow_name: str,
    run_id: str,
    residual_details: Dict[str, Any],
    extra: Dict[str, Any] = None
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
    with open(path, "w") as f:
        toml.dump(manifest, f)
