
# Naming Conventions & Module Layout (Shim)

---
**Canonical Package List:**
All file/module naming and layout must reflect the official package requirements listed in the overview doc above.

This doc defines canonical file/module names, a ports-and-adapters layout, and conventions for naming that make the Shim discoverable and maintainable.

Goals:
- File names are self-descriptive and short.
- Module names map to directories and packages.
- Public API entry-points are clearly labeled and discoverable.
- Ports & adapters pattern maximizes backend+lib substitution.

1) High-level module layout (packages)

### Registry Path Constants
Define registry entry-point paths in a single file for consistency:

```python
# src/computable_flows_shim/registry_paths.py
TRANSFORM_REGISTRY_PATH = "computable_flows_shim.transforms"
MANIFOLD_REGISTRY_PATH  = "computable_flows_shim.manifolds"
OP_REGISTRY_PATH        = "computable_flows_shim.ops"
```

Reference these constants in all registry lookups to prevent drift.

- `src/computable_flows_shim/`
  - `__init__.py` (public exports)
  - `api.py` (one-call happy path: `run`, `compile`, `run_certified`)
  - `controller.py` (RED/AMBER/GREEN automaton)
  - `telemetry.py` (ParquetFlightRecorder + TelemetrySink)
  - `runtime/` (engine + primitives)
    - `__init__.py`
    - `step.py` (run_flow_step)
    - `primitives.py` (F_Dis, F_Proj, F_Multi, F_Con)
  - `energy/` (compiler & IR)
    - `compile.py` (compile_energy)
    - `specs.py` (EnergySpec dataclasses)
  - `multi/` (multiscale transforms)
    - `__init__.py`
    - `wavelets.py` (TransformOp, WaveletOp wrappers)
  - `fda/` (certificates)
    - `__init__.py`
    - `certificates.py` (eta/gamma/LW_apply)
  - `tuner/` (gap dial)
    - `__init__.py`
    - `auto.py` (run_with_auto_tuner)
  - `manifolds/` (registry and adapters)
    - `__init__.py`
    - `manifolds.py` (Manifold interface)
  - `ops/` (adapters for linear ops)
    - `__init__.py`
    - `fft.py`, `warp.py`, `fft_channel.py` (named per op)
  - `cli/` (flows commands)
    - `__init__.py`
    - `hud.py`, `run.py`, `cert.py`, `viz.py`
    - CLI entry points registered in `pyproject.toml`:
      ```toml
      [project.scripts]
      cf-run = "computable_flows_shim.cli.run:main"
      cf-hud = "computable_flows_shim.cli.hud:main"
      ```
      This ensures reproducible command names across environments.
  - `serialization/` (manifest, run load/save)
  - `backends/` (JAX implementation and potential adapters)
    - `jax_backend.py`
    - `torch_backend.py` (future)

2) File naming rules (human-readable)
- Use `snake_case.py` for module names.
- For modules that expose a single public class, name the file after the class (e.g., `wavelet_op.py` for `WaveletOp`).
- Avoid numeric suffixes (no `db4` files); use descriptive `wavelets.py` and parameterize.

3) API and Registry naming
- Expose registries explicitly: `transform_registry`, `manifold_registry`, `op_registry`.
- Registry items provide `register_transform(name, constructor)` and `get_transform(name)`.

4) Ports & Adapters
- Ports: stable internal interfaces (e.g., `TransformOp` dataclass, `Manifold` Protocol, `TelemetrySink` Protocol).
- Adapters: backend or library-specific wrappers live under `backends/` or `adapters/` (e.g., `adapters/jaxwt_adapter.py` or `backends/jax_wavelet_adapter.py`).

5) Naming consistency for key concepts
- Use `eta_dd` and `gamma` as the canonical variable names for certificates.
- Use `gap_dial` for the primary global sparsity dial; per-scale `gap_dial_scales` or `lambda_j` for per-scale.
- `Flight Recorder` -> `telemetry` (module `telemetry.py`) with `ParquetFlightRecorder`.
- `Flight Controller` -> `controller` (module `controller.py`) with `run_certified` API.

6) Dtype & float policy
- Global config key: `dtype` ("float32" | "float64").
- Expose helper: `get_global_dtype()` and `set_global_dtype()` in `api.py`.

7) Examples and configs
- All examples and user specs must be Python DSL files: `examples/hvac.py`, `examples/rf.py`.
- Run manifests remain TOML.

8) Requirements & extras
- `requirements.txt` should list core + optional extras: `jax[jaxlib]`, `jaxwt` (wavelets), `riemax`/`rieoptax` (manifolds), `duckdb`, `pyarrow`, `pandas`.

9) Next steps
- Add doc `docs/shim_build/11_naming_and_layout.md` (this file).
- Refactor docs to use these canonical names and link to registries.

## Doc Build Linkage
For MkDocs auto-indexing, add:

```
docs/
  shim_build/
    index.md  (links to all spec docs)
```

This enables full spec navigation and distribution.

Notes:
- This naming scheme avoids magic numbers and makes files self-explanatory.
- Adopt ports & adapters to keep the Shim agnostic of specific libs (e.g., jaxwt) and make it easy to swap adapters.
