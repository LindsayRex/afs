# AFS: Automatic Flow Synthesizer

This repository contains the engine for a novel computational paradigm: casting computation as a physical process of energy minimization over a state space. This is a "physics-first" approach, where the physics of a problem directly defines a transparent, stable, and performant algorithm.

The ultimate goal is to create an **Automatic Flow Synthesizer (AFS)**, an AI that can automatically discover the optimal computational flow for a given business problem.

## The Core Engine: The Computable Flow Shim

The heart of this project is the **Computable Flow Shim**, a small, reliable, and reusable runtime engine built on JAX. It is not a complete application, but a "Functional Core" that the AFS will drive.

The Shim is built on the composition of five primitive, continuous-time dynamical flows:
1.  **Dissipative (`F_Dis`):** Gradient descent, for minimizing energy.
2.  **Projective (`F_Proj`):** Proximal operators, for enforcing constraints.
3.  **Conservative (`F_Con`):** Symplectic integrators, for preserving physical quantities.
4.  **Multiscale (`F_Multi`):** Wavelet transforms, for representing information efficiently.
5.  **Stochastic (`F_Ann`):** Langevin dynamics, for exploration.

By composing these primitives, we can construct algorithms to solve a vast class of optimization problems.

## Development Methodology

This project is built with a strict, verifiable methodology designed for correctness and collaboration with AI agents. Any contributor (human or AI) **must** adhere to these principles.

1.  **Architectural Pattern:** We use a **Functional Core, Imperative Shell** pattern. All mathematical logic is pure and testable; all side effects are isolated.
    *   **See:** `Design/shim_build/17_design_pattern.md`

2.  **Workflow:** We use **Test-Driven Development (TDD)**. We write a failing test first, then write the code to make it pass.
    *   **See:** `copilot-instructions.md`

3.  **Documentation:** We maintain a **QA Log** for every component built, providing a clear audit trail.
    *   **See:** `qa_logs/`

## Getting Started

1.  **Understand the Math:** The foundational mathematics are described in `background/On_Compitable_Flows_v2.1.md`.
2.  **Understand the Architecture:** The Shim's architecture and naming conventions are in `Design/shim_build/`.
3.  **Follow the Rules:** Read and adhere to the `copilot-instructions.md`.

To set up the environment:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest -q
```

## JAX Configuration and Type System

The AFS system uses JAX for high-performance numerical computing with a strict type system and configuration policy.

### Dtype Policy
All AFS computations use **float64** as the default dtype for numerical stability in differential geometry:

- **Default dtype**: `float64` (for real numbers - numerical stability)
- **Complex dtype**: `complex128` (for complex numbers)
- **High precision**: `float64` (default - for when precision is critical)
- **Low precision**: `float32` (for memory-constrained operations)
- **Lowest precision**: `float16` (for extreme memory constraints)

### JAX Environment Configuration
Configure JAX before importing AFS modules:

```python
# Auto-configure for current platform
from computable_flows_shim import configure_jax_environment
configure_jax_environment()

# Or configure manually
import jax
jax.config.update('jax_default_dtype', jnp.float64)
```

### Environment Variables
```bash
# JAX platform and XLA flags
export JAX_PLATFORM_NAME=cpu  # or gpu, tpu
export AFS_JAX_PLATFORM=auto  # auto-detect platform
export AFS_DISABLE_64BIT=true # use float32 instead of default float64

# XLA optimization flags (auto-set based on platform)
export XLA_FLAGS='--xla_cpu_multi_thread_eigen=true --xla_enable_fast_math=true'
```

### Type Enforcement
Use the centralized type system for consistent arrays:

```python
from computable_flows_shim import create_array, zeros, get_dtype, enforce_dtype

# Create arrays with enforced dtypes
x = create_array([1.0, 2.0, 3.0])  # Uses default float64
y = zeros((10, 10), dtype='low_precision')  # Uses float32

# Enforce dtypes on existing arrays
z = enforce_dtype(some_array, 'default')  # Convert to float64
```

### Platform-Specific XLA Flags

#### CPU Development
```bash
export XLA_FLAGS='--xla_cpu_multi_thread_eigen=true --xla_cpu_enable_fast_math=true --xla_cpu_enable_xprof_traceme=true --xla_enable_fast_math=true --xla_optimization_level=3'
```

#### GPU Production
```bash
export XLA_FLAGS='--xla_gpu_enable_fast_min_max=true --xla_gpu_enable_llvm_module_compilation_parallelism=true --xla_gpu_enable_async_all_reduce=true --xla_enable_fast_math=true --xla_optimization_level=3'
```

#### Debug Mode
```bash
export AFS_DEBUG=true
export XLA_FLAGS='--xla_dump_hlo_as_text=true --xla_enable_dumping=true --xla_dump_to=logs/xla_dumps/'
```

## Launching the Flow Dynamics HUD

The AFS Flow Dynamics HUD provides real-time visualization of optimization flows, including sparsity monitoring and system health.

### Prerequisites
- Node.js and npm (for web dependencies)
- Python (for local web server)

### Quick Start
```bash
# Install web dependencies
cd src/ux
npm install

# Start the dashboard server
npm run dev
# or manually:
# python -m http.server 8000

# Open in browser: http://localhost:8000/sparsity_hud_demo.html
```

### Features
- **Sparsity Visualization**: Real-time monitoring of solution compression (0.0 = sparse, 1.0 = dense)
- **Color-coded Feedback**: Blue (good compression) → Green (balanced) → Red (needs regularization)
- **Interactive Animations**: Compression waves, elastic effects, and smooth transitions
- **Health Monitoring**: System status based on optimization stability and sparsity levels
- **Flow Selection**: Switch between different optimization flows by name for comparative analysis
- **Theme & Sound Controls**: Customizable interface with dark/light themes and optional sound effects

The dashboard integrates with the telemetry system to provide visual feedback during flow execution.

## Local Documentation Archive
This repository contains a local archive of the documentation for its key dependencies. This ensures that development is based on a stable, version-specific set of APIs. The documentation can be found in the `archive/` directory.

Key packages include:
- **JAX:** `archive/jax-docs-archive/`
- **Optax:** `archive/optax-docs/`
- **Orbax:** `archive/orbax-docs/`
- **Jax-Wavelet-Toolbox (jaxwt):** `archive/jaxwt-docs/`
- **NetworkX:** `archive/networkx-docs/`
- **DuckDB:** `archive/duckdb-docs/`
- **PyArrow:** `archive/pyarrow-docs/`
- **frozendict:** `archive/frozendict-docs/`
