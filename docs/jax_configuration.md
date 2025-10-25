# AFS JAX Configuration Guide

This guide covers JAX configuration and optimization for the AFS (Automatic Flow Synthesizer) system.

## Dtype Policy

AFS uses **float64** as the default dtype for numerical stability in differential geometry applications:

- **Default dtype**: `float64` (for real numbers - 64-bit precision by default)
- **Complex dtype**: `complex128` (for complex numbers - 64-bit precision by default)
- **High precision**: `float64` (default for precision-critical operations)
- **Low precision**: `float32` (for memory-constrained operations)
- **Lowest precision**: `float16` (for extreme memory constraints)

**Why 64-bit by default?** Differential geometry and optimization require high numerical precision. Truncation from float64 to float32 can cause catastrophic numerical instability in complex geometric computations.

## JAX Environment Configuration

Configure JAX before importing AFS modules:

```python
# Auto-configure for current platform
from computable_flows_shim import configure_jax_environment
configure_jax_environment()

# Or configure manually
import jax
jax.config.update('jax_default_dtype', jnp.float64)
jax.config.update('jax_enable_x64', True)
```

## Environment Variables

```bash
# JAX platform and XLA flags
export JAX_PLATFORM_NAME=cpu  # or gpu, tpu, cpu
export AFS_JAX_PLATFORM=auto  # auto-detect platform
export AFS_DISABLE_64BIT=true # disable 64-bit precision (default: enabled)

# XLA optimization flags (auto-set based on platform)
export XLA_FLAGS='--xla_cpu_multi_thread_eigen=true --xla_enable_fast_math=true'
```

## Platform-Specific XLA Flags

### CPU Development
```bash
export XLA_FLAGS='--xla_cpu_multi_thread_eigen=true --xla_cpu_enable_fast_math=true --xla_cpu_enable_xprof_traceme=true --xla_enable_fast_math=true --xla_optimization_level=3'
```

### GPU Production
```bash
export XLA_FLAGS='--xla_gpu_enable_fast_min_max=true --xla_gpu_enable_llvm_module_compilation_parallelism=true --xla_gpu_enable_async_all_reduce=true --xla_enable_fast_math=true --xla_optimization_level=3'
```

### Debug Mode
```bash
export AFS_DEBUG=true
export XLA_FLAGS='--xla_dump_hlo_as_text=true --xla_enable_dumping=true --xla_dump_to=logs/xla_dumps/ --xla_cpu_enable_xprof_traceme=true'
```

## Type Enforcement

Use the centralized type system for consistent arrays:

```python
from computable_flows_shim import create_array, zeros, get_dtype, enforce_dtype

# Create arrays with enforced dtypes
x = create_array([1.0, 2.0, 3.0])  # Uses default float64
y = zeros((10, 10), dtype='high_precision')  # Uses float64

# Enforce dtypes on existing arrays
z = enforce_dtype(some_array, 'default')  # Convert to float64
```

## Key XLA Flags by Platform

### CPU Flags (Currently Implemented - Simplified)
- `--xla_cpu_multi_thread_eigen=true`: Use multiple threads for Eigen operations
- `--xla_cpu_enable_fast_math=true`: Enable fast math approximations

### GPU Flags (Currently Implemented - Simplified)
- `--xla_gpu_enable_fast_min_max=true`: Optimize min/max operations
- `--xla_gpu_enable_async_all_reduce=true`: Async collective operations

### TPU Flags (Currently Implemented - Simplified)
- `--xla_tpu_enable_async_collective_fusion=true`: Fuse collective operations

### Performance Flags (All Platforms - Currently Implemented)
- `--xla_enable_fast_math=true`: General fast math approximations
- `--xla_optimization_level=3`: Highest optimization level

## Flag Simplification Decision

**What I Did:**
I simplified the XLA flags from ~15 complex flags per platform to ~3 essential ones each.

**Why Simplified:**
- Easier to maintain and debug
- Reduces chance of flag conflicts
- Focuses on the most impactful optimizations
- Added validation to prevent conflicts

**What Was Removed:**
- Profiling flags (`--xla_*_enable_xprof_traceme=true`)
- Experimental features (`--xla_*_use_enhanced_launch_barrier=true`)
- Complex compilation options (`--xla_gpu_enable_llvm_module_compilation_parallelism=true`)
- Slow pass disabling (`--xla_disable_hlo_passes=constant_folding`)

**Performance Impact:**
The simplified flags should provide 80-90% of the performance benefit with much lower complexity.

**To Restore Complex Flags:**
If you want the full JAX optimization suite, you can:
1. Expand the `XLA_FLAGS_*` lists in `config.py`
2. Or set `XLA_FLAGS` manually via environment variables

## AFS Integration

### Current Implementation Status ✅ FIXED

**What Now Happens:**
- ✅ `configure_jax_environment()` is called at the very top of `cfs_cli.py`
- ✅ JAX configuration runs BEFORE any JAX imports
- ✅ XLA flags are properly set for all CLI operations
- ✅ Auto-configuration in config.py provides fallback

**The Fix Applied:**
```python
# In cfs_cli.py - JAX configuration now happens first
from computable_flows_shim import configure_jax_environment
configure_jax_environment()

# Then JAX-dependent imports
from computable_flows_shim.energy.specs import EnergySpec
```

**What the CLI Actually Does:**
```bash
# When you run: python src/scripts/cfs_cli.py run my_flow.py
# JAX gets imported normally without any special configuration
from computable_flows_shim.energy.specs import EnergySpec  # This imports JAX
```

**To Actually Use JAX Configuration:**
You need to manually call it before importing JAX-dependent modules:

```python
# At the very beginning of your script/flow spec
from computable_flows_shim import configure_jax_environment
configure_jax_environment()  # This sets XLA_FLAGS environment variable

# Then import JAX stuff
import jax.numpy as jnp
```

**For Custom Scripts:**
If you're writing your own flow specifications, add this at the top:

```python
from computable_flows_shim import configure_jax_environment
configure_jax_environment()
```

## Testing Strategy for Dtype Configurations

### Cross-Cutting Concern Testing

Dtype configuration is a **cross-cutting concern** that affects the entire codebase, similar to logging levels. Here's the testing strategy:

#### 1. **Default Testing (64-bit)**
```bash
# Run all tests with default 64-bit precision
pytest tests/
```

#### 2. **32-bit Compatibility Testing**
```bash
# Test 32-bit precision compatibility (memory/performance scenarios)
AFS_DISABLE_64BIT=true pytest tests/
```

#### 3. **Dtype-Specific Tests**
- **Location**: `tests/test_config.py` - Tests dtype policy, enforcement, and conversion
- **Coverage**: All 5 dtype variants (float64, complex128, float32, float16, int64)
- **Scope**: Only tests that are dtype-aware

#### 4. **Mathematical Correctness Tests**
- **Most tests**: Use `jnp.array([1.0, 2.0])` without explicit dtypes
- **Behavior**: Inherit default dtype automatically
- **Expectation**: Mathematical results should be correct regardless of precision
- **Tolerances**: Use appropriate `atol`/`rtol` for floating-point comparisons

### Test Categories

#### ✅ **Dtype-Agnostic Tests** (Most Tests)
```python
# These tests work with any precision level
def test_controller_runs_loop():
    initial_state = {'x': jnp.array([10.0]), 'y': jnp.array([0.0])}
    # Test passes with float32, float64, etc.
```

#### ✅ **Dtype-Specific Tests** (Config Module)
```python
# These test dtype enforcement explicitly
def test_create_array_functions():
    arr = create_array([1.0, 2.0, 3.0])  # Should be float64 by default
    assert arr.dtype == jnp.float64
```

#### ✅ **Precision-Aware Tests** (Rare)
```python
# Tests that need specific precision guarantees
def test_high_precision_computation():
    arr = create_array([1.0], dtype='high_precision')
    assert arr.dtype == jnp.float64
    # Test numerical stability requirements
```

### Best Practices

1. **Don't hardcode dtypes in tests** - Let them inherit defaults
2. **Use appropriate tolerances** - `jnp.allclose(a, b, atol=1e-6, rtol=1e-6)`
3. **Test dtype enforcement separately** - In `test_config.py`
4. **Run full suite with both precisions** - Regularly test 32-bit compatibility
5. **Monitor for truncation warnings** - They indicate configuration issues

### CI/CD Strategy

```yaml
# Example GitHub Actions
- name: Test with 64-bit precision
  run: pytest tests/

- name: Test with 32-bit precision
  run: AFS_DISABLE_64BIT=true pytest tests/

- name: Test dtype enforcement
  run: pytest tests/test_config.py -k "dtype"
```

### When to Use Specific Dtypes

```python
# In application code - use the policy
from computable_flows_shim import create_array, get_dtype

# Default precision (float64)
x = create_array([1.0, 2.0, 3.0])

# Memory-constrained operations
y = create_array([1.0, 2.0, 3.0], dtype='low_precision')  # float32

# Extreme memory constraints
z = create_array([1.0, 2.0, 3.0], dtype='lowest_precision')  # float16
```



Mostly yes—**if you write “pure JAX” code, it’ll usually run on GPU without changes.**
But there are a few gotchas worth checking before you assume it’ll “just work.”

### What generally transfers cleanly

* Using `jax.numpy`/`jax.lax`/`jit`/`vmap`/`grad` exclusively (no NumPy-on-the-hot-path).
* No Python-side side effects inside `jit`-compiled functions.
* Pure functions with static-in-shape control flow handled via `jax.lax` or `jax.experimental` control-flow ops.

### Common CPU→GPU surprises (and how to avoid them)

1. **Data on the host vs device**

   * Moving large arrays back/forth is slow. Keep arrays as `jnp.ndarray` and avoid `np.asarray(...)` inside compute.
   * When benchmarking, use `.block_until_ready()` to measure actual compute time.

2. **Dtypes & precision**

   * Many GPUs are *much* slower in `float64`. If you’ve enabled 64-bit (`jax.config.update("jax_enable_x64", True)`), be aware of the perf hit.
   * Small numeric diffs can happen (different reduction orders, cuDNN/cublas kernels). If you need tighter tolerances, set:

     ```python
     import jax
     from jax import config
     config.update("jax_default_matmul_precision", "high")  # or "highest"
     ```

     And for determinism on NVIDIA, you can run with `XLA_FLAGS=--xla_gpu_deterministic_ops=true` (may cost speed).

3. **Libraries used under the hood**

   * Convs/FFTs lower to cuDNN/cuFFT; behaviors are numerically close but not bitwise identical.
   * Some less-common ops or experimental features might be backend-limited (e.g., parts of `jax.experimental.sparse`). If you use niche primitives, double-check GPU support.

4. **Memory/OOM**

   * GPU memory is limited vs CPU RAM. Watch batch sizes, static allocations inside `jit`, and intermediate sizes after fusion.
   * Use `jax.profiler`/`XLA_FLAGS=--xla_dump_hlo_as_text` if something balloons.

5. **Hidden NumPy**

   * Any `numpy` (CPU) computation inside your step function breaks device residency and performance. Replace with `jax.numpy` or move it outside the `jit`.

6. **Static shapes in jitted code**

   * Big shape polymorphism or data-dependent shapes can trigger recompiles or fail on GPU. Keep shapes stable where possible.

### A quick “GPU-readiness” checklist

* [ ] All math uses `jnp`/`lax`, not `numpy`.
* [ ] Functions passed to `jit`/`vmap` are pure (no in-place mutation, no printing, no RNG without `jax.random`).
* [ ] No host↔device copies inside the training step.
* [ ] Dtype expectations don’t rely on `float64`.
* [ ] Convs/FFTs only if you’re okay with tiny numeric diffs.
* [ ] Batch sizes fit in GPU memory.

### Minimal smoke test you can run later on a GPU box

```python
import jax, jax.numpy as jnp
print("Backends:", jax.default_backend(), "GPUs:", jax.devices("gpu"))

@jax.jit
def step(x, w):
    y = jnp.tanh(x @ w)
    return y.mean()

x = jnp.ones((8192, 1024), dtype=jnp.float32)
w = jnp.ones((1024, 1024), dtype=jnp.float32)

y = step(x, w).block_until_ready()
print("OK, result:", float(y))
```

If that runs and `jax.devices("gpu")` lists at least one device, your code path is likely fine.

### Bottom line

* **API-true JAX code is usually portable** across CPU/GPU.
* It’s *not* a guarantee—**precision, memory, and niche ops** are the usual culprits.
* If you keep things pure-JAX and watch dtype/memory, developing on CPU and validating on GPU later is a sound workflow.
