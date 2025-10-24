# New Component Quick Start Guide

## Overview

This guide provides standardized templates and checklists for creating new components in the AFS (Automatic Flow Synthesizer) project. It consolidates the essential requirements from `copilot-instructions.md`, hygiene specifications, and design patterns.

**Key Principle:** Design by Contract (DbC) is primary for mathematical components; Test-Driven Development (TDD) is primary for infrastructure. Both ensure correctness and prevent regressions.

## When to Use Each Methodology

### Design by Contract (DbC) - Primary for Mathematical Components
**Use for:** Energy functions, gradient computations, prox operators, runtime primitives, FDA certificates, optimization algorithms

**Why:** Mathematical correctness requires formal verification of preconditions, postconditions, and invariants. DbC provides computational proofs of mathematical properties.

**Pattern:**
1. **Define Contract:** Specify mathematical properties (preconditions, postconditions, invariants)
2. **Write Contract Tests:** Tests that verify mathematical correctness
3. **Implement:** Code that satisfies the contract
4. **Verify:** Tests prove the implementation meets mathematical requirements

### Test-Driven Development (TDD) - Primary for Infrastructure
**Use for:** CLI tools, logging, configuration, file I/O, UI components, data serialization

**Why:** Infrastructure requires behavioral verification. TDD ensures components behave correctly in all scenarios.

**Pattern:**
1. **Write Failing Test:** Define expected behavior
2. **Implement Minimal Code:** Make test pass
3. **Refactor:** Clean up while maintaining test coverage
4. **Verify:** All tests pass

## Component Types & Templates

### 1. Mathematical Function Component

**Use Case:** Energy functions, gradient computations, prox operators, certificate calculations

**Template:**
```python
"""
Mathematical component: [Brief description]

Contract:
- Preconditions: [Mathematical requirements on inputs]
- Postconditions: [Mathematical guarantees on outputs]
- Invariants: [Properties that must hold throughout execution]
"""

import jax.numpy as jnp
from computable_flows_shim.core import numerical_stability_check
from computable_flows_shim import get_logger
from pydantic import BaseModel, Field
from typing import Optional

# Data structures (Pydantic models)
class [ComponentName]Config(BaseModel):
    """Configuration for [component description]."""
    name: str = Field(..., min_length=1, max_length=100)
    dtype: str = Field("float64", pattern="^(float32|float64)$")
    # Add component-specific fields

class [ComponentName]Result(BaseModel):
    """Result structure for [component description]."""
    value: float
    metadata: Optional[dict] = None
    # Add component-specific result fields

# Core mathematical function (Functional Core)
@numerical_stability_check
def [component_name](
    input_data: jnp.ndarray,
    config: [ComponentName]Config
) -> [ComponentName]Result:
    """
    [Detailed mathematical description]

    Args:
        input_data: [Shape and dtype requirements]
        config: Configuration parameters

    Returns:
        [ComponentName]Result: [What is computed and guaranteed]

    Contract:
    - Pre: [Mathematical preconditions]
    - Post: [Mathematical postconditions]
    - Invariant: [Invariants maintained]
    """
    logger = get_logger(__name__)

    # Input validation (Imperative Shell)
    if not jnp.isarray(input_data):
        raise ValueError("input_data must be a JAX array")

    # Mathematical computation (Functional Core)
    # ... pure JAX operations only ...

    result = [ComponentName]Result(
        value=computed_value,
        metadata={"computation_time": "tracked_if_needed"}
    )

    return result
```

**Checklist:**
- [ ] Uses `@numerical_stability_check` decorator
- [ ] All operations use `jax.numpy` (never `numpy`)
- [ ] Input/output validation in Imperative Shell
- [ ] Pure mathematical logic in Functional Core
- [ ] Pydantic models for all data structures
- [ ] Comprehensive docstring with contract
- [ ] Logger integration for observability

### 2. Infrastructure Component

**Use Case:** CLI tools, configuration management, file operations, UI components

**Template:**
```python
"""
Infrastructure component: [Brief description]

This component handles [responsibility] using TDD methodology.
"""

import argparse
from pathlib import Path
from typing import Optional
from computable_flows_shim import get_logger, configure_logging

class [ComponentName]:
    """[Component description]."""

    def __init__(self, config: Optional[dict] = None):
        self.logger = get_logger(__name__)
        self.config = config or {}

    def [main_method](self, *args, **kwargs):
        """[Method description]."""
        self.logger.debug(f"Starting {self.__class__.__name__}.[main_method]")
        try:
            # Implementation
            result = self._[main_method]_impl(*args, **kwargs)
            self.logger.info(f"{self.__class__.__name__}.[main_method] completed successfully")
            return result
        except Exception as e:
            self.logger.error(f"{self.__class__.__name__}.[main_method] failed: {e}")
            raise

    def _[main_method]_impl(self, *args, **kwargs):
        """Private implementation."""
        # Core logic here
        pass

def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="[Component description]")
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                       default='WARNING', help='Set logging level')
    parser.add_argument('--log-output', choices=['stderr', 'stdout', 'file', 'null'],
                       default='stderr', help='Log output destination')
    parser.add_argument('--log-file', help='Log file path (when output=file)')

    # Component-specific arguments
    parser.add_argument('[component_args]')

    args = parser.parse_args()

    # Configure logging
    configure_logging(
        level=args.log_level,
        output=args.log_output,
        log_file=getattr(args, 'log_file', None)
    )

    # Run component
    component = [ComponentName]()
    component.[main_method](args)

if __name__ == '__main__':
    main()
```

**Checklist:**
- [ ] TDD approach: Tests written first
- [ ] Proper error handling and logging
- [ ] CLI integration with logging options
- [ ] Configuration management
- [ ] Comprehensive test coverage

### 3. Telemetry-Integrated Component

**Use Case:** Components that need systematic mathematical operation tracking

**Template:**
```python
"""
Telemetry-integrated component: [Brief description]

This component integrates with the telemetry system for systematic
tracking of mathematical operations and performance metrics.
"""

from computable_flows_shim.telemetry import TelemetryManager
from computable_flows_shim import get_logger

class [ComponentName]:
    """[Component description with telemetry integration]."""

    def __init__(self, telemetry_manager: Optional[TelemetryManager] = None):
        self.logger = get_logger(__name__)
        self.telemetry = telemetry_manager or TelemetryManager()

    def [mathematical_operation](self, *args, **kwargs):
        """[Operation description]."""
        start_time = time.perf_counter()

        try:
            # Pre-operation telemetry
            self.telemetry.record_event(
                event_type="operation_start",
                data={
                    "operation": "[operation_name]",
                    "input_summary": self._summarize_inputs(*args, **kwargs)
                }
            )

            result = self._[mathematical_operation]_impl(*args, **kwargs)

            # Post-operation telemetry
            duration = time.perf_counter() - start_time
            self.telemetry.record_sample(
                run_id=self._get_current_run_id(),
                phase="COMPUTING",  # or appropriate phase
                iter=self._get_current_iteration(),
                t_wall_ms=duration * 1000,
                E=getattr(result, 'energy', None),
                grad_norm=getattr(result, 'gradient_norm', None),
                # ... other telemetry fields
                phi_residual=getattr(result, 'physics_residual', None),
                invariant_drift_max=getattr(result, 'invariant_drift', None)
            )

            return result

        except Exception as e:
            # Error telemetry
            self.telemetry.record_event(
                event_type="operation_error",
                data={
                    "operation": "[operation_name]",
                    "error": str(e),
                    "duration_ms": (time.perf_counter() - start_time) * 1000
                }
            )
            raise
```

**Checklist:**
- [ ] TelemetryManager integration
- [ ] Pre/post-operation event recording
- [ ] Sample recording with mathematical metrics
- [ ] Error event tracking
- [ ] Proper run_id and iteration tracking

## Testing Patterns

### DbC Testing (Mathematical Components)
```python
import pytest
import jax.numpy as jnp
from hypothesis import given, strategies as st

class Test[ComponentName]:
    """Design by Contract tests for [ComponentName]."""

    def test_preconditions(self):
        """Test precondition violations are caught."""
        # Test invalid inputs are rejected
        pass

    def test_postconditions(self):
        """Test postcondition guarantees hold."""
        # Test mathematical properties of outputs
        pass

    def test_invariants(self):
        """Test invariants are maintained."""
        # Test properties that must hold throughout execution
        pass

    @given(st.floats(min_value=-1e6, max_value=1e6))
    def test_numerical_stability(self, value):
        """Test numerical stability across input ranges."""
        # Test with extreme values
        pass

    def test_dtype_consistency(self):
        """Test consistent dtype handling."""
        # Test float32 vs float64 behavior
        pass
```

### TDD Testing (Infrastructure Components)
```python
import pytest
from unittest.mock import patch, MagicMock

class Test[ComponentName]:
    """TDD tests for [ComponentName]."""

    def test_initialization(self):
        """Test component initializes correctly."""
        component = [ComponentName]()
        assert component is not None

    def test_main_functionality(self):
        """Test core functionality works."""
        component = [ComponentName]()
        result = component.[main_method](test_input)
        assert result is not None

    def test_error_handling(self):
        """Test error conditions are handled."""
        component = [ComponentName]()
        with pytest.raises(ExpectedException):
            component.[main_method](invalid_input)

    def test_logging_integration(self):
        """Test logging is properly configured."""
        with patch('computable_flows_shim.get_logger') as mock_logger:
            component = [ComponentName]()
            # Verify logger usage
            pass

    def test_cli_integration(self):
        """Test CLI interface works."""
        # Test argument parsing and execution
        pass
```

## JAX Configuration Checklist

### For New Components:
- [ ] Call `configure_jax_environment()` before JAX imports
- [ ] Use `float64` as default dtype for numerical stability
- [ ] Handle platform-specific XLA flags appropriately
- [ ] Test on both CPU and GPU platforms

### Environment Variables:
```bash
# Set these in development/CI
export JAX_PLATFORM_NAME=cpu  # or gpu, tpu
export AFS_JAX_PLATFORM=auto
export AFS_DISABLE_64BIT=false  # Use float64 by default
```

## Integration Checklist

### Pre-Commit Verification:
- [ ] JAX functions used throughout (no `numpy`)
- [ ] `@numerical_stability_check` on all math functions
- [ ] Pydantic models for all data structures
- [ ] Functional Core/Imperative Shell pattern followed
- [ ] Telemetry integration for mathematical operations
- [ ] CLI logging configuration
- [ ] Comprehensive test coverage (DbC for math, TDD for infra)
- [ ] Documentation with contract specifications

## Reference Documents

### Core Specifications:
- **`copilot-instructions.md`**: Mandatory development rules
- **`Design/shim_build/03b_energy_spec_hygiene_spec.md`**: Hygiene requirements
- **`Design/shim_build/17_design_pattern.md`**: Functional Core/Imperative Shell
- **`Design/shim_build/21_telematry.md`**: Telemetry field specifications
- **`Design/shim_build/23_logs_&_debugging.md`**: Logging infrastructure

### Functional Specifications:
- **`Design/shim_build/01_shim_overview_architecture.md`**: System architecture
- **`Design/shim_build/02_primitives_operator_api.md`**: Primitive operations
- **`Design/shim_build/05_fda_certificates.md`**: Certificate mathematics
- **`Design/shim_build/15_schema.md`**: Data schemas

### Examples:
- **`src/computable_flows_shim/logging.py`**: Logging infrastructure
- **`src/scripts/cfs_cli.py`**: CLI integration
- **`src/computable_flows_shim/controller.py`**: Controller pattern
- **`tests/test_*.py`**: Testing patterns

## Quick Decision Tree

```
New Component Needed?
├── Mathematical function? (energy, gradients, certificates)
│   ├── Use DbC methodology
│   ├── Apply mathematical function template
│   └── Focus on contract verification
│
├── Infrastructure component? (CLI, config, I/O)
│   ├── Use TDD methodology
│   ├── Apply infrastructure template
│   └── Focus on behavioral verification
│
└── Needs telemetry integration?
    ├── Apply telemetry template
    ├── Integrate TelemetryManager
    └── Record mathematical operations
```

This guide ensures consistent, correct, and maintainable component development across the AFS project.</content>
<parameter name="filePath">j:\Google Drive\Software\afs\Design\shim_build\24_new_component_quick_start.md