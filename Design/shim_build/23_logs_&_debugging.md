# AFS SDK Logging & Debugging Infrastructure Specification

## Overview

This document specifies the logging and debugging infrastructure for the AFS (Computable Flows Shim) SDK. The system provides consistent, lightweight, and configurable logging capabilities that integrate with our Design by Contract, Test-Driven Development, and Functional Core/Imperative Shell architecture patterns.

## Core Requirements

### Functional Requirements
- **Consistent Logging**: Unified logging API across all SDK modules
- **Configurable Levels**: Standard log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- **Structured Output**: JSON-formatted logs for machine readability
- **Performance-Aware**: Minimal overhead when logging is disabled
- **Hierarchical Control**: Module-specific and global logging configuration
- **Multiple Outputs**: Console, file, and programmatic access
- **SDK-Friendly**: Easy configuration for end users without code changes

### Non-Functional Requirements
- **Lightweight**: No heavy dependencies beyond Python standard library
- **Thread-Safe**: Safe for concurrent use in JAX environments
- **JIT-Compatible**: Logging operations don't break JAX compilation
- **Configurable**: Environment variables and programmatic control
- **Observable**: Rich context for debugging mathematical operations

## Architecture Integration

### Functional Core / Imperative Shell
- **Functional Core**: Mathematical operations remain pure (no logging)
- **Imperative Shell**: Logging occurs in shell wrappers and orchestration code
- **Boundary Logging**: Log at JAX/Python boundaries for observability

### Design by Contract Integration
- Log contract violations and precondition failures
- Include contract context in log messages
- Enable/disable contract logging independently

### TDD Integration
- Test logging behavior with contract tests
- Verify appropriate logs are emitted under failure conditions
- Test log format and structured data correctness

## Implementation Design

### Core Components

#### 1. SDKLogger Configuration Class
```python
class SDKLogger:
    @staticmethod
    def configure(
        level: str = "WARNING",
        format: str = "json",
        output: str = "stderr",
        log_file: Optional[str] = None,
        enable_performance_logging: bool = False
    ):
        """Configure SDK-wide logging with validation."""
```

**Configuration Options:**
- `level`: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
- `format`: "json" (structured) or "text" (human-readable)
- `output`: "stderr", "stdout", "file", or "null"
- `log_file`: Path for file output (when output="file")
- `enable_performance_logging`: Enable timing decorators

#### 2. Module Logger Factory
```python
def get_logger(name: str) -> logging.Logger:
    """Get a configured logger for a specific module."""
    return logging.getLogger(f'afs.{name}')
```

#### 3. JSON Formatter
```python
class JSONFormatter(logging.Formatter):
    def format(self, record) -> str:
        """Format log record as JSON with structured data."""
```

#### 4. Performance Logging Decorator
```python
def log_performance(logger: logging.Logger, operation: str):
    """Decorator for timing critical operations."""
```

### File Structure

```
src/computable_flows_shim/
├── logging.py              # Core logging infrastructure
├── __init__.py            # Export configure_logging function
└── [existing modules]     # Updated to use logging
```

### Module Integration Pattern

#### Before (No Logging)
```python
def compile_energy(spec: EnergySpec) -> CompiledEnergy:
    # Pure compilation logic
    return CompiledEnergy(...)
```

#### After (With Logging)
```python
def compile_energy(spec: EnergySpec) -> CompiledEnergy:
    logger = get_logger(__name__)
    logger.debug("Starting energy compilation", extra={
        'spec_terms': len(spec.terms),
        'has_regularizer': spec.regularizer is not None
    })

    try:
        result = _compile_energy_pure(spec)  # Pure function
        logger.info("Energy compilation successful", extra={
            'compiled_terms': len(result.f_components)
        })
        return result
    except Exception as e:
        logger.error("Energy compilation failed", extra={
            'error_type': type(e).__name__,
            'spec_summary': str(spec)[:200]  # Truncated for brevity
        })
        raise
```

## Configuration Methods

### 1. Environment Variables
```bash
# Set global logging level
export AFS_LOG_LEVEL=DEBUG

# Configure output destination
export AFS_LOG_OUTPUT=file
export AFS_LOG_FILE=afs_debug.log

# Enable performance logging
export AFS_LOG_PERFORMANCE=true

# Set log format
export AFS_LOG_FORMAT=json
```

### 2. Programmatic Configuration
```python
from computable_flows_shim import configure_logging

# Basic configuration
configure_logging(level="DEBUG")

# Advanced configuration
configure_logging(
    level="INFO",
    format="json",
    output="file",
    log_file="optimization.log",
    enable_performance_logging=True
)
```

### 3. CLI Integration
```bash
# Enable debug logging for troubleshooting
afs run --log-level DEBUG --log-file debug.log

# Performance analysis
afs run --log-level INFO --log-performance
```

## Log Categories and Usage Patterns

### 1. Mathematical Operation Logging
```python
@log_performance(logger, "certificate_computation")
def estimate_eta_dd(operator, v):
    logger.debug("Computing η_dd certificate", extra={
        'operator_shape': operator.shape,
        'vector_norm': jnp.linalg.norm(v)
    })
    # ... computation
    result = compute_eta_dd(operator, v)
    logger.debug("η_dd computed", extra={
        'eta_dd_value': float(result),
        'computation_time_ms': timing_info
    })
    return result
```

### 2. Contract Violation Logging
```python
def validate_spec(spec: EnergySpec) -> bool:
    logger = get_logger(__name__)
    if not spec.terms:
        logger.error("Contract violation: EnergySpec must have at least one term", extra={
            'contract': 'EnergySpec.terms.non_empty',
            'spec_id': getattr(spec, 'id', 'unknown')
        })
        return False
    return True
```

### 3. Performance Monitoring
```python
@log_performance(logger, "optimization_step")
def run_optimization_step(state, compiled_energy):
    # Optimization logic
    pass
```

### 4. Error Context Logging
```python
try:
    result = run_certified(spec, initial_state)
except OptimizationFailure as e:
    logger.error("Optimization failed", extra={
        'failure_type': e.failure_type,
        'final_energy': e.final_energy,
        'certificate_violations': e.violations,
        'phase': e.phase,
        'iterations_completed': e.iterations
    })
    raise
```

## Structured Log Schema

### Standard Fields
```json
{
  "timestamp": "2025-10-23T14:30:15.123Z",
  "level": "DEBUG",
  "module": "afs.energy.compiler",
  "message": "Energy compilation started",
  "extra": {
    "spec_terms": 5,
    "has_regularizer": true
  }
}
```

### Performance Logging Fields
```json
{
  "timestamp": "2025-10-23T14:30:15.500Z",
  "level": "DEBUG",
  "module": "afs.fda.certificates",
  "message": "certificate_computation completed",
  "extra": {
    "duration_ms": 45.67,
    "success": true,
    "operation": "estimate_gamma_lanczos"
  }
}
```

### Error Context Fields
```json
{
  "timestamp": "2025-10-23T14:30:16.000Z",
  "level": "ERROR",
  "module": "afs.runtime.engine",
  "message": "Certificate violation detected",
  "extra": {
    "eta_dd": 1.2,
    "gamma": 0.8,
    "phase": "AMBER",
    "iteration": 42
  }
}
```

## Performance Considerations

### Lazy Evaluation
- Log level checking before expensive operations
- String formatting only when logging is enabled
- Avoid logging in hot paths unless DEBUG level

### Memory Management
- Bounded log buffers for high-frequency operations
- Structured data prevents string concatenation overhead
- File rotation for long-running processes

### JAX Compatibility
- Logging outside JIT-compiled functions
- No logging in pure functions (Functional Core)
- Log at JAX/Python boundaries

## Testing Strategy

### Unit Tests
```python
def test_logging_configuration():
    configure_logging(level="DEBUG", output="null")
    logger = get_logger("test")
    assert logger.isEnabledFor(logging.DEBUG)

def test_performance_logging_decorator():
    logger = get_logger("test")
    configure_logging(level="DEBUG", output="null")

    @log_performance(logger, "test_operation")
    def dummy_operation():
        time.sleep(0.01)
        return "result"

    result = dummy_operation()
    assert result == "result"
    # Verify log was emitted with timing data
```

### Integration Tests
- Test log aggregation across modules
- Verify structured data in JSON output
- Test environment variable configuration
- Validate performance impact when disabled

### TDD Approach
- Write failing tests for logging behavior first
- Test error conditions and context logging
- Verify log format compliance
- Test configuration edge cases

## Migration Strategy

### Phase 1: Infrastructure Setup
1. Create `logging.py` module
2. Add basic configuration to `__init__.py`
3. Update CLI to accept logging options

### Phase 2: Module Integration
1. Add logging to core modules (energy, runtime, controller)
2. Implement performance decorators on expensive operations
3. Add contract violation logging

### Phase 3: Advanced Features
1. Add FDA certificate logging
2. Implement telemetry integration
3. Add log analysis utilities

### Phase 4: Optimization
1. Performance tuning based on profiling
2. Memory usage optimization
3. Documentation and examples

## Usage Examples

### SDK User: Basic Troubleshooting
```python
import computable_flows_shim as cfs

# Enable debug logging
cfs.configure_logging(level="DEBUG")

# Run optimization with detailed logging
result = cfs.run_certified(spec, initial_state)
```

### SDK User: Performance Analysis
```python
import computable_flows_shim as cfs

# Enable performance logging
cfs.configure_logging(
    level="INFO",
    enable_performance_logging=True,
    log_file="performance.log"
)

result = cfs.run_certified(spec, initial_state)
# Check performance.log for timing data
```

### Developer: Module-Specific Debugging
```python
import logging
import computable_flows_shim as cfs

# Enable debug for specific module
cfs.configure_logging(level="DEBUG")
logging.getLogger('afs.fda.certificates').setLevel(logging.DEBUG)

# Only FDA certificate operations will log at DEBUG level
result = cfs.run_certified(spec, initial_state)
```

## Security Considerations

- No sensitive data in default logs
- Configurable log redaction for privacy
- Safe serialization of complex objects
- File permission handling for log files

## Future Extensions

- Distributed logging for multi-GPU setups
- Log aggregation and analysis tools
- Integration with external monitoring systems
- Log-based performance regression detection
- Automated log analysis for debugging assistance

## Conclusion

This logging infrastructure provides the consistent, lightweight debugging capability needed for the AFS SDK while maintaining compatibility with our architectural patterns. The system enables effective troubleshooting without compromising performance or purity of the functional core.</content>
<parameter name="filePath">j:\Google Drive\Software\afs\Design\shim_build\23_logs_&_debugging.md
