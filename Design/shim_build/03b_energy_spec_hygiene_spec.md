# 03b_energy_spec_hygiene_spec.md

# Energy Specification Hygiene and Validation

## Overview

This specification defines input validation and numerical safety requirements for EnergySpec and related components in the AFS SDK. It balances SDK flexibility (allowing creative problem-solving) with safety (preventing catastrophic failures) using a layered validation approach.

## Validation Philosophy

### Core Principles
- **Functional Core, Imperative Shell**: Validation resides in the Imperative Shell; the Functional Core remains pure mathematical functions
- **80/20 Rule**: Basic validation catches 80% of user errors with 20% of implementation effort
- **Configurable Safety**: Users control validation strictness based on their expertise level
- **Fail Fast, Guide Well**: Clear error messages with actionable suggestions
- **SDK Flexibility**: Allow experimentation while preventing obvious mistakes

### Design Constraints
- **JAX Dtype Consistency**: All validation respects JAX dtype enforcement (`float32`/`float64`)
- **Numerical Stability**: NaN/Inf detection prevents silent failures
- **Composable**: Validation doesn't break SDK building-block nature
- **Performance**: Minimal overhead in default mode

## Validation Architecture

### Layered Validation Levels

#### Level 1: Basic (Always On)
- **Purpose**: Prevent catastrophic crashes from obvious mistakes
- **Scope**: Required fields, basic types, JAX compatibility
- **Performance**: Minimal overhead (<1% of execution time)
- **Errors**: Clear, actionable messages

#### Level 2: Flexible (Default)
- **Purpose**: Reasonable safety with experimentation allowed
- **Scope**: Range checks, shape validation, warnings for edge cases
- **Performance**: Low overhead (~5% of execution time)
- **Behavior**: Warns but allows potentially problematic configurations

#### Level 3: Strict (Opt-in)
- **Purpose**: Full mathematical and safety validation for production
- **Scope**: Complete schema validation, mathematical consistency, performance checks
- **Performance**: Higher overhead (~10-20% of execution time)
- **Behavior**: Rejects invalid configurations with detailed explanations

#### Level 4: Expert (Opt-in)
- **Purpose**: Maximum flexibility for advanced users/research
- **Scope**: No validation, raw access to internals
- **Performance**: Zero overhead
- **Behavior**: "You know what you're doing" mode

### Functional Core vs Imperative Shell

#### Imperative Shell Responsibilities
- Input validation before Core function calls
- Output validation after Core function returns
- Error handling and user guidance
- Validation configuration management

#### Functional Core Responsibilities
- Pure mathematical computations
- No input/output validation
- No error handling
- No side effects

## EnergySpec Validation Rules

### Basic Validation (Level 1 - Always On)

#### Required Fields
```python
# Must be present and non-empty
name: str                    # Non-empty string, 1-100 characters
terms: List[TermSpec]        # At least 1 term, max 50 terms
dtype: str                   # Must be "float32" or "float64"
```

#### Type and Format Checks
```python
# Basic type validation
assert isinstance(name, str) and len(name.strip()) > 0
assert isinstance(terms, list) and 1 <= len(terms) <= 50
assert dtype in ["float32", "float64"]
```

#### JAX Compatibility
```python
# Ensure JAX can handle the specification
assert all(isinstance(term, TermSpec) for term in terms)  # No raw dicts
assert all(hasattr(term, 'atom') for term in terms)       # Required TermSpec fields
```

### Flexible Validation (Level 2 - Default)

#### Range and Reasonableness Checks
```python
# Weight ranges (warnings, not errors)
for term in terms:
    if not (1e-6 <= term.weight <= 1e6):
        warn(f"Term weight {term.weight} is unusual. Typical range: 1e-6 to 1e6")

# Atom existence
for term in terms:
    if term.atom not in ATOM_REGISTRY:
        warn(f"Unknown atom '{term.atom}'. Available: {list(ATOM_REGISTRY.keys())}")
```

#### Shape and Dimensionality Hints
```python
# Suggest but don't enforce
if hasattr(spec, 'expected_input_shape'):
    if spec.expected_input_shape is not None:
        hint(f"Expected input shape: {spec.expected_input_shape}")
```

### Strict Validation (Level 3 - Opt-in)

#### Complete Schema Validation
```python
# Full Pydantic-style validation
class EnergySpecModel(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    terms: List[TermSpecModel] = Field(..., min_items=1, max_items=50)
    dtype: Literal["float32", "float64"]
    
    @validator('terms')
    def validate_terms_compatibility(cls, v):
        # Check mathematical compatibility between terms
        return v
```

#### Mathematical Consistency
```python
# Energy functional properties
def validate_mathematical_consistency(spec):
    # Check convexity assumptions
    # Verify differentiability requirements
    # Validate convergence properties
    pass
```

#### Performance Validation
```python
# Check for performance anti-patterns
def validate_performance(spec):
    # Warn about float64 on GPU
    if spec.dtype == "float64" and jax.default_backend() == "gpu":
        warn("float64 on GPU is 10-100x slower than float32")
    
    # Check term complexity
    total_complexity = sum(term.estimated_complexity for term in spec.terms)
    if total_complexity > PERFORMANCE_THRESHOLD:
        warn(f"High complexity energy ({total_complexity}). Consider simplification")
```

## NaN/Inf Safety (Numerical Stability)

### Input Validation
```python
def validate_inputs(x, spec):
    """Pre-computation input checks"""
    # Basic shape and type
    assert jnp.isarray(x), "Input must be JAX array"
    assert x.dtype in [jnp.float32, jnp.float64], f"Unsupported dtype: {x.dtype}"
    
    # Numerical validity
    if jnp.isnan(x).any():
        raise NumericalError("Input contains NaN values")
    if jnp.isinf(x).any():
        raise NumericalError("Input contains infinite values")
    
    # Shape consistency with spec
    if hasattr(spec, 'expected_shape') and spec.expected_shape:
        assert x.shape == spec.expected_shape, f"Shape mismatch: {x.shape} vs {spec.expected_shape}"
```

### Output Validation
```python
def validate_outputs(energy, gradient, spec):
    """Post-computation output checks"""
    # Energy validation
    assert jnp.isfinite(energy), f"Energy is not finite: {energy}"
    assert jnp.isreal(energy), f"Energy is complex: {energy}"
    
    # Gradient validation
    assert jnp.isfinite(gradient).all(), "Gradient contains non-finite values"
    assert gradient.shape == input_shape, f"Gradient shape mismatch: {gradient.shape}"
    assert gradient.dtype == spec.dtype, f"Gradient dtype mismatch: {gradient.dtype}"
    
    # Numerical stability checks
    grad_norm = jnp.linalg.norm(gradient)
    if grad_norm > 1e6:
        warn(f"Very large gradient norm: {grad_norm}. Possible numerical instability")
```

### Runtime Monitoring
```python
def monitor_numerical_stability(func):
    """Decorator for numerical stability monitoring"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            
            # Check for numerical issues
            if jnp.isnan(result).any() or jnp.isinf(result).any():
                logger.error(f"Numerical instability detected in {func.__name__}")
                raise NumericalInstabilityError(f"NaN/Inf in {func.__name__}")
            
            return result
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}")
            raise
    return wrapper
```

## Error Handling and User Guidance

### Error Message Standards
```python
# Good error messages
ValueError: EnergySpec.name cannot be empty. 
Example: EnergySpec(name="my_energy_function", ...)

# Even better with context
ValidationError: TermSpec[2].weight is negative (-0.5). 
Weights must be positive for convergence guarantees. 
For regularization terms, try values like 0.01-1.0. 
See: https://afs-docs.com/energy-specs#weights
```

### Warning System
```python
# Configurable warning levels
warnings.filterwarnings('error', category=NumericalWarning)  # Strict mode
warnings.filterwarnings('default', category=NumericalWarning)  # Default mode
warnings.filterwarnings('ignore', category=NumericalWarning)  # Expert mode
```

## Configuration API

### Global Configuration
```python
import afs

# Set validation level
afs.configure(validation='basic')    # Default
afs.configure(validation='strict')   # Full validation
afs.configure(validation='expert')   # No validation

# Per-operation configuration
energy = afs.compile_energy(spec, validation='strict')
```

### Environment Variables
```bash
# For CI/production
export AFS_VALIDATION=strict
export AFS_NUMERICAL_WARNINGS=error

# For development
export AFS_VALIDATION=basic
export AFS_NUMERICAL_WARNINGS=warn
```

## Implementation Phases

### Phase 1: Basic Validation (1-2 weeks)
- Implement Level 1 validation in Imperative Shell
- Add NaN/Inf input/output checks
- Basic error messages
- Integration with existing JAX dtype enforcement

### Phase 2: Flexible Validation (2-3 weeks)
- Add Level 2 validation with warnings
- Performance monitoring
- Configuration API
- Enhanced error messages

### Phase 3: Strict Validation (3-4 weeks)
- Full Pydantic integration for Level 3
- Mathematical consistency checks
- Comprehensive testing
- Documentation updates

### Phase 4: Expert Features (2-3 weeks)
- Level 4 expert mode
- Advanced monitoring
- Performance optimization
- SDK ecosystem integration

## Testing Strategy

### Validation Testing
- **Unit Tests**: Each validation rule
- **Integration Tests**: End-to-end with different validation levels
- **Error Message Tests**: Verify helpful guidance
- **Performance Tests**: Validation overhead measurement

### Numerical Stability Testing
- **Edge Cases**: NaN/Inf inputs and outputs
- **Precision Tests**: float32 vs float64 behavior
- **Scale Tests**: Very large/small values
- **Convergence Tests**: Numerical stability over iterations

## Migration and Compatibility

### Backward Compatibility
- Default behavior unchanged (Level 2 validation)
- Existing code continues to work
- New validation is opt-in

### Deprecation Strategy
- Warn about deprecated patterns
- Provide migration guides
- Gradual rollout of stricter validation

## Success Metrics

- **User Experience**: 90% of validation errors have actionable fixes
- **Performance**: <5% overhead in default mode
- **Flexibility**: Expert mode allows all legitimate use cases
- **Safety**: Catches 95% of catastrophic errors
- **Maintainability**: Clear separation between validation and computation</content>
<parameter name="filePath">j:\Google Drive\Software\afs\Design\shim_build\03b_energy_spec_hygine_spec.md