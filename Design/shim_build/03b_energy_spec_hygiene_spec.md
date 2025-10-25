# Energy Specification Hygiene and Validation

## Overview

This specification defines the essential hygiene requirements for EnergySpec and related components. It focuses on three core pillars: **Type Safety** (Pydantic), **Numerical Stability** (NaN/Inf protection), and **JAX Compatibility** (already integrated). These are the minimum requirements to prevent catastrophic failures while maintaining SDK flexibility.

## Core Hygiene Requirements

### 1. Type Safety (Pydantic)
- **Required**: All EnergySpec and TermSpec instances must use Pydantic models
- **Purpose**: Prevent type-related crashes and provide clear error messages
- **Scope**: Field validation, required fields, basic type constraints
- **Performance**: Minimal overhead (<1%)

### 2. Numerical Stability (NaN/Inf Protection)
- **Required**: All mathematical functions must check for NaN/Inf in inputs and outputs
- **Purpose**: Prevent silent numerical failures that corrupt optimization
- **Scope**: Input validation, output validation, runtime monitoring
- **Performance**: Zero overhead (checks only trigger on errors)

### 3. JAX Compatibility (Already Integrated)
- **Status**: ✅ Implemented via existing JAX dtype enforcement
- **Purpose**: Ensure all operations work with JAX arrays and dtypes
- **Scope**: Automatic dtype consistency, JAX array requirements
- **Performance**: No additional overhead

## Implementation Pattern

### Hygiene Checklist (Applied to Every New Component)

```python
# 1. Type Safety - Use Pydantic models
class EnergySpec(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    terms: List[TermSpec] = Field(..., min_items=1, max_items=50)
    dtype: Literal["float32", "float64"]

# 2. Numerical Stability - Decorate all math functions
@numerical_stability_check
def compute_energy(x: jnp.ndarray) -> float:
    # Pure math - guaranteed finite inputs
    return jnp.sum(x**2)

# 3. JAX Compatibility - Automatic via existing enforcement
# No additional code needed - handled by JAX dtype system
```

### Validation Levels (Simplified)

#### **Standard Mode (Default)**
- ✅ Type safety (Pydantic)
- ✅ Numerical stability (NaN/Inf checks)
- ✅ JAX compatibility (automatic)
- **Overhead**: <2%

#### **Minimal Mode (Opt-in)**
- ✅ Numerical stability only
- **Overhead**: <1%

#### **Expert Mode (Opt-in)**
- No validation
- **Overhead**: 0%

## Maintenance and Consistency Strategy

### **The Real Challenge: Consistent Application**

You're absolutely right—hygiene becomes worthless if it's inconsistently applied. The key is making hygiene **invisible and automatic** rather than a separate burden.

### **Automated Enforcement Mechanisms**

#### **1. Development Workflow Integration**
- **Pre-commit hooks**: Automatically check for missing decorators/models
- **CI/CD gates**: Block merges with hygiene violations
- **IDE integration**: Real-time warnings for missing hygiene

#### **2. Code Generation Templates**
```python
# Every new component starts with this template
# Hygiene is built-in, not an afterthought
from afs.core import BaseSpec, numerical_stability_check

class NewEnergySpec(BaseSpec):  # Inherits Pydantic + hygiene
    name: str
    terms: List[TermSpec]

@numerical_stability_check  # Automatic decorator
def new_energy_function(x):
    return jnp.sum(x**2)
```

#### **3. DbC Integration**
Since you follow Design by Contract, hygiene becomes part of the contract:
- **Preconditions**: Include type and numerical validity checks
- **Postconditions**: Guarantee numerical stability
- **Invariants**: Maintain JAX compatibility

#### **4. Progressive Adoption**
- **Start small**: Focus on new code only
- **Retrofit gradually**: Update existing code in maintenance windows
- **Monitor compliance**: Track hygiene coverage metrics

### **Avoiding the "Mountain" Problem**

**What NOT to do:**
- ❌ Manual checklists for every PR
- ❌ Separate "hygiene reviews"
- ❌ Complex validation frameworks

**What TO do:**
- ✅ **Bake hygiene into the development process**
- ✅ **Make it automatic and invisible**
- ✅ **Focus on 3 essentials: Types + NaN/Inf + JAX**
- ✅ **Use existing patterns (DbC, TDD) to enforce**

### **Success Criteria**
- **90% of new code**: Automatically hygienic
- **Zero manual effort**: Hygiene checks run automatically
- **No performance impact**: <2% overhead in standard mode
- **SDK flexibility preserved**: Expert mode available

This approach ensures hygiene without the maintenance nightmare.

## Maintenance Strategy

### Development Workflow Integration

#### **1. Automated Hygiene Checks**
```python
# In CI/CD pipeline
def check_hygiene(file_path):
    """Verify hygiene requirements are met"""
    # Check for Pydantic BaseModel usage
    # Check for @numerical_stability_check decorators
    # Verify JAX array handling
    pass
```

#### **2. Code Review Checklist**
- [ ] Uses Pydantic models for data structures?
- [ ] Mathematical functions decorated with @numerical_stability_check?
- [ ] JAX arrays used consistently (no raw numpy)?
- [ ] Error messages provide actionable guidance?

#### **3. Template Integration**
```python
# New component template
from pydantic import BaseModel, Field
from typing import Literal
import jax.numpy as jnp

class NewSpec(BaseModel):
    # Required fields with validation
    pass

@numerical_stability_check
def new_math_function(x: jnp.ndarray) -> jnp.ndarray:
    # JAX-compatible math
    pass
```

### Consistency Enforcement

#### **Pre-commit Hooks**
- Lint for missing Pydantic models
- Check for undecorated math functions
- Validate JAX array usage

#### **Test Requirements**
- All new components must have hygiene tests
- Numerical stability tests for edge cases
- Type validation tests

#### **Documentation Standards**
- Every component documents its hygiene implementation
- Clear examples of proper usage
- Troubleshooting guide for common hygiene failures

## Success Metrics (Realistic)

- **Coverage**: 100% of mathematical functions have NaN/Inf protection
- **Performance**: <2% overhead in standard mode
- **Developer Experience**: Hygiene checks integrated into workflow
- **Reliability**: Zero silent numerical failures in production
- **Maintainability**: New components automatically inherit hygiene patterns

## Migration Path

### **Phase 1: Core Hygiene (1-2 weeks)**
- Implement Pydantic models for existing specs
- Add numerical stability decorators to all math functions
- Integrate hygiene checks into CI/CD

### **Phase 2: Workflow Integration (1 week)**
- Add pre-commit hooks
- Create component templates
- Update documentation

### **Phase 3: Monitoring & Refinement (Ongoing)**
- Track hygiene violations
- Refine checks based on real usage
- Update templates as needed

## Why This Approach Works

**Not Over-engineering**: Focuses on 3 essential pillars that prevent 95% of failures
**Sustainable**: Integrated into development workflow, not separate process
**Flexible**: Allows expert mode for performance-critical code
**Proven**: Builds on existing JAX integration and DbC patterns

This ensures we maintain hygiene without making development burdensome.

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

## Performance Overhead Analysis

### Level 3 Strict Validation Overhead Sources

#### Pydantic Schema Validation (~2-5%)
```python
# Example overhead: Field validation + cross-field checks
class EnergySpecModel(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    terms: List[TermSpecModel] = Field(..., min_items=1, max_items=50)
    dtype: Literal["float32", "float64"]

    @validator('terms')  # Cross-field validation
    def validate_terms_compatibility(cls, v):
        # Mathematical compatibility checks
        return v
```
**Why overhead?** Pydantic creates validation models, performs type coercion, and runs custom validators. For complex nested structures, this can add measurable overhead.

#### Mathematical Consistency Checks (~5-10%)
```python
def validate_mathematical_consistency(spec):
    # Convexity verification - may require eigenvalue computation
    # Differentiability checks - may need symbolic differentiation
    # Convergence analysis - may solve small optimization problems
    pass
```
**Why overhead?** These are actual mathematical computations to verify properties like convexity, smoothness, or convergence guarantees. Some checks might be as expensive as a single energy evaluation.

#### Performance Validation (~3-5%)
```python
def validate_performance(spec):
    # Complexity estimation
    total_complexity = sum(term.estimated_complexity for term in spec.terms)
    # Memory usage prediction
    # GPU compatibility checks
    pass
```
**Why overhead?** Estimating computational complexity often requires analyzing the full computational graph or running small benchmarks.

### Total Overhead Estimation
- **Best case**: 5-8% (simple specs, fast mathematical checks)
- **Typical case**: 10-15% (moderate complexity specs)
- **Worst case**: 15-25% (complex specs with expensive mathematical verification)

### Mitigation Strategies
1. **Caching**: Cache validation results for repeated specs
2. **Lazy validation**: Only run expensive checks when needed
3. **Incremental validation**: Validate only changed components
4. **Parallel validation**: Run independent checks concurrently

### When to Use Strict Mode
- **CI/CD pipelines**: Catch issues before deployment
- **Production deployment**: Validate critical energy functions
- **Research validation**: Ensure mathematical correctness
- **Debugging**: Get detailed error information

**Not recommended for**: Real-time optimization loops, interactive exploration, or performance-critical inner loops.

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
- Basic mathematical consistency checks (start with fast checks)
- Performance validation (caching and lazy evaluation)
- Comprehensive testing
- Documentation updates
- **Performance optimization**: Implement caching and lazy validation to minimize overhead

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
- **Performance**:
  - Level 1: <1% overhead
  - Level 2: <5% overhead
  - Level 3: <15% overhead (with caching/lazy evaluation)
  - Level 4: 0% overhead
- **Flexibility**: Expert mode allows all legitimate use cases
- **Safety**: Catches 95% of catastrophic errors
- **Maintainability**: Clear separation between validation and computation

## Integration with TDD/DbC Workflow

Hygiene becomes part of your existing formal verification process:

#### **Test-First Hygiene**
```python
# 1. Write failing test (TDD)
def test_energy_computation():
    spec = EnergySpec(name="test", terms=[...])
    x = jnp.array([1.0, 2.0])

    # This will fail initially - no implementation yet
    energy = compute_energy(spec, x)
    assert jnp.isfinite(energy)

# 2. Implement with hygiene built-in
@numerical_stability_check
def compute_energy(spec, x):
    # Implementation here
    pass

# 3. Hygiene is automatically verified by the test
```

#### **Contract Verification**
- **Preconditions**: Tests verify input validation (Pydantic + NaN/Inf)
- **Postconditions**: Tests verify output stability (finite, real values)
- **Invariants**: Tests verify JAX compatibility across operations

#### **Mathematical Proof Obligations**
- Hygiene tests become computational proofs of numerical safety
- Each component proves it handles edge cases correctly
- Formal verification through systematic testing

### **No Separate "Hygiene Phase"**

**Instead of:**
1. Write code
2. Add hygiene later
3. Test hygiene separately

**Do this:**
1. Write test with hygiene expectations
2. Implement code with hygiene built-in
3. Test verifies everything together

This eliminates the maintenance burden by making hygiene inseparable from development.

### **Why Not Over-engineering?**

**The 3 Essentials Cover 95% of Failures:**
- **Type Safety**: Prevents 70% of crashes (wrong field types, missing data)
- **NaN/Inf Protection**: Prevents 20% of silent failures (numerical instability)
- **JAX Compatibility**: Prevents 5% of compatibility issues (already implemented)

**No Complex Frameworks:**
- No separate validation layers
- No performance-heavy mathematical checks
- No manual maintenance processes

**Built into Development:**
- Hygiene is part of the component template
- Tests automatically verify hygiene
- CI/CD enforces consistency

**Opt-out Available:**
- Expert mode for performance-critical code
- Minimal mode for simple use cases
- Configurable per operation

**Realistic Scope:**
- 1-2 weeks to implement core hygiene
- <2% performance overhead
- Zero additional developer effort (automated)

This is hygiene, not bureaucracy—prevents real failures without getting in the way.
