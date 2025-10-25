# Ruff Linting Configuration for Scientific Python Code

**Version**: 1.0
**Date**: October 25, 2025
**Purpose**: Definitive linting configuration for mathematical/scientific Python codebases

---

## Overview

This document specifies the Ruff linting configuration for the AFS (Computable Flows) scientific Python codebase. The configuration balances strict code quality standards with the mathematical naming conventions essential for scientific computing.

### Key Principles

1. **Mathematical Rigor**: Allow standard mathematical notation (uppercase matrices, operators)
2. **Code Quality**: Maintain strict standards for everything except domain-specific naming
3. **Scientific Standards**: Follow patterns established by SciPy, NumPy, and other scientific libraries
4. **Practical Enforcement**: Use per-file-ignores rather than global rule changes

---

## Core Configuration

### [tool.ruff]

```toml
[tool.ruff]
# Only scan src/ and tests/ directories
include = ["src/**/*.py", "tests/**/*.py"]

exclude = [
    # Standard Python excludes
    ".bzr",
    ".direnv",
    ".eggs",
    ".git",
    ".git-rewrite",
    ".hg",
    ".mypy_cache",
    ".nox",
    ".pants.d",
    ".pytype",
    ".ruff_cache",
    ".svn",
    ".tox",
    ".venv",
    "__pypackages__",
    "_build",
    "buck-out",
    "build",
    "dist",
    "node_modules",
    "venv",
]

# Same line length as black
line-length = 88
indent-width = 4

# Assume Python 3.12
target-version = "py312"
```

### [tool.ruff.lint]

```toml
[tool.ruff.lint]
# Enable comprehensive rule set for maximum error detection
select = [
    "E",   # pycodestyle errors
    "W",   # pycodestyle warnings
    "F",   # Pyflakes
    "I",   # isort
    "B",   # flake8-bugbear
    "C4",  # flake8-comprehensions
    "UP",  # pyupgrade
    "N",   # pep8-naming
    "A",   # flake8-builtins
    "BLE", # flake8-blind-except
    "FBT", # flake8-boolean-trap
    "C90", # McCabe complexity
    "COM", # flake8-commas
    "DTZ", # flake8-datetimez
    "EM",  # flake8-errmsg
    "EXE", # flake8-executable
    "FA",  # flake8-future-annotations
    "ISC", # flake8-implicit-str-concat
    "ICN", # flake8-import-conventions
    "G",   # flake8-logging-format
    "INP", # flake8-no-pep420
    "PIE", # flake8-pie
    "T10", # flake8-debugger
    "PERF",# Perflint
    "FURB",# refurb
    "LOG", # flake8-logging
    "RUF", # Ruff-specific rules
]

# Ignore some rules that don't apply to your project
ignore = [
    "E501", # line too long, handled by black
    "B008", # do not perform function calls in argument defaults
    "C901", # too complex (McCabe)
    "COM812", # trailing comma missing (conflicts with formatter)
    "ISC001", # implicit string concatenation (conflicts with formatter)
    "A003", # class attribute shadows builtin (too strict for some cases)
    "EM101", # exception message should not use f-string (too opinionated)
    "EM102", # exception message should not use .format() (too opinionated)
    "TRY003", # avoid specifying long messages outside exception class (too opinionated)
    "FBT001", # boolean positional arg in function definition (too strict)
    "FBT002", # boolean default positional argument (too strict)
    "FBT003", # boolean positional value in function call (too strict)
    "N999", # invalid module name (too strict for some cases)
]

# Allow fix for all enabled rules (when `--fix`) is provided.
fixable = ["ALL"]
unfixable = []

# Allow unused variables when underscore-prefixed.
dummy-variable-rgx = "^(_+|(_+[a-zA-Z0-9_]*[a-zA-Z0-9]+?))$"
```

### [tool.ruff.lint.per-file-ignores]

```toml
[tool.ruff.lint.per-file-ignores]
# Allow mathematical naming conventions in scientific code
"src/**/*.py" = [
    "N806",  # Mathematical variables: A, ATA, ATb, L_matrix, X, Y, T (matrices/operators)
    "N802",  # Mathematical functions: F_Dis, F_Proj, F_Multi, F_Con, L_apply, g_prox_in_W
    "N803",  # Mathematical arguments: L_apply, L_matrix
    "RUF002",  # Greek sigma in docstrings (mathematical notation)
    "RUF003",  # Greek sigma in comments (mathematical notation)
]
"tests/**/*.py" = [
    "N806",  # Mathematical variables in tests
    "N802",  # Mathematical functions in tests
    "N803",  # Mathematical arguments in tests
]

# Keep existing ignores
"__init__.py" = ["F401"]
"tests/**/*" = ["T201", "T10"]
"scripts/**/*" = ["T201", "T10"]
"setup.py" = ["INP001"]
"conftest.py" = ["INP001"]
```

### [tool.ruff.lint.isort]

```toml
[tool.ruff.lint.isort]
known-first-party = ["computable_flows_shim"]
```

### [tool.ruff.lint.flake8-tidy-imports]

```toml
[tool.ruff.lint.flake8-tidy-imports]
# Disallow all relative imports.
ban-relative-imports = "all"
```

### [tool.ruff.lint.flake8-import-conventions]

```toml
# Declare the conventions for import aliases.
[tool.ruff.lint.flake8-import-conventions.aliases]
"jax.numpy" = "jnp"
```

### [tool.ruff.format]

```toml
[tool.ruff.format]
# Like Black, use double quotes for strings.
quote-style = "double"

# Like Black, indent with spaces, rather than tabs.
indent-style = "space"

# Like Black, respect magic trailing commas.
skip-magic-trailing-comma = false

# Like Black, automatically detect the appropriate line ending.
line-ending = "auto"

# Enable auto-formatting of code examples in docstrings. Markdown,
# reStructuredText code/literal blocks and doctests are all supported.
#
# This is currently disabled by default, but it is planned for this
# to be opt-out in the future.
docstring-code-format = false

# Set the line length limit used when formatting code snippets in
# docstrings.
#
# This only has an effect when the `docstring-code-format` setting is
# enabled.
docstring-code-line-length = "dynamic"
```

---

## Mathematical Naming Conventions

### Variables (N806 - Allowed Uppercase)

**Single Letters (Matrices/Operators):**
- `A`, `B`, `C` - General matrices/operators
- `X`, `Y`, `Z` - State/input/output variables
- `L` - Laplacian operators, linear operators
- `W` - Wavelet transforms, weight matrices
- `T` - Temperature, time, transformation matrices

**Compound Expressions:**
- `ATA` - A^T A (matrix multiplication)
- `ATb` - A^T b (transposed multiplication)
- `L_matrix` - Laplacian matrix
- `L_op` - Linear operator
- `L_apply` - Linear operator application function
- `effective_L_apply` - Effective linear operator

### Functions (N802 - Allowed MixedCase)

**Flow Primitives:**
- `F_Dis` - Dissipative flow operator
- `F_Proj` - Projective/Proximal flow operator
- `F_Multi` - Multiscale flow operator
- `F_Con` - Conservative flow operator

**Mathematical Functions:**
- `L_apply` - Apply linear operator
- `g_prox_in_W` - Proximal operator in wavelet space
- `L_w_space` - Linear operator in wavelet space

### Arguments (N803 - Allowed MixedCase)

- `L_apply` - Linear operator application function
- `L_matrix` - Laplacian matrix

### Documentation (RUF002/RUF003 - Greek Letters)

- `σ` (sigma) - Standard deviation, stress, surface tension
- Other Greek letters in mathematical expressions

---

## Related Configurations

### Pylint Configuration

```toml
[tool.pylint.main]
init-hook = "import sys; sys.path.insert(0, 'src')"
```

### Pytest Configuration

```toml
[tool.pytest.ini_options]
pythonpath = [".", "src"]
testpaths = ["tests"]
addopts = [
    "--strict-markers",
    "--junitxml=outputs/test-results.xml",
    "--log-file=logs/pytest_debug.log",
    "--log-level=INFO",
    "--log-file-level=DEBUG",
    "--log-file-format=%(asctime)s - %(levelname)s - %(message)s",
    "--log-file-date-format=%Y-%m-%d %H:%M:%S",
    "--ignore=jax-docs",
    "--ignore=archive",
    "--ignore=background",
    "-m not optional",
    "-ra",
    "-q"
]
python_files = ["test_*.py", "*_test.py", "*_integration.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
markers = [
    "experiments: marks tests as experiment tests (deselect with '-m \"not experiments\"')",
    "tuning: marks tests as parameter tuning tests (deselect with '-m \"not tuning\"')",
    "integration: marks tests as integration tests (deselect with '-m \"not integration\"')",
    "unit: marks tests as unit tests (deselect with '-m \"not unit\"')",
    "fda: marks tests as FDA compliance tests (deselect with '-m \"not fda\"')",
    "optional: opt-in tests that require large external data or long runtime"
]
filterwarnings = [
    # JAX dtype truncation warnings should now be rare since we enable x64 by default
    # If they appear, they're important to investigate
    "default"
]
```

---

## Results and Impact

### Error Reduction

- **Before Configuration**: 151+ linting errors
- **After Configuration**: 61 linting errors
- **Eliminated**: ~90 mathematical naming convention errors
- **Remaining**: 61 actual code quality issues

### Error Categories Eliminated

1. **N806** (67+ instances): Mathematical variables (A, ATA, ATb, L_matrix, etc.)
2. **N802** (20+ instances): Mathematical functions (F_Dis, F_Proj, L_apply, etc.)
3. **N803** (arguments): Mathematical parameters
4. **RUF002/RUF003**: Greek letters in documentation

### Remaining Issues (61 total)

**High Priority:**
- INP001 (3): Missing `__init__.py` files
- BLE001 (6): Blind exception catching
- DTZ005 (4): Datetime without timezone

**Medium Priority:**
- RUF015 (2): Single element slice inefficiencies
- RUF005 (2): Slice concatenation syntax

**Low Priority:**
- Various style issues (RUF022, UP047, N815, B023, B007, F401, G004, RUF006)

---

## Rationale and Best Practices

### Why Per-File-Ignores Over Global Changes

1. **Scientific Standards**: Follows SciPy/NumPy patterns
2. **Surgical Precision**: Only ignores domain-specific naming
3. **Maintainability**: Clear documentation of exceptions
4. **Gradual Adoption**: Easy to modify for specific files

### Mathematical Naming Justification

Scientific computing uses established conventions:
- **Linear Algebra**: Uppercase for matrices (A, B, L)
- **Operators**: MixedCase for mathematical functions (L_apply, F_Dis)
- **Documentation**: Greek letters for mathematical notation

### Configuration Philosophy

1. **Comprehensive Coverage**: Enable all relevant rule categories
2. **Minimal Ignores**: Only ignore what's mathematically necessary
3. **Clear Documentation**: Every ignore has a comment explaining why
4. **Regular Review**: Re-evaluate ignores as codebase evolves

---

## Maintenance and Updates

### When to Add New Ignores

1. **Mathematical Necessity**: New mathematical naming patterns
2. **Scientific Standards**: Conventions used in the field
3. **Documentation**: Always add explanatory comments

### When to Remove Ignores

1. **Code Evolution**: When mathematical naming can be improved
2. **Better Alternatives**: When clearer naming is possible
3. **Standards Changes**: When community conventions evolve

### Regular Audits

- Review ignore list quarterly
- Validate against SciPy/NumPy practices
- Ensure ignores remain minimal and justified

---

## References

- **SciPy Ruff Configuration**: https://github.com/scipy/scipy/blob/main/pyproject.toml
- **NumPy Ruff Configuration**: https://github.com/numpy/numpy/blob/main/pyproject.toml
- **Ruff Documentation**: https://docs.astral.sh/ruff/
- **PEP 8**: https://peps.python.org/pep-0008/
- **Scientific Python Guidelines**: Various community standards</content>
<parameter name="filePath">j:\Google Drive\Software\afs\Design\shim_build\25_ruff_lint_scientific.md
