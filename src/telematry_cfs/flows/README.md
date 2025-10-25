# Flow Specifications Directory

This directory contains flow specifications written in Python DSL for the Computable Flow Shim.

## Directory Structure

```
src/telematry_cfs/flows/
├── {flow_name}.py          # Flow specification files
├── README.md              # This file
└── __pycache__/           # Python bytecode (ignored)
```

## Flow Specification Format

Each flow specification is a Python file that defines:

### Required Attributes
- `spec`: `EnergySpec` object defining the energy functional
- `op_registry`: `Dict[str, Op]` mapping operator names to implementations
- `initial_state`: `Dict[str, jnp.ndarray]` with initial optimization variables

### Optional Attributes
- `step_alpha`: `float` - step size (default: 0.1)
- `num_iterations`: `int` - number of iterations (default: 100)
- `flow_name`: `str` - display name (default: filename without .py)

### Example: `quadratic_flow.py`

```python
import jax.numpy as jnp
from computable_flows_shim.energy.specs import EnergySpec, TermSpec, StateSpec
from computable_flows_shim.api import Op

class IdentityOp(Op):
    def __call__(self, x):
        return x

# Flow specification
spec = EnergySpec(
    terms=[TermSpec(type='quadratic', op='I', weight=1.0, variable='x', target='y')],
    state=StateSpec(shapes={'x': [10], 'y': [10]})
)

op_registry = {'I': IdentityOp()}

# Initial state
target_y = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
noise = jax.random.normal(jax.random.PRNGKey(42), (10,)) * 2.0
initial_state = {'x': target_y + noise, 'y': target_y}

# Parameters
step_alpha = 0.1
num_iterations = 50
flow_name = "quadratic_demo"
```

## Usage

### Running a Flow
```bash
# From project root
python src/scripts/cfs_cli.py run quadratic_flow
python src/scripts/cfs_cli.py run my_custom_flow.py
```

### Checking Certificates
```bash
python src/scripts/cfs_cli.py cert quadratic_flow
```

### Starting Dashboard
```bash
python src/scripts/cfs_cli.py hud
```

## Development Workflow

1. **Create**: Write your flow specification in `flows/{name}.py`
2. **Validate**: Run `cfs cert {name}` to check certificates
3. **Execute**: Run `cfs run {name}` to execute the flow
4. **Analyze**: Results saved to `../fda_run_{timestamp}/` with telemetry

## File Naming Convention

- Use snake_case for flow names: `wavelet_deconvolution.py`
- Avoid special characters and spaces
- The flow_name attribute can provide a more readable display name

## Best Practices

- Keep flow specs focused on a single optimization problem
- Use descriptive variable names and comments
- Test with small problem sizes first
- Include convergence criteria in comments
- Version control your flow specs alongside results
