# The user's entire program file: recon_job.py

from computable_flows_shim.api import run_certified
from my_project.operators import build_measurement_operator

# 1. Define the Energy Functional using the Atom Library
spec = {
    "dtype": "float32",
    "state": {"main": {"shape": (256, 256)}},
    "terms": [
        {
            "type": "quadratic",
            "op": "A",
            "target": "y_observed",
            "weight": 1.0,
        },
        {
            "type": "l1_wavelet",
            "transform": "db4",
            "weight_key": "lambda_sparsity", # Tuned by GapDial
        },
        {
            "type": "tv_iso",
            "weight_key": "lambda_tv", # Tuned by GapDial
        },
    ],
    "tuner": {
        "lambda_sparsity": {"range": [1e-3, 1.0], "log": True},
        "lambda_tv": {"range": [1e-3, 1.0], "log": True},
    }
}

# 2. Provide the concrete operators and data
A = build_measurement_operator(...)
y_observed = load_data(...)
init_state = {"main": jnp.zeros((256, 256))}

# 3. Run the certified flow
final_state, report = run_certified(
    spec=spec,
    initial_state=init_state,
    ops={"A": A},
    data={"y_observed": y_observed}
)