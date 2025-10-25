oh, I didn't capture the
# Multiscale & Wavelet Module

---
**Wavelet/Transform Requirements:**
Use jaxwt for all differentiable wavelet transforms. Optionally support CR-Wavelets, S2WAV, and S2BALL for extended transforms. All registry and API patterns must be compatible with the official package list in the overview doc above.

## Mathematical Definition
Let $\mathcal{W}$ be the multiscale transform:
$$
\mathcal{F}_{\text{Multi}}^+(z) = \mathcal{W}z,\qquad \mathcal{F}_{\text{Multi}}^-(u) = \mathcal{W}^\top u
$$
where $\mathcal{W}$ is unitary ($\mathcal{W}^\top \mathcal{W} = I$) or a tight frame.

## Supported Transforms
- **Default (1D):** Haar, Daubechies-4/8 ("haar", "db4", "db8")
- **2D/Images:** 2D separable wavelets
- **Graphs:** Graph wavelets via Laplacian spectral filters (tight frames)
- **Fourier:** Use DFT for stationary/periodic problems (unitary)

## Properties
- $\mathcal{W}$ must be invertible (unitary/tight frame)
- For graphs, $\mathcal{W}$ is built from Chebyshev polynomials of the Laplacian

## API Example
```python
from computable_flows_shim.multi import make_wavelet
W = make_wavelet("haar", levels=5)
```

## JAX Code Stub
```python
@dataclass(frozen=True)
class WaveletOp:
	name: str
	levels: int
	forward: Callable[[Array], Array]   # e.g., jaxwt.dwt_nD
	inverse: Callable[[Array], Array]   # e.g., jaxwt.idwt_nD

def F_Multi_forward(x, W):
	return W.forward(x)

def F_Multi_inverse(u, W):
	return W.inverse(u)

## TransformOp and Frame-aware Details

We define a standard registry item `TransformOp` that exposes frame properties for tuning and certificates:

```python
@dataclass(frozen=True)
class TransformOp:
	name: str
	forward: Callable[[Array], Array]
	inverse: Callable[[Array], Array]
	frame: str = "unitary"  # "unitary" | "tight" | "general"
	c: float = 1.0            # tight-frame constant if frame == "tight"
```

### Frame handling notes
- **unitary**: W^T W = I. No rescaling needed.
- **tight(c)**: W^T W = c I. Rescale norms and thresholds by c in prox and FDA.
- **general**: use analysis/synthesis pair (W, \tilde{W}), compute L_W = W L \tilde{W}.

## Validation & Manifest recording
Before finalizing a transform for a run, perform these validations and record the results in the run manifest (under `transforms.validation`):

- Forward/Inverse consistency: for random probe vectors x, check that
	`|| inverse(forward(x)) - x ||_2 / ||x||_2 < tol` (default tol=1e-6).
- Tight-frame energy check: when `frame == 'tight'`, verify that
	`| ||forward(x)||_2^2 - c * ||x||_2^2 | / (c * ||x||_2^2) < tol` (default tol=1e-3).

Record measured statistics (mean, std of residuals), pass/fail flags, and the seed/probe used so runs are fully reproducible.

### Chirplets and Graph Wavelets
- **Chirplets**: implemented via FFT + chirp multipliers; typically `tight` with c ~ 1.02. Provide dual windows for synthesis.
- **Graph wavelets**: implement analysis via Chebyshev polynomial filters of graph Laplacian; treat as `tight` with explicit c when available.

### Example: registry use
```python
W = TransformOp(
	name="db4", forward=lambda x: jaxwt.dwt_nD(x, 'db4'),
	inverse=lambda u: jaxwt.idwt_nD(u, 'db4'),
	frame='unitary'
)
```
```

## Operator Transformation
- Transform core operator $L$ to $L_{\mathcal{W}} = \mathcal{W} L \mathcal{W}^\top$
- Used for spectral gap and diagonal dominance checks

## Sparsity in $\mathcal{W}$-space
$$
	ext{sparsity}(\mathcal{W}x) = \frac{\#\{i: |(\mathcal{W}x)_i| > \tau\}}{\text{dim}(x)}
$$

## Telemetry & Controller Integration
- All transform/frame metadata, certificate checks, and phase transitions are logged via Flight Recorder events and telemetry rows.
- Controller phases (RED/AMBER/GREEN) enforce build checklist: transforms are validated and normalized before run; all transitions and results are recorded.
- API: transform registry and frame handling are called by the controller, which manages phase logic and telemetry recording.
```python
def sparsity_fraction_in_W(x, W, tau=1e-8):
	wx = W.forward(x)
	return (jnp.abs(wx) > tau).mean()
```

## Extensibility
- Plug in new transforms by subclassing or registering
- All transforms must provide forward/inverse and support JAX jit

For Lens selection and compressibility probes, see `11a_fda_hooks.md` which defines `LensPolicy` and Builder Mode probe hooks.

### LensPolicy and MultiscaleSchedule
LensPolicy (spec):

```python
LensPolicy(
	candidates=[TransformRef('db4'), TransformRef('haar')],
	probe_metrics=['compressibility', 'reconstruction_error'],
	selection_rule='min_recon_error @ target_sparsity'
)
```

MultiscaleSchedule (spec):

```python
MultiscaleSchedule(
	mode='residual_driven',
	levels=5,
	activate_rule='residual>tau'
)
```

Builder Mode probes:
- Lens probe: run short compressibility/reconstruction tests and emit `LENS_SELECTED`.
- Coarse-to-fine rehearsal: after a GREEN warm-start, run with coarse bands and progressively unlock finer scales.

## Engineering Note
- Always decompose state into multiscale domain before flow analysis
- Fast wavelet transform (FWT) is $O(N)$

---

See FDA and runtime docs for how multiscale is used in certificates and flows.

## Config Policy
- Use Python DSL for transform and manifold declarations (see `11_naming_and_layout.md`).
