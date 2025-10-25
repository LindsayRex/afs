
# Extensibility & Backends

---
**Extensibility Requirements:**
All backend adapters must wrap official packages (JAX, jaxwt, jaxlie, Optax, etc.) and follow the requirements in the overview doc above.

## Op and Prox Registry
- Register new ops via Python entry points
- Add new prox maps with decorators
- Plug in new backends (JAX, Torch) by implementing adapter API

## Transform and Manifold Registry

Example entry-point names: `computable_flows_shim.transforms` and `computable_flows_shim.manifolds`.
Example entry-point names and expected signatures:

- `computable_flows_shim.transforms` -> callable returning a list of (name, constructor) or registering via decorator.

	Signature:
	```python
	def register_transforms() -> list[tuple[str, Callable[..., TransformOp]]]:
			return [("db4", lambda **kwargs: TransformOp(...))]
	```

- `computable_flows_shim.manifolds` -> callable returning a list of (name, adapter_constructor)

	Signature:
	```python
	def register_manifolds() -> list[tuple[str, Callable[..., Manifold]]]:
			return [("SE3", lambda **kwargs: SE3Manifold(**kwargs))]
	```

- `computable_flows_shim.ops` -> callable returning op constructors

	Signature:
	```python
	def register_ops() -> list[tuple[str, Callable[..., Op]]]:
			return [("fft_channel", lambda **kwargs: FFTChannelOp(**kwargs))]
	```

- `computable_flows_shim.backends` -> callable returning backend adapters

	Signature:
	```python
	def register_backends() -> list[tuple[str, Callable[..., BackendAdapter]]]:
			return [("jax", lambda **kwargs: JaxBackend(**kwargs))]
	```

Registration example (setup.cfg / pyproject):

```ini
[options.entry_points]
computable_flows_shim.transforms =
		mypkg.transforms = mypkg.transforms:register_transforms
computable_flows_shim.manifolds =
		mypkg.manifolds = mypkg.manifolds:register_manifolds
computable_flows_shim.ops =
		mypkg.ops = mypkg.ops:register_ops
computable_flows_shim.backends =
		mypkg.backends = mypkg.backends:register_backends
```

Adapters should follow the ports-and-adapters pattern: the adapter returns an object conforming to the `TransformOp`, `Manifold`, `Op`, or `BackendAdapter` interface

## Adding a New Backend
- Implement backend API in `backends/`
- Ensure all primitives and transforms are supported
- Register backend in config

## Plugin System
- Surface for ops/prox is kept minimal for stability
- Any repo can contribute ops without modifying core

## Engineering Notes
- JAX is default; Torch backend can be added later
- Registry pattern ensures extensibility without code drift


See ops and primitives docs for extension details.

## Flight Recorder & Controller Extensibility
Telemetry and event capsules are extensible: new metrics, events, and capsule artifacts can be added via registry hooks.
Controller phases (RED/AMBER/GREEN) and builder mode are extensible: new gates, certificate checks, and tuning policies can be registered.
Manifest, telemetry, and events are designed for plugin compatibility and AI/CLI/Notebook inspection.
For mathematicians and formal-methods users, an optional Tensor Logic front-end (see `13_tensor_logic_frontend.md`) allows declarative tensor program input, compiled to the same runtime. This does not affect core extensibility or backend adapters.
