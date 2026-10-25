# API reference

The [capability index](https://sunyrain.github.io/ChemWorld-Public/?lang=en#index/combustion)
is the interactive API reference for the current public foundation. Each implemented
capability has an inline Python interface, parameter/default table, complete example,
explicit input and output captured by executing that example. You can copy and run
the code without opening the implementation source.

Select **Usage** for a complete walkthrough, **API & parameters** for signatures and
action fields, or **Examples & output** for the runnable code and its results. Additional
examples are expandable. Planned capabilities are labelled and have no callable API.

Inputs and outputs default to a field/value/unit table with expandable nested groups.
Switch to **JSON** to inspect the full structure; **Copy JSON** works in either view.

Examples use synthetic inputs or packaged presets. Some demonstrate initialization,
configuration or a rejected request; their scope is stated beside the code. They are
representative entry points, not an exhaustive listing of every Python symbol or a new
scientific qualification. Numerical last digits can vary with the platform and solver
version. See the [current foundation](current-foundation.md) for delivery boundaries.

## Run an example

Install the current checkout using [Getting started](getting_started.md). Save the
copied Python code as `example.py`, then run `uv run --no-sync python example.py`.
For example, convert a temperature using the public quantity API:

```python
from chemworld.foundation.units import Quantity

result = Quantity(26.85, "degC").to("K")
print(result.value, result.unit)
```

```text
300.0 K
```

## Runtime and authoring contracts

Apparatus examples show the actual `action_schema()` fields for the initialized
session, including units, required fields, bounds and choices. Bounds can change with
state: use `validate_action()` before execution. A request's receipt distinguishes a
committed action from a rejection. Reading `observe()` returns the cached public
packet; acquiring a new measurement uses `execute({"operation": "measure", ...})`.

| Topic | Reference |
| --- | --- |
| Installation and first environment | [Getting started](getting_started.md) |
| Typed operation payloads | [Action schema](action_schema.md) |
| Supported process operations | [Operations](operations.md) |
| World component compatibility | [Composition contract](world-composition-contract.md) |
| Composition examples | [Composition examples](world-composition-examples.md) |
| Capability coverage | [World capability map](world-capability-map.md) |

Core agent-facing calls are also available on the Gymnasium adapter's `env.unwrapped`:

```python
env.unwrapped.task_prompt()
env.unwrapped.available_actions()
env.unwrapped.action_schema("heat")
env.unwrapped.validate_action(action)
env.unwrapped.agent_view_bundle(observation, info)
```
