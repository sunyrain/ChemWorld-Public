# API reference

> Current main: [October foundation update](current-foundation.md). Existing figures and numerical evidence below describe the frozen August runtime.

ChemWorld exposes the same runtime surface as v0.2.0. Start from the agent-facing helpers, then use the deeper contracts when authoring worlds or evaluation code.

| Topic | Reference |
| --- | --- |
| Installation and first environment | [Getting started](getting_started.md) |
| Typed operation payloads | [Action schema](action_schema.md) |
| Supported process operations | [Operations](operations.md) |
| World component compatibility | [Composition contract](world-composition-contract.md) |
| Composition examples | [Composition examples](world-composition-examples.md) |
| Capability coverage | [World capability map](world-capability-map.md) |

Core agent-facing calls are available on `env.unwrapped`:

```python
env.unwrapped.task_prompt()
env.unwrapped.available_actions()
env.unwrapped.action_schema("heat")
env.unwrapped.validate_action(action)
env.unwrapped.agent_view_bundle(observation, info)
```
