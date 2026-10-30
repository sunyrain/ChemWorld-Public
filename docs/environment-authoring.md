<!-- Generated from https://github.com/sunyrain/ChemWorld/blob/main/docs/environment-authoring.en.md; edit the upstream guide. -->

# Design your own experimental environment

Choose supported physical models and equipment, declare operations, instruments and
resources, then connect your own agent. Users define the research question, exploration
strategy, reward and evaluation. The current entry points are `WorldCompositionSpec`
and `ChemWorldRuntime`; installed module declarations define the supported capabilities.

## Customize a working environment

This example reuses the native surface-reaction apparatus and changes both research
vessels' temperature bounds, operation duration and total operation budget. Its vessels,
finite gas feeds, catalysts and thermal models come from the existing builder. Parameters
are synthetic examples, not predictions for a real catalyst. No agent action sequence is prescribed.

```python
import json
from pathlib import Path

from chemworld.runtime import ChemWorldRuntime
from chemworld.world.composition import WorldCompositionSpec
from chemworld.world.surface_world import surface_world_spec

design = surface_world_spec(terminal_assay=False).to_dict()
design["composition_id"] = "my-two-vessel-study"
design["task"]["objective"] = "Investigate how heating changes measured gas composition"
design["task"]["budget"] = 24
for component in design["components"]:
    if component["kind"] == "thermal":
        component["parameters"]["temperature_range_K"] = [330, 410]
        component["parameters"]["duration_range_s"] = [1, 60]

spec = WorldCompositionSpec.from_dict(design)
Path("world.json").write_text(
    json.dumps(spec.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
)
session = ChemWorldRuntime(composition=spec, observation_noise_mode="keyed")
try:
    session.reset(seed=12)
    print(session.task_prompt()["text"])
    print([item["operation"] for item in session.available_actions()])
    print(session.action_schema("heat"))
finally:
    session.close()
```

Save the script and `world.json` in your own working directory. Parsing and runtime
compilation check compatibility and parameters; naming a module does not install a
new physical capability. For other domains, consult the
[domain workflows](https://github.com/sunyrain/ChemWorld/blob/main/docs/native_domains.md)
and [installed capabilities](https://github.com/sunyrain/ChemWorld/blob/main/docs/native_capabilities.md)
before choosing a supported builder or model declaration.

## Let your agent operate the environment

Your host loads `world.json`, creates a session and gives the agent the research question,
`task_prompt()`, action schemas and acquired native observations. The agent chooses its
next action; the host calls `validate_action(action)` and `execute(action)`, saves the
execution record and returns public results. Reading `observe()` neither acquires a
measurement nor advances the clock.

Read `result.observation.to_dict()["native"]["channels"]` and preserve units, quality,
sample times and uncertainty. Missing values are not zero; cached readings are not new
evidence. The world declaration, full `result.info` and session snapshots belong to the
host and may contain physical parameters or internal diagnostics withheld from the agent.

An unscored environment's prompt uses the declared `objective`. It does not require
optimizing a built-in score or submitting an agent manifest. `trajectory_validity` in
structured metadata describes record/replay checks, not agent performance. A reward is
optional; see the [ordinary Python reward example](environment.md#write-your-own-reward-or-evaluator).
Termination follows the declared world: this example needs no terminal assay, while
other environments may explicitly require a terminal instrument.

## Record, resume and inspect

- Save actions and native measurements; use the existing trajectory logger and replay interfaces.
- `session.snapshot()` saves private state; `ChemWorldRuntime.from_snapshot(...)` continues compatible sessions.
- Account for model usage separately from physical operations and reserve capacity for a final report if needed.
- Research Workbench and Blender Lab are deferred; use the [environment API and CLI](environment.md).

## Historical research protocols

The Work I world-fork protocol linked from the [archive guide](legacy.md) specifies
single-variable interventions and frozen evidence for a controlled study. It is not the
general authoring process for a new environment. Reproducing that study requires its bound
version; start here when designing an environment for your own agent.
