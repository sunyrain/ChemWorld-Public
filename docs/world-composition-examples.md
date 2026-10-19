# Public world-authoring examples

These requests are retained from the frozen release and still compile on current
main. Their old pattern labels are historical: current reports use
`component-network`, with vessel-qualified state owners. See
[current foundation](current-foundation.md).

ChemWorld uses one object hierarchy for authored compositions and the 15 registered reference
tasks: components form a world, a task contract supplies the operating surface, a scenario fixes
initial state and seeds, and execution produces a trajectory. A controlled world fork is a
narrower object derived from a valid world; it is not another spelling of general composition.

The examples below are construction requests. Successful compilation establishes a valid public
contract, not physical qualification, task-space enumeration or agent performance.

## Example matrix

| Authoring case | Declared components | What it demonstrates |
| --- | --- | --- |
| Single process module | phase + mandatory observation | A world may expose one process component while still carrying the observation and lifecycle surface required for execution. |
| Cross-module world | reaction + thermal + observation | Two process components share material, temperature, time, energy and measurement interfaces. |
| Multi-stage world | reaction + thermal + phase + separation + observation | Reaction and downstream purification operations are compiled into one task contract and one shared runtime state. |
| Controlled fork | one valid parent world plus one private-law intervention | The public task surface stays fixed while exactly one declared private component changes for attribution. |

The corresponding public files are:

- `examples/world-authoring/composed-equilibrium-characterization-v0.1.json`;
- `examples/world-authoring/composed-reaction-assay-v0.1.json`;
- `examples/world-authoring/composed-reaction-purification-v0.1.json`; and
- `examples/world-authoring/mechanism-fork-v0.1.json` and
  `examples/world-authoring/material-law-fork-v0.1.json` for controlled forks.

The phrase “single process module” excludes the mandatory observation component. Every executable
Each composition must still expose public measurement and a reachable terminal evaluation.

## Frozen pre-launch reference paths

Compilation alone does not establish that an authoring example can complete a lifecycle. The
The published use-case qualification therefore freezes two deterministic paths in
`examples/world-authoring/use-case-reference-paths-v0.1.json` and executes them in the example
tests.

- U02 uses the `phase + observation` example with the supported `balanced` objective and a
  five-action budget. It adds solvent and reagent to create a measurable sample, performs one pH
  measurement, terminates and commits one final assay. UV--Vis remains an available but unselected
  instrument.
- U03 uses the multi-stage reaction--phase--separation example with the supported `balanced`
  objective. Its first submitted action deliberately attempts phase separation before phase
  preparation and must roll back with `precondition_failed` while preserving physical state. The
  following 18 actions execute the legal reaction, measurement, extraction, purification,
  termination and final-assay recovery path.

These paths are public construction and lifecycle checks, not agent-performance measurements or
physical-laboratory validation.

## Compile an example

```python
import json
from pathlib import Path

import chemworld

request = json.loads(
    Path(
        "examples/world-authoring/composed-reaction-purification-v0.1.json"
    ).read_text(encoding="utf-8")
)
compiled = chemworld.compile_world_composition(request)

print(compiled.compatibility.pattern)
print(compiled.task_spec.allowed_operations)
print(compiled.to_public_dict()["task"]["resources"])
```

The compiler checks component dependencies, state ownership, parameter units and bounds, operation
and instrument surfaces, resource reachability and lifecycle closure before an environment is
constructed. The multi-stage example deliberately declares its complete separation surface so the
component-owned operations and task-owned operations can be compared directly.

## How the 15 reference tasks fit the same contract

The registered tasks are reference points, not the extent of the composition space. Their world
surfaces fall into eight declared component patterns:

| Component pattern | Registered reference tasks |
| --- | --- |
| reaction + thermal + observation | `reaction-to-assay`, `reaction-optimization-standard`, `reaction-safety-constrained`, `reaction-mechanism-explanation`, `low-budget-characterization`, `public-private-generalization` |
| reaction + thermal + phase + separation + observation | `reaction-to-purification`, `purity-yield-tradeoff`, `tool-agent-planning` |
| phase + separation + observation | `partition-discovery` |
| reaction + thermal + crystallization + observation | `reaction-to-crystallization` |
| reaction + thermal + distillation + observation | `reaction-to-distillation` |
| reaction + thermal + continuous flow + observation | `flow-reaction-optimization` |
| reaction + electrochemistry + observation | `electrochemical-conversion` |
| phase + observation | `equilibrium-characterization` |

Tasks sharing a component pattern remain distinct through the task-contract overlay: objective,
budget, episode mode, allowed operations and instruments, observation and termination policies,
success metrics, world split and seeds. Scenarios then bind concrete initial state and hidden
parameters without changing that public hierarchy.

The machine-readable mapping is
`examples/world-authoring/reference-task-contract-map-v0.1.json`. It is checked against the public
task registry and the composition compatibility checker. The mapping does not claim that an
authoring example reproduces a frozen registered task byte for byte, and it does not transfer a
qualification result from a reference task to a new composition.

## Composition and controlled forks stay separate

General authoring selects a compatible component set and task surface. A controlled fork starts
from one already valid world, changes exactly one registered private-law target and preserves the
public action, observation, instrument, resource, failure, scoring and task contract. Use the
composition examples to create worlds and the fork examples only for controlled attribution.

This separation is why the same hierarchy can describe both open construction and controlled
branching without treating the 15 reference tasks as an exhaustive benchmark catalogue.
