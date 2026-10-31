<!-- Generated from https://github.com/sunyrain/ChemWorld/blob/main/docs/environment.en.md; edit the upstream guide. -->

# Use the programmable environment

ChemWorld provides physical models, composable equipment, experimental operations,
native-unit observations, resource records and replay. Users own their agents,
objectives, rewards and evaluation methods. Free exploration requires no preset
agent and no leaderboard submission.

## Install and inspect capabilities

```bash
git clone https://github.com/sunyrain/ChemWorld-Public.git
cd ChemWorld-Public
uv sync --locked --extra dev
uv run --no-sync chemworld capabilities --format summary
uv run --no-sync python -m chemworld.examples.user_reward
```

Installed-wheel users run their environment's `chemworld` and
`python -m chemworld.examples.user_reward` directly. The distribution name remains
`chemworld-bench`; this change does not rename or publish a package.

## Connect your agent

Use the [environment authoring guide](environment-authoring.md) to choose equipment,
operation bounds and resources, then connect your own agent.

```python
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.surface_world import surface_world_spec

session = ChemWorldRuntime(composition=surface_world_spec())
try:
    session.reset(seed=12)
    print([item["operation"] for item in session.available_actions()])
    print(session.action_schema("add_component"))
    # Your agent chooses the action. No preset policy or reward is required.
    action = {"operation": "add_component", "component": "CO-feed", "amount_mol": 6e-5}
    result = session.execute(action)
    print(result.observation.to_dict())
finally:
    session.close()
```

Consume native channels with their units, quality, sample identity and timestamps.
An unmeasured value is not zero. Reading an existing observation performs no new
measurement. Supported operations depend on the declared world and current state.
Keep host diagnostics private: do not forward the entire `result.info` or a private
snapshot to a restricted agent.

The runtime exposes operations, public history and a document workspace.
Your application owns the research loop, persistent model context and evaluation.
The old study-specific interactive adapter is archived; it is not required to use these APIs.

## Write your own reward or evaluator

`chemworld.examples.user_reward` uses an ordinary Python function over a newly
acquired pressure reading. Pass `reward_fn=None` to use no reward. Changing the
reward does not change the physics. Its target and scale are demonstration choices,
not official success criteria.

The runtime retains reward, task termination and historical scoring fields used by
existing research and Gym consumers. This entry-point cleanup does not alter their
execution semantics. New applications can ignore the old reward and compute their
own from public observations and resources, or use a separate host evaluator.
Private-reference evaluation belongs on the trusted host, not in agent feedback.

## Environment tools

- `chemworld capabilities`: inspect installed module declarations.
- `chemworld render --composition world.json --actions action.json --format json`: execute actions.
- `chemworld verify --submission trajectory.jsonl`: verify replay, not agent performance.
- `session.snapshot()` and `ChemWorldRuntime.from_snapshot(...)`: save and continue supported sessions.

## Interface release status

Research Workbench and Blender Lab are the two retained internal interface prototypes;
their public release remains deferred. Task Lab, Lab Console, Experiment Explorer and
Asset Preview have been removed from the source tree. Current sdists and wheels exclude
these interfaces and their dedicated backends, and do not register `chemworld-lab`.
Use the Python API, `chemworld` environment CLI or agent tools. The static documentation
site provides guides and recorded examples.

See [native sessions](native_foundation.md),
[domain workflows](native_domains.md),
[shared laboratories](native_shared_lab.md)
and [distribution scope](community_release.md).
[Offline research](offline_research.md) demonstrates one optional evaluation template.
Historical rankings, baselines and step-wise LLM adapters have [explicit legacy entry points](legacy.md).
