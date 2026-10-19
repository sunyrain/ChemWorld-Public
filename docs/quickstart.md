# 5-minute quickstart

This path creates one environment, inspects the public contract and commits a valid measurement lifecycle. It uses no API key or external model provider.

## Install

```bash
git clone https://github.com/sunyrain/ChemWorld-Public.git
cd ChemWorld-Public
uv sync --locked --extra dev
```

ChemWorld supports Python 3.11 and 3.12.

## Inspect before acting

```python
import gymnasium as gym
import chemworld

env = gym.make("ChemWorld", task_id="reaction-to-assay", seed=0)
observation, info = env.reset(seed=0)

print(env.unwrapped.task_prompt()["text"])
print(env.unwrapped.action_schema("heat"))
print(env.unwrapped.available_actions()[:3])
```

The public surface tells the agent what it may do. It does not expose evaluator-owned laws or hidden state.

## Run a complete example

```bash
uv run --no-sync python examples/demo_manual_event_sequence.py
```

Then open [the first notebook](notebooks.md) to see validation, intermediate HPLC feedback, final assay and resource accounting step by step.

## Check current main

```bash
uv run --no-sync pytest -q
uv run --no-sync python scripts/build_readme_visuals.py --check
```

Both checks work offline after installation.

[Current runtime and historical reproduction](current-foundation.md).
