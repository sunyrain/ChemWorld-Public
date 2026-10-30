<!-- Generated from https://github.com/sunyrain/ChemWorld/blob/main/docs/agents.md; edit the upstream guide. -->

# Connect an agent

Start with the [environment API](environment.md) and [authoring guide](environment-authoring.md). You can supply any planner and choose your own reward, or use no reward. The task-scored runner and preset adapters below are retained historical applications; their CLI is `python -m chemworld.legacy.cli`.

ChemWorld agents choose one typed operation at a time from public task state. The environment keeps
hidden world state evaluator-owned, validates every request, records resource use and writes an exact
replay trajectory. Start offline before introducing an external model.

## Capability matrix

| Path | Network or key | Intended use | Public status |
| --- | --- | --- | --- |
| Custom Python agent | No | Your own planner or policy | Stable protocol |
| Built-in classic agents | No | Smoke tests and reproducible baselines | Stable |
| Replay agent | No | Re-run a previously captured model trace | Stable |
| DeepSeek `LiveLLMAgent` | `DEEPSEEK_API_KEY` | Operation-level online model experiments | Optional adapter |
| Codex subscription client | Existing `codex login` | Audited structured calls through a ChatGPT subscription | Optional adapter |
| Frozen SB3 policy | Local checkpoint and `.[rl]` | Evaluation of an already trained RL policy | Optional extra |

The command-line `--agent` flag intentionally exposes only constructor-free built-ins. Agents that
need credentials, model selection, replay paths, checkpoints or workspaces are created in Python and
passed to `run_agent` explicitly.

## Minimal custom agent

Subclassing `BaseAgent` supplies a zero-external-resource manifest. Use only public task information
and public history when choosing an action.

```python
from chemworld.agents.base import BaseAgent, HistoryRecord
from chemworld.eval.runner import run_agent


class TutorialAgent(BaseAgent):
    name = "tutorial"

    recipe = (
        {"operation": "add_solvent", "volume_L": 0.030, "solvent": 1},
        {"operation": "add_reagent", "amount_mol": 0.012},
        {"operation": "add_catalyst", "catalyst": 2, "catalyst_amount_mol": 0.0004},
        {"operation": "heat", "target_temperature_K": 350.0, "duration_s": 1200.0,
         "stirring_speed_rpm": 800.0},
        {"operation": "measure", "instrument": "hplc"},
        {"operation": "quench"},
        {"operation": "terminate"},
        {"operation": "measure", "instrument": "final_assay"},
    )

    def act(self, history: list[HistoryRecord]) -> dict[str, object]:
        return dict(self.recipe[len(history)])


run_agent(
    env_id="ChemWorld",
    agent=TutorialAgent(),
    world_split="public-dev",
    budget=8,
    objective="balanced",
    seed=0,
    task_id="reaction-to-assay",
    output_path="runs/tutorial-agent.jsonl",
)
```

For a capable agent, prefer `act_with_context(context)` or
`act_with_public_view(context, public_view)`. The official runner detects these methods and provides
the current legal operations, resource state, observation summaries and lifecycle status without
exposing hidden truth.

## Built-in offline agents

List tasks and run a deterministic connectivity check:

```bash
chemworld tasks list
python -m chemworld.legacy.cli run --task reaction-to-assay --agent scripted_chemistry --seed 0
```

Other constructor-free names include `random`, `lhs`, `greedy`, `gp_bo`, `rf_ei`,
`safe_gp_bo`, `tool_using_llm_stub` and `llm_replay`. Use `make_agent(name)` from
`chemworld.legacy.agent_registry` when composing runs in Python.

## Live DeepSeek adapter

Keep the key out of source files and shell history:

```bash
export DEEPSEEK_API_KEY="..."
```

```python
from chemworld.legacy.agents.live_llm import LiveLLMAgent
from chemworld.eval.runner import run_agent
from chemworld.providers.deepseek import DeepSeekClient

client = DeepSeekClient(model="deepseek-v4-pro", thinking=True)
agent = LiveLLMAgent(client, role_id="public-example")

run_agent(
    env_id="ChemWorld",
    agent=agent,
    world_split="public-dev",
    budget=18,
    objective="balanced",
    seed=0,
    task_id="reaction-to-assay",
    output_path="runs/deepseek-reaction-to-assay.jsonl",
)
```

Model identity, retries, token use and estimated cost are recorded separately. Provider failure is
not silently replaced with another model or a host-generated action.

## Codex subscription adapter

Install the Codex CLI, complete `codex login` with a ChatGPT subscription, and construct
`CodexSubscriptionClient` from `chemworld.providers.codex_subscription`. Pass that client to
`LiveLLMAgent` using the same runner pattern above. Subscription usage has no per-run USD price, so
the accounting receipt reports that limitation rather than inventing a cost.

A persistent research context, study design and prediction protocol belong to the caller.
The former study-specific interactive adapter is archived; the generic runtime,
agent runner, public history and document workspace remain available.

## Reproducibility and safety boundary

- Never commit API keys, provider response dumps or private reasoning.
- Do not compare live-agent scores unless task, seed, model identity, prompt contract and resource
  limits are fixed.
- Validate and replay every submitted trajectory before reporting a score.
- ChemWorld is a software-model environment, not a generator of physical laboratory procedures.
- The checked-in live-agent evidence is finite qualification evidence, not a claim that one provider
  or agent is generally superior.
