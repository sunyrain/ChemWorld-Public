# 接入 Agent

新应用从[环境API](environment.md)和[环境作者指南](environment-authoring.md)开始。可使用自己的规划器与奖励，也可完全不评分。下文按任务评分的runner与预设适配器属于保留的历史应用；其命令入口为`python -m chemworld.legacy.cli`。

ChemWorld Agent 每一步都从公开任务状态中选择一个有类型的操作。隐藏世界状态由评测器持有；环境
负责校验请求、记录资源并生成可精确回放的轨迹。建议先跑离线 Agent，再接入外部模型。

## 能力矩阵

| 路径 | 网络或凭据 | 用途 | 公开状态 |
| --- | --- | --- | --- |
| 自定义 Python Agent | 不需要 | 自己的规划器或策略 | 稳定协议 |
| 内置经典 Agent | 不需要 | 连通性检查与可复现基线 | 稳定 |
| Replay Agent | 不需要 | 重放已经捕获的模型轨迹 | 稳定 |
| DeepSeek `LiveLLMAgent` | `DEEPSEEK_API_KEY` | 操作级在线模型实验 | 可选适配器 |
| Codex subscription client | 已完成 `codex login` | 通过 ChatGPT subscription 调用结构化输出 | 可选适配器 |
| Interactive Codex experiment | 已完成 `codex login` | 每个完整实验保留一个模型上下文 | 高级 API |
| Frozen SB3 policy | 本地 checkpoint 与 `.[rl]` | 评估已经训练好的 RL 策略 | 可选 extra |

命令行 `--agent` 只暴露无需构造参数的内置 Agent。需要凭据、模型、replay 文件、checkpoint 或
workspace 的 Agent，应在 Python 中显式创建并传给 `run_agent`。

## 最小自定义 Agent

继承 `BaseAgent` 会自动声明零外部资源。决策时只能使用公开任务信息和公开历史。

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

更完整的 Agent 可以实现 `act_with_context(context)` 或
`act_with_public_view(context, public_view)`。官方 runner 会提供当前合法操作、资源、观测摘要和
生命周期状态，但不会泄露隐藏真值。

## 离线 Agent

```bash
chemworld tasks list
python -m chemworld.legacy.cli run --task reaction-to-assay --agent scripted_chemistry --seed 0
```

其他无需参数的名称包括 `random`、`lhs`、`greedy`、`gp_bo`、`rf_ei`、
`safe_gp_bo`、`tool_using_llm_stub` 和 `llm_replay`。Python 中可使用
`chemworld.legacy.agent_registry.make_agent(name)`。

## DeepSeek 在线适配器

不要把 key 写入源码：

```bash
export DEEPSEEK_API_KEY="..."
```

```python
from chemworld.legacy.agents.live_llm import LiveLLMAgent
from chemworld.eval.runner import run_agent
from chemworld.providers.deepseek import DeepSeekClient

agent = LiveLLMAgent(
    DeepSeekClient(model="deepseek-v4-pro", thinking=True),
    role_id="public-example",
)

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

模型身份、重试、token 与估算成本会独立记录；Provider 失败不会被静默替换成其他模型或宿主动作。

## Codex subscription

安装 Codex CLI，并用 ChatGPT subscription 完成 `codex login`。然后从
`chemworld.providers.codex_subscription` 创建 `CodexSubscriptionClient`，按上面的方式传给
`LiveLLMAgent`。subscription 无法提供逐次调用的美元价格，因此 receipt 会明确记录这一限制。

`InteractiveCodexExperimentAgent` 用于在一个完整实验内保持同一 Codex 上下文。它要求显式指定
隔离 workspace 和 role ID；正式比较前应阅读类说明和对应测试。

## 可复现与安全边界

- 不提交 API key、Provider 原始响应或隐藏推理。
- 任务、seed、模型身份、prompt 合同和资源上限不一致时，不比较 live-agent 分数。
- 对外报告分数前必须验证并回放轨迹。
- ChemWorld 是软件模型环境，不生成真实实验室操作规程。
- 仓库中的 Agent 证据是有限资格测试，不代表某个 Provider 或 Agent 普遍更优。
