# 五分钟开始

这条路径会创建一个环境、读取公开任务合同，并完成一条合法的测量生命周期。整个过程不需要 API key 或外部模型 Provider。

## 安装

```bash
git clone https://github.com/sunyrain/ChemWorld-Public.git
cd ChemWorld-Public
uv sync --locked --extra dev
```

ChemWorld 支持 Python 3.11 和 3.12。

## 先检查，再操作

```python
import gymnasium as gym
import chemworld

env = gym.make("ChemWorld", task_id="reaction-to-assay", seed=0)
observation, info = env.reset(seed=0)

print(env.unwrapped.task_prompt()["text"])
print(env.unwrapped.action_schema("heat"))
print(env.unwrapped.available_actions()[:3])
```

公开接口会告诉 Agent 可以做什么，但不会泄露评测者拥有的世界规律或隐藏状态。

## 运行完整示例

```bash
uv run --no-sync python examples/demo_manual_event_sequence.py
```

随后打开[第一个 notebook](notebooks.md)，逐步查看动作验证、中间 HPLC、终检和资源账本。

## 检查当前基座

```bash
uv run --no-sync pytest -q
uv run --no-sync python scripts/build_readme_visuals.py --check
```

安装完成后，这两个检查都可以离线运行。

[当前运行时与旧版复现](current-foundation.md)。
