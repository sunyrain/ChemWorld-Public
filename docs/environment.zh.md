# 使用可编程环境

ChemWorld提供多领域物理模型、可组合设备、实验操作、原生单位观测、资源记录和回放。
Agent、实验目标、奖励函数和评价方法由使用者定义。自由探索不要求使用某个内置Agent，也不要求提交排行榜。

## 安装与查看能力

```bash
git clone https://github.com/sunyrain/ChemWorld-Public.git
cd ChemWorld-Public
uv sync --locked --extra dev
uv run --no-sync chemworld capabilities --format json
uv run --no-sync python -m chemworld.examples.user_reward
```

安装wheel后直接运行其环境中的`chemworld`和`python -m chemworld.examples.user_reward`。
分发名暂时保留`chemworld-bench`，本次没有更名或发布新版本。

## 接入自己的Agent

先按[环境作者指南](environment-authoring.md)选择设备、设置操作边界和资源，再接入自己的Agent。

```python
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.surface_world import surface_world_spec

session = ChemWorldRuntime(composition=surface_world_spec())
try:
    session.reset(seed=12)
    print(session.available_actions())
    print(session.action_schema("add_component"))
    # 用自己的Agent决定下一步；不需要继承内置Agent或使用固定奖励。
    action = {"operation": "add_component", "component": "CO-feed", "amount_mol": 6e-5}
    result = session.execute(action)
    print(result.observation.to_dict())
finally:
    session.close()
```

读取原生`observation.native.channels`中的数值、单位、质量和样品时间。尚未测量不等于零；
读取已有观测不会生成新测量。允许的操作和实体由世界声明及当前状态决定。
宿主负责保留内部执行诊断；不要把完整`result.info`或私有快照直接发送给受限Agent。

已有工具调用研究路径提供MCP操作、公开历史、测量文件和可选笔记本。
`InteractiveCodexExperimentAgent`是仍在使用的提供方接入实现，不是自由探索能力的唯一入口。
独立第三方MCP宿主的启动接入尚未在本轮验证；Python接口可以直接由用户控制。

## 奖励与评价由用户定义

`chemworld.examples.user_reward`仅展示普通Python函数：根据新取得的压力读数计算用户目标，
也可以传入`reward_fn=None`完全不打分。更换奖励不改变环境物理。
示例的目标值和缩放是示例作者选择，不是官方成功标准。

当前底层仍保留供已有研究／Gym消费者使用的`reward`、任务终结及历史评分字段；
它们不是新应用必须采用的目标。本轮归档入口没有改动这些执行语义。
可直接忽略旧奖励，根据公开观测和资源消耗计算自己的奖励，或在宿主侧编写独立评价。
若评价需要私有真值，应由可信评价端消费，不能回传给被评测Agent。

## 环境工具

- `chemworld capabilities`：从模块声明读取安装能力。
- `chemworld render --composition world.json --actions action.json --format json`：执行给定操作。
- `chemworld verify --submission trajectory.jsonl`：检查记录能否回放，不评价Agent好坏。
- `session.snapshot()`与`ChemWorldRuntime.from_snapshot(...)`：保存和延续支持的物理会话。

## 界面发布状态

当前只保留Research Workbench与Blender Lab两组内部界面原型，均暂缓公开发布。
Task Lab、轻量Lab控制台、Experiment Explorer和Asset Preview已从源码移除。
当前sdist／wheel不包含这些界面及专用后端，也不注册`chemworld-lab`命令。
用户通过Python API、`chemworld`环境CLI或Agent工具接口操作基座；静态文档站展示说明和案例。

进一步阅读[原生基座](native_foundation.md)、[领域工作流](native_domains.md)、
[共享实验室](native_shared_lab.md)和[发行范围](community_release.md)。
[离线研究例子](https://github.com/sunyrain/ChemWorld-Public/blob/main/examples/demo_offline_research.py)展示一种用户评价模板，不是全环境强制协议。
旧排行榜、基线和逐步LLM适配见[历史入口](legacy.md)。
