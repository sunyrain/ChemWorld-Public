# API 参考

> 当前基座见[10月更新](current-foundation.md)。下文已有图示与数值证据对应8月冻结运行时。

ChemWorld v0.3.0 保持与 v0.2.0 相同的运行时接口。可以先从 Agent-facing helper 开始，再进入世界构建和评测合同。

| 主题 | 参考文档 |
| --- | --- |
| 安装与第一个环境 | [Getting started](getting_started.md) |
| 有类型操作 payload | [Action schema](action_schema.md) |
| 支持的过程操作 | [Operations](operations.md) |
| 世界组件兼容性 | [Composition contract](world-composition-contract.md) |
| 世界组合示例 | [Composition examples](world-composition-examples.md) |
| 能力覆盖 | [World capability map](world-capability-map.md) |

核心 Agent 接口位于 `env.unwrapped`：

```python
env.unwrapped.task_prompt()
env.unwrapped.available_actions()
env.unwrapped.action_schema("heat")
env.unwrapped.validate_action(action)
env.unwrapped.agent_view_bundle(observation, info)
```
