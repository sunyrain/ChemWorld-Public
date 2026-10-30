<!-- Generated from https://github.com/sunyrain/ChemWorld/blob/main/docs/action_schema.md; edit the upstream guide. -->

# 编写 Action 与 Recipe

Action 是一次操作，Recipe 是一组按顺序执行的 Action。两者都使用普通字典/JSON，因此同一份内容
可以交给 Gym 环境、baseline、轨迹回放和提交验证器。

## Action：提交一个动作

最小 Action 只包含操作名：

```python
{"operation": "terminate"}
```

带参数的常见例子：

```python
{"operation": "add_solvent", "volume_L": 0.03, "solvent": 1}
{"operation": "heat", "target_temperature_K": 350.0, "duration_s": 1200.0, "stirring_speed_rpm": 800.0}
{"operation": "measure", "instrument": "final_assay"}
```

不要凭记忆填写范围。运行时 schema 会告诉你必填字段、单位、上下界和类别选项：

```python
schema = env.unwrapped.action_schema("heat")
```

## Recipe：描述一条实验路线

```python
recipe = [
    {"operation": "add_solvent", "volume_L": 0.03, "solvent": 1},
    {"operation": "add_reagent", "amount_mol": 0.012},
    {"operation": "heat", "target_temperature_K": 350.0, "duration_s": 1200.0, "stirring_speed_rpm": 800.0},
    {"operation": "terminate"},
    {"operation": "measure", "instrument": "final_assay"},
]
```

Recipe 适合固定 baseline、教程和回放。它只是动作清单，并不提前保证每一步合法：环境仍会结合
当前任务、物料状态、预算和前置条件逐步检查。

## 动作失败时看哪里

| 信号 | 含义 | Agent 通常应该做什么 |
| --- | --- | --- |
| `precondition_failed` | 当前状态还不能执行该操作 | 补齐前序步骤或换一个合法操作 |
| `unsafe` / `unsafe_by_task_limit` | 触发环境或任务风险边界 | 降低强度、缩短时间或改变路线 |
| `high_cost` | 成本超过任务阈值 | 减少测量或选择更便宜的流程 |
| `low_selectivity` | 当前路线选择性不足 | 调整条件或先获取信息 |
| `constitution_failed` | 状态账本或守恒检查失败 | 将运行视为环境/执行失败并保留证据 |

这些信号是可学习反馈，不是可以忽略的日志。需要查看当前合法动作时，优先使用
`available_actions()` 和 `validate_action()`。

!!! note "与现实设备的边界"
    Action schema 描述可评测的虚拟实验操作，不等同真实机器人指令。设备驱动、权限、联锁和安全
    审批应由独立 adapter 处理。
