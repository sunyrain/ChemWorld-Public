# 认识操作语言

Operation 是 Agent 在 ChemWorld 中真正能做的事：投料、改变条件、测量、分离或结束实验。
它刻意保持在人类可读的实验动作与机器可验证的 Gym action 之间。

## 一条操作长什么样

每个操作都有一个 `operation` 名称，并按需携带参数：

```python
{"operation": "heat", "temperature_K": 350.0, "duration_s": 1200.0}
```

常见参数包括：

| 字段 | 表示什么 |
| --- | --- |
| `volume_L` | 加入或处理的体积 |
| `amount_mol` | 物质的量 |
| `temperature_K` | 目标温度 |
| `duration_s` | 持续时间 |
| `stirring_rpm` | 搅拌速率 |
| `phase` | `organic`、`aqueous` 等相选择 |
| `instrument` | 要调用的测量工具 |

字段名保持英文，因为它们属于稳定 API；界面和文档负责提供中文解释与单位。

## 从投料到终检

| 阶段 | 常用操作 | 会改变什么 |
| --- | --- | --- |
| 准备 | `add_solvent`、`add_reagent`、`add_catalyst` | 物料、体积与配方身份 |
| 反应 | `heat`、`cool`、`stir`、`wait` | 时间、温度、转化、能耗与风险 |
| 测量 | `measure` | 公开观测、样品量、成本与预算 |
| 分相 | `add_extractant`、`mix`、`settle`、`separate_phase` | 相组成、夹带与回收 |
| 后处理 | `wash`、`dry`、`concentrate`、`crystallize`、`filter`、`distill` | 纯度、回收率与物流账本 |
| 结束 | `terminate`、`final_assay` | 阶段切换或最终评分 |

操作是有状态的。连续两次 `heat` 会从当前温度与组成继续推进；没有形成两相时调用
`separate_phase` 会被拒绝，而不是假装成功。

## 执行前先问环境

```python
available = env.unwrapped.available_actions()
schema = env.unwrapped.action_schema("heat")
check = env.unwrapped.validate_action(
    {"operation": "heat", "temperature_K": 350.0, "duration_s": 1200.0}
)
```

这三个接口分别回答：现在能做什么、参数应该怎么写、这条具体动作是否合法。Agent 不需要通过
反复失败来猜规则。

## 这套语言刻意不做什么

Operation 不是现实实验室 SOP，也不是设备控制协议。它优先保证四件事：含义清楚、失败可解释、
状态变化可审计、轨迹可以确定性回放。需要具体字段时继续查看
[Action 与 Recipe](action_schema.md)。
