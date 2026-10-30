<!-- Generated from https://github.com/sunyrain/ChemWorld/blob/main/docs/operations.md; edit the upstream guide. -->

# 认识操作语言

Operation 是 Agent 在 ChemWorld 中真正能做的事：投料、改变条件、测量、分离或结束实验。
它刻意保持在人类可读的实验动作与机器可验证的 Gym action 之间。

## 一条操作长什么样

每个操作都有一个 `operation` 名称，并按需携带参数：

```python
{"operation": "heat", "target_temperature_K": 350.0, "duration_s": 1200.0, "stirring_speed_rpm": 800.0}
```

常见参数包括：

| 字段 | 表示什么 |
| --- | --- |
| `volume_L` | 加入或处理的体积 |
| `amount_mol` | 物质的量 |
| `target_temperature_K` | 夹套目标温度，不是物料瞬时实测温度 |
| `duration_s` | 持续时间 |
| `stirring_speed_rpm` | 搅拌速率 |
| `phase` | `organic`、`aqueous` 等相选择 |
| `instrument` | 要调用的测量工具 |

字段名保持英文，因为它们属于稳定 API；界面和文档负责提供中文解释与单位。

## 从投料到终检

| 阶段 | 常用操作 | 会改变什么 |
| --- | --- | --- |
| 准备 | `add_solvent`、`add_reagent`、`add_component`、`add_catalyst` | 有限目录投料、混合载体组成、体积 |
| 反应 | `heat`、`wait`、`quench` | 时间、温度、转化、能耗与风险 |
| 测量 | `measure` | 公开观测、样品量、成本与预算 |
| 分相 | `add_extractant`、`mix`、`settle`、`separate_phase` | 相组成、夹带与回收 |
| 后处理 | `wash`、`dry`、`concentrate`、`cool_crystallize`、`filter_crystals`、`distill` | 纯度、回收率与物流账本 |
| 结束 | `terminate`，再 `measure` + `instrument="final_assay"` | 停止与终检是不同事件 |

`cool`、`stir`、`crystallize`、`filter` 不是当前的独立 operation ID。
通用混合溶剂不会因首次加入而全部锁定；具体任务仍可通过合同约束纯介质。
`add_component` 的 `component` 取自公开 `feed_catalog`，不按猜测的隐藏物种名投料。
容器路由、持续控制等扩展只在对应任务或 composition 允许时使用；从 schema 查询，不硬编码动作向量长度。

## 仪器 ID 与信号不是同一层

基础仪器为 `hplc`、`gc`、`uvvis`、`ph_meter`、`final_assay`；
当前核心也支持显式声明的 `nmr`、`ir`、`ms`，部分 full-process 合同另开放 `particle_size`。
每个任务的允许项仍读 `allowed_instruments`，不能把低层支持理解成每个任务都允许。
新光谱仪使用 `configure_instrument` 设置扫描次数、分辨率和稀释倍数，再用 `measure` 采集。
有限匿名校准通道不是结构鉴定模型；已有谱包中 IR/NMR-like 特征也不自动授权独立测量。

操作是有状态的。连续两次 `heat` 会从当前温度与组成继续推进；没有形成两相时调用
`separate_phase` 会被拒绝，而不是假装成功。

## 执行前先问环境

```python
available = env.unwrapped.available_actions()
schema = env.unwrapped.action_schema("heat")
check = env.unwrapped.validate_action(
    {"operation": "heat", "target_temperature_K": 350.0, "duration_s": 1200.0, "stirring_speed_rpm": 800.0}
)
```

这三个接口分别回答：现在能做什么、参数应该怎么写、这条具体动作是否合法。Agent 不需要通过
反复失败来猜规则。

## 这套语言刻意不做什么

Operation 不是现实实验室 SOP，也不是设备控制协议。它优先保证四件事：含义清楚、失败可解释、
状态变化可审计、轨迹可以确定性回放。需要具体字段时继续查看
[Action 与 Recipe](action_schema.md)。
