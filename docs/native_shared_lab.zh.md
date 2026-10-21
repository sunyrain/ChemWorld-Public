# 一个有限的共享实验室时钟

> 中文预览译文；技术接口与适用范围请同时核对[英文原文](https://sunyrain.github.io/ChemWorld-Public/explore/?lang=en#guides/chemworld/native_shared_lab)。

在受保护的仓库环境中使用自包含的示例：

```
uv run --no-sync python examples/native_shared_lab.py
```

`shared_lab_world()` 的作者设计了两种不同的液体容器，一个是有限功率加热器，另一个是预编程的 HPLC。它利用现有的会话、原生本地求解器、采样法则、设备内存和交易。不涉及任何外部化学后端、模型密钥或分布式调度器。

```
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.lab_world import shared_lab_world

lab = ChemWorldRuntime(composition=shared_lab_world())
lab.reset(seed=17)
lab.execute({"operation": "queue_lab_heat", "duration_s": 30,
             "target_temperature_K": 350})
lab.execute({"operation": "queue_lab_measure"})
lab.execute({"operation": "select_vessel", "vessel": "sample"})
lab.execute({"operation": "queue_lab_measure"})
result = lab.execute({"operation": "wait", "duration_s": 120})
print(result.info["actual_duration_s"])  # 30: next heater completion
print(lab.lab_status())                  # analyzer result is not yet visible
```

默认的 `CompositionPhysics.lab_clock=None` 是原始的 **本地** 时钟。 显式地选择 `LabClockSpec` 可以选择整个会话的共享模式。 无论是否选择，所有已启用的设备都会一起前进。 请求的等待是上限：它会在下一次完成事件后返回。 另一个等待会启动同一物理时间上的下一个 FIFO 任务。 在等待之间，零时间请求可以排队，但永远不能在已采集的样本上进行后向修改。 在相同的时间戳下，持续的序列号决定完成的顺序。

分析样本于 **服务开始**，而不在于提交队列或发布结果。现有的采样器能够处理真实的样本提取和持续的色谱柱带效应。结果会传递。 `sampled_at` 以及 `available_at` 在...上 `lab` 时钟；只有已完成的包才会显示 `lab_status()` 或者 `info["lab_clock"]`. 调查过程不消耗任何资源或引入随机性。检测器故障会导致无效的完成作业，而不是恢复样本。 过程、采样和检测器 RNG 的状态是独立的。

物理时间单位在整个系统中只计一次。分析工作会分别计算实际的繁忙时间，而采样费用在服务开始时只计一次。排队/控制/等待的通话消耗了正常的活动预算；工作记录会显示单独的获取历史。如果出现数值错误，则会丢弃所有未提交的片段。首先，物理停止会将所有系统重新调整到最早的端点，提交该片段，然后需要进行完整的实验室重置才能继续进行。已提交的事件仍然在记录中。

私有设备 `snapshot().to_dict()` 和 `ChemWorldRuntime.from_snapshot(...)` 能够精确地恢复设备所有权、队列订单、待处理的 SSA 事件以及未发布的数据包。绝不允许向任何代理提供私有状态快照作为观察证据。

## 明确的限制 {#explicit-limits}

一个或两个独立的、原生的、封闭式热处理/SSA 容器；最多一个加热器和一个非终端的原生离线分析仪。 默认的有限会话最多允许 64 个任务（可配置为 256 个），并保留所有完成记录。 热处理和 SSA 可以是独立的容器，绝不涉及在一个容器中隐含的混合求解器。 仅限队列、选择、等待和终止操作在初始共享模式中可用。 不支持部分实体重置、终端/现场仪器、容器间/开放流以及替代相策略，而是选择默默地混合时钟。 色谱分析方法的配置需要在共享会话之前进行。 这些是合成的开发示例，而不是实验室化学或机器人协调的说法。

排队式检测读数支持按顺序和有键的噪声。有键模式使用现有的观测-坐标基本结构，以及实验室命名空间和持久的排队序列。轮询不会改变键、物理采样或 SSA 串流。检测器的种子和坐标输入保持在私有的快照中。可选的 `lab_schedule.duration_range_s` 声明公共等待范围降至 1 微秒；如果没有，则普通的一秒下限保持不变。这允许在短时间内离线采集时，而无需改变仪器的声明的分析时间或底层物理速率。
