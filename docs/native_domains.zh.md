# 运行集成后的原生示例

> 中文预览译文；技术接口与适用范围请同时核对[英文原文](https://sunyrain.github.io/ChemWorld-Public/explore/?lang=en#guides/chemworld/native_domains)。

这些是固定的合成开发工作流程，而不是自主的模型运行或与真实实验室数据的拟合。它们使用与原生领域相同的 `ChemWorldRuntime`、Gym、CLI、交易系统和回放功能。不需要任何模型密钥或可选的化学后端。预装的入口点位于 wheel 和 sdist 中，不需要导入任何测试辅助工具或仅限于仓库的配置。

从受保护的存储环境：

```
uv run --no-sync python -m chemworld.examples.native_domains --output /tmp/native-domains
uv run --no-sync chemworld verify --submission /tmp/native-domains/heater.jsonl
uv run --no-sync python -m chemworld.examples.native_capabilities --format json
```

在虚拟环境中安装了内置轮后，对应的输入是 `python -m chemworld.examples.native_domains --output /tmp/native-domains`。 使用 `--workflow NAME` 选择一个，或重复使用以选择多个。 输出结果是实际的动作/读数轨迹和终端快照，而不是虚构的分数。

| 工作流程 | 它锻炼的是什么？ |
| --- | --- |
| `heater` | 一个校准后的有限加热装置：气体表面反应 → 液相吸附 → 热能 SPM。 相同的装置移动，而不是复制。 每个容器都保留自身的材料和本地经过的时间。 |
| `membrane` | 在有限的进水/泵/RO系统/渗透水和清洗/废液物料库存中，进行三次清洗/冲洗循环；污垢仍然存在，且没有进行补水/重置。 |
| `mineral` | 碳酸盐/方解石的动力学过程，有限的酸性物质输入，相转移，采样和pH值。 |
| `shared-clock` | 独立的、低复制率的 A/B 和 SSA 型号，以及确定性的 A/B 型号，向一个离线分子计数器发出两个竞争性请求，导致结果延迟，并且持续运行 RNG。 |
| `distillation` | 有限的托盘残留，保留接收器，定时分液收集，完全回流和回流变化。 |
| `crystal` | 播种、有限冷却/生长、过滤、重新悬浮、温和溶解以及第二次冷却间隔。 |
| `solid` | 二元分离 → 加热 → 局部平衡状态下冷却，并读取原生相。 |

## 设备和时间界限 {#device-and-time-boundaries}

`reconnect_heater` 是一种在明确编写的系统中，不涉及物理时间或空间的连接动作。 **本地时钟** 组成部分。首先进行暂停控制；世界上必须只有一个热源设备，并且该设备必须与一个具有兼容原生温度域的目标进行匹配。该设备的身份、校准和设置必须保持不变；材料不得移动。使用 `select_vessel` 然后 `heat` 为了在新的目标位置运行。它不实现机器人的运动、空间碰撞避免或通用仪器的热插拔。如果连接被拒绝，则遵循正常的程序，但不会改变材料、设备或物理时间。

`shared-clock`的工作流程与众不同：所有启用的容器在第39条的有限调度器下，同时老化。它不进行**和**的关联。SSA以及ODE在单个容器内进行，分子捕获是零载体体积的伯努利采样，而不是液体分取。不要将使用相同种子多次重置解释为独立的实验。

使用 `--noise-mode keyed` 选择现有的已标记的探测器原型。例如，使用世界/进程种子 7 和独立的观察种子 12。每个输出都包含解析后的组成和字面动作序列，保存在 `*-recipe.json` 文件中。共享时钟示例明确声明了亚秒级的等待，以便快照可以保留一个繁忙的一秒分析器和一个未执行的 SSA 事件；它不会缩短分析器的物理分析时间。

一个由用户编写的共享时钟，使用 `LabClockSpec(vessel_capacity=4, ...)` 允许四个独立的闭环热/SSA容器。 默认值为两个；显式整数容量为 1–64，与有限的任务队列 `capacity` 区分。 1、2 和 4 个容器路径具有直接的功能覆盖。 所有声明的容器都会演化到相同的事件终点，包括未选择的容器。 更改声明顺序或选择另一个容器不会选择数值组装顺序。 一个物理加热器和一个离线分析仪提供持久的FIFO请求。如果切片失败，则会丢弃样本，RNG变化；现有的被拒绝操作的成本/风险惩罚已记录在案。超出已声明的有限设备、多个设备和同一容器的开放连接。ODE/SSA混合反应仍然缺乏充分的支持。

`chemworld.examples.closed_shared_labs.reactive_closed_lab()` 作者设计了四个独立的合成系统：一个基于热的A/B反应，热SPM电池、有限的吸附接触以及低复制SSA. `phase_closed_lab()` 结合了有限的晶体生长与独立的 Al/Cu 固体样品。 使用任何一种配方构建 `ChemWorldRuntime(composition=...)`，并使用 `select_vessel`、`queue_lab_heat`、可选的 `queue_lab_measure` 和 `wait`。 电力驱动是一种明确编写的初始程序；这些配方不添加控制器。 只有一个有限的热源在不同系统中移动，并且每个模型都保持自身的温度范围。 一种等温电池可以推进，但无法接受加热。 所有策略必须明确声明 `shared_clock=True`；单独安装模块并不能赋予这种能力。 物理上的截止点将所有系统同步到实际时间，并停止实验室。 这些组合使用现有的原生方程和合成参数；它们不建立材料的匹配。

`chemworld.examples.open_shared_labs.open_shared_lab("transport")` 将有限的资源（如泵/管道/换热器）与一个封闭式的热瓶结合起来。选择 `"membrane"` 对于有限的 RO 型号，保留污垢、清洗/废弃物，以及一个单独的容器。选择该装置的 `reactor`，打电话 `set_flow_rate`，然后使用 `wait` 为了全面提升实验室的水平。选择 `bottle` 以及 `queue_lab_measure` 在保持流动的同时，占据其一台离线色谱仪。现有的本地 `run_flow` 该操作无法在此引入第二个时钟。

每个有限的装置都会一次性运行，包括其来源和收集容器。配置的容器容量计算物理容器的数量：对于运输示例，为三个；对于污垢膜示例，为五个。有限的来源、收集限制或清洗结束，将所有系统与实际停止状态对齐。全球材料、泵的工作和热量保持在原始记录中；连接系统的参考信息出现在结果诊断中。在编译过程中，额外的路径、共享的储水池、未建模的循环以及重叠的策略所有权将被拒绝。这些配方重复使用原始的有限流动、稀释/等温膜和热方程。

## 残余值和热量计算器的选择 {#residual-and-caloric-solver-choices}

默认情况下，`BatterySPMSpec` 和 `TransientElectrodeSpec` 将使用 `numerical_form="residual"`。 将 `numerical_form="ode"` 设置为显式选择其原始 ODE 路径。 这种选择与物理模型一起保存，并在快照恢复后仍然有效。 即使求解失败仍然是一个数值错误，它不会选择其他算法。

两条路径都使用相同的有限物料库存、运输和反应方程、能量约定、设备限制和运行时交易。对于 SPM，残余路径同时解决电流和电压的约束条件，包括恒定电流、恒定电压和有限源电流的限制。非终止事件确定限制过渡。代数值出现在求解器诊断中，并从物理状态重新计算，它们不是另一个持久的库存。平面电极保持其现有的氧化还原、扩散和双层电荷模型以及等温能量范围。

常见的后端可以将常规的块索引-1 残差系统简化为：BDF或者 Radau。这并不是一个任意索引。DAE求解器。诊断功能分别识别解析和有限差分导数；常用的物理模型使用解析速率矩阵和显式数值状态/时间导数。

`FiniteTransportSpec.caloric_algorithm` 默认使用 `"residual"`，消耗相同的液体热焓、分析热容和仪器热容。 明确的 `"brentq"` 选择仍然可用。 `caloric_residual_system` 还可以提供针对目标能量和已声明的物理参数的原生温度响应。 如果缺少分析参数的导数，则报告为不可用。 这些数值选择均不扩大底层合成材料的适用范围。

## 能力和验证范围 {#capability-and-validation-scope}

请看 [生成的模块表](native_capabilities.md)。它的来源是已安装的声明和它们的范围说明，而不是重复的可用性列表。实际的数值域来自编写好的物理和设备规格，并且受到任务权限的限制。两个模块的存在并不意味着它们之间的所有耦合都得到支持。SPM，稀释 0DRO并且，有限的二元平衡状态并不完整。PyBaMM, WaterTAP或者CALPHAD替代品。

导出实际的公共行动范围、材料连接以及已安装的仪器声明，用于生成一个食谱（或原始的组成文件）：

```
uv run --no-sync python -m chemworld.examples.native_capabilities --composition /tmp/native-domains/heater-recipe.json
```

本报告记录当前任务和活动容器的状态；未来的容器选择或物理状态变化可能会影响可用性。它重用运行时提供的公共合同，而不是维护一个单独的手动维护的范围表。

验证命令：

```
uv run --no-sync pytest tests/test_native_domains_end_to_end.py -v
uv run --no-sync python scripts/smoke_test_wheel.py --native-foundation-only --output /tmp/native-install.json
```

后者不仅构建了各个分布，还使用锁定默认依赖项，在仓库之外安装它们。它检查是否存在可选的上游库，运行七个示例，并在另一个进程中验证每个轨迹。这是一个本地安装测试。 **不** 包括 PyPI 的上传或正式发布。 接受记录中并未声称未实际测试的平台。
