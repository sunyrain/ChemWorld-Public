<p align="center"><a href="README.md">English</a> · <strong>简体中文</strong></p>

<p align="center">
  <img src="docs/assets/chemworld-readme-banner.png" alt="ChemWorld：可编程的化学世界与自主实验。概念实验室包含反应器、膜组件、电池和晶体。" width="1200">
</p>

<p align="center"><strong>为能够开展实验的智能体提供原生仿真基座。</strong></p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.11%20%7C%203.12-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.11 和 3.12"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-167D8D?style=flat-square" alt="MIT 许可证"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/Development-0.5.0.dev0-B16D39?style=flat-square" alt="开发版本 0.5.0.dev0"></a>
  <a href="docs/native_domains.md"><img src="https://img.shields.io/badge/Examples-No%20model%20key-34495E?style=flat-square" alt="示例无需模型密钥"></a>
</p>

<p align="center">
  <a href="#快速开始">快速开始</a> ·
  <a href="#实验领域">能力范围</a> ·
  <a href="#迭代里程碑">迭代里程碑</a> ·
  <a href="docs/native_domains.md">完整示例</a> ·
  <a href="docs/native_foundation.md">原生 API</a> ·
  <a href="docs/agents.zh.md">接入智能体</a> ·
  <a href="https://sunyrain.github.io/ChemWorld-Public/?lang=zh">中文交互网站</a>
</p>

## 构建一个世界，让智能体探索它

ChemWorld 为自主智能体提供一个可以采取行动的化学世界：加入物料、加热反应器、分离混合物、执行测量，再选择下一次实验。结果来自物理模型与持续演化的状态。样品会发生变化，库存会被消耗，设备保留使用历史，先前的选择会影响后续能够发生的过程。

你定义化学体系、装置与任务。ChemWorld 通过同一套运行时处理物理演化、原生测量、资源核算与回放，并提供 **Python、Gym、CLI 和 Agent 四种接口**。

你可以用它开发实验策略、研究智能体如何利用观测，或构建可复现的仿真实验练习。改变世界、可用信息或研究预算时，实验过程依然可以被检查。

## 迭代里程碑

沿 **ChemWorld 主仓库**的开发历程归并主要升级，日期表示开发交付阶段，不对应公开仓库的集中同步日期。[当前基座](docs/current-foundation.zh.md)记录公开版本的支持范围，也可以在[交互首页](https://sunyrain.github.io/ChemWorld-Public/?lang=zh#home)浏览时间线。

| 日期 | 能力升级 |
| --- | --- |
| **2026-10-25** | **自定义环境、流程网络与模型互操作.** 公共版本已支持自定义设备、资源和奖励。主库进一步加入多回路有限流程网络、半电池 DFN 读数序列化及来源完整的 Cantera 气相机理导出，并完善物性拟合诊断；这些新开发增量尚未纳入公共包。 |
| **2026-10-24** | **从物性计算扩展到耦合过程与系统分析.** 新增水/蒸汽、湿空气、分子与多流体热力学；连接机理生成、有限反应流、膜与水处理、多相 CEF 相图。扩展电池族、阻抗、随机混合过程和 SBML 分析，并统一轨迹字段与续算语义；各模型保留验证范围。 |
| **2026-10-23** | **扩展原生热力学与工程计算.** 纯流体 Helmholtz 物性、饱和与闪蒸连接活度模型、六种 UNIFAC 变体和立方状态方程混合规则；扩展带来源的物性导数与拟合，并补齐流量计、阀门、泵、两相流和设备几何计算。 |
| **2026-10-21–22** | **科学求解、物性数据与模型互操作。** 共同残差／DAE 基础，有限电池模型与诊断，带来源的维里气体物性与热过程，持久 Thevenin 电路；扩展选定的 CALPHAD／ChemSage 模型和 SBML／SED-ML 子集。 |
| **2026-10-20** | **可组合的多领域原生基座。** 覆盖气相、体相与表面反应，以及分离、膜、结晶、矿物、固相与随机反应网络；共享时钟、持久设备和测量队列连接跨领域流程，Python、Gym、CLI 与 Agent 统一使用续算和精确回放。 |
| **2026-10-09–13** | **可编程实验室与科研记录。** 带身份的样品、多容器与物料端口、持续控制和有限连续流，可配置 NMR／IR／MS 合成光谱与重复重结晶；完整实验历史、版本化假设笔记，以及主仓库的本地科研工作台原型。 |
| **2026-09-16–18** | **分相全流程实验。** 纯化、结晶和蒸馏使用分相库存及选择性采样，保留晶体粒群、热演化与晶种来源；自主电化学实验和实验后评估扩展连续实验流程。 |
| **2026-09-06** | **可视化实验室原型。** 主仓库接入可选 Blender 场景、移动操作臂和显式载体运输，将公开观测连接到三维场景，复用 Core 的执行、预算和回放；该原型尚非真实硬件驱动。 |
| **2026-08-13–15** | **有资源约束的连续实验。** 任务预算核算操作、物料和过程时间，并预留最终测量资源；跨轮状态、会话续接与失败恢复保留已完成操作及其消耗。 |

## 快速开始

安装 [uv](https://docs.astral.sh/uv/getting-started/installation/)，使用 Python 3.11 或 3.12。原生示例不需要模型密钥，也不依赖可选的上游化学求解器。

```bash
git clone https://github.com/sunyrain/ChemWorld-Public.git
cd ChemWorld-Public
uv sync --locked --extra dev

# 运行一次完整的表面反应与体相反应实验。
uv run --no-sync python -m chemworld.examples.native_surface_research --bulk --mobile --terminal-assay
```

示例执行 **12 个操作**：从有限库存加料、加热、测量、等待、转移物料，在接收容器中继续实验，最后完成终检。JSON 输出包含每个动作的结果、实际物理时长，以及带有原生单位的仪器读数。

从[环境作者指南](docs/environment-authoring.zh.md)设计世界，或运行`uv run --no-sync python -m chemworld.examples.user_reward`使用自己的奖励。

## 基座提供什么

| 能力 | 具体行为 |
| --- | --- |
| 独立定义世界与任务 | 声明物料、容器、设备和物理模型，再分别绑定目标、权限、预算和结束条件。同一套装置可以承载不同研究问题。 |
| 跨领域复用设备 | 加热器通过共同热接口连接兼容容器；标定保留在设备上，物料通过声明的端口流动。完整示例让同一个加热器跨越三个领域。 |
| 保留实验后果 | 催化剂覆盖度、膜污染、电池退化与仪器历史属于持久实体。后续操作使用前序实验留下的状态。 |
| 共同推进兼容物理过程 | 耦合过程共享库存、求解器与能量核算；可选的实验室共享时钟推进独立容器，并安排设备队列和延迟读数。 |
| 通过明确的仪器合同观察 | 读数保留单位、形状、质量和采样时刻。仪器合同定义噪声、量程与样品消耗；世界内部状态与智能体可见观测保持分离。 |
| 检查并回放实验 | 事务在提交前检查声明的交换和守恒。历史记录保留结果与资源消耗；快照支持继续执行，轨迹可以通过回放验证。 |

## 实验领域

当前开发基座包含下列原生模型，连接同一套状态、操作、测量和事务合同。

| 领域 | 原生能力 | 从这里开始 |
| --- | --- | --- |
| 反应与催化 | 体相动力学、气相机制、有限表面位点、可逆双位点相互作用与耦合热过程 | [表面反应示例](src/chemworld/examples/native_surface_research.py)、[机制导入](docs/native_mechanisms.md) |
| 热力学与相过程 | 状态相关物性、TP/PH 闪蒸、液液平衡、有限精馏、浓缩、吸附与结晶 | [物理模型指南](docs/native_foundation.md) |
| 输运与分离 | 有限储罐、受压力与功率约束的泵、换热、反渗透、持久污染与清洗 | [液体输运示例](src/chemworld/examples/native_liquid_transport.py)、[膜污染示例](src/chemworld/examples/native_membrane_fouling.py) |
| 电化学与电池 | 守恒扩散、有限电极、SPM/SPMe/DFN 与粒子分布模型、恒流/恒压/静置循环、热响应与 SEI | [电池及领域示例](docs/native_domains.md) |
| 水化学与矿物 | 稀溶液碳酸盐平衡、有限方解石溶解与沉淀、归属于物相的表面几何 | [矿物示例](src/chemworld/examples/native_mineral_research.py) |
| 随机生化网络 | Gillespie 事件、可逆与酶反应网络、整数库存，以及相互分离的过程随机性和测量随机性 | [生化世界](docs/native_biochemical.md) |
| 固相材料 | 有限置换型平衡、选定 TDB/CEF 与 ChemSage 输入、受支持的多亚点阵与准化学模型 | [原生基座指南](docs/native_foundation.md) |
| 仪器与调度 | 程序色谱、持久仪器历史、共享加热器/分析仪队列与定时读数 | [共享实验室](docs/native_shared_lab.md) |

这些模型具有明确的方程、参数和适用范围。各电池模型拥有自己的离散化与材料合同，膜过程和固相模型也保留各自的有限范围。组合新系统前，请阅读[模型范围](docs/extension-roadmap.zh.md)及[模块声明](docs/native_capabilities.md)。

## 运行一条连续实验流程

[完整领域示例](docs/native_domains.md)展示多个能力怎样通过一系列操作连接起来。

| 示例 | 实验过程 |
| --- | --- |
| 共享加热器 | 将同一个已标定加热器从表面反应器移至吸附容器，再移至热电池系统 |
| 膜过程 | 使用有限进料与洗水完成三次运行/清洗循环，保留膜污染历史 |
| 矿物 | 加酸，推进碳酸盐/方解石化学，转移物相并采样测量 pH |
| 共享时钟 | 推进独立的随机与确定性容器，让竞争请求共享一个离线分析仪 |
| 精馏 | 从全回流切换到收集馏分，始终使用有限库存 |
| 结晶 | 冷却、过滤、重悬、升温再冷却，保留物料与样品 |
| 固溶体 | 在升温与降温过程中观察二元分相 |

运行加热器示例并验证保存的轨迹：

```bash
uv run --no-sync python -m chemworld.examples.native_domains --workflow heater --output runs/native-domains
uv run --no-sync chemworld verify --submission runs/native-domains/heater.jsonl
```

用 `--workflow membrane`、`mineral`、`shared-clock`、`distillation`、`crystal` 或 `solid` 选择其他流程。这些示例使用固定脚本，无需模型服务；接入自己的策略后，可以让策略决定实验动作。检测噪声默认为顺序随机模式，`--noise-mode keyed` 可选择按键确定的噪声模式。

## 一个运行时，四种使用方式

![ChemWorld 架构：物理世界与任务定义共同配置运行时；Python、Gym、CLI 和 Agent 交换动作与观测；运行时记录实验历史。](docs/assets/chemworld-runtime-architecture.png)

世界定义物理上能够发生什么；任务定义研究者允许做什么，以及要达成什么目标。四种接口通过同一个实验会话发送动作并接收观测。会话记录物理状态、测量和资源使用，供检查与回放。

### Python

```python
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.surface_world import surface_world_spec

session = ChemWorldRuntime(composition=surface_world_spec())
session.reset(seed=12)
session.execute({
    "operation": "add_component", "component": "CO-feed", "amount_mol": 2e-5,
})
reading = session.execute({"operation": "measure", "instrument": "gas_pressure"})
for channel in reading.observation.channels:
    print(channel.channel_id, channel.value, channel.unit, channel.quality)
session.close()
```

### Gym、CLI 与智能体

| 入口 | 用途 | 接口 |
| --- | --- | --- |
| Gym | 接入策略与强化学习 | `ChemWorldEnv(composition=spec)` |
| CLI | 执行 JSON 动作序列、检查轨迹 | `chemworld render`、`chemworld verify` |
| Agent runner | 由自己的规划器或策略选择类型化操作 | `run_agent(..., composition=spec.to_dict())` |

[会话指南](docs/native_foundation.md)说明原生观测、Gym 兼容投影与快照续算。[智能体指南](docs/agents.zh.md)介绍自定义 Python 策略、离线基线、已保存轨迹回放和可选的在线模型适配器。默认示例可以离线执行，外部模型由使用者显式选择。

### 设计自己的实验环境

通过[环境作者指南](docs/environment-authoring.zh.md)声明设备、操作与资源，接入自己的Agent。
奖励和评价由用户定义；也可只观察实验过程，不计算分数。

```bash
uv run --no-sync python -m chemworld.examples.user_reward
```

旧Student Lab命令已退役。Research Workbench与Blender Lab仍是内部原型，暂不随公共包发布。

## 文档与网站

**[浏览中文版网站](https://sunyrain.github.io/ChemWorld-Public/?lang=zh)** · **[English website](https://sunyrain.github.io/ChemWorld-Public/?lang=en)**

总览图谱、能力索引和领域指南共享同一条导航。实验案例播放已保存的公开测试观测；静态网站不会启动实时实验室或执行新的物理计算。

| 目标 | 文档 |
| --- | --- |
| 构建并运行原生世界 | [原生基座指南](docs/native_foundation.md) |
| 执行完整示例 | [连续实验流程](docs/native_domains.md) |
| 接入规划器、模型或策略 | [智能体接入](docs/agents.zh.md) |
| 查看模型假设与限制 | [模型范围与扩展](docs/extension-roadmap.zh.md) |
| 检查已注册操作与状态归属 | [原生能力声明](docs/native_capabilities.md) |
| 使用队列与延迟测量 | [共享实验室时间](docs/native_shared_lab.md) |
| 阅读英文介绍 | [English README](README.md) |

## 开发与扩展

扩展应成为其他领域也能复用的原生能力：

1. **明确模型。** 定义方程、单位、参数范围和有限参考案例。
2. **声明归属与接口。** 明确哪些状态属于物料、容器或设备，以及它们如何迁移、保持并交换物质或能量。
3. **接入共同运行时。** 通过模块声明贡献过程、操作或仪器合同。
4. **验证实际行为。** 检查物理极限与守恒，再通过受支持的接口执行、序列化并回放完整实验。

上游科学工具为方程与有限参考比较提供依据。实际执行使用 ChemWorld 原生实现，可选参考依赖独立安装。复用的代码和数据保留其许可证及来源。

运行基座定向检查：

```bash
uv run --no-sync pytest tests/test_domain_modules.py tests/test_native_foundation_end_to_end.py tests/test_lab_clock.py -q
```

开发和打包检查见 [CONTRIBUTING.md](CONTRIBUTING.md)。可通过 [Issue](https://github.com/sunyrain/ChemWorld-Public/issues)提交可复现问题、模型建议或接口改进。提议新领域时，请说明预期实验、控制方程与适用范围。

**开发状态：** 当前源码版本为 `0.5.0.dev0`。集成检查覆盖十条流程、四种入口、两种检测噪声，以及新进程中的续算。独立 wheel/sdist 安装检查在源码目录之外执行示例和轨迹回放。模型边界与受支持组合以指南为准；下列历史结果只对应其固定运行时。

<details>
<summary>早期教程与冻结证据</summary>

教程无需模型服务密钥，继续固定在 `v0.3.0`：

| Notebook | 内容 | 运行 |
| --- | --- | --- |
| `01_first_experiment` | 检查 Reaction-to-Assay 合同、验证动作、使用 HPLC 并完成终检 | [在 Colab 中打开](https://colab.research.google.com/github/sunyrain/ChemWorld-Public/blob/v0.3.0/notebooks/01_first_experiment.ipynb) |
| `02_reaction_to_purification` | 继续执行萃取、洗涤、干燥与浓缩 | [在 Colab 中打开](https://colab.research.google.com/github/sunyrain/ChemWorld-Public/blob/v0.3.0/notebooks/02_reaction_to_purification.ipynb) |
| `03_controlled_world_change` | 固定公开实验，仅改变一个已注册的世界组分 | [在 Colab 中打开](https://colab.research.google.com/github/sunyrain/ChemWorld-Public/blob/v0.3.0/notebooks/03_controlled_world_change.ipynb) |

Notebook 中保存的输出是确定性的公开演示，不是基准结果或优化后的真实实验流程。

### 冻结证据

以下结果和图示属于冻结的八月运行时。它们作为历史证据保留，不用于证明当前 main 的资格。请使用[固定源码快照](https://github.com/sunyrain/ChemWorld-Public/tree/db98e8002fac0512ed9084a3c8c27985cb9fc405)复现；上方 Notebook 链接也继续固定版本。

![基于历史证据的智能体生命周期、回滚恢复与受控世界变化](docs/assets/representative-agent-and-world-change.svg)

- 64/64 个已注册任务—世界单元及 1,786/1,786 条参考配方完成。
- 52/52 个覆盖生成的组合完成，其中包含 8/8 个非参考反应—精馏世界。
- 八条确定性生命周期检查了全部 89 个提交动作：88 个提交成功，1 个按设计回滚。
- 六组受控父子世界产生了 24 条无需模型服务的轨迹。
- 一个独立智能体完成了包含 15 个动作的公开仪器生命周期。

这些是有限软件模型的资格结果，不代表普适化学精度、通用智能体智能或真实实验室迁移能力。

原始全树发布验证器属于固定的八月源码，见[证据文档](docs/evidence.zh.md)。可以运行 `uv run --no-sync python scripts/build_readme_visuals.py --check` 检查冻结图示的一致性。

</details>

## 范围、引用与许可证

本公开仓库包含仿真基座、接口、示例、文档、定向测试及保留的历史公开证据。稿件、私有研究配置、凭据和原始模型服务响应不属于本仓库的公开内容。

引用信息见 [CITATION.cff](CITATION.cff)。ChemWorld 使用 [MIT 许可证](LICENSE)。
