# 原生扩展与模型范围

当前开发源码已经包含下列有限模型，统一使用ChemWorld的状态、操作、观测和事务接口。
开源工具用于方程参考和可选数值对照，运行原生模型不需要安装这些求解器。
合成模型的适用范围如下，不等同于完整复刻上游软件或预测特定真实材料。

| 领域 | 已实现的原生范围 | 示例或说明 |
| --- | --- | --- |
| 共同基座 | Gym、CLI、Agent共用实验会话；模块声明、物料／设备所有权、提交前守恒、物理单位与保存续算 | [原生会话](native_foundation.md) |
| 表面催化 | 有限CO／O2吸附与氧化、可选体相反应、共同热推进；可逆双位点、覆盖度相互作用与位点交换 | `python -m chemworld.examples.native_multisite_surface` |
| 热力学与相过程 | 状态相关物性、TP／PH闪蒸、两液相稳定性、有限动态蒸馏、浓缩、吸附再生及结晶／重溶 | `python -m chemworld.examples.native_distillation_research`；`python -m chemworld.examples.native_crystal_research` |
| 输运与电极 | 有限罐、压头／功率受限泵与换热；平面／球形守恒扩散和有限扩散—BV电极 | `python -m chemworld.examples.native_liquid_transport`；`python -m chemworld.examples.native_electrode_research` |
| 电池 | 球形单粒子扩散、恒流／恒压／静置、集中合成热模型、有限溶剂SEI与锂损失 | `python -m chemworld.examples.native_battery_degradation` |
| 水相与矿物 | 稀碳酸盐／水平衡、有限方解石、带种子的溶解／沉淀及归属固相的表面几何；298.15 K | `python -m chemworld.examples.native_mineral_research` |
| 膜分离 | 有限进料／收集罐、恒温稀NaCl反渗透、浓差极化、持续合成污染与有限纯水冲洗 | `python -m chemworld.examples.native_membrane_fouling` |
| 随机生化 | 恒容恒温Gillespie事件、可逆和显式酶网络、整数加料与分子捕获；过程和检测随机性分开 | [生化世界](native_biochemical.md)、[SSA](native_stochastic.md) |
| 机理与仪器 | 有限YAML／JSON机理导入、P-log／Chebyshev、NASA7气相热力学；程序色谱和持续仪器状态 | [机理导入](native_mechanisms.md)；`python -m chemworld.examples.programmed_chromatography --output chromatography-run` |
| 二元固体 | 一／两种替位相、理想混合与L0／L1项、全局支撑线检查及热路径中的局部平衡 | `python -m chemworld.examples.native_solid_solution` |
| 共享实验室时间 | 一至两个独立热／SSA容器、一台切换热设备和一台离线分析仪；FIFO请求、原子时间片及延迟读数 | [共享实验室时钟](native_shared_lab.md) |

源码检出后，Python命令加前缀`uv run --no-sync`。这些示例不需要模型密钥。
生化示例运行`uv run --no-sync python examples/native_biochemical.py`。

## 集成范围与后续科学扩展

共享时钟通过会话配置显式开启，所有已启用容器同时老化，快照保留设备队列、待发SSA事件与未发布读数。
局部时钟仍为默认。共享模式支持顺序及keyed检测噪声，可显式声明亚秒等待范围；目前不包含开放流路与替代相策略，完整范围见使用说明。
[组合示例](native_domains.md)通过四个入口、两种噪声及新进程续算检查七条连续历史；
wheel／sdist独立安装检查在仓库外运行默认噪声示例及逐条回放。

后续科学扩展仍须分别规定有限方程、参数和对照：电池电解液输运、析锂／裂纹、高浓度非理想水相、
膜反向流动与化学清洗、空间表面动力学、晶体破碎／聚并、非恒温SSA／ODE混合以及多子晶格固体。
现有SPM、膜和二元固体均有明确范围，不代表完整PyBaMM、WaterTAP或CALPHAD能力。

## 开源参考与原生实现

[英文说明](extension-roadmap.md)列出Cantera、PyBaMM、Reaktoro、WaterTAP、IDAES、
COPASI、RMG-Py、CatMAP和pycalphad等参考项目。先确定方程、适用域、状态归属和交换规则，
再接入同一求解与实验会话，并检查解析极限、守恒和有限参考条件。
复用的数据夹具保留来源和许可证，例如`tests/fixtures/mechanisms/`中的Cantera 3.1.0机理。
外部求解器保持可选；原生重写不免除实际复用代码或数据的许可义务。
