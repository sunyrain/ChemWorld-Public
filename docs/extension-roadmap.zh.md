# 原生扩展与模型范围

当前 Python 后端已包含下列模型。科学计算接口显式声明参数和适用范围；接入世界的模型
复用 ChemWorld 的状态、操作、观测与事务合同。上游软件包用于可选数值对照，运行不依赖
其求解器。

| 领域 | 已实现范围 | 指南 |
| --- | --- | --- |
| 共同基座 | 模块声明、状态归属、共同残差／DAE 基础、原生单位、有限拟合、Python／Gym／CLI／Agent 续算与回放 | [原生会话](native_foundation.md) |
| 热力学与物性 | 活度／超额性质、六种 UNIFAC 变体、立方 alpha／混合规则、纯流体 Helmholtz 物性／求根／饱和／闪蒸、物性导数与拟合 | [热力学 API](native_foundation.md#thermodynamic-and-engineering-apis) |
| 工程流体 | 摩阻、管件、流量计、阀门、泵、填料床／塔、气体与两相流、明渠、粒径分布、设备几何、大气与太阳计算 | [工程 API](native_foundation.md#engineering-flow-meter) |
| 表面与体相反应 | 有限气相／体相／表面反应、多位点吸附与氧化、覆盖度相互作用和共同热过程 | [机理](native_mechanisms.md) |
| 分离与输运 | 有限 TP／PH 闪蒸、液相稳定性、动态塔、吸附／再生、结晶、膜流动／污堵／冲洗及守恒扩散 | [连续流程](native_domains.md) |
| 电池与电极 | SPM、有限 SPMe／DFN、粒径分布、有限溶剂 SEI、选定阻抗／SOH／OCV 诊断，以及持久 Thevenin RC／双节点热模型 | [电化学](native_foundation.md) |
| 水相与矿物 | 稀碳酸盐平衡、有限方解石溶解／沉淀和相归属表面几何 | [连续流程](native_domains.md) |
| 固相 | 选定 CALPHAD／ChemSage 模型及来源输入、有限平衡与热过程消费 | [固相模型](native_foundation.md) |
| 生化 | 确定性与随机反应网络，支持范围内的 SBML 规则／事件及 SED-ML／COMBINE 合同 | [生化世界](native_biochemical.md) |
| 共享实验室时间 | 有限异构容器组、已声明开放流路、多台持久设备、FIFO 测量队列与延迟读数 | [共享时钟](native_shared_lab.md) |

## 当前边界

来源、单位、相态假设和有效域都是模型输入的一部分。已实现方程并不代表每种材料均有
经过验证的参数。来源表身份审查、更广电池模型族和实验互操作仍在开发；目前实现并不
等同于 CoolProp、thermo、chemicals、fluids、PyBaMM、pycalphad 或 COPASI 的全部功能。

通用缔合／多流体 EOS、无约束相包络与通用流程求解仍待扩展。统一 CPU 内核重构属于
规划，本次不包含编译执行、SIMD 或新的默认后端。局部容器时间仍为默认，共享时间需
显式启用，并遵守指南中支持的模型组合。

## 参考与复用

原生模型通过解析极限、守恒和有限参考案例检查。独立实现及数值一致性不免除复用
源材料或数据的署名与许可义务。工程方法包含保留来源的表格和公式，不能笼统声称
所有方法都未接触上游源码。调用方提供的流体／数据文件保留各自许可，不自动随包分发。

同步与验证见[当前基座](current-foundation.md)，环境配置见[安装指南](getting_started.md)。
