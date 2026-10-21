# 原生整数随机化学

> 中文预览译文；技术接口与适用范围请同时核对[英文原文](https://sunyrain.github.io/ChemWorld-Public/explore/?lang=en#guides/chemworld/native_stochastic)。

将 `ReactionNetworkSpec(..., stochastic={"volume_L": 0.01, "temperature_K": 298.15})` 声明并传递整数计数给 `chemworld.world.stochastic_world.stochastic_world(network, counts, seed=7)`。然后，使用返回的组合与 `ChemWorldRuntime`、`reset()` 和 `execute({"operation": "wait", "duration_s": 10})` 结合使用。这与确定性世界中的运行时、交易和会计路径相同。

每个基本方向都是一个不可逆的过程。 `mass_action` 与...反应 `parameters={"k": ...}`. A↔B 表示两个通道，而不是潜在的负向净倾向。速率常数保留了网络的 mol/L 和秒级约定。对于分子数 m，随机常数乘以组合的速率为： `k*(N_A*V)**(1-m)*prod(factorial(r_i))`. 非整数的化学计量比、定制化订单、活动、持续控制、隐式的 ODE/SSA 混合物以及分子数超过三个的情况，均不在本有限的实施范围内。

该阶段账本拥有所有材料。它的mol这些数值代表精确的分子数量，除以精确的阿伏伽德罗常数；转换仅允许使用四个浮点数 ULPs（基本单位），不允许使用分数剂量。数量的范围限定在 2\*\*40，以保持整数分辨率的明确性。该过程的延续性属于容器的ProcessLedger并且不包含任何第二次物料库存：PCG64状态、绝对待处理事件、处理时间、事件计数以及速率/库存签名。公开状态预测不包含这些信息。 真实的即时干预会使待处理事件无效，同时保持处理过程。RNG连续读取；而读取操作不会产生这样的效果。

`snapshot().to_dict()` / `ChemWorldRuntime.from_snapshot(payload)` 能够保存待执行的事件以及精确的后续执行，包括在另一个已安装的进程中。纯粹的 `advance_stochastic` API 还可以返回单个事件的时间/标识，以便进行数值验证。数值事件预算会回溯整个候选方案，而不是报告化学失效或消耗物理时间。常见的运行时仍然适用其声明的请求/测量资源规则。

仅支持恒定体积和温度。没有关于热量/能量模型的声明或符合生物化学准确性的拟合。在编译时，拒绝使用分段式破坏性仪器；采用专门的整数采样以及两个预设的生物化学参数是独立的。NATIVE-EXT-37 任务。 无COPASI或者其他上游的模拟器会被运行时加载。

开发资格：`uv run --no-sync pytest -q tests/test_stochastic_kinetics.py` ; `uv run --no-sync python scripts/validate_native_ssa.py --output /tmp/ssa-new-attempt.json`。 后者拒绝覆盖之前的尝试。 固定的4000次运行设计，所有结果和统计标准都位于“世界基础”工作流中。
