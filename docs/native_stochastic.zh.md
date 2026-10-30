<!-- Generated from https://github.com/sunyrain/ChemWorld/blob/main/docs/native_stochastic.zh.md; edit the upstream guide. -->

# 原生整数随机化学

通过 `ReactionNetworkSpec(..., stochastic={"volume_L": 0.01, "temperature_K": 298.15})`
声明网络，以整数分子数调用 `stochastic_world`，再使用共同运行时的 `wait`、快照和恢复。

默认算法为 `gillespie-direct-v1`；声明 `algorithm="gibson-bruck-next-reaction-v1"`
可使用按依赖更新的反应时钟。直接法保留下一事件，next-reaction 保留各通道时钟、
倾向和内部 Poisson 余量；读取不消耗过程 RNG，快照保存续算状态。

本版还包含显式分区的混合 ODE/跳跃过程和有限自适应 tau-leap。它们各自保留算法合同，
tau-leap 是近似法，不等同于精确 SSA；没有任意自动分区或空间 kMC 声明。
调用 `chemworld.examples.stochastic_tau_leap.tau_world(seed=51, copies=10000)`，
再交给 `ChemWorldRuntime`，可使用自带近似模型。

整数库存、温度/体积范围、采样限制及回滚规则见[英文完整说明](https://sunyrain.github.io/ChemWorld-Public/docs/native_stochastic/)。
