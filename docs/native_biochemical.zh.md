# 有限的低复制世界

> 中文预览译文；技术接口与适用范围请同时核对[英文原文](https://sunyrain.github.io/ChemWorld-Public/docs/native_biochemical/)。

两个合成世界运行在原生会话和任务-36的直接策略 SSA 上： `biochemical_world("reversible")` (A与B的关系) 以及 `biochemical_world("enzyme")` (E+S→ES, ES→E+S, ES→E+P)。这些是数值示例，而不是经过生物学验证的系统。 合成 E 是 N2，而 ES 是 CH2N2O，因此结合保持了原子的存在。 这种反应不涉及迈克尔-门腾还原，也没有隐式的 ODE/SSA 切换。

在没有模型密钥或 COPASI 的情况下运行：

```
uv run --no-sync python examples/native_biochemical.py
uv run --no-sync pytest tests/test_biochemical_worlds.py tests/test_stochastic_kinetics.py -q
```

普通 `add_component` 动作接受 mol 单位，但每次剂量必须精确地表示分子数量（在浮点数转换舍入误差范围内）。 0/1 索引对应可逆过程， E/S 对应酶。 此处不支持任何小数舍入、载体添加、热控或流控。 元素之间的转换以精确的整数原子为准，而不是宏观的 mol 容差。 mol 账本仍然是唯一的库存；计数是推导出来的视图。

`molecule_counter` 报告带独立高斯噪声的有符号校准计数（标准差为 1 个计数，每个分子的响应为一个单位），绝不输出事件真值。`molecule_capture` 以 0.1 的概率独立选择每个溶质分子，将选中的分子转移到仪器废弃物中，并使用相同的探测器噪声进行测量。这是 **理想化的分子捕获** ，不是抽取液体等分试样；载液体积保持不变。分析在离线仪器时钟上耗时 1 s，反应器不演化。两种读取的成本均为 0.01。捕获真值和采样随机数生成器保持私有。空通道或噪声通道不会被隐含真值悄然替换。模型不声称支持饱和效应或未知物种识别。

“种子”的定义在世界级别声明，检测器“种子”在会话重置时使用，捕获使用从世界“种子”中导出的独立持久流。读取不会影响后续反应或捕获；实际的剂量/捕获会使待处理的事件失效。快照保留所有后续状态，以及材料/批次/容器的初始化会恢复已声明的离散状态（包括其初始的种子）。因此，多次重置是确定性的重新初始化，而不是独立的副本：为独立的进程副本创建一个新的世界，并使用新的种子。使用固定装置可以容忍材料的重置。快照包含评估者的真实信息，而不是代理的观察。

可选的外部引用块（无默认依赖，可以选择新的输出）：

```
uv run --no-sync --with python-copasi==4.48.309 python scripts/validate_native_biochemical.py --output /tmp/biochemical-attempt.json
```

固定发展的设计和完整的 8,288 个端点数据位于 `workstreams/world_foundation/NATIVE_BIOCHEMICAL_ACCEPTANCE.md` 和 `NATIVE_BIOCHEMICAL_RESULTS.json` 中。它比较瞬态/有效平衡的二项分布、有限的 COPASI 端点分布以及体积/计数比例缩放的 ODE 归一化。有限分布的一致性并不能证明所有动力学状态或实验室预测的有效性。一个通用的分子计数器和合成捕获是明确的简化，而不是对特定物理实验的建模。
