<!-- Generated from https://github.com/sunyrain/ChemWorld/blob/main/docs/native_mechanisms.zh.md; edit the upstream guide. -->

# 有限原生气相反应机制

> 中文预览译文；技术接口与适用范围请同时核对[英文原文](https://sunyrain.github.io/ChemWorld-Public/docs/native_mechanisms/)。

这是开发中的功能，并非通用的 Cantera/RMG 解释器，也不代表化学行为已通过真实实验验证。它使用 ChemWorld 现有的反应网络、共享物料库存与能量积分器、热设备、事务和回放机制。原生执行过程不导入 Cantera。

## 导入与运行 {#import-and-run}

在仓库锁定的环境中运行：

```
uv run --no-sync python -m chemworld.physchem.mechanism_import \
  tests/fixtures/mechanisms/cantera-3.1.0/h2o2.yaml \
  --phase ohmech --output /tmp/native-h2o2.json
uv run --no-sync python -m chemworld.examples.native_gas_mechanism \
  tests/fixtures/mechanisms/cantera-3.1.0/h2o2.yaml \
  --phase ohmech --amounts '{"H2":0.00002,"O2":0.00001,"AR":0.00007}'
```

该转换器拒绝覆盖已存在的输出。 `load_mechanism` 读取导出的原生 JSON。 原始的热源、单位、原始反应声明和原始来源信息在传输过程中得以保留。 当该模块安装在非结算区域时，相同的模块命令可以接受明确的本地来源路径；测试用例不作为隐藏的运行时数据进行打包。

该作者设计了一种10mL的船，在700 K处进行有限功率加热，达到750 K的温度。 `heat → pressure measurement → wait → pressure measurement → terminate`。 压力测量仪利用现有的物理测量时间，因此总的物理时间包括测量。 只有通用压力测量仪才暴露：现有的 CO/O2/CO2 分析方法 **不** 被重新标记为通用型 H2/O2 或任意物种分析仪。物料库存是私有的状态，而不是人为的测试结果。

## 支持的有限模型范围 {#supported-finite-contract}

-   支持一个明确选定的理想气相，包含 1–64 个本地物种和 1–256 个反应。所有选定的物种与反应均须保留，否则整次导入失败。
-   NASA7 标准状态的热力学，基本阿伦尼乌斯公式，第三体效应，林德曼/特劳公式，P-log 和切byshev公式。在两个方程的双方都明确指定碰撞伙伴，即使在净化学计量平衡中，它们仍然保持各自的动力学顺序。重复的通道仍然是独立的反应。
-   P-log 函数计算在相同压力下的所有项，在 **之前进行** 的对数插值。切byshev 使用反温和对数压力缩放的坐标。两者都拒绝T/P超出已声明的范围。在终端处，最多八个机器 ULPs 按照物料库存到压力的浮点数舍入进行标准化；不提供任何物理外推或最近压力钳制。
-   基于原生浓度/速率的基准值为 mol/L 和 mol/(L·s)。 输入长度 m/cm/mm，数量 mol/kmol，时间和单位（s/分钟），以及支持的活化能单位，将被一次性转换。 这是一个有限扩展。 `units: {volume: L, quantity: mol}` 可以被使用 **而不是** 长度声明；L 不应被视为长度。P-log 需要明确指定 `temperature-range` （一种额外的、具有有限域的合同，并非 Cantera 的无限制 P-log 格式所必需的）。
-   压力是从当前理想气体库存和温度计算得出的。可逆气体速率使用NASA7以及 `C° = P°/(1000 RT)`mol/L，而不是默默地重复使用历史上的液体标准浓度。内部能量是 `Σ n_i (h_i(T)−RT)`，而热容是它的导数；反应热则由共享的能量平衡一次计算。
-   原生热元件的限制来源于容器的明确声明的热模型。历史液体的默认值以及公众规定的至少 1 秒的热/等待时间保持不变。数学批处理积分器也支持亚秒级的刚性轨迹。

可执行的 Python 脚本，外部物种包括，自定义反应顺序，不支持负 A 拟合，不支持未平衡的动力学和热力学，以及未实现的模型均导致失败。 运输和未选择的相仍然保留原始元数据，但未实现模型。 `h2o2.yaml` 和其 BSD 许可证在 `tests/fixtures/mechanisms/cantera-3.1.0/` 下保留。

## 验证 {#validation}

单独的 `import_chemkin(mechanism_path, thermo_path)` 接口支持有限的 NASA7 气相子集：
基元反应、第三体、Lindemann、Troe、PLOG、CHEB、REV、FORD 和 RORD 声明。
它展开本地递归 INCLUDE，接受 ELEM/ELEMENTS，并保留源单位与反应级数。
MOLECULES 浓度单位通过精确阿伏伽德罗常数转换为原生 mol/L。
命名反应区段、表面／等离子体／离子相及非理想模型等未支持内容会使整个导入失败。
Cantera 3.2 的 ck2yaml 尚拒绝 RORD，该上游往返限制仍保留。
这个接口不由 YAML/JSON 命令行自动选择；其资格范围与下述比较分开。

```
uv run --no-sync pytest tests/test_pressure_dependent_mechanisms.py tests/test_reaction_rate_contracts.py -q
uv run --no-sync --with 'cantera==3.1.0' python scripts/validate_native_mechanisms.py \
  --output /tmp/mechanism-comparison.json
```

可选的参考命令运行 45 个速率网格案例、9 条加热轨迹和 3 个平衡极限。开发笔记在执行之前，先处理了分母和容差。最终的证据以及第一次失败的终点运行结果都保留在 `NATIVE_MECHANISM_ACCEPTANCE.md` 旁边。这并不能证明其他机制、平台、实验室结果或发布构建。
