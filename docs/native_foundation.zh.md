# 原生物理会话

## 工质与分子过程 {#working-fluids-and-molecular-processes}

10 月 24 日更新加入原生水/蒸汽、冰、湿空气、不可压缩工质、分子状态方程和有限过程路径。以下代码使用已安装的公开包。[能力索引](https://sunyrain.github.io/ChemWorld-Public/explore/?lang=zh#index/working-fluids)还提供机理生成、表面控制、过滤、三元平衡、析锂和 SBML 分析的函数签名、输入表与 JSON/键值表输出。

### IF97：液态水

IF97 液态水 TP 求解；焓和热容按质量计。冰、湿空气和不可压缩工质各有独立的范围合同。

```python
import json
from chemworld.physchem.water_if97_solvers import water_if97_tp

inputs = {'temperature_K': 300, 'pressure_Pa': 3000000, 'phase': 'liquid'}

state = water_if97_tp(inputs["temperature_K"], inputs["pressure_Pa"], phase=inputs["phase"])
output = state.to_dict()

print(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False))
```

```json
{
  "region": 1,
  "temperature_K": 300,
  "pressure_Pa": 3000000,
  "mass_density_kg_m3": 997.8529400984821,
  "enthalpy_J_kg": 115331.27302143883,
  "internal_energy_J_kg": 112324.81798237875,
  "entropy_J_kg_K": 392.2947924026257,
  "cp_J_kg_K": 4173.012184067787,
  "cv_J_kg_K": 4121.201603587441,
  "speed_of_sound_m_s": 1507.73920966903,
  "gibbs_energy_J_kg": -2357.164699348876,
  "isothermal_compressibility_Pa_inv": 4.4638212280219395e-10,
  "thermal_expansion_K_inv": 0.0002773545334266113,
  "stability_scope": "IF97_stable_region",
  "specific_volume_m3_kg": 0.001002151679686694
}
```

### PC-SAFT：局部混合物响应

合成二元 PC-SAFT 局部均相响应；机械稳定性不等于全局相稳定性，参数不代表真实物质。

```python
import json
from chemworld.physchem.pcsaft import PCSAFTModel, SAFTSegment
from chemworld.physchem.molecular_eos import molecular_eos_state

inputs = {'temperature_K': 400,
 'molar_density_mol_m3': 20,
 'mole_fractions': [0.3, 0.7]}

model = PCSAFTModel(
    (SAFTSegment("light", 1, 3.4e-10, 120, "synthetic"),
     SAFTSegment("heavy", 2, 3.8e-10, 180, "synthetic")),
    ((0, 0.02), (0.02, 0)), (100, 1000), "synthetic")
state = molecular_eos_state(model, inputs["temperature_K"],
    inputs["molar_density_mol_m3"], tuple(inputs["mole_fractions"]))
output = state.to_dict()

print(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False))
```

```json
{
  "temperature_K": 400,
  "molar_density_mol_m3": 20,
  "mole_fractions": [
    0.3,
    0.7
  ],
  "pressure_Pa": 66375.50309041252,
  "compressibility_factor": 0.9978922592287066,
  "residual_helmholtz_J_mol": -7.017506251226959,
  "residual_internal_energy_J_mol": -17.109782171114848,
  "residual_enthalpy_J_mol": -24.119674911785385,
  "residual_entropy_J_mol_K": -0.025230689799719725,
  "residual_cv_J_mol_K": 0.017203159119927498,
  "residual_cp_J_mol_K": 0.10310302534629337,
  "dP_dT_rho_Pa_K": 166.79375449322313,
  "dP_drho_T_Pa_m3_mol": 3311.7804844346783,
  "mechanically_stable": true,
  "log_fugacity_coefficients": [
    0.0010175146195439816,
    -0.0034472285464053946
  ],
  "stability_scope": "homogeneous_mechanical_only",
  "energy_reference": "ideal_gas_at_same_T_rho_composition"
}
```

## 热力学与工程接口 {#thermodynamic-and-engineering-apis}

以下示例使用当前 Python 包，输入包含物理单位及显式系数；输出由展示代码实际生成。
能力网站同时提供函数签名和 JSON／键值表切换。

已交付热力学包括活度导数与有限拟合、由调用方提供基团数据的六种 UNIFAC 变体、
Twu／PPR78 与 GE 立方混合，以及共同均相求值接口。纯流体 Helmholtz 支持密度求根、
饱和、TP／PH／PS／T-rho 闪蒸及有限热过程。下面用理想气体极限演示接口；真实流体
需要显式来源系数与适用范围。现已加入选定分子、缔合与多流体模型，分别保留局部适用域与参数来源。

工程方法覆盖摩阻、管件、阀门、流量计、泵、填料床／塔、两相流、明渠、粒径分布
及设备几何，各方法保留自己的物理假设与工作域。

### 纯流体 Helmholtz 状态 {#helmholtz-state}

自编理想气体系数演示：输入 K 与 mol/m³，输出有量纲物性。真实流体需提供来源系数；此处只检查均相局部稳定。

```python
import json
from chemworld.physchem.helmholtz import HelmholtzFluid
from chemworld.physchem.helmholtz_terms import HelmholtzTerm

inputs = {'temperature_K': 400, 'molar_density_mol_m3': 20}

fluid = HelmholtzFluid(
    fluid_id="synthetic-monatomic", casrn="", molar_mass_kg_mol=0.04,
    gas_constant_J_mol_K=8.3, reducing_temperature_K=300,
    reducing_molar_density_mol_m3=1000, minimum_temperature_K=100,
    maximum_temperature_K=1500, maximum_pressure_Pa=1e9,
    ideal_terms=(
        HelmholtzTerm("IdealGasHelmholtzLead", {"a1": 0, "a2": 0}),
        HelmholtzTerm("IdealGasHelmholtzLogTau", {"a": 1.5}),
    ), residual_terms=(), source="Self-authored ideal-gas demonstration",
)
state = fluid.state(**inputs)
output = {key: state.to_dict()[key] for key in (
    "pressure_Pa", "enthalpy_J_mol", "cp_J_mol_K", "cv_J_mol_K",
    "compressibility_factor", "stability_scope",
)}

print(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False))
```

```json
{
  "pressure_Pa": 66400.0,
  "enthalpy_J_mol": 8300.0,
  "cp_J_mol_K": 20.75,
  "cv_J_mol_K": 12.45,
  "compressibility_factor": 1.0,
  "stability_scope": "homogeneous_local_only"
}
```

### 活度热力学 {#activity-thermodynamics}

合成二元 Margules 参数用于展示超额和偏摩尔性质；组分 A/B 不代表经过验证的真实材料。

```python
import json
from chemworld.physchem.equilibrium import ActivityModelSpec
from chemworld.physchem.activity_thermodynamics import activity_thermodynamic_report

inputs = {'temperature_K': 343.15,
 'composition': {'A': 0.252, 'B': 0.748},
 'parameters': {'A:A|B': 1.2, 'A:B|A': 0.8}}

model = ActivityModelSpec("synthetic-binary", ("A", "B"), "margules", inputs["parameters"])
report = activity_thermodynamic_report(
    model, inputs["composition"], temperature_K=inputs["temperature_K"])
output = {key: report.to_dict()[key] for key in (
    "activity_coefficients", "gibbs_excess_J_mol", "enthalpy_excess_J_mol",
    "entropy_excess_J_mol_K", "partial_molar_gibbs_excess_J_mol",
)}

print(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False))
```

```json
{
  "activity_coefficients": {
    "A": 1.7482386468981486,
    "B": 1.0928664265178667
  },
  "gibbs_excess_J_mol": 591.1491189544142,
  "enthalpy_excess_J_mol": 0.0,
  "entropy_excess_J_mol_K": -1.7227134458820172,
  "partial_molar_gibbs_excess_J_mol": {
    "A": 1593.7711326575798,
    "B": 253.367371022332
  }
}
```

### 物性导数 {#property-derivatives}

显式热容关联式给出温度导数与所选参数的 Jacobian，保留单位与有效域；这是合成参数示例。

```python
import json
from chemworld.physchem.specs import PropertyCorrelation
from chemworld.physchem.dippr import dippr_temperature_response, dippr_parameter_jacobian

inputs = {'temperature_K': 300,
 'coefficients': {'A': 25.0, 'B': 30.0, 'C': 800.0, 'D': 10.0, 'E': 400.0}}

curve = PropertyCorrelation(
    "synthetic-capacity", "dippr107", inputs["coefficients"],
    {"temperature": "K"}, "J/(mol*K)", "ideal_gas_heat_capacity",
    {"temperature": (200, 500)}, "Synthetic API demonstration")
response = dippr_temperature_response(curve, temperature_K=inputs["temperature_K"])
jacobian = dippr_parameter_jacobian(curve, [inputs["temperature_K"]], parameters=["A", "C"])
output = {"temperature_response": response.to_dict(), "parameter_jacobian": jacobian.to_dict()}

print(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False))
```

```json
{
  "temperature_response": {
    "correlation_id": "synthetic-capacity",
    "temperature_K": 300,
    "values": [
      33.47979179011391,
      0.05154913751893453,
      -2.6191592747468737e-05,
      -1.4136319377386219e-06
    ],
    "output_unit": "J/(mol*K)",
    "warnings": [],
    "temperature_basis": "K",
    "orders": [
      0,
      1,
      2,
      3
    ],
    "value_units": [
      "J/(mol*K)",
      "(J/(mol*K))/K^1",
      "(J/(mol*K))/K^2",
      "(J/(mol*K))/K^3"
    ]
  },
  "parameter_jacobian": {
    "temperatures_K": [
      300
    ],
    "parameters": [
      "A",
      "C"
    ],
    "values": [
      [
        1.0,
        -0.017602090544385513
      ]
    ],
    "output_unit": "J/(mol*K)",
    "warnings": [
      []
    ],
    "parameter_basis": "declared source-coefficient coordinates, no normalization"
  }
}
```

### 工程流量计 {#engineering-flow-meter}

先由质量流率计算压差，再由压差反算流率。安装条件、流体参数和关联式适用域须满足声明。

```python
import json
from chemworld.physchem.flow_meters import LiquidOrificeSpec, liquid_orifice_pressure, liquid_orifice_mass_flow

inputs = {'pipe_diameter_m': 0.07366,
 'orifice_diameter_m': 0.05,
 'mass_flow_kg_s': 7.0,
 'density_kg_m3': 999.1,
 'viscosity_Pa_s': 0.0011}

spec = LiquidOrificeSpec(inputs["pipe_diameter_m"], inputs["orifice_diameter_m"], taps="corner",
    installation_note="ISO concentric sharp edge; compliant straight runs and roughness assumed")
properties = {key: inputs[key] for key in ("density_kg_m3", "viscosity_Pa_s")}
forward = liquid_orifice_pressure(spec, mass_flow_kg_s=inputs["mass_flow_kg_s"], **properties)
inverse = liquid_orifice_mass_flow(spec, differential_pressure_Pa=forward.differential_pressure_Pa,
    **properties)
output = {"forward": forward.to_dict(), "recovered_mass_flow_kg_s": inverse.mass_flow_kg_s}

print(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False))
```

```json
{
  "forward": {
    "mass_flow_kg_s": 7.0,
    "reynolds": 109997.7391596048,
    "discharge_coefficient": 0.6102129110865264,
    "differential_pressure_Pa": 13455.336179859189,
    "permanent_pressure_loss_Pa": 7213.34049191436,
    "method": "iso5167"
  },
  "recovered_mass_flow_kg_s": 6.999999999999911
}
```

---

## 会话与表面反应示例

这些开发示例使用已安装的当前运行时环境，无需 API 密钥或 Cantera。它们描述了一个有限的合成 CO 氧化模型，而不是一个通用的化学预测模型或新发布的版本。Python 3.11 或 3.12 是验证目标。

在隔离环境中安装 wheel 或 sdist 后：

```
python -m chemworld.examples.native_surface_research --bulk --mobile --terminal-assay
python -m chemworld.examples.reuse_native_equipment
```

在开发环境中，请在 Python 命令前添加前缀 `uv run --no-sync`. 举例来说，第一个例子处理有限的物料库存，进行加热、测量、输送，并完成任务。 `--no-surface` 使用与惰性或仅用于大体积系统的相同装置； `--bulk` 增加了已声明的大规模反应。 `--mobile` 附着一种由材料构成的表面。如果没有这种表面，当气体移动时，静止的表面仍然停留在其容器上。

第二个例子使用了相同的加热器型号，在惰性、大体积、表面和组合系统中。它在两个批次之间在反应器中初始化新鲜材料，并检查保留的加热器保持其身份和校准。此外，它将一个编译的物理世界与两个任务联系起来：直接终止和终止后进行所需的最终检测。它们的物理响应在终止前是协调一致的。

## 通过其他入口使用相同的会话。 {#use-the-same-session-through-other-entry-points}

```
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.surface_world import surface_world_spec

spec = surface_world_spec(terminal_assay=True)
session = ChemWorldRuntime(composition=spec, observation_noise_mode="keyed")
session.reset(seed=12)
result = session.execute({"operation": "add_component", "component": "CO-feed",
                          "amount_mol": 2e-5})
reading = session.execute({"operation": "measure", "instrument": "gas_pressure"})
for channel in reading.observation.channels:
    print(channel.channel_id, channel.value, channel.unit, channel.quality)
session.close()
```

`ChemWorldEnv(composition=spec)` 在相同的运行时提供 Gym `reset` / `step` 接口。`chemworld.eval.runner.run_agent` 接受 `composition=spec.to_dict()` 作为代理。这两个入口点都保留了交易结果、物理持续时间、资源核算以及 `info` 中的原生观察。

从 CLI 执行 JSON 动作，将 `spec.to_dict()` 保存为 `world.json`，并将每个动作分别保存为 JSON 文件，然后运行：

```
chemworld render --composition world.json --seed 12 --noise-mode keyed \
  --actions feed.json pressure.json --format json --trajectory run.jsonl \
  --snapshot continuation.json
chemworld verify --submission run.jsonl
```

这CLI拒绝覆盖现有的轨迹或快照。快照和轨迹的来源信息包含用于评估者后续操作和回放的私有模拟器状态/配置；这些信息不是代理的观察结果。JSON可以重新排序地图的关键，因此物理回放使用数值容差（即CLI默认值为 `1e-5`，而分类结果必须一致。回放需要原始的运行时语义；当前的“主程序”不能替代冻结的纸质物理模型。

## 测量、时间以及重复使用 {#readings-time-and-reuse}

原生通道拥有物理值、单位、不确定性、质量、坐标和采集时间等属性。无效或未观测到的通道是明确的；有效的物理读数不需要符合 `[0, 1]` 的要求。固定的 Gym 标量视图是一种推导出的兼容性投影。使用 `info["native_observation"]` 或会话结果来获取完整的通道记录。调用 `observe()` 返回缓存的读数，但不获取新的样本。

“原位采集”通过相同的物理集成器，推动世界发展。离线分析则保留采集的快照，并分别计算分析时间。对于需要先进行最终检测的任务，首先会暂停生产，然后检测完成后停止；它不会再次启动已暂停的生产流程。

`session.snapshot().save(path)` 和 `ChemWorldRuntime.from_snapshot(SessionSnapshot.load(path))` 支持私有续航，并与兼容的绑定一起使用。`session.reset()` 创建一个新的世界。仅评估 `reset(options={"scope": "material", "vessel_ids": ["reactor"]})` 替换该材料，同时保留已连接的固定设备及其校准。部分初始化保留时钟、预算和 RNG 的状态；这并非自由代理操作。`SessionSnapshot` 也可以从 `chemworld.runtime` 中导出。

当前的私人会话使用 `chemworld-session-v2`。它们的相位的身份和材料的坐标顺序得以保留。JSON包括保留的样本；待处理事件、设备排队等。RNG并且资源事件仍然属于同一连续性的一部分。配置比较使用已安装的模型代码。在恢复本地扩展时，提供相同的 `domain_modules` 注册表；保存的绑定中使用的可调用名称描述了已安装的代码，并且永远不会作为导入。

较早版本的 v1 会话需要明确的 `SessionSnapshot.migrate_v1(payload)` 调用。 这仅在运行时语义与编码匹配时进行转换，并且在省略汇编顺序时，确保顺序明确。 多个记录，如果没有保存的顺序，则需要原始的排序信息。 恢复操作仍然会检查完整的执行绑定。 较早的物理世界输入会保留其支持的边界转换；这并不允许重新解释已冻结的会话证据。

## 模型和接受范围 {#model-and-acceptance-boundaries}

这种气体反应、五步表面速率、有限的材料转移、共享的内部能量/比热容平衡以及加热器模型，都是可执行的物理模型。测试中出现的数值失效、过程失效和无效读数，是接口上的测试，而不是实际的失效机制。热力学停止使用真实事件。属性表和速率参数是有限的合成合约。表面流动、任意的热力学和新的电极动力学，则不在这个基础部分。

原生的接受范围包括固定的原生/健身房/CLI/代理历史记录，以及两种观察噪声模式，直接/检测结果，回放和保留的设备。安装检查同时构建了两个档案，并在不进行检查的情况下运行安装的模块，其中 Cantera 不存在，并且从子环境中移除了模型密钥。报告的平台结果仅适用于实际运行；这是一种开发接受，而不是正式发布或声称普遍的化学有效性。请参阅 [的分发边界和](current-foundation.zh.md#source-synchronization) ，了解包装和冻结发布的细节。

## 当前状态下的特性 {#properties-at-the-current-state}

`ComponentPropertyPackage.evaluate_state(...)` 评估一个纯状态；`evaluate_property_state(packages, amounts_mol, ...)` 评估一个组合。两者都使用现有的组件 ID、已声明的别名和 CAS 标识符。提供 `temperature_K`、`pressure_Pa`、`phase` 以及请求的 `quantities`。报告保留了每个属性的单位和有效性警告。例如：

```
from chemworld.physchem.curated_properties import curated_property_package

report = curated_property_package("7732-18-5").evaluate_state(
    temperature_K=400, pressure_Pa=1000, phase="gas",
    quantities=("cp", "enthalpy", "internal_energy", "density"),
)
cp = report.properties["cp"].quantity().to("J/(mol*K)")
```

能量值包括摩尔值 J/mol、比热容 J/(mol K)、体积 m³/mol 和质量密度 kg/m³。 默认的能量数据明确指出，在 298.15 K 时，H=0；它不提供形成焓或绝对熵值。 当需要提供其他数据时，请提供 `CaloricReference`。 熵和化学势需要熵值。 目前提供的六个软件包提供气体 Cp 值和蒸汽压力；它们的默认材料相不意味着液体的 Cp 值或输运数据。 缺失的数值会导致错误。

气体状态默认采用理想气体模型。 明确指定 `CubicEOSSpec` 选择 PR/SRK 的体积、势能、偏差和导数，这些参数对应于请求的蒸汽/液体相。 这是一个单相评估；相平衡属于闪蒸模型。 它的 Cp 导数遵循所选的外部根，不支持根拓扑结构的改变。 `validity_policy="warn"` 或 `"ignore"` 可以评估超出相关范围，但 API 在两种情况下都会保留无效域的记录。

原生反应堆和相关设备的使用者可以共享 `PropertyPackageThermodynamics` 与...一起 `SpeciesCaloricModel` 数据。他们从当前候选状态的相同评估中获得 U(n,T)、 Cv 和成分导数。 常见的积分器同时处理时间和热的计算；材料混合则针对温度求解相同的 U 关系。 接受的状态和定义的运行区间必须在声明的能量域内。 这种能量的运行时间不声称存在非理想的封闭体积。EOS保持平衡。现有的恒定 Cv 声明委托给此评估路径；较早的液体的体积 Cp 模型和固定容量的相区仍然是明确选择的模型。 变量热量的相变需要瞬时/潜热闭合。

在已安装的环境中运行 `python -m chemworld.examples.state_property_research`，用于进行属性查询和实际的变参数-Cv反应器历史。 它的温度依赖关系是合成的，并保留了原始的天然U数据；它是一个接口示例，而不是新的材料数据。 生产评估不依赖任何后端参考。 可选的 `physchem-ref` 工具仅用于 `workstreams/world_foundation/NATIVE_PROPERTY_ACCEPTANCE.md` 中记录的，分别针对同一方程和模型偏差的检查。

## TP/PH：闪光和有限相位库存 {#tpph-flash-and-finite-phase-inventories}

`VLEPropertyPackage` 结合了现有的属性包、明确的气体焓参考以及已声明的液体活性模型。`tp_flash(...)` 更新...K根据当前的液体和蒸汽组成；以及可选的现有气体EOS更新了蒸汽的势能和逸散焓。液体的焓是与相同饱和压强/活性的自由能相关的温度导数，与气相的基准值相同。`ph_flash(..., pressure_Pa=..., enthalpy_J=...)` 保持总的焓值不变，并求解 T。纯饱和状态使用焓的杠杆法则。缺失或消失的相具有零的物料库存。无法达到目标焓值和非收敛情况需要明确指出。

TP当没有提供入口 H 时，它会返回一个状态查询；此时，它的热交换功能将失效。 通过明确的 `feed_enthalpy_J`，TP报告了维持指定 T 值的所需热量。PH报告显示没有额外的热量输入，并提供了其实际的焓残差。它不会通过将残差分配给加热器来平衡能量。

这个第一个模型支持一种液体和一种气体，以及两种/三种组分的理想液体或经过认证的凸形二元马格鲁斯液体。 其他的活动模型和未经认证/非凸形的液体需要独立的 LLE/稳定性求解器。 关键的相变和三相状态不在该模型范围内。 罗奥液体热力学模型忽略了液体的体积/泊松功；其默认的液体占有量是一种独立的几何填充近似。 没有任何声称能够精确计算液体内部能量/PV-功的平衡。

`FlashSystemSpec` 声明这是一个有限的恒压热力系统。 原生的 `vapor_liquid_equilibrium` 组件通过现有的物理协调器，选择其焓策略。 候选 H 被一次性整合，PH 关闭所有候选，并接受的液体/气体数量占据正常的记录。 正常的设备、进料、材料路由、测量、交易、资源和快照服务都使用相同的库存。 注册的液体种类明确声明其载体表示，从而防止从几何体积推断出第二个旧溶剂-体积库存。 元素和载体平衡检查保持活动状态。

执行 `python -m chemworld.examples.native_flash_research` 命令，用于 PH 查询，然后进行实际的有限供料、跨相加热、定时相位探针、转移和终止操作。 `phase_state` 是一种明确的、无噪音的合成传感器，而不是实验性相位识别的声明。 请参阅 `workstreams/world_foundation/NATIVE_FLASH_ACCEPTANCE.md`，了解固定 27 状态的热比较和发展边界。

## 严格的液体稳定性，以及有限的清洗 {#strict-liquid-stability-and-finite-washing}

`liquid_tpd_minimum(activity_model, composition, temperature_K=...)` 在一个确定的 simplex 初始集合上，最小化实际的切线平面距离。`equilibrate_liquids(activity_model, amounts_mol, temperature_K=...)` 最小化受约束的吉布斯能量，精细化相的化学势平衡，检查组分平衡，并在两个结果相上重复 TPD 的搜索。稳定的进料返回一个液体。剩余的负向分割 TPD 拒绝不支持的额外相。二元/三元模型最多支持两个液体；有限的数值搜索并不能证明对于任意的活性函数，都能实现全面的全局优化。

`mixing_chemical_potentials_RT` 报告 `ln(x_i gamma_i)`，相对于常见的纯液体标准，并除以 RT。 常见的纯参考术语在等温液体相中相互抵消；这些不是人为构建的绝对化学势。 活度系数使用现有的参数模型，与性质和接触计算共享。

`StrictLiquidSystemSpec` 声明一个等温材料定律，以及成分的摩尔保留率和有限洗涤策略。 `liquid_equilibrium` 组件及其匹配 `CompositionPhysics.liquid_models` 该功能使常规混合/冲洗、进料、计时、测量、物料分配和快照服务得以正常运行。实际的清洗液来自一个有限的、有明确名称的供应；被丢弃的材料是明确的分子输出。可混溶的清洗液保留一种混合的液体。液滴吸附使用与经验接触模型相同的库存转移函数。相位探针报告液体数量和第二液体的比例，作为合成的原始通道。混合热和PV的工作在这一等温切片中明确未建模；没有暗示任何热控制操作。

较早的分配系数接触模型仍然是一个独立的模型。 它的 `minimum_tpd_like` 包含一个经验性的分割驱动，并且明确地标记为 `thermodynamic_stability_certified=False`。 严格的液体平衡状态不使用该驱动，也不会在求解器失败后返回到该驱动。

执行 `python -m chemworld.examples.native_liquid_research` 程序，进行有限混合、清洗、定时观察、转移和终止操作。该示例中的 Margules 参数和组分身份均为合成数据。请参阅 `workstreams/world_foundation/NATIVE_LLE_ACCEPTANCE.md`，了解关于共 tangent、三元和实际 PhasePy 验证的信息。

## 保守型扩散原模型 {#conservative-diffusion-primitive}

`chemworld.physchem.finite_volume` 提供在 SI 单位中，二维和三维的一维菲克算子。`DiffusionMesh` 定义物理体积和面面积；`ConservativeDiffusion.evaluate(amounts_mol)` 返回摩尔导数和带符号的面速率，但不进行时间推进。正向指定的边界流速意味着 mol/(m² s) 中的向外流速。球形中心需要对称性。

`operator.couple(existing_inventory_keys, process_id=..., owner_id=..., boundary_enthalpies_J_mol=...)` 为现有的 `CoupledSystem` 提供一个过程和材料端口。对于开放边界，需要明确的边界热焓。BDF/Radau 使用已组装的稀疏网格。外向固定流耗尽在实际事件处停止。调用者保留了物料库存的所有权。这是后续电极/粒子模型的共享空间基本单元；这些数学消费者不实现电池或新的 PDE 运行时。

## 有限动态蒸馏 {#finite-dynamic-distillation}

`DynamicColumnSpec` 复用 `ShortcutColumnSpec` 定义硬件范围，并用 `VLEPropertyPackage` 定义热力学定律。模型包含 2–8 块平衡塔板，以及有限的再沸器、冷凝器和产品收集器。各隔室推进实际组分量与焓 H；task-18 的 PH 求解在每个候选状态闭合相分配。蒸气采出与液体溢流使用声明的有限持液量定律。热流与物料流统一进入一次共享的 BDF 推进。此模型不包含完整塔板流体力学、压降网络或同时发生的反应。

A `dynamic_distillation` 组件及其匹配 `CompositionPhysics.column_models` 选择此策略。 `dynamic_column_world_spec(spec, initial, ...)` 通过明确提供的属性和初始物料库存，构建一个原生装置。 `distill`，通过加热/等待和定时观察，使用共同的协调器；干燥、冲洗、温度、收集和原生容器-体积边界，都停留在其实际根部。产品历史记录了收集量随时间的变化。 `collect_fraction` 将实际的液体 n 和 H 转移到其各自的保留相中；它不会重置柱子或进行回收。

蒸馏控制包括目标温度、持续时间和有限时间 `reflux_ratio`。 原生事件JSON此外，该系统也接受带有标准要求的回流字段的数字 `full_reflux: 1`； `full_reflux: 0` 恢复有限的收集。 保持不变的固定 Gym 数字载体明确拒绝此附加字段，并引导用户到事件。JSON. 原始的 `distillation` 组件仍然保持为标记的 `shortcut_estimate`，并且没有记录任何阶段历史或在动态集成失败时的备用方案。

PhaseLedger单独拥有大量 n 型材料和文具。材料占用单独的类型槽。更换材料会保持设备的热量和校准，然后通过以下方式使新液体和剩余的物质达到平衡：PH. 批量重置使用已声明的初始热边界，同时保留硬件设置。快照保留所有组件和控制。合成相探针观察回热器组件。液体的体积保持与任务 18 的几何堆积近似，没有精确的液体U/PV- 索赔申请

运行 `python -m chemworld.examples.native_distillation_research` 针对五个实际操作：蒸馏、定时测量、有限收集、回流变化和停止。该示例不需要任何可选的参考软件包。其 Cp 值、潜热和蒸气压数据都是明确的合成数据。固定动态网格、IDAES 和 MESH 的稳定比较、芬斯极限和时间精细化等内容已记录。 `workstreams/world_foundation/NATIVE_DYNAMIC_DISTILLATION_ACCEPTANCE.md`.

## 有限的真空浓度 {#finite-vacuum-concentration}

`CoupledConcentrationSpec` 采用与闪蒸和蒸馏相同的状态依赖的 VLE 属性和 PH 闭合过程。 容器和接收瓶各自拥有自己的物料库存，并且与 H 相互作用。 实际存在的容器蒸气通过抽样口导出。 现有设备将捕获的蒸气导向瓶；未捕获的蒸气穿过明确的组件/H 边界。 捕获的蒸气在 PH 和有限冷却口冷却后，仍然保持在瓶的物料库存中。 没有单独的瞬时冷凝计算或潜热扣除。

该 `vacuum_concentration` 组件和对应的 `CompositionPhysics.concentration_models` 进出口选择策略，共同决定操作。`concentration_world_spec(spec, initial, ...)` 搭建一个原生的装置。`concentrate` 能够实现加热和蒸汽输送。加热/等待以及随时间进行的现场测量，继续进行相同的物理过程。加热暂停保持蒸汽输送设置不变。操作停止关闭蒸汽和加热，使其恢复到初始状态；随后的被动测量可以冷却瓶子。在未满足溶剂终点的情况下，持续运行，是一个有效的结果。

锅、瓶、冷凝器气体饱和、残余液体、目标回收、温度和容器体积是有限的边界。 经过认可的时间、加热器的输送、冷却和出口积分，一旦确定，便不再改变。 原生材料的接收，可以验证捕获、组件提取和输送的焓值； 使用相同的操作名称，并不允许新的出口。 数值上的错误会回溯到当前的材料、H和时间。PhaseLedger拥有所有当前的散装材料；按类型区分流体 H 和固定热容材料。材料更换可以保持身体的热量和校准，并且…PH在不改变时间的情况下，实现了新的充电，同时保留了实际的库存、控制和累计的边界收据。

明确声明为不挥发的组分，在模型压力下使用液相 Cp/H 参考，气相分配严格为零。缺失的气相物性仍不可用。当气相分率舍入为一时，共享 RR 报告仍独立保留痕量液相分率；只要存在正量的不挥发物料，未定义的 beta=1 目标就返回 null。该模型范围支持固定压力下含一种或两种挥发性物种、以及可选不挥发目标物的理想混合物，不包含同时发生的反应。液体体积仍采用声明的几何持液量近似；忽略液相 PV/Poynting 功。

`simulate_vacuum_concentration(CoupledConcentrationRequest(...))` 和原生操作使用相同的贡献和公共求解器。现有的 `VacuumConcentrationRequest` 输入明确地规定了一个固定的评估运行方案。它们的差分近似仍然被称为 `declared_fixed_operating_profile`，共享相位-对流和求解器工具，并且永远不会作为因原生求解失败而采用的备用方案。其现有的服务现在会收取实际的运行时间以及传递的热量，包括提前停止。

运行 `python -m chemworld.examples.native_concentration_research`，用于浓度测量、定时测量、加热暂停、等待和终止。该示例（可安装）使用合成属性，不需要额外的参考包。固定九个控制网格、两个材料固定装置、瑞利保持极限以及原生快照的延续，都记录在 `workstreams/world_foundation/NATIVE_CONCENTRATION_ACCEPTANCE.md` 中。

## 持续的有限吸附剂接触 {#persistent-finite-sorbent-contact}

`SorbentKineticsSpec` 提供具有竞争力的朗缪尔亲和力，以及本地LDF包括速率和吸附能。流体的浓度和温度来自 `CoupledSystem` 使用的相同参数。一旦结合的分子进入其物料库存。温度依赖的亲和力和整体/结合的 U 偏移量必须使用相同的吸附能。载体热容与共同的温度平衡相结合；不单独计费吸附热。不均匀的组分迁移率使用已记录的有限空位占据约束来保持共享的位点。这是一种零维、单位级的液体接触，涉及两个竞争组分，没有轴向床、压力降或微粒分辨率。

`sorbent_contact` 选择 `CompositionPhysics.sorbent_models` 以及现有的耦合热策略。一个固定的载体在其设备槽内保持其负载；一个移动的载体移动实际的质量、位置、热容和结合分子，并使用其材料部分。材料替换保持固定的负载和设备的热量，通过标准的 U 型模型混合新鲜材料。实际的干燥/热/等待过程共享一个时钟和求解器。求解器失败会回退流体、负载、温度和物理时间；已确定的失败行为的成本/风险惩罚仍然适用。

`CondensedBackgroundSpec` 明确定义了特定体积下的 Cv 和 U 值。L-仅限运输方式，内容精简PV/Poynting 过程被忽视。它不涉及分子载体或气体数据的产生。 原生的吸附装置必须声明一个有限的载体来源容器。 现有的 add\_solvent 功能会扣除实际库存，并将其能量转移出去； 路由传输包括溶质 U、载体 U 和可移动的结合材料。 静止的清洗源没有主动的热过程。 资源耗尽会拒绝输入，而正常的材料/批次初始化会保留有限的供应。 明确的实体替换和新世界的重置仍然是生命周期的边界。

现有的独立等温干燥请求现在委托给相同的瞬时有限流LDF这项工作保留了明确的有限新鲜床请求和最终的废弃出口，并且没有提供未提供的热量数据。 实验室的运行方式拥有每容器 20 克未使用的媒体套件； 每次实际的干燥包都会扣除并通过认证的途径输出。kg确认实体接收。重复的干燥或材料重新设置无法制造新的介质。不会保留第二次使用的床材库存。

通过吸附、有限洗涤、加热、定时测量和停止，对`uv run --no-sync python -m chemworld.examples.native_sorbent_research`中的八个现有操作进行模拟。所有构成数据都是明确的合成数据。`workstreams/world_foundation/NATIVE_SORBENT_REGENERATION_ACCEPTANCE.md`中记录了固定12个条件的接触网格、独立的平衡参考和三个原生循环。

## 有限的瞬态工作电极 {#finite-transient-working-electrode}

`electrode_world_spec()` 创造了一个原生、有限的二维 O + e ↔ R 系统。 共享的守恒扩散算符以及 `CoupledSystem` 的推进，共同影响了两种物种的分布，并结合了布特-沃尔姆交换和双层电荷。 罗宾边界包括第一半电池的扩散阻抗；外部边界是封闭的。 正向电流对应于阳极。 在返回候选方案进行交易验证之前，会检查材料和带电荷的关系。 电气工作会被记录，但不声称完全关闭电池能量或反极电解的化学反应。

```
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.electrode_world import electrode_world_spec

session = ChemWorldRuntime(composition=electrode_world_spec())
session.reset(seed=7)
session.execute({"operation": "configure_electrical_control", "potential_V": 0.1})
session.execute({"operation": "wait", "duration_s": 5})
reading = session.execute({"operation": "measure", "instrument": "electrical_meter"})
print(reading.observation.native_record())
```

将 `ramp_duration_s` 设置为从当前设定值开始的线性电位变化控制动作。 此配置不影响时间； `wait` 和仪表的一秒实时采集使用相同的物理协调器。 测量结果显示终端 V 和总 A 值，但不包括内部浓度或界面电位。 这种合成型仪表是无噪音且无功耗的。

这些数据在材料相的特定区域内仅存在一次。电荷和累积的电能属于电极设备；材料重置可以保留这些数据，而明确的设备更换则保持实际电解液数据的完整性。快照可以保存控制程序、物料库存和电荷信息。静态设备与流动液体保持分离；动态或未支撑的材料状态仍然需要一种兼容的预先策略。

第一种模型使用恒定的扩散系数、足够的支撑电解质、等温运行和有限的串联电阻。它排除了迁移、对流、均相反应、电流控制和全电池的热物理现象。现有的固定扩散层和 RC 模型仍然是更简单、更明确的替代方案，它们共享相同的电化学常数和 BV 因素。

运行 `uv run --no-sync python -m chemworld.examples.native_electrode_research`，进行三个控制，等待并进行原生测量，然后终止。 固定的数值检查和开发失败情况记录在 `workstreams/world_foundation/NATIVE_ELECTRODE_ACCEPTANCE.md` 中。

## 程序化色谱分析和持续的仪器状态 {#programmed-chromatography-and-persistent-instrument-state}

`chemworld.world.chromatography_world.chromatography_world_spec("hplc")` （或者） `"gc"`) 创建一个有限的、合成的三种分析物世界，包括源瓶、接收器、材料路由、废弃物以及一个持久的色谱仪。它使用与其它原生世界相同的运行时间、资源记录、测量记录、快照和回放功能；不需要任何模型提供者或 API 密钥。

```
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.chromatography_world import chromatography_world_spec

session = ChemWorldRuntime(composition=chromatography_world_spec("hplc"))
session.reset(seed=12)
session.execute({"operation": "configure_chromatography",
                 "instrument": "programmed_hplc", "program_index": 1})
# Initially the receiver contains blank solvent. Route a finite source aliquot
# into it to measure a standard or sample (see the complete example below).
result = session.execute({"operation": "measure", "instrument": "programmed_hplc"})
print(result.observation.channel("peak_1_area"))
```

程序 0 保持不变（298.15，K，HPLC 移动相比例 0.2）；程序 1 在 30 秒内逐渐增加到 0.8 的比例，用于 HPLC 或 338.15 的 K 或 GC。程序 2 使用在 0、15 和 30 秒的时间点，在 15 秒达到 20% 的变化。这些程序重用现有的保留时间方程，具有明确的程序延迟和时间积分的洗脱标准；恒定情况回归到现有的等离子/等温方法。曲线重用高斯峰的基础。

每次采集消耗 0.0002 L，成本 0.05 货币，并且在不改变主机的情况下，增加 60 秒的离线分析时间。实际的分析物残余属于设备拥有的状态。一个空白操作会将此物料库存转移到记录的外部废弃物出口；这并不是重置。使用计数、增益/基准漂移、程序和校准在材料更换和快照恢复过程中保持不变。完全重置会恢复声明的初始仪器。

公开记录包含信号曲线、拟合后的峰面积和浓度估计，以及单位和质量信息，以及已声明的峰模板和所需的响应校准，以便在不使用模拟器状态的情况下重现反向拟合。 这种有限的仪器假设已知的峰的身份/形状；它无法识别未知的保留机制。 `calibrate_chromatography` 记录了最后获得的公开峰面积，以及已声明的 `standard_1_M`、`standard_2_M` 和 `standard_3_M`；至少三个标准采集与现有的响应校准相符。 它不检查隐藏的浓度、检测器增益或基线。 饱和、重叠和不完整的分离会使估计失效，并且未完成的采样。 重新读取缓存的观察结果不会消耗任何样本、时间或随机抽取。

对于两台仪器以及所有三种程序（包括 JSONL 回放和独立的残余/废品检查），完整地执行标准-样品-空白序列，共执行两次。

```
uv run --no-sync python -m chemworld.examples.programmed_chromatography --output /tmp/chromatography-example
```

选择一个新的输出目录；现有的结果不会被覆盖。相同的模块是从已安装的轮（wheel）运行的。这是一个有限的开发示例，而不是一个真实的柱：三个合成分析物、对称的峰、合并的残差、固定的离线运行时间以及一个不随时间变化的宿主。连续的流动相使用、柱的空间运输和仪器的热能利用均未建模，因此无法进行能量守恒或一般的梯度-柱相关性声明。请参阅 [的开发验收](current-foundation.zh.md#source-synchronization) 。

## 原生单颗粒锂电池 {#native-single-particle-lithium-battery}

`battery_world_spec()` 安装了一个有限的电池，每个电极都包含一个球形颗粒群。每个颗粒群都使用与平面电极相同的守恒扩散算子。内部的锂离子转移将两个表面与施加的电流连接起来；电压遵循合成的 OCV 曲线和对称的布特-沃默过压。电流、锂离子和提供的电功在共同的求解器中进行，并伴随着实际的表面和电压截止事件。

```
from chemworld.physchem.battery_models import BatterySPMSpec
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.battery_world import battery_world_spec

spec = BatterySPMSpec()
session = ChemWorldRuntime(composition=battery_world_spec(spec))
session.reset(seed=4)
session.execute({"operation": "configure_electrical_control",
                 "current_A": spec.current_at_c_rate(0.5)})
session.execute({"operation": "wait", "duration_s": 600})
reading = session.execute({"operation": "measure", "instrument": "battery_meter"})
```

相同的电控制作用仅接受以下一种支持的量：对于该电池，接受 `current_A` 或 `potential_V`；对于有限的工作电极，接受 `potential_V`。两者都采用标量线性插值和原生计量协议；并且不支持源拒绝模式和限制。正极电池电流表示放电。C-rate 的定义基于初始可用的锂容量，而不是从后续测量容量重新计算。原生计量仅暴露 V 终端和 A。其一秒采集操作，会同步该电池的状态。

锂电状态、已签署的充电和工作数据属于电池设备，并且只会被保存一次。材料重置会保留这些数据；更换实际的电池需要明确地记录新的设备状态。表面切割会保留真实的时间和物料库存。将电流设置为零，可以实现后续的放松。共享账本使用正向的电工作，因此，已交付的电池工作会被记录为相反的符号。

这些参数是合成的，并且使用均质无扩散电解液。默认的等温选项仍然可用；电解液的运输问题仍然未解决，并且可选的有限SEI扩展方法如下。运行 `uv run --no-sync python -m chemworld.examples.native_battery_research` 以查看放电/休息/充电历史。PyBaMM仅由可选的参考块使用；普通安装无需使用此示例。数值范围、所有开发失败和固定比较都记录在 `workstreams/world_foundation/NATIVE_BATTERY_SPM_ACCEPTANCE.md` 中。

### 周期性循环以及一致的合成热模型 {#cycles-and-a-consistent-synthetic-thermal-model}

通过传递 `BatterySPMSpec(thermal=BatteryThermalSpec(...))` 来改变温度。细胞的化学能量是从相同的线性 OCV 函数中整合的，并且通用的求解器平衡了终端的电工作、化学储存、加热输入和环境损失。动能、欧姆损耗和扩散损耗是独立地推导和检查的；它们不会再次被视为反应热。 OCV 曲线没有明确的温度依赖性，因此可逆熵热为零。正 OCV 坡度被排除在这个凸面热模型之外。`series_resistance_ohm` 可选地添加了实际的电压降和焦耳加热。

`battery_world_spec(spec, thermal_device=ThermalDeviceSpec(...))` 连接现有的加热器或等温浴。`heat`、`wait` 和热量暂停/恢复功能均使用相同的设备；`stirring_speed_rpm` 是 `heat` 的可选功能。一个具有有限功率的加热器，会逐渐改变实际温度。理想的浴缸应从设定的值开始，并记录其实际的补偿热量。

使用 `queue_electrical_stage` 与以下任一值：`current_A` / `potential_V`，一个有限的 `stage_duration_s`，以及可选地一个原生终端条件：`stop_voltage_below_V`、`stop_voltage_above_V` 或 `stop_abs_current_A`。零电流阶段是休息状态。这些阶段附加到源端控制队列；`wait` 推进公共控制器，该控制器也用于现有的热/气体程序。未安装任何针对电池的特定调度器。完成的队列切换到零电流。明确的 `configure_electrical_control` 清空队列并安装手动步进/斜坡。电压控制根据实际曲线计算电流；无法达到的电压保持在源端电流限制，实际电压和限制状态被记录。

每个阶段的转换记录了最终的读取值和实际经过的时间。物理中断事件保留已达到的状态；数值错误会回退完整的候选操作，包括之前的候选阶段，同时保留诊断信息。快照精确地保存了队列、光标、温度和物料库存。材料初始化是一个评估边界：它保留电池 SOC/profiles，并使用声明的初始温度，该温度记录在它的生命周期收据中。持续循环则不需要进行此类初始化。

运行 `uv run --no-sync python -m chemworld.examples.native_battery_cycles` 两次CC/CV/休息/充电循环。九次热量比较，可选择PyBaMM并且，`workstreams/world_foundation/NATIVE_BATTERY_THERMAL_ACCEPTANCE.md` 中记录了两个原生周期。

### 有限的 SEI 增长和保持的退化 {#finite-sei-growth-and-retained-degradation}

将 `sei=SEISpec()` 添加到 `BatterySPMSpec` 中，以实现溶剂扩散受限的 SEI。 具体的总反应是 `2 Li + EC -> Li2CO3 + C2H4`，对应于负极界面上两个锂离子和两个电子。 EC（碳酸盐）保留了乙烯，并且在电池的设备槽中，活性锂离子浓度同时存在。 膜的厚度和电阻均来自碳酸盐储备；不存在独立的厚度库存或每次循环容量的减少。

侧电流为 `-2 F r`。负向掺杂产生 `I + 2 F r`，而正极则产生 I。 颗粒边界和负向 BV 关系都使用该主电流。 膜电阻的变化通过测量终端电压来确定。 它的耗散和反应热与热模型中的化学状态函数相同。 不可逆的侧过程在电化学驱动力的边界处停止； 高速率或不利条件不会被无声地推导出向自发增长的模式。

最初的电影明确地存在于最初的元素库存中。一个闭合的细胞会消耗其实际的有限EC；未使用隐藏的溶剂储备。分析过程中的溶剂保持检查，则引入了明确的外部供料。物种公式使 Li/C/H/O在运行时交易中，这些检查具有实际意义。材料初始化保留整个细胞的化学历史；更换细胞使用新的明确设备库存。快照和代理回放保留了薄膜、溶剂、气体、活性锂、电气控制和温度。

源队列中，有 32 个待处理的任务。已完成的任务会保留在历史记录中，并在添加新任务时从待处理列表中移除。`stage_index` 保持不变，而 `stage_offset` 负责管理保留的任务列表。因此，重复执行有限数量的程序不会耗尽整个队列的容量。

连续运行 `uv run --no-sync python -m chemworld.examples.native_battery_degradation` 十个周期。该合成模型包括有限的溶剂、薄膜电阻以及双电子计算；但不包括电镀、开裂、电解质运输和气压力学。固定分析，九个条件。PyBaMM并且，每十周的检查记录在 `workstreams/world_foundation/NATIVE_BATTERY_SEI_ACCEPTANCE.md` 中。 窗式放电充电和锂损失等效充电分别报告，两者不能直接替代使用。

### 有限的泵、管道和换热器的输送 {#finite-pump-pipe-and-heat-exchanger-transport}

`transport_world_spec()` 安装了一个真实的有限流量和接收器，通过一个受限的泵、管道以及可选的埃尔根床阻力连接。`set_flow_rate` 在 mL/min 中设置一个请求，但不改变时间；`run_flow` 或 `wait` 同时改变两个物料库存。零请求停止泵送。`run_flow` 可以通过 `target_temperature_K` 设置功用温度；否则，使用已安装的值。

该泵能够实现根据所需流量、压力和轴功率限制所允许的最大流量。 `PumpSpec` 以及 `pump_operating_point` 接受连接设备的单调压力-负载规律，以便膜进行重复使用。 在一个示例中，流量达到约 12.21 mL/分钟，针对 20 mL/分钟的请求，并在有限的拾取脚部停止，而不会对未执行的时间进行充电。

常见的有限物料库存求解器，会调整组件的数量和焓值。 液体性质使用候选的 T/x 时的原生软件包；换热器使用泵加热后的实际质量流量。 换热过程通过液压工作加热流体，轴的效率是显而易见的外部损失，而供热系统使用现有的 Nu/U/NTU 功能。 供料和供料共享压力和高度；管道/换热器具有极低的残留，而供热系统是一个明确的恒定温度边界，而不是一个未追踪的冷却剂库存。

材料槽拥有自身的流体焓；容器槽拥有容器的热量。 泵拥有其自身的规格和控制。 默认批次初始化保留有限的进料及其局部年龄。 显式接收材料替换保留实际容器的热量，并在不改变时间的情况下重新关闭温度。 来源耗尽、完全接收器和出口属性限制停止过程；数值失败会撤销候选方案，并进行诊断。

`flow_meter` 通过原生渠道报告 mL/min、Pa、W、K 和 L。 原生、Gym、CLI 和代理界面共享执行、会计和精确回放功能。 运行 `uv run --no-sync python -m chemworld.examples.native_liquid_transport`。 限制性的合成模型排除反应、回流、气泡、双相流以及经验性的工业泵曲线。 现有的空间 PFR 实验保持其声明的几何模型。 验证和可选的 fluids/ht 比较在 `workstreams/world_foundation/NATIVE_TRANSPORT_ACCEPTANCE.md` 中进行。

## 有限的晶体生长和溶解 {#finite-crystal-growth-and-dissolution}

`crystal_thermal` 组件安装 `CompositionPhysics.crystal_models` 除了明确的热力学模型，现有的“seed”、“cool\_crystallize”、“heat”、“wait”、“filter”和“resuspend”操作，都使用相同的有限悬浮液在实际温度下进行。 预设的设定值提供有限的加热或冷却；它不会事先分配溶解的分子。 PhaseLedger拥有当前的悬浮液分子。 组分行会分解固体，生成颗粒数量、直径、种子来源和被包裹的杂质；这些不是额外的库存。

所提出的范特霍夫曲线和有限径向生长/收缩定律，共享了液/固热量的参考。一个速率/能量汇集器和ODE，以及后端，用于处理材料、温度和功的积分。几何坐标的单位为m，以及mol/mol，用于重建每个候选者的实际分子库存。直径为零的根部可以移除宿主并释放其实际的隐藏物质。有限的杂质捕获只发生在序列化库存分辨率以下（默认值为总杂质的1e-8）；实际剩余库存和数值修正得以保留并与未变动的独立平衡进行比较。

核聚过程使用明确的时间步长，通常为 30 秒：实际分子进入核聚体储备，并在间隔末端与正在生长的群体合并。这种有限分辨率的近似方法具有独立的间隔精细化检查。不存在聚集、断裂、多形或轴向种群模型。所提出的简化的 Cv 和 U 需要极少的简化的 PV 工作；缺失的相态热力学数据仍然不可用。同时进行的整体反应和晶体热过程不在本范围内。

原生的理想过滤器保留所有颗粒、固体目标、种子和被遮挡的杂质，以及 8% 的湿母液。它的 480 秒的延迟，实现了相同的活性热控效果。重新悬浮可以将历史滤液在自身温度下储存，并通过 U 值混合已声明的进入载体，而无需立即溶解。现有的资源卡限制了新鲜载体和种子。材料替换保持装置的温度；快照保留了组、控制和资源延续。当前的容器转移需要在混合占据的天然晶体隔间时保持相同的温度。数值失败会倒退物理状态和时间，并保留已确定的失败行为的惩罚。

该传统实验室浆料也使用了有限的径向贡献，并具有明确的浓缩焓边界。它的载体 Cp 值没有被重新标记为 Cv 值。传统的 96%/92% 机械保留近似仍然独立，并且不能证明完全的原生 U 值的平衡。在浆料的温度保持之前，先进行独立的、预设温度的结晶过程。作为独立的、预设温度的结晶提供者，仍然是一个明确的近似参考，而不是在原生求解器失效时的备用方案。

使用合成属性，且不使用任何可选的参考库，运行 `uv run --no-sync python -m chemworld.examples.native_crystal_research` 8 次，实际运行时间为 901 秒。 记录了固定接触、出生-优化和双周期的覆盖情况，以及所有尝试的结果，这些信息都记录在 `workstreams/world_foundation/NATIVE_CRYSTAL_THERMAL_ACCEPTANCE.md` 和 JSON 中。

## 有限的碳酸盐水化学 {#finite-carbonate-water-chemistry}

`aqueous_world_spec` 安装了已申报的 `aqueous_mineral` 组分，并使用实际的水、H+/OH-、CO2\*、碳酸氢盐/碳酸盐、Ca、Na、Cl 以及有限的纯钙质。 浓度使用实际剩余的水 kg；几何液体体积是一个单独的申报的残留量。 元素、碳、电荷、水解以及碳酸盐的反应紧密相关。 有限的钙质可能出现或减少；在求解后，不会提供任何平衡的离子或矿物储备。 导出的物种信息需要与 PhaseLedger 进行比较，并且不是另一种材料库存。

相同的关闭机制同样适用 `add_component` 从有限的中性溶液储备中 `wait`仅用水 `sample` 以及无损的 `aqueous_ph` 电极。采样方法能够保留独立的矿物相。现有的pH计模型使用氢活动，型号为RNG；较早的简化酸碱模型则适用于历史任务。两种观察噪声计划都支持精确的快照和资源的持续性。无效的输入和数值错误，在现有的交易规则下，能够保留物理库存和时间。

这种合成系统仅适用于298.15 K，低压和离子强度 <=.1 mol/kg。 常数包括：pKw=14、pKa1=6.35、pKa2=10.33、log10 Ksp=-8.48、Davies A=.509/b=.3、中性γ=1和水活动度=1。 实际的水仍然参与反应元素的平衡。 这些平衡常数并不能提供热量数据、温度依赖性或矿物动力学信息。 矿物的流动和热混合不在该范围内。

在不使用可选库的情况下运行 `uv run --no-sync python -m chemworld.examples.native_aqueous_research`。 `scripts/check_native_aqueous.py --reaktoro-prefix <compatible-prefix>` 运行固定 33 种条件的模块，包括与 Reaktoro 2.13.0 进行 32 次比较，使用相同的自定义标准状态和活动，以及一个域拒绝和四个分析限制。 参照对象不在默认运行时依赖集中。 它的 CLI 在绕过观察到的 Windows 回调-最终化失败之前，会清除结果；计算失败会保持非零的退出状态。 覆盖和所有结果都在 `workstreams/world_foundation/NATIVE_AQUEOUS_MINERAL_ACCEPTANCE.md` 和它的 JSON 中。

## 有限的碳酸钙动力学和材料表面所有权 {#finite-calcite-kinetics-and-material-surface-ownership}

`mineral_world_spec` 在相同的碳酸盐化学条件下，安装一个 `mineral_kinetics` 类型的设备，使其与该系统协同工作。 单个 BDF 类型的设备可以促进有限量的碳酸钙的生成，同时在当前固态物质的条件下，快速进行水溶液反应。 面积减小的规律为 dn/dt=-A k(1-Ω)，其中 A=A\_ref(n/n\_ref)^(2/3)。 PhaseLedger 拥有分子；一个具有参考区域和参考 mol 的相边界材料，而无需将该参考区域视为另一个原子库存。

原生水相和碳酸钙相各自保持自身的相态。一个固体的 aliquot 携带与参考几何形状相对应的比例。水相转移/采样保留固体及其表面。设备保持原位。第一个模型支持每个隔间一个发展的矿物家族。不同的来源或不同的大小历史需要分开的隔间。同一批次中的相同进展部分可以在不改变其参考区域或库存的情况下进行分割和重新组合。

在反应完全结束时，实际溶解的原子仍然存在，并移除被消耗的表面。无种子超饱和状态没有面积，不会发生核化。声明的零速率/表面保持其库存。化学耗尽或数值饱和容差会导致反应停止，同时支撑的等温保持状态填补剩余的物理时间间隔；这些时间分别记录。有限的酸性输入改变了相同的水状态，在随后的动力学时间之前。pH电极使用其实际获取间隔后的物种状态。

任务 32 的温度、活动和热量限制仍然适用。 这既不是一种反应-运输数据库，也不是独立演化的晶体尺寸的分布。 执行 `uv run --no-sync python -m chemworld.examples.native_mineral_research`，进行 10 个原生操作、相转移以及在没有上游依赖的情况下，使用 1402 秒的共享时钟历史。

可选的 `scripts/check_native_minerals.py` 比较了十二个固定轨迹/120个状态，以及九个早期时间限制。 它的独立参考使用 Reaktoro 2.13.0 约束的吉布斯化学和 DOP853 的集成；它不调用原生代数求解器/ODE 后端，也不将上游隐式有限时间步视为精确的连续轨迹。 所有失败和结果都在 `workstreams/world_foundation/NATIVE_MINERAL_KINETICS_ACCEPTANCE.md` 和其 JSON 的摘要中。

## 有限稀释反渗透 {#finite-dilute-reverse-osmosis}

`membrane_world_spec()` 安装了一个有限的、充分混合的进料室，一个持久的膜/液压驱动系统，以及一个有限的渗透液收集器。 初始示例包含 1 个 L 的水，以及稀释的 NaCl。 原生的物理世界容量来自已声明的设备硬件； 遗留模板的大小范围保持不变。

`set_flow_rate` 要求水流的扫荡速度为 mL/分钟。`run_flow` 和 `wait` 使用相同的有限物料库存求解器。任务-24 泵的功能，通过所需的膜压力、管道阻力、最大压力和轴功率来限制实际流量。压力是从耦合的溶液扩散和浓度极化方程中推导的，而不是根据要求进行的排斥。剩余的进料是浓缩液：不存在重复的浓缩液库存。

`membrane_meter` 作为原生材料，能够展现水流的特性。m/s，膜的压力Pa, 轴功率（单位：W），进料/出料/收集的盐度mol/m^3 并收集L. 瞬时渗透盐度与收集者的累积组成不同。 `collected_fraction` 指的是收集的水，除以当前保留的水总量；在明确的材料替换后，它不代表对所有历史进水的总回收。 针对特定任务的回收利用实际的输入/输出历史。 零流量出口盐度报告模型的右限。

在盐渗透率有限的情况下，零驱动渗透浓度趋向于整体浓度。理想的过滤膜则具有有限的渗透阈值。这两种情况是不同的。反向驱动不受支持；有限的吸取和收集能力，没有前向泵驱动，以及浓度域的限制，都保留了达到状态/时间。数值计算失败会撤销候选方案。批量初始化同时保留了实际的液体库存和它们的局部年龄。明确的材料替换记录会保留交换过程，同时保持膜的身份和容器的身份。

这个第一个模型是在298.15度的等温条件下进行的。K在理想的低渗透压和水密度为1000的情况下kg/m^3 以及可忽略的盐分体积百分比。总体/界面浓度保持在 0–100 之间。mol/m^3. 液体动力作用，对一个无出口的腔体施加压力，实现几乎没有压缩存储和膜/管道残留。 没有任何回流、可压缩的瞬态或收集器回流。 污垢是明确的选项，具体描述如下。 理想的混合焓为零；边界将实际的泵工作作为热量移除。 这种范围不预测溶液的温度或高压的焓值。

运行 `uv run --no-sync python -m chemworld.examples.native_membrane` 两次，每次持续半小时，并记录原始数据。独立使用 WaterTAP 和 SD/CP 进行本地比较，这些比较使用相同的理想边界属性。它们不声称提供完整的 WaterTAP 流程表或其默认属性包。所有测试和尝试都在 `workstreams/world_foundation/NATIVE_MEMBRANE_ACCEPTANCE.md` 中进行。

## 持续的膜污染以及有限的冲洗 {#persistent-membrane-fouling-and-finite-rinsing}

将 `MembraneSpec(fouling=MembraneFoulingSpec())` 传递给 `membrane_world_spec`，使其包含一种惰性物质。SiO2污染物，一个有限的纯净水清洗罐和一个有限的废弃物罐。 污染物进入设备拥有的薄膜；相同的薄膜库存设置相同的RO流体/泵解决方案。吸附过程将材料返回到进料。不引入第二层膜、时钟或质量复制。

`set_flow_rate` 接受可选的 `flow_path="process"` 或 `"rinse"`。 省略路径后，它将保持当前的选择。 仅进行清洁的膜，有限的传输路径和旧的 PFR 不采用冲洗路径。 切换理想阀门不会消耗时间和液体； `wait` 或 `run_flow` 实际执行操作。 冲洗旁路隔离进水，将清洗液带到膜表面，并将所有分离的污染物送入废弃物。 它没有残留物，也没有膜渗透。 水的使用和泵的工作遵循实际流量； 零流量不会进行清洁。 空洗或满载的废弃物在达到预定时间时停止。

批次边界和更换进料，可以保持膜和附着的薄膜。更换膜时，使用常规的设备维护流程，并记录移除的附着物料库存。这种评估初始化不是自由的清洁过程。公共计量器记录洗涤/废弃物体积、冲洗流量和废弃物污染物质量；它不会暴露隐藏的污垢参数或附着物料库存。

这是合成的稀悬浮液模型：SiO2 的偏体积为零，仅增加水力阻力，不含渗透、反应或热量项。298.15 K 的恒温边界移除轴功。冲洗路径不模拟化学清洗、反洗或泵与阀的瞬态。清洁极限保留 task-34 的 WaterTAP 对照；污染过程采用独立质量衡算、解析脱附及时间步细化进行验证。

使用相同的有限容量的容器，对 `uv run --no-sync python -m chemworld.examples.native_membrane_fouling` 进行三个处理/冲洗循环。九个固定的开发循环单元以及所有尝试的结果都记录在 `workstreams/world_foundation/NATIVE_MEMBRANE_FOULING_ACCEPTANCE.md` 和 JSON 中。

## 有限的二进制固态平衡 {#finite-binary-solid-equilibrium}

`solid_world_spec()` 安装一个有限的A/B样品、现有的热控设备以及一种合成相计。`SolidSolutionSpec` 声明，该样品包含一个或两个理想混合的取代相。L0/L1红里-基斯特项。求解器会检查每种已声明相的每一个稳定点的共线。一个 nominal 相可以分解为两个成分域；网格只初始化共线求解。现有的相谱记录存储每个域的实际A/B物料库存，在设备设置中没有单独的质量复制功能。

`heat` 和 `wait` 均使用通用的有限相容热量积分器。 在每个候选的热量值下，稳定的相平衡确定温度和相的组成。 现有的有限功率加热器/冷却器提供实际的热量；时间和能源使用在确定后。 材料和批次的更换保持样品保持器的热量，并使其与新鲜材料达到平衡；容器的更换也初始化了保持器。 这些是记录的评估器初始化，而不是自由的实验操作。

该仪表显示了已承诺的相位的组成/比例以及温度。K. 原生/健身房/CLI/ 代理会和独立流程JSON后续部分采用相同的实现方式。该模块声明其自身的 290–850K有效范围；历史上的液体作用范围不会静止地阻止一个固体在430摄氏度以下。K. 授权限制和设备/容器限制仍然缩小了可接受的请求范围。

运行 `uv run --no-sync python -m chemworld.examples.native_solid_solution`，以加热混合后的固体，然后将其冷却回两个成分区域。 这是一个合成的二元、恒压、瞬时局部平衡模型，具有常见的恒定Cp值。 Al/Cu 公式仅提供元素计数，而不是一个拟合的 Al-Cu 合金、一个核/生长模型、一个磁性或多亚相描述，或一个运行时TDB解释器。可选的 pycalphad 参考使用相同的合成能量，并明确地将其四舍五入后的气体常数与原生常数对齐。运行时不需要任何上游求解器。固定 33 个单元的开发块以及所有尝试都位于 `workstreams/world_foundation/NATIVE_SOLID_SOLUTION_ACCEPTANCE.md` 中。JSON.

## 选择的原生 TDB/CEF 材料 {#selected-native-tdbcef-material}

`TdbDatabase` 接受明确的 TDB 文本，并具有明确的 `InputSource` 格式。 `CefPhaseModel` 评估所选相的亚晶格、空位、带电成分、有序性、磁性和热力学贡献，而无需依赖上游的运行时依赖。 `CefMaterialSpec` 添加了人工设定的温度区间、固定压力、热容和机械原子-摩尔体积。 `calphad_world_spec(model, element_amounts_mol=..., temperature_K=...)` 将该材料安装到现有的固态/热力学策略和有限功率设备中。

物料库存的标识是真实的。TDB元素和数量是mol非空位的原子。带电的成分和空位属于已承诺相的区域比例；它们不会产生新的物料库存。该模型在 `calphad_material` 版本化的物理声明中存储一次。其设备记录仅引用这种关联。`calphad_phase_meter` 读取已承诺的温度、元素数量和区域比例。它不会合成衍射谱，也不会假定真实的元素是旧的。A/B计数器。

`heat` / `wait` 实现了相同的有限物料总焓。内部结构和相对较低的热容都来源于共同的约束根；温度来源于共同的热能根。在这个仅涉及热量的过程中，数值未知数中不再存在恒定元素，并且每个贡献都必须具有零的物质速率。在材料替换过程中，保持恒定的热量；快照延续、历史回放、故障处理和热控制都保持了其原有的协议。

该消费者选择了一个局部内部的固相或液相，但没有选择一个全局稳定的多相体系。 线性化合物可能具有有效的热响应，但其个体元素的潜在值不确定；没有报告任何作为确定化学势的任意乘数。 负的有效热容、未满足的约束条件以及未支持的导数边界导致系统无法收敛。 气体 EOS 和一般的相选择需要各自声明的模型。 机械体积是作者定义的几何形状，而不是从 TDB 推导出的压力导数。

`ChemSageDatabase.from_file(path, source=...)` 能够接受与同一 Gibbs/材料界面的原始声明性 ChemSage 和 DAT 输入。 这种不可变输入保留了原始来源、相的发生情况、抑制标志、原始的化学计量、所有区间和模型元数据。 重复的标签使用 `NAME#OCCURRENCE1` 等； `phase_records` 暴露了原始标签和记录。 G 和 Cp 的描述可以编译为通用的热量评估器，包括参考 H/S、潜热、扩展的功率和磁性项。 QKTO 使用声明的化学组和方向的 Kohler/Toop 指数。 辅助电子平衡限制了 CEF 的内部结构，而无需添加原子。

源抑制需要明确的 `activate_suppressed=True`。非零的过压系数需要明确的 `excess_pressure_unit="Pa"` 或 `"bar"`。对于 SUBG/SUBQ 四元相，选择 `MQMQAPhaseModel`。它消耗源协调数及其混合/互逆的递归，参考对，熵模型，以及与化学组和三元项的定向 G/Q/R 相互作用。四元相的组成是内部的；实际的原子数量仍然存在于现有的物料库存中。它与吉布斯响应、约束的组成和 `CefMaterialSpec` 与 CEF 共享。原始对原子和四元电荷必须保持一致；该模型拒绝矛盾的输入，并保持其原始声明。没有明确能量效应的遗留注释仍然可见于诊断中。非零的 MQMQA 源辅助平衡需要单独声明的消费者。CEF 消费者拒绝 MQMQA 输入。不支持的真实气体、水溶液、体积和组的覆盖物理也明确被拒绝。JSON 存储原始输入和声明的选项，然后通过已知的原生编解码器重新编译。

完整的七个文件 ChemSage 资格认证，涵盖 114 个原始阶段声明：111 个有效阶段通过 666 个响应点；三个不一致的源阶段明确被拒绝，从而留下 18 个原始数值点未执行。一个独立的 96 个输入、576 点的合成块，涵盖了两种离子晶格上的额外离子化学组。这些是软件开发资格，而不是对所有导入材料的测量准确性的声明。固定覆盖、独立参考和保留的失败情况，已记录在 `workstreams/world_foundation/NATIVE_CONV41D3_EXPERIMENT.md` 中。

## 双位点表面相互作用 {#two-site-surface-interactions}

将 `MultisiteSurfaceSpec` 传递给已有的 `surface_world_spec(surface_spec=...)`。这两个站点类别都属于同一个载体，具有不同的有限容量，分别对应 CO/O 和 CO 的物料库存，并且两个类别之间可以进行交换。现有的表面处理工厂、热控系统、气体仪器以及物料输送操作均参与实验。不存在第二种催化剂服务或时间循环。

新的机制利用可逆吸附/氧化，以及每个正向/反向对的共同过渡态。标准吉布斯势和热能共享已声明的 U/Cv 表和参考熵。一个对称的 `interaction_matrix_J_mol` 贡献 G\_ex=U\_ex=1/2 n.epsilon.n/N\_ref。相同的能量及其梯度进入热平衡；这并不是额外的外部反应-热源。状态-槽的声明参考已存在的吸附物/位点数量，因此载体分割、合并和材料转移能够保留大量的能量。被拒绝的是一个冲突的作者编写的热能表。

历史上的五步机制仍然是默认设置，并且保持其精确速率。它包含不可逆的氧化反应。选择该历史规格会恢复原始模型；从明确的可逆模型中移除 B，并不会默默地将 A 反应转化为旧的不可逆动力学。可选的旧批量反应仍然可以通过通用的求解器进行贡献；可逆对验证关注的是已声明的新表面通道。

运行 `uv run --no-sync python -m chemworld.examples.native_multisite_surface`。该示例使用了有限的燃气来源和现有的现场检测延迟。它包括 120 秒的请求加热/等待时间和 10 秒的采集时间，这被计算一次。固定 57 个单元的开发比较使用CatMAP“它实际与当地相互作用以及速率常数核函数的计算，这些核函数是独立求解的稳态方程的结果；它不进行描述符拟合。”CatMAP“他的映射器”。所有尝试都保持在 `workstreams/world_foundation/NATIVE_MULTISITE_RESULTS.json`。这是一个有限的合成均一场模型，不包含空间上的 kMC 或真实催化剂预测。

## 跨领域示例 {#cross-domain-examples}

已安装的 [原生领域的流程](native_domains.md) 包括共享设备的使用、有限的流程库存、延迟的随机读数，以及回放功能。 [能力表](native_capabilities.md) 它是由已安装的模块声明生成的；受支持的组合域仍然明确地被界定。

## 参数和测量输入 {#parameter-and-measurement-inputs}

运行 `uv run --no-sync python -m chemworld.examples.scientific_inputs` 以进行一个完整的导入示例：一个现有的水性质包，来自两个实验组的六个合成冷却曲线记录，以及通过原生模型声明应用的两个热损失参数。该示例会发出...JSON包括来源、不确定性、训练/验证数量以及公共压力读数。它不进行拟合。

`chemworld.data.scientific_inputs.ScientificTable.from_json(text)` 读取参数表或测量表。列声明一个数量、单位、角色、有限的有效范围和参考状态。每个单元包含 `value`、`unit`、标准 `uncertainty`、`uncertainty_note` 和 `quality`。一个数据源具有明确的引用、版本、许可证声明和 `synthetic`、`measured` 或 `literature` 标签。未知的不确定性保持 `None`；它必须有解释说明。摄氏温度值获得温度偏移，而不确定性仅使用刻度。

每一行提供 `row_id`、`source_id`、`group_id`、`sample_id`、`split` 以及所有已声明的单元格。 `sample_id` 标识一个采集，包括所有其通道；需要重复采集时，需要使用不同的 ID。 分割的类型为 `train`、`validation`、`test` 或 `unused`。 实验组中的所有行必须共享一个分割。 参数行使用 `parameters`。 缺失或无效的观测值将保留在表中，并带有质量标志；条件和参数必须有效。 导入器会拒绝重复的标识、未知的列、不兼容的单位以及超出其声明范围的有效值。

`table.to_csv()` 撰写一篇内容广泛的CSV使用行标识和每数量五个单元格字段，例如 `pressure.value` 和 `pressure.uncertainty`。 `ScientificTable.from_csv(text, metadata)` 使用相同的合同进行读取； `metadata` 是 `table.to_dict()`，但删除了 `rows` 条目。 没有进行任何分割或随机化。 `table.summary()` 报告完整计数，而 `table.rows_for("validation")` 选择已声明的“保留”状态。

`chemworld.physchem.property_inputs.PropertyInputCard` 导入现有的组件/相关格式，包括每个相关源和不确定性记录。其 `evaluate_state` 方法使用现有的属性评估器。对于能量和熵的需求，需要明确导入的卡路里参考值。经过整理的示例保留了未知的上游表重分配项和未量化的不确定性；成功的解析并不意味着获得许可或校准。

`chemworld.world.parameter_inputs.bind_parameter_row(physics, table, row_id, targets)` 将一个参数行应用于新的模型配置。该应用提供 `ModelParameterTarget` 该函数独立于输入，定义已安装的模型/版本、绑定、数值字段路径和数量合同。该函数返回新的 `physics` 以及进口 `receipt`；通过常规的构造函数将该配置传递给它。模型编解码器会验证相关的更改。测量表不能作为参数绑定，并且导入操作不会将观察结果或物料库存写入正在运行的世界。

### 有限的原生校准 {#finite-native-calibration}

`chemworld.data.calibration.NativeCalibration` 通过在全新的原生会话中执行固定历史，来匹配一个有限参数的白名单。提供一个 `ScientificTable`、一个 `CalibrationParameter` 声明和一个 `NativeCalibrationExperiment`，用于每个数据组。每个 `CalibrationReadout` 将一个观测列与实际的测量步骤和公共通道关联，并可选地包含一个采样时间列。参数路径可以处理现有的连续模型配置字段；网络结构、任务目标和实时物料库存无法进行匹配。

只有训练数据进入了带有不确定性加权的、有限的最小二乘方法。 成功的拟合执行验证组，并报告原始单位、加权误差、活动边界和局部雅可比矩阵的误差。 无效的输入观测值仍然在排除计数中；有效的观测值需要声明的正的标准不确定性。 失败的前向历史保留实际读数、操作结果、时间、成本和采样信息。 它们停止拟合，而不是产生人为的残差惩罚。 前向边界计数实际的组执行情况，包括有限差分评估。

运行预定义的恒定热模拟，以及与 SPM 和其他混杂的热力学示例一起进行：

```
uv run --no-sync python -m chemworld.examples.native_calibration --output $env:TEMP/chemworld-calibration.json
```

该热力学示例使用了明确的理想合成压力计和固定数据扰动。 SPM 示例从公开的终端电压中提取了一个扩散系数。 联合的热量损失/热容拟合表明，即使能够获得良好的曲线拟合，也可能无法确定所有参数。 这些示例建立了校准流程； 经验材料验证需要独立获取的数据以及固定的验证设计。

### DIPPR：相关性分析和能量需求 {#dippr-correlations-and-caloric-requests}

`PropertyCorrelation` 接受 `dippr100`、`101`、`102`、`104`、`105`、`106`、`107`、`114`、`115`、`116`、`127`（每个都带有 `dippr` 前缀），以及 `dippr100_reciprocal`、`dippr105_reciprocal` 和 `dippr106_reciprocal`。 声明支持的 `property_id`，输出单元，绝对 K 输入，系数以及材料有效性区间。 `property_equation_contracts()` 暴露所需的/可选参数、属性维度、方程来源和领域。 材料参数仍然需要自己的来源和不确定性元数据。

现有的 `evaluate_correlation` 和属性包使用这些声明。 `correlation_temperature_derivative` 返回解析的第一个导数； `integrate_correlation` 返回一个确定的积分，或者，与 `over_temperature=True` 结合使用，返回 T 除以积分的结果。 它的数值误差估计和积分评估计数仅描述自适应积分，不包括域检查。 失败的积分和跨越极点的区间将被拒绝。

`caloric_correlation_change`、`evaluate_caloric_correlation`以及现有的相-路径焓报告都使用相同的方程。输出单位转换会调整结果，同时保留特征温度和指数。气体 Cv-到-熵的转换使用已声明的理想气体参考值。其他相仍然需要在 H 和 U 之间进行交换之前，明确定义其状态。

固定开发模块将 142 个案例与化学品 1.5.2 进行比较；生产评估不涉及化学品。这涵盖了配方行为，而不是一个新的材料数据库。多项式库存管理保持不变，等待后续的流程集成任务；完整的化学品扩展仍在进行中。

### 只读电池阻抗（开发） {#read-only-battery-impedance-development}

现有的SPM、SPMe和DFN通过`chemworld.physchem.battery_impedance.battery_impedance`，实现了对局部静止点的分析。它利用了这些设备的实际物料库存速率和电气约束，而不是独立设计的电路。

```
from chemworld.physchem.battery_models import BatterySPMSpec
from chemworld.physchem.battery_impedance import battery_impedance

cell = BatterySPMSpec()
state = cell.initial_state()
spectrum = battery_impedance(cell, [1e-4, 1e-3, 1e-2], state=state)
report = spectrum.to_dict()  # per-frequency real/imaginary ohm and failures
```

这不会改变时间，消耗实验资源，引入新的仪器，或修改快照。正向电流意味着放电；被动阻抗为 `-delta_V/delta_I`，与 `exp(+i*2*pi*f*t)` 矢量相关。频率必须严格为正。不稳定的浓度曲线或停止边界将被拒绝，而不是被默默地放松。热和副反应的组合尚未得到支持。目前这种近静态 BV 模型没有双层电容，因此它们的谱不能被解释为包括其 RC 弧。

`residual_frequency_response` 在 `physchem.residual_frequency` 中的共享功能，也支持一般的已声明的残差、非对角线质量矩阵和代数传递。它利用现有的导数模式和稀疏线性求解器，保留了奇异频率失效，而不是采用最小二乘解。物理输入/输出单元和导数来源仍然在报告中。这是一种局部响应，而不是非线性稳定性证书。

首个独立开发实验块将六个 SPM 案例与连续球形扩散对照，涵盖四种径向分辨率和平坦 OCV 极限。后续 SPMe/DFN 实验块依据已发布的 PyBaMM 25.4.2 方程，以 CasADi 独立求导、SciPy 求解，检查端子阻抗及三个电解质区域各自的平均盐浓度复数响应。十个案例各测 13 个频率，均通过固定对照及原生／参考解的网格细化界限。原先的均匀径向参考网格对五个 SPMe 案例分辨率不足（总体 5/10）；该结果单独保留。仅将参考径向网格改为边界细化，即得到 10/10，未改变原生物理模型或误差界限。这是有限、合成且物性恒定的适用域，不代表任意材料、非稳态工作点或真实实验已获验证。NATIVE-CONV-45 的其他模型族仍待完成。固定输入、界限与全部尝试可通过工作流中的 `NATIVE_CONV45_IMPEDANCE_EXPERIMENT.md` 查阅。

### 采用非线性电极 OCV {#sourced-nonlinear-electrode-ocv}

`physchem.battery_materials.ElectrodeOCV` 代表一个有界多项式，以及指数和双曲正切函数。它存储了材料、来源、模型兼容性以及化学计量学领域的参数。使用 `from_parameter_row` 来明确指定从现有 `ScientificTable` 中的列；振幅使用 V，而比例/位移则使用 `dimensionless`。原始的参数值、单位和不确定性说明仍然存在于 JSON 中。

```
from chemworld.physchem.battery_materials import ElectrodeOCV
from chemworld.physchem.battery_models import BatterySPMSpec

# table is a validated ScientificTable of kind "parameters".
ocv = ElectrodeOCV.from_parameter_row(
    table, "electrode-row", material_id="my-electrode",
    polynomial=("constant_V", "linear_V"),
    tanh_terms=(("transition_V", "transition_scale", "transition_center"),),
)
spec = BatterySPMSpec(negative_ocv=ocv)
```

一个明确的曲线取代了该电极的线性OCV领域。实际SPM，SPMe 和DFN电压定律会消耗它，并且DFN它利用耦合的雅可比矩阵中的解析斜率。化学势和积分电极能量都来自相同的表达式。公共曲线评估会拒绝超出其声明范围的点；求解器可能会显式地评估其解析延展，而电池的物理停止表面仍然决定了可接受的状态。

该固定八单元的开发块，对比了两个 Chen2020 OCV 关系，三个连续的放电/休息/充电轨迹，以及三个局部光谱，与已发布的 PyBaMM 25.4.2 函数和模型：8/8 通过。在该块中，几何和输送仍然是合成的。这并不是一个完整的 LGM50 参数化。目前，使用热或侧反应模型进行非线性 OCV 被拒绝；CONV-45 的完整材料、输送、热以及其他模型范围仍然开放。所有输入和结果都在 `workstreams/world_foundation/NATIVE_CONV45_OCV_EXPERIMENT.md` 和 `NATIVE_CONV45_OCV_RESULTS.json` 中。

### 平衡电极 SOH {#equilibrium-electrode-soh-development}

`physchem.battery_soh` 通过使用相同的 OCV 功能，解决了电压窗口几何问题。`ElectrodeSOHSpec` 接受已知的电极容量，并可以选择 `known="cyclable_lithium"`（活性电极 Li 充电）或 `known="cell_capacity"`。`HalfCellSOHSpec` 明确地表示，相对于 Li/Li+ 的正极工作电极，在 0 V 处进行分析，这是一种平衡分析，而不是原生半电池的动态过程。容量为 C；`ScientificTable` 的参数绑定可以明确地将 Ah 或 mAh 转换为原始单位，同时保留原始的误差和来源。没有添加任何估计器：原生公共观测校准仍然是现有的数据校准工作流程，并且本地隐式敏感性并不能证明参数的可识别性。

```
from chemworld.physchem import soh_from_battery, battery_spec_at_soc
from chemworld.physchem.battery_models import BatterySPMSpec

cell = BatterySPMSpec(minimum_voltage_V=3.38, maximum_voltage_V=3.92)
state = cell.initial_state()
soh = soh_from_battery(cell, state)  # active Li only, not electrolyte salt or locked SEI
if soh.status == "unique":
    solution = soh.branch()
    energy_J = soh.theoretical_energy_j()  # reversible OCV integral, not loaded work
    fresh_cell = battery_spec_at_soc(cell, soh, .5)  # a NEW initial configuration
```

这不会改变时间、消耗资源、放松实时浓度曲线，或重置现有设备。 SPM、SPMe 和 DFN 读出其实际电极。 `at_soc()` 和 `soc_at_voltage()` 使用所选的平衡分支。 获得多个结果需要明确的分支索引；不完整的隔离会保留未解决的区域，并且不能被视为唯一。 在已声明的电极面上，已知容量的处理不会扩大范围。 无法进行物料库存、没有电压窗口解决方案以及数值/等级不确定性具有不同的状态。 共享的有限区间/Krawczyk搜索使用现有的点-根求解器；向外的浮点边界不是正式的精确实数证明。 截断根必须限定一个完整的物理路径，而不是仅仅解决两个端点方程。

离线示例：`uv run --no-sync python examples/native_battery_soh.py`。 预设的 12 个单元开发块，比较已发布的 PyBaMM 25.4.2 和 SOH 方程，以及手动代数和独立多项式根，包括多种/不可行/无法识别的情况。 输入和所有尝试都在 `workstreams/world_foundation/NATIVE_CONV45_SOH_EXPERIMENT.md` 和对应的 `NATIVE_CONV45_SOH_RESULTS*.json` 中。 针对 MSMR 和 SOH 的老化推断、湿电池精度以及剩余的任务-45 模型家族均不在该块中。

### 多粒子电极（正在开发中，科学资质待定） {#many-particle-electrodes-development-scientific-qualification-pending}

`BatteryMPMSpec` 是一种明确的理想-均匀-电解质MPM. 每一根电极都保持其总的活性体积和材料系数，在 `ParticleSpec` 中。 可选的 `ParticleSizeDistribution` 声明将该体积分配到正半径上。 每一个尺寸都有其自身的实际径向库存、扩散和界面电流；这些电流在共享的电极电势下同时求解，从未预先从体积比例中分配。 在没有尺寸分布的情况下，声明的粒径半径给出均一分布的SPM限制。

```
from chemworld.physchem import BatteryMPMSpec, ParticleSizeDistribution, mpm_fields
from chemworld.world.battery_world import battery_world_spec

# source is the common InputSource for this distribution.
sizes = ParticleSizeDistribution((5e-6, 15e-6), (.5, .5), source)
cell = BatteryMPMSpec(negative_distribution=sizes, positive_distribution=sizes)
fields = mpm_fields(cell, cell.initial_state(), current_A=.2)
world = battery_world_spec(cell)
```

直接权重是归一化的体积分数。 `from_weights(..., basis=...)` 明确地将正相对数值、面积或体积权重进行转换和归一化。 `from_parameter_table` 保留了原始的常见数据源行、单位和不确定性； `um` 是一种显式长度转换，而不是推断的单位。 `lognormal_size_distribution` 在指定输入基中，明确地离散化了给定的截断密度，同时保留了中位数、对数标准差、上下限和四元数计数。 它不会恢复未观察到的分布尾部。

所有尺寸均使用现有的电池设备插座、控制源、共享时钟、原生测量、快照和回放功能。表面读数是区域平均值；总体化学计量是体积加权。停止功能检查所有尺寸，不仅仅是平均值。在零终端电流下，可以重新分配 Li 元素在不同尺寸之间的比例。电极化学能、耗散。SOH总和和只读的本地阻抗使用相同的 bin。一个源自非线性OCV必须明确声明与 `MPM` 的兼容性；已存储的旧版本兼容性列表不会被扩展。

离线工作流程：`uv run --no-sync python -m chemworld.examples.battery_mpm --output /tmp/mpm.jsonl`。 功能检查包括：真 CC/CV/源限制、解析导数、守恒、能量平衡、以及两种测量噪声模式和跨过程连续性。 预先准备好的九个单元的已发布模型比较，已在 `workstreams/world_foundation/NATIVE_CONV45_MPM_EXPERIMENT.md` 中准备，但尚未在 **中运行** 。 不要从这些功能检查中推断出合格的网格分辨率、测量单元的精度、复合电极或热/老化能力。

### 历史温度校准 {#historical-temperature-calibration}

`temperature_convert(value_K, current, desired)` 支持 ITS-90 以及完整的 ITS-68、ITS-48、ITS-27 和 ITS-76 范围。 明确接受历史上的 IPTS/EPT 别名。 每个历史尺度使用一个立方校准，并共享有限的根求解器进行反运算； 域使用实际的映射端点。 `temperature_scale_conversion` 还会返回相同的映射的导数。 开尔文/摄氏温度单位系统和世界温度语义保持不变。

```
from chemworld.foundation.units import Quantity
from chemworld.physchem import temperature_quantity, water_refractive_index_iapws

calibrated = temperature_quantity(Quantity(26.85, "degC"), "IPTS-68", "ITS-90")
index = water_refractive_index_iapws(calibrated.value, 997)
```

`temperature_input` 在现有系统中保留原始输入和可追溯的来源。InputValue请注意：`temperature_observation` 保持原始的标量通道，并在原始观测中添加了一个派生通道，该通道包含样本/时间戳和原生数据。JSON单程往返测量。测量不确定性采用一阶导数传播方法；未知的校准不确定性保持未确定。此标尺适配器明确拒绝了包含无效数值读数和需要按点进行不确定性的曲线。

`its90_68_difference` 是一种独立的近似多项式方法。 它的默认设置 `published_overlap` 填补了旧文档中低温部分的空白，并使用了已发表的中间多项式。 明确的 `convention="chemicals_1_5_2"` 复制了旧 API 在该空白中的零修正以及其连接选择。 这样的零是兼容性行为，而不是表明尺度一致的证据。 两种方法都没有主张热力学-温度的准确性，也没有默默地修改热量参考。 完整的节点/区间、跨函数和源-库存开发证据已记录在基础验收文件中； 早期源失效以及与 CAS 的 285 个未解决的主-组件映射仍然可见。

### 在原生世界中，正式的 SBML 型号 {#formal-sbml-models-on-the-native-world}

给定一个带有来源信息的 `SBMLModel`，声明其数值特征以及该仪器可能观测到的数量：

```
from chemworld.physchem.sbml_system import SBMLObservable
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.sbml_world import sbml_world

world = sbml_world(model, maximum_step_s=0.02,
                   observables=(SBMLObservable("amount", "A"),))
runtime = ChemWorldRuntime(composition=world, observation_noise_mode="keyed")
runtime.reset(seed=4)
runtime.execute({"operation": "wait", "duration_s": 0.25})
runtime.execute({"operation": "measure", "instrument": "formal_quantity_meter"})
reading = runtime.observe()  # declared mol channel with detector uncertainty
saved = runtime.snapshot().to_dict()  # private continuation; keep outside agent payloads
restored = ChemWorldRuntime.from_snapshot(saved)
```

受限的 Level 3 版本 2 配置文件支持明确的 3D 区域、源单元、动力学定律、初始/分配/速率规则以及带有延迟、取消、捕获分配和执行优先级的横向不等式事件。不支持的语义将被拒绝。事件需要一个由人工编写的有限的最大步数；任意快速的交叉和切点没有普遍的保证。

一个现有的设备实体拥有所有实时数据、分区体积、参数和待处理事件。现有的共享实验室时钟拥有时间。 `numerical_form="residual"` 选择通用的残差求解器。源数据变为 mol、L，以及在显式边界处的时间；模型公式保持其原始基础。浓度速率规则包括体积链项，并且事件浓度分配使用执行前后的体积。

工作空间没有物理化学储备或液体热容。 严格的坐标系统并不能建立元素、电荷、能量或载体-质量的闭合关系，并且无法进行热/材料采样操作。 测量是瞬时的，不消耗任何样品；其允许的量/浓度/体积通道可以传输 mol、mol/L 或 L，并声明存在高斯不确定性。 公共状态省略了随时间演变的坐标和队列。 材料准备保持了正式实体；完整的实验重置会重新初始化它。 失败的事件链会回溯完整的进展。 完整的私有快照可以保存待处理事件和检测器的延续，跨越不同的过程。

固定原生的世界块通过了 36 个运行时合同以及 770 个独立的封闭形式比较。完全的 SDK 状态明确表示为错误：18 模型的 COPASI 块存在三个保留差异，而 RoadRunner 存在一个源模型初始化差异。官方的 SBML 数量/事件案例以及所有失败的尝试都保留在基础证据记录中。 SED-ML/COMBINE 协议和完整的入口资格验证仍然是独立的任务。
