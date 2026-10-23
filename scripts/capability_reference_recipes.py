"""Executable documentation recipes, using only the installed public API.

These fixed, synthetic examples check that the documented call and output agree.
They are software demonstrations, not material validation or qualification data.
Coverage: every implemented capability has a recipe; planned nodes have no API.
Measure successful execution, JSON output and complete catalogue coverage. Keep
all failures in the build report; publish the reference only after every recipe
passes. No model provider, external dataset or scientific reference is required.
"""

# ruff: noqa: E501, RUF001 -- bilingual documentation and complete copyable code strings

from pprint import pformat
from textwrap import dedent

RECIPES = {}
BINDINGS = {}


def add(key, nodes, imports, inputs, body, apis, zh, en, *, output_zh=None, output_en=None):
    code = (
        "import json\n"
        + dedent(imports).strip()
        + "\n\n"
        + "inputs = "
        + pformat(inputs, width=78, sort_dicts=False)
        + "\n\n"
        + dedent(body).strip()
        + "\n\n"
        + "print(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False))\n"
    )
    RECIPES[key] = {
        "code": code,
        "targets": apis,
        "inputs": inputs,
        "notes": {"zh": zh, "en": en},
        "output_notes": {
            "zh": output_zh or "下方为此代码实际打印的完整 JSON；只选择代码中列出的返回字段。",
            "en": output_en
            or "The complete JSON printed by this code is shown below. The code explicitly selects the returned fields to display.",
        },
    }
    for node in nodes.split():
        BINDINGS.setdefault(node, []).append(key)


def runtime(key, nodes, imports, factory, actions, apis, zh, en):
    add(
        key,
        nodes,
        imports + "\nfrom chemworld.runtime import ChemWorldRuntime",
        {"seed": 7, "actions": actions},
        f"""
session = ChemWorldRuntime(composition={factory})
session.reset(seed=inputs["seed"])
output = {{"action_schemas": {{action["operation"]: session.action_schema(action["operation"])
                            for action in inputs["actions"]}}, "steps": []}}
try:
    for action in inputs["actions"]:
        step = session.execute(action)
        output["steps"].append({{
            "action": action,
            "transaction_status": step.info["transaction_status"],
            "actual_duration_s": step.info["actual_duration_s"],
            "measurement_status": step.info["measurement_status"],
            "readings": {{c.channel_id: {{"value": c.value, "unit": c.unit}}
                         for c in step.observation.channels if c.observed}},
            "terminated": step.terminated,
        }})
        if step.info["transaction_status"] != "committed":
            raise RuntimeError(step.info.get("error_message", step.info))
finally:
    session.close()
""",
        [
            *apis,
            "chemworld.runtime:ChemWorldRuntime.execute",
            "chemworld.runtime:ChemWorldRuntime.action_schema",
        ],
        zh,
        en,
    )


add(
    "helmholtz-state", "advanced-thermo",
    """
from chemworld.physchem.helmholtz import HelmholtzFluid
from chemworld.physchem.helmholtz_terms import HelmholtzTerm
""",
    {"temperature_K": 400, "molar_density_mol_m3": 20},
    """
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
""",
    ["chemworld.physchem.helmholtz:HelmholtzFluid", "chemworld.physchem.helmholtz:HelmholtzFluid.state"],
    "自编理想气体系数演示：输入 K 与 mol/m³，输出有量纲物性。真实流体需提供来源系数；此处只检查均相局部稳定。",
    "Self-authored ideal-gas coefficients demonstrate inputs in K and mol/m³ and dimensional properties. Real fluids require sourced coefficients; this call checks homogeneous local stability.",
)

add(
    "activity-properties", "activity-thermo",
    """
from chemworld.physchem.equilibrium import ActivityModelSpec
from chemworld.physchem.activity_thermodynamics import activity_thermodynamic_report
""",
    {"temperature_K": 343.15, "composition": {"A": 0.252, "B": 0.748},
     "parameters": {"A:A|B": 1.2, "A:B|A": 0.8}},
    """
model = ActivityModelSpec("synthetic-binary", ("A", "B"), "margules", inputs["parameters"])
report = activity_thermodynamic_report(
    model, inputs["composition"], temperature_K=inputs["temperature_K"])
output = {key: report.to_dict()[key] for key in (
    "activity_coefficients", "gibbs_excess_J_mol", "enthalpy_excess_J_mol",
    "entropy_excess_J_mol_K", "partial_molar_gibbs_excess_J_mol",
)}
""",
    ["chemworld.physchem.equilibrium:ActivityModelSpec", "chemworld.physchem.activity_thermodynamics:activity_thermodynamic_report"],
    "合成二元 Margules 参数用于展示超额和偏摩尔性质；组分 A/B 不代表经过验证的真实材料。",
    "Synthetic binary Margules parameters demonstrate excess and partial-molar properties; A/B do not denote validated real materials.",
)

add(
    "property-derivatives", "calibration caloric",
    """
from chemworld.physchem.specs import PropertyCorrelation
from chemworld.physchem.dippr import dippr_temperature_response, dippr_parameter_jacobian
""",
    {"temperature_K": 300, "coefficients": {"A": 25., "B": 30., "C": 800., "D": 10., "E": 400.}},
    """
curve = PropertyCorrelation(
    "synthetic-capacity", "dippr107", inputs["coefficients"],
    {"temperature": "K"}, "J/(mol*K)", "ideal_gas_heat_capacity",
    {"temperature": (200, 500)}, "Synthetic API demonstration")
response = dippr_temperature_response(curve, temperature_K=inputs["temperature_K"])
jacobian = dippr_parameter_jacobian(curve, [inputs["temperature_K"]], parameters=["A", "C"])
output = {"temperature_response": response.to_dict(), "parameter_jacobian": jacobian.to_dict()}
""",
    ["chemworld.physchem.dippr:dippr_temperature_response", "chemworld.physchem.dippr:dippr_parameter_jacobian"],
    "显式热容关联式给出温度导数与所选参数的 Jacobian，保留单位与有效域；这是合成参数示例。",
    "An explicit heat-capacity correlation yields temperature derivatives and a selected-parameter Jacobian, retaining units and validity range. Parameters are synthetic.",
)

add(
    "flow-meter-inverse", "fluid-engineering transport",
    """
from chemworld.physchem.flow_meters import LiquidOrificeSpec, liquid_orifice_pressure, liquid_orifice_mass_flow
""",
    {"pipe_diameter_m": 0.07366, "orifice_diameter_m": 0.05, "mass_flow_kg_s": 7.,
     "density_kg_m3": 999.1, "viscosity_Pa_s": 0.0011},
    """
spec = LiquidOrificeSpec(inputs["pipe_diameter_m"], inputs["orifice_diameter_m"], taps="corner",
    installation_note="ISO concentric sharp edge; compliant straight runs and roughness assumed")
properties = {key: inputs[key] for key in ("density_kg_m3", "viscosity_Pa_s")}
forward = liquid_orifice_pressure(spec, mass_flow_kg_s=inputs["mass_flow_kg_s"], **properties)
inverse = liquid_orifice_mass_flow(spec, differential_pressure_Pa=forward.differential_pressure_Pa,
    **properties)
output = {"forward": forward.to_dict(), "recovered_mass_flow_kg_s": inverse.mass_flow_kg_s}
""",
    ["chemworld.physchem.flow_meters:LiquidOrificeSpec", "chemworld.physchem.flow_meters:liquid_orifice_pressure", "chemworld.physchem.flow_meters:liquid_orifice_mass_flow"],
    "先由质量流率计算压差，再由压差反算流率。安装条件、流体参数和关联式适用域须满足声明。",
    "Calculate differential pressure from mass flow, then invert it to recover flow. Installation conditions, fluid parameters and the correlation domain must satisfy the declaration.",
)


add(
    "identity",
    "identity identity-elements identity-formula",
    "from dataclasses import asdict\nfrom chemworld.physchem import ComponentSpec, get_element, get_isotope",
    {"formula": "H2O", "element": "O", "isotope": "[13C]"},
    """
component = ComponentSpec("water", inputs["formula"], cas_number="7732-18-5")
output = {"component": component.to_dict(),
          "element": get_element(inputs["element"]).to_dict(),
          "isotope": asdict(get_isotope(inputs["isotope"]))}
""",
    [
        "chemworld.physchem:ComponentSpec",
        "chemworld.physchem:get_element",
        "chemworld.physchem:get_isotope",
    ],
    "组分用分子式与 CAS 显式标识。元素、同位素数据随公开包提供；不需要下载数据库。",
    "Identify the component with a formula and CAS number. Element and isotope data ship with the public package.",
)

add(
    "source",
    "identity-sources",
    """
from chemworld.data.scientific_inputs import InputSource
from chemworld.physchem.data_conflicts import DataSourceProvenance
""",
    {"source_id": "demo", "citation": "Synthetic documentation inputs", "version": "1"},
    """
source = InputSource(DataSourceProvenance(inputs["source_id"], 0,
    inputs["citation"], inputs["version"], "CC0"), "synthetic")
output = source.to_dict()
""",
    [
        "chemworld.data.scientific_inputs:InputSource",
        "chemworld.physchem.data_conflicts:DataSourceProvenance",
    ],
    "声明示例数据来源；synthetic 表示自定义合成输入，不把它标成文献测量。",
    "Declare provenance for synthetic inputs without presenting them as measured literature data.",
)

add(
    "units",
    "conservation",
    "from chemworld.foundation.units import Quantity",
    {"value": 26.85, "unit": "degC", "target_unit": "K"},
    """
converted = Quantity(inputs["value"], inputs["unit"]).to(inputs["target_unit"])
output = {"value": converted.value, "unit": converted.unit}
""",
    ["chemworld.foundation.units:Quantity", "chemworld.foundation.units:Quantity.to"],
    "摄氏温度转换为绝对温度。单位转换不等于历史温标校准；物料守恒需由实际流程事务核对。",
    "Convert Celsius to absolute temperature. Unit conversion is distinct from temperature-scale calibration; material conservation is checked by process transactions.",
)

add(
    "temperature",
    "temperature",
    """
from chemworld.foundation.units import Quantity
from chemworld.physchem import temperature_quantity
""",
    {"value": 226.85, "unit": "degC", "source_scale": "ITS-68", "target_scale": "ITS-90"},
    """
result = temperature_quantity(Quantity(inputs["value"], inputs["unit"]),
                              inputs["source_scale"], inputs["target_scale"])
output = {"value": result.value, "unit": result.unit}
""",
    ["chemworld.physchem:temperature_quantity"],
    "读数先按声明单位处理，再在温标的有效范围内校准；超出范围会抛出 ValueError。",
    "Interpret the stated unit, then calibrate within the scale domain. Out-of-range readings raise ValueError.",
)

add(
    "virial",
    "density",
    "from chemworld.physchem import virial_derivatives",
    {
        "method": "Tsonopoulos",
        "temperature_K": 510,
        "parameters": {"Tc": 425.2, "Pc": 3800000, "omega": 0.193},
    },
    """
values = virial_derivatives(inputs["method"], inputs["temperature_K"], inputs["parameters"])
output = dict(zip(["B_m3_mol", "dB_dT_m3_mol_K", "d2B_dT2_m3_mol_K2", "d3B_dT3_m3_mol_K3"], values))
""",
    ["chemworld.physchem:virial_derivatives", "chemworld.physchem:virial_correlation"],
    "显式临界参数示例；T、Tc 使用 K，Pc 使用 Pa。输出 B 及前三阶温度导数，不自动验证材料适用性。",
    "Explicit illustrative critical parameters: T and Tc in K, Pc in Pa. Returns B and its first three temperature derivatives, without establishing material validity.",
)

add(
    "viscosity",
    "viscosity",
    "from chemworld.physchem import water_viscosity_iapws, wilke_viscosity",
    {
        "temperature_K": 298.15,
        "density_kg_m3": 998,
        "mole_fractions": [0.4, 0.6],
        "viscosities_Pa_s": [1e-5, 2e-5],
        "molecular_weights_g_mol": [28, 44],
    },
    """
output = {"water_viscosity_Pa_s": water_viscosity_iapws(inputs["temperature_K"], inputs["density_kg_m3"]),
          "mixture_viscosity_Pa_s": wilke_viscosity(inputs["mole_fractions"], inputs["viscosities_Pa_s"], inputs["molecular_weights_g_mol"])}
""",
    ["chemworld.physchem:water_viscosity_iapws", "chemworld.physchem:wilke_viscosity"],
    "水黏度与合成二元气体的 Wilke 混合示例；质量密度为 kg/m³，黏度为 Pa·s，分子量为 g/mol。",
    "Water viscosity and Wilke mixing for a synthetic binary gas: density in kg/m³, viscosity in Pa·s, molecular weight in g/mol.",
)

add(
    "conductivity",
    "conductivity",
    "from chemworld.physchem import conductivity_value, conductivity_derivatives",
    {"method": "Sheffy_Johnson", "temperature_K": 300, "parameters": {"MW": 47, "Tm": 280}},
    """
value = conductivity_value(inputs["method"], inputs["temperature_K"], **inputs["parameters"])
output = {"conductivity_W_m_K": value,
          "temperature_derivatives": conductivity_derivatives(inputs["method"], inputs["temperature_K"], inputs["parameters"])}
""",
    ["chemworld.physchem:conductivity_value", "chemworld.physchem:conductivity_derivatives"],
    "液相导热关联式的显式参数示例，MW 为 g/mol，Tm 与温度为 K；不可外推为完整传热设备模型。",
    "Explicit liquid conductivity correlation: MW in g/mol, Tm and temperature in K. This correlation is not a complete heat-transfer apparatus.",
)

add(
    "interfaces",
    "interfaces",
    "from chemworld.physchem.interfaces import laplace_pressure\nfrom chemworld.physchem.property_reports import PropertyEvaluation",
    {"surface_tension_N_m": 0.072, "radius_m": 0.001},
    """
tension = PropertyEvaluation("surface_tension", "demo", "explicit", inputs["surface_tension_N_m"], "N/m", {})
result = laplace_pressure(tension, curvature_1_per_m=1/inputs["radius_m"],
                         curvature_2_per_m=1/inputs["radius_m"])
output = result.to_dict()
""",
    ["chemworld.physchem.interfaces:laplace_pressure"],
    "球面 Laplace 压差示例。表面张力用 N/m、半径用 m，读取返回报告的单位与有效性字段。",
    "Spherical Laplace pressure: surface tension in N/m and radius in m. The report carries units and validity.",
)

add(
    "solubility",
    "solubility",
    "from chemworld.physchem.henry import henry_convert\nfrom chemworld.physchem.solubility import solid_solubility",
    {
        "henry_value": 101325,
        "source_scale": "SI",
        "target_scale": "KHpx",
        "T_K": 300,
        "Tm_K": 350,
        "fusion_enthalpy_J_mol": 10000,
    },
    """
output = {"converted_henry": henry_convert(inputs["henry_value"], inputs["source_scale"], inputs["target_scale"]),
          "solid_mole_fraction": solid_solubility(inputs["T_K"], inputs["Tm_K"], inputs["fusion_enthalpy_J_mol"])}
""",
    ["chemworld.physchem.henry:henry_convert", "chemworld.physchem.solubility:solid_solubility"],
    "示例分别演示 Henry 标度转换与理想活度近似的固体溶解度；其他标度可能需要额外溶剂参数。",
    "Separate examples for Henry-scale conversion and ideal-activity solid solubility. Other scale pairs may require solvent parameters.",
)

add(
    "optics",
    "optics",
    "from chemworld.physchem.electro_optics import water_refractive_index_iapws, brix_to_refractive_index",
    {"temperature_K": 300, "density_kg_m3": 997, "brix": 10},
    """
output = {"water_refractive_index": water_refractive_index_iapws(inputs["temperature_K"], inputs["density_kg_m3"]),
          "brix_refractive_index": brix_to_refractive_index(inputs["brix"])}
""",
    [
        "chemworld.physchem.electro_optics:water_refractive_index_iapws",
        "chemworld.physchem.electro_optics:brix_to_refractive_index",
    ],
    "水的折射率与 Brix 标定分别计算；Brix 换算不代表任意混合物的糖含量测定。",
    "Compute water refraction and the Brix calibration separately. Brix conversion is not a sugar assay for arbitrary mixtures.",
)

add(
    "combustion",
    "combustion combustion-products combustion-heat",
    "from chemworld.physchem.combustion import combustion_data\nfrom chemworld.foundation.units import Quantity",
    {"formula": "CH4", "formation_enthalpy_J_mol": -74520, "amount_mol": 2},
    """
fuel = combustion_data(inputs["formula"], method="Stoichiometry", Hf=inputs["formation_enthalpy_J_mol"])
heat = fuel.heat(Quantity(inputs["amount_mol"], "mol"), basis="HHV")
output = {"stoichiometry": dict(fuel.stoichiometry), "HHV_J_mol": fuel.HHV,
          "LHV_J_mol": fuel.LHV, "finite_heat": {"value": heat.value, "unit": heat.unit}}
""",
    ["chemworld.physchem.combustion:combustion_data"],
    "采用示例生成焓与完全燃烧产物。放热的焓变为负；不求解火焰温度、速率或平衡。",
    "Use an illustrative formation enthalpy and complete-combustion products. Exothermic enthalpy is negative; no flame temperature, kinetics or equilibrium is solved.",
)

add(
    "assessment",
    "assessment",
    "from chemworld.physchem.safety_environment import ppmv_to_mass_concentration",
    {"ppmv": 10, "molecular_weight_g_mol": 44, "temperature_K": 298.15, "pressure_Pa": 101325},
    """
output = {"mass_concentration_mg_m3": ppmv_to_mass_concentration(**inputs)}
""",
    ["chemworld.physchem.safety_environment:ppmv_to_mass_concentration"],
    "将气体体积分数换算为质量浓度；此处没有暴露限值，也不作安全判定。",
    "Convert gas volume fraction to mass concentration. This example supplies no exposure limit and makes no safety determination.",
)

add(
    "fuel",
    "fuel-quality",
    "from chemworld.physchem.combustion import octane_index",
    {"research_octane": 95, "motor_octane": 85, "quantity": "AKI"},
    """
output = {"AKI": octane_index(**inputs)}
""",
    ["chemworld.physchem.combustion:octane_index"],
    "显式 RON/MON 合成输入，计算抗爆指数；不推断未测量燃料的辛烷值。",
    "Compute the antiknock index from explicit illustrative RON/MON values, without predicting unmeasured octane numbers.",
)

runtime(
    "surface",
    "surface handling physical-readouts conservation",
    "from chemworld.world.surface_world import surface_world_spec",
    "surface_world_spec(include_bulk=True)",
    [
        {"operation": "add_component", "component": "CO-feed", "amount_mol": 6e-5},
        {"operation": "add_component", "component": "O2-feed", "amount_mol": 3e-5},
        {"operation": "wait", "duration_s": 2},
        {"operation": "measure", "instrument": "gas_composition"},
    ],
    ["chemworld.world.surface_world:surface_world_spec"],
    "使用公开包自带的合成 CO 氧化装置；进料量为 mol，等待为 s。读数仅来自已观测通道。",
    "Use the packaged synthetic CO-oxidation apparatus. Feed amounts are in mol, waits in seconds; only observed channels are printed.",
)

runtime(
    "aqueous",
    "aqueous",
    "from chemworld.world.aqueous_world import aqueous_world_spec",
    "aqueous_world_spec()",
    [{"operation": "wait", "duration_s": 2}, {"operation": "measure", "instrument": "aqueous_ph"}],
    ["chemworld.world.aqueous_world:aqueous_world_spec"],
    "有限碳酸盐水相预设；等待推进状态，再读取带单位和可见性标记的 pH 通道。",
    "A finite carbonate aqueous preset: advance its state, then read the observable pH channel.",
)

runtime(
    "mineral",
    "mineral",
    "from chemworld.world.mineral_world import mineral_world_spec",
    "mineral_world_spec()",
    [{"operation": "wait", "duration_s": 10}, {"operation": "measure", "instrument": "aqueous_ph"}],
    ["chemworld.world.mineral_world:mineral_world_spec"],
    "有限方解石与水相预设，固相几何及物料历史保留在同一装置。",
    "Finite calcite and aqueous phases retain solid geometry and inventory history in one apparatus.",
)

runtime(
    "electrode",
    "electrode electrical-readouts",
    "from chemworld.world.electrode_world import electrode_world_spec",
    "electrode_world_spec()",
    [
        {"operation": "configure_electrical_control", "potential_V": 0.1},
        {"operation": "wait", "duration_s": 1},
        {"operation": "measure", "instrument": "electrical_meter"},
    ],
    ["chemworld.world.electrode_world:electrode_world_spec"],
    "合成电极电位阶跃；电位为 V，时间为 s，输出电流等原生通道。",
    "A potential step on the synthetic electrode: potential in V, time in s, and native current/readout channels.",
)

runtime(
    "spm",
    "spm battery-spm-model",
    "from chemworld.world.battery_world import battery_world_spec",
    "battery_world_spec()",
    [
        {"operation": "configure_electrical_control", "current_A": 0.02},
        {"operation": "wait", "duration_s": 2},
        {"operation": "measure", "instrument": "battery_meter"},
    ],
    [
        "chemworld.world.battery_world:battery_world_spec",
        "chemworld.physchem.battery_models:BatterySPMSpec",
    ],
    "合成 SPM 电池的短时恒流读数；此示例不验证真实电芯或长期老化。",
    "A short constant-current SPM demonstration; it does not validate a measured cell or long-term aging.",
)

for key, nodes, model_import, model in (
    (
        "spme",
        "spme-dfn battery-spme-model",
        "from chemworld.physchem.battery_models import BatterySPMSpec\nfrom chemworld.physchem.battery_electrolyte import BatteryElectrolyteSpec",
        "BatterySPMSpec(electrolyte=BatteryElectrolyteSpec())",
    ),
    (
        "dfn",
        "spme-dfn battery-dfn-model",
        "from chemworld.physchem.battery_dfn import BatteryDFNSpec",
        "BatteryDFNSpec()",
    ),
):
    runtime(
        key,
        nodes,
        model_import + "\nfrom chemworld.world.battery_world import battery_world_spec",
        f"battery_world_spec({model})",
        [
            {"operation": "configure_electrical_control", "current_A": 0.02},
            {"operation": "wait", "duration_s": 1},
            {"operation": "measure", "instrument": "battery_meter"},
        ],
        ["chemworld.world.battery_world:battery_world_spec"],
        f"{key.upper()} 合成默认参数；显式等温短时示例，端电压不代表材料精度验证。",
        f"Synthetic default {key.upper()} parameters; an explicitly isothermal short-time example, not a validation of material accuracy.",
    )

runtime(
    "thevenin",
    "impedance",
    "from chemworld.examples.native_thevenin import default_spec\nfrom chemworld.world.thevenin_world import thevenin_world_spec",
    "thevenin_world_spec(default_spec())",
    [
        {"operation": "configure_electrical_control", "current_A": 0.02},
        {"operation": "wait", "duration_s": 1},
        {"operation": "measure", "instrument": "thevenin_meter"},
    ],
    ["chemworld.world.thevenin_world:thevenin_world_spec"],
    "本例演示扩展电池族中的 Thevenin 时域流程；并非频域阻抗扫描。参数来自自带合成表。",
    "This example demonstrates the Thevenin time-domain member of the extended battery family, not an impedance sweep. Parameters come from packaged synthetic tables.",
)

for key, nodes, factory, imports, actions in (
    (
        "membrane",
        "membrane",
        "membrane_world_spec()",
        "from chemworld.world.membrane_world import membrane_world_spec",
        [
            {"operation": "set_flow_rate", "flow_rate_mL_min": 20},
            {"operation": "run_flow", "duration_s": 1},
            {"operation": "measure", "instrument": "membrane_meter"},
        ],
    ),
    (
        "transport",
        "transport",
        "transport_world_spec()",
        "from chemworld.world.transport_world import transport_world_spec",
        [
            {"operation": "set_flow_rate", "flow_rate_mL_min": 10},
            {"operation": "run_flow", "duration_s": 2},
            {"operation": "measure", "instrument": "flow_meter"},
        ],
    ),
):
    runtime(
        key,
        nodes,
        imports,
        factory,
        actions,
        [imports.split(" import ")[0].replace("from ", "") + ":" + factory.split("(")[0]],
        "使用有限库存合成装置；流量用 mL/min，推进时间用 s。操作字段受该装置配置约束。",
        "Use a finite-inventory synthetic apparatus: flow in mL/min and duration in s. Operation fields depend on the bound apparatus.",
    )

# Existing maintained examples expose the complete fixed workflow as an API.
for key, nodes, module, kwargs in (
    ("flash", "flash", "native_flash_research", {}),
    ("extraction", "extraction", "native_liquid_research", {"amplitude": 3}),
    ("distillation", "distillation", "native_distillation_research", {}),
    ("crystal", "crystal", "native_crystal_research", {}),
    ("adsorption", "adsorption", "native_sorbent_research", {}),
    ("devices", "devices", "reuse_native_equipment", {}),
    ("caloric", "caloric", "state_property_research", {}),
    ("calibration", "calibration", "scientific_inputs", {}),
):
    add(
        key,
        nodes,
        f"from chemworld.examples.{module} import run_example",
        kwargs,
        "output = run_example(**inputs)",
        [f"chemworld.examples.{module}:run_example"],
        "运行公开包维护的完整合成示例，所有模型预设均随包提供。下方展开其全部返回值；示例用于学习接口，不代表真实材料验证。",
        "Run the maintained synthetic example shipped with the package. All presets are included; the full returned value is shown below. This demonstrates the API, not measured material validity.",
    )

for key, nodes, workflow in (
    ("shared", "scheduler ssa discrete-readouts", "shared-clock"),
    ("persistent", "persistence", "device-lifecycle"),
    ("solid", "calphad", "solid"),
):
    add(
        key,
        nodes,
        "from tempfile import TemporaryDirectory\nfrom chemworld.examples.native_domains import run_workflow",
        {"workflow": workflow, "noise_mode": "keyed"},
        """
with TemporaryDirectory() as directory:
    result = run_workflow(inputs["workflow"], directory, noise_mode=inputs["noise_mode"])
    output = {key: value for key, value in result.items() if key != "trajectory"}
""",
        ["chemworld.examples.native_domains:run_workflow"],
        "运行固定合成工作流并验证导出轨迹的回放。临时目录在示例结束后清理，输出省略本机路径。",
        "Run the fixed synthetic workflow and verify its exported replay. The temporary directory is removed at exit; host-specific paths are omitted from output.",
    )

for key, nodes, module, function in (
    ("thermal-cycle", "battery-thermal-cycle", "native_battery_cycles", "cycle_session"),
    ("sei", "battery-sei", "native_battery_degradation", "degradation_session"),
):
    add(
        key,
        nodes,
        f"from chemworld.examples.{module} import {function}",
        {"seed": 7},
        f"""
{"session" if key == "thermal-cycle" else "spec, session"} = {function}()
try:
    output = {{"measurement_schema": session.action_schema("measure"),
               "wait_schema": session.action_schema("wait")}}
finally:
    session.close()
""",
        [f"chemworld.examples.{module}:{function}"],
        "构建设备与模型并查看测量合同；这是配置示例，不是长期循环运行。",
        "Construct the apparatus/model and inspect its measurement contract. This is a configuration example, not a long cycling run.",
    )

add(
    "ode",
    "numerics numerics-ode",
    "from chemworld.physchem.solver_backend import solve_ode",
    {"initial_mol": 2, "rate_per_s": 0.7, "times_s": [0, 1, 2]},
    """
result = solve_ode(lambda t, y: -inputs["rate_per_s"] * y, [inputs["initial_mol"]],
                   time_span_s=(0, 2), evaluation_times_s=inputs["times_s"])
result.raise_for_failure("first-order decay")
output = {"success": result.success, "time_s": result.t.tolist(), "amount_mol": result.y[0].tolist()}
""",
    ["chemworld.physchem.solver_backend:solve_ode"],
    "求解 dn/dt = −kn 的合成一阶衰减。初值为 mol，k 为 s⁻¹；返回时间与状态数组。",
    "Solve synthetic first-order decay dn/dt = −kn. Initial amount is in mol and k in s⁻¹; returns time and state arrays.",
)

add(
    "dae",
    "numerics-dae",
    """
import numpy as np
from chemworld.physchem.residual_system import ResidualSystem, ResidualVariable, ResidualEquation, ResidualDerivatives
from chemworld.physchem.residual_solver import consistent_initial_conditions
""",
    {"initial_state": [2, 0.2], "rate_per_s": 0.7},
    """
system = ResidualSystem("quadratic", (
    ResidualVariable("x", "mol", lower=0),
    ResidualVariable("z", "mol^2", role="algebraic", lower=0)),
    (ResidualEquation("decay", "mol/s"), ResidualEquation("constraint", "mol^2")),
    lambda t, y, v, p: np.array([v[0] + p[0]*y[0], y[1] - y[0]**2]),
    ResidualDerivatives("finite_difference"),
    (ResidualVariable("k", "1/s", role="parameter", lower=0),), (inputs["rate_per_s"],))
result = consistent_initial_conditions(system, inputs["initial_state"], time_s=0)
if not result.success:
    raise RuntimeError(str(result))
output = {"success": result.success, "state": result.point.state.tolist(), "rate": result.point.rate.tolist()}
""",
    [
        "chemworld.physchem.residual_system:ResidualSystem",
        "chemworld.physchem.residual_solver:consistent_initial_conditions",
    ],
    "index-1 示例：x′ = −kx，代数约束 z = x²。修正不一致的 z 初值，并求一致导数；这里演示初始化，不是整段积分。",
    "An index-1 example: x′ = −kx with z = x². Correct the inconsistent algebraic initial value and compute consistent rates; this demonstrates initialization, not a full integration.",
)

add(
    "diffusion",
    "numerics-space",
    """
from chemworld.physchem.finite_volume import DiffusionMesh, ConservativeDiffusion
""",
    {"length_m": 0.01, "cells": 2, "diffusivity_m2_s": 1e-9, "amounts_mol": [0.01, 0]},
    """
mesh = DiffusionMesh.planar(inputs["length_m"], inputs["cells"], area_m2=1)
model = ConservativeDiffusion(mesh, inputs["diffusivity_m2_s"])
result = model.evaluate(inputs["amounts_mol"])
output = {"cell_volumes_m3": mesh.cell_volumes_m3.tolist(),
          "amount_rates_mol_s": result.amount_rates_mol_s.tolist(),
          "face_rates_mol_s": result.face_rates_mol_s.tolist()}
""",
    [
        "chemworld.physchem.finite_volume:DiffusionMesh.planar",
        "chemworld.physchem.finite_volume:ConservativeDiffusion",
        "chemworld.physchem.finite_volume:ConservativeDiffusion.evaluate",
    ],
    "两单元平面扩散，默认两端零通量。输入为每个单元的物质的量，输出是速率，调用本身不推进时间。",
    "Two planar cells with zero-flux boundaries. Inputs are cell inventories; outputs are rates. This call does not advance time.",
)

add(
    "air-fuel",
    "combustion-streams",
    "from chemworld.physchem.combustion_streams import air_fuel_ratio_solver",
    {"ratio": 10, "basis": "mole", "n_fuel": 1},
    """
output = dict(zip(["n_air", "n_fuel", "mole_ratio", "mass_ratio", "volume_ratio"], air_fuel_ratio_solver(**inputs)))
""",
    ["chemworld.physchem.combustion_streams:air_fuel_ratio_solver"],
    "两流摩尔基规格：给出空气/燃料比与燃料摩尔流量，反求空气流量。使用一致的摩尔流量单位。",
    "Two-stream molar specification: supply the air/fuel ratio and fuel molar flow to solve air flow. Use consistent molar-flow units.",
)

add(
    "impact",
    "environment-assessment",
    """
from chemworld.data.assessments import AssessmentValue
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.safety_environment import emission_impact
from chemworld.foundation.units import Quantity
""",
    {"emitted_mass_kg": 2, "synthetic_factor": 30, "horizon_years": 100},
    """
source = DataSourceProvenance("demo", 0, "Synthetic factor, not an IPCC value", "1", "CC0")
factor = AssessmentValue("GWP", "synthetic-100yr", "demo-gas", "demo-record",
    inputs["synthetic_factor"], "dimensionless", "valid", source,
    inputs["synthetic_factor"], {"horizon_years": inputs["horizon_years"]})
result = emission_impact({"demo-gas": Quantity(inputs["emitted_mass_kg"], "kg")}, {"demo-gas": factor})
output = result.to_dict()
""",
    [
        "chemworld.data.assessments:AssessmentValue",
        "chemworld.physchem.safety_environment:emission_impact",
    ],
    "使用明确标记的合成影响因子演示排放质量换算。库存不自动视为排放；不同方法或时间尺度不能混用。",
    "Convert an explicitly emitted mass using a labelled synthetic impact factor. Inventory is not automatically emitted mass; methods and horizons must agree.",
)

runtime(
    "bulk-reaction",
    "thermal-reaction",
    "from chemworld.world.surface_world import surface_world_spec",
    "surface_world_spec(include_surface=False, include_bulk=True)",
    [
        {"operation": "add_component", "component": "CO-feed", "amount_mol": 6e-5},
        {"operation": "add_component", "component": "O2-feed", "amount_mol": 3e-5},
        {"operation": "heat", "target_temperature_K": 400, "duration_s": 2},
        {"operation": "measure", "instrument": "gas_composition"},
    ],
    ["chemworld.world.surface_world:surface_world_spec"],
    "启用体相反应、关闭表面反应的合成预设。加热设定值不是瞬时物料温度，能量与反应共同推进。",
    "Synthetic preset with bulk reactions enabled and surface reactions disabled. The heating target is not an instantaneous material temperature; energy and reaction advance together.",
)

add(
    "continuation",
    "continuation",
    """
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.surface_world import surface_world_spec
""",
    {"seed": 7, "action": {"operation": "wait", "duration_s": 2}},
    """
session = ChemWorldRuntime(composition=surface_world_spec())
session.reset(seed=inputs["seed"])
saved = json.loads(json.dumps(session.snapshot().to_dict(), allow_nan=False))
restored = ChemWorldRuntime.from_snapshot(saved)
try:
    first = session.execute(inputs["action"])
    second = restored.execute(inputs["action"])
    output = {"original_status": first.info["transaction_status"],
              "restored_status": second.info["transaction_status"],
              "same_continuation": session.snapshot().to_dict() == restored.snapshot().to_dict()}
    assert output["same_continuation"]
finally:
    session.close()
    restored.close()
""",
    [
        "chemworld.runtime:ChemWorldRuntime.snapshot",
        "chemworld.runtime:ChemWorldRuntime.from_snapshot",
    ],
    "快照包含私有状态与随机数状态，用于宿主续跑。它不是给 Agent 的公开观测；不要把快照当作测量证据。",
    "A host continuation includes private state and RNG state. It is not an agent-visible observation or measurement evidence.",
)

add(
    "public-readings",
    "assay resource-accounting observatory interface-runtime entrypoints",
    """
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.surface_world import surface_world_spec
""",
    {"seed": 7, "instrument": "gas_composition"},
    """
session = ChemWorldRuntime(composition=surface_world_spec(terminal_assay=True))
session.reset(seed=inputs["seed"])
try:
    result = session.execute({"operation": "measure", "instrument": inputs["instrument"]})
    output = {"channels": [c.to_dict() for c in session.observe().channels],
              "accounting": {k: result.info[k] for k in (
                  "actual_duration_s", "measurement_cost", "sample_consumed", "cost_components")},
              "lab_status": session.lab_status()}
finally:
    session.close()
""",
    [
        "chemworld.runtime:ChemWorldRuntime.observe",
        "chemworld.runtime:ChemWorldRuntime.execute",
        "chemworld.runtime:ChemWorldRuntime.lab_status",
    ],
    "测量会计时、计费并可能消耗样品。observe() 与 lab_status() 只读已有结果，可供本地界面展示；此处未启动 Web 服务。",
    "Acquisition charges time/cost and may consume sample. observe() and lab_status() read existing results for local UI clients; this example does not start a web server.",
)

add(
    "gym",
    "entrypoints interface-gym",
    """
from chemworld.envs import ChemWorldEnv
from chemworld.world.surface_world import surface_world_spec
""",
    {"seed": 7, "action": {"operation": "wait", "duration_s": 2}},
    """
env = ChemWorldEnv(composition=surface_world_spec())
try:
    observation, info = env.reset(seed=inputs["seed"])
    observation, reward, terminated, truncated, info = env.step(inputs["action"])
    output = {"observation_keys": list(observation), "reward": float(reward),
              "terminated": terminated, "truncated": truncated,
              "transaction_status": info["transaction_status"]}
finally:
    env.close()
""",
    [
        "chemworld.envs:ChemWorldEnv",
        "chemworld.envs:ChemWorldEnv.reset",
        "chemworld.envs:ChemWorldEnv.step",
    ],
    "Gymnasium 的 reset 返回二元组，step 返回五元组；与原生会话使用同一装置规格。",
    "Gymnasium reset returns two values and step returns five, using the same apparatus specification as the native runtime.",
)

add(
    "agent-actions",
    "interface-agent",
    """
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.surface_world import surface_world_spec
""",
    {"action": {"operation": "wait", "duration_s": 2}},
    """
session = ChemWorldRuntime(composition=surface_world_spec())
session.reset(seed=7)
try:
    output = {"validation": session.validate_action(inputs["action"]),
              "action_schema": session.action_schema(inputs["action"]["operation"]),
              "available_operations": [a["operation"] for a in session.available_actions()]}
finally:
    session.close()
""",
    [
        "chemworld.runtime:ChemWorldRuntime.validate_action",
        "chemworld.runtime:ChemWorldRuntime.available_actions",
        "chemworld.runtime:ChemWorldRuntime.action_schema",
    ],
    "Agent 可先读取当前动作合同，再验证候选动作。验证不执行动作，已安装、任务允许和当前可执行是不同条件。",
    "An agent can inspect the current action contract and validate a candidate without executing it. Installed, task-allowed and currently executable are distinct conditions.",
)

add(
    "notes",
    "research-loop",
    """
from pathlib import Path
from tempfile import TemporaryDirectory
from chemworld.agents.notebook_versions import NotebookVersions
""",
    {
        "initial_text": "# Research notes\n",
        "prediction": "Prediction: waiting will change the measured composition.\n",
    },
    """
with TemporaryDirectory() as directory:
    path = Path(directory) / "notes.md"
    path.write_text(inputs["initial_text"], encoding="utf-8")
    notebook = NotebookVersions(path, inputs["initial_text"])
    revision = notebook.commit(inputs["initial_text"] + inputs["prediction"],
                               public_cursor={}, message="Record before executing")
    readback = notebook.read(revision["revision"])
    output = {"revision": readback["revision"], "text": readback["text"],
              "revision_count": notebook.log()["total"]}
""",
    [
        "chemworld.agents.notebook_versions:NotebookVersions",
        "chemworld.agents.notebook_versions:NotebookVersions.commit",
        "chemworld.agents.notebook_versions:NotebookVersions.read",
    ],
    "研究循环中的执行前笔记存档示例。修订管理不推进环境，也不自动证明预测成立；留出检验需另行设计。",
    "Archive a pre-execution research note. Revision management neither advances the environment nor verifies the prediction; held-out testing needs its own design.",
)

add(
    "replay",
    "replay",
    """
from pathlib import Path
from tempfile import TemporaryDirectory
from chemworld.examples.native_domains import run_workflow
from chemworld.data.logging import load_jsonl
from chemworld.eval.verify import verify_records
""",
    {"workflow": "heater"},
    """
with TemporaryDirectory() as directory:
    run = run_workflow(inputs["workflow"], directory, noise_mode="keyed")
    records = load_jsonl(Path(run["trajectory"]))
    replay = verify_records(records, tolerance=1e-8)
    output = {"actions": run["actions"], "replay_steps": run["replay_steps"], "verified": replay.verified}
    assert replay.verified
""",
    ["chemworld.data.logging:load_jsonl", "chemworld.eval.verify:verify_records"],
    "生成固定合成轨迹，再验证同一执行语义下的回放。临时目录不保留；需要长期保存时改为自己的输出目录。",
    "Generate a fixed synthetic trajectory and verify replay under its execution semantics. Use a persistent output directory if you want to retain the files.",
)

for key, task_id in (
    ("partition-task", "partition-discovery"),
    ("equilibrium-task", "equilibrium-characterization"),
    ("coupled-tasks", "reaction-to-crystallization"),
    ("distillation-task", "reaction-to-distillation"),
    ("flow-task", "flow-reaction-optimization"),
    ("electro-task", "electrochemical-conversion"),
):
    add(
        key,
        key + " notebooks",
        "from chemworld.runtime import ChemWorldRuntime",
        {"task_id": task_id, "seed": 7},
        """
session = ChemWorldRuntime(task_id=inputs["task_id"])
session.reset(seed=inputs["seed"])
try:
    task = session.task_info()
    output = {key: task[key] for key in (
        "task_id", "objective", "budget", "allowed_operations", "allowed_instruments")}
    output["available_operations"] = [a["operation"] for a in session.available_actions()]
finally:
    session.close()
""",
        [
            "chemworld.runtime:ChemWorldRuntime",
            "chemworld.runtime:ChemWorldRuntime.reset",
            "chemworld.runtime:ChemWorldRuntime.task_info",
        ],
        "初始化对应任务并读取任务目标、动作许可与当前可执行操作。这是 Notebook 起点，不是已经完成的优化实验；种子固定为 7。",
        "Initialize this task and inspect its objective, permissions and currently available operations. This is a notebook starting point, not a completed optimization experiment; seed is fixed at 7.",
    )

SBML = """<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
<model id="decay" timeUnits="second" substanceUnits="mole" extentUnits="mole" volumeUnits="litre">
<listOfUnitDefinitions><unitDefinition id="per_second"><listOfUnits>
<unit kind="second" exponent="-1" scale="0" multiplier="1"/>
</listOfUnits></unitDefinition></listOfUnitDefinitions>
<listOfCompartments><compartment id="cell" spatialDimensions="3" size="1" units="litre" constant="true"/></listOfCompartments>
<listOfSpecies><species id="A" compartment="cell" initialAmount="1" hasOnlySubstanceUnits="true" boundaryCondition="false" constant="false"/></listOfSpecies>
<listOfParameters><parameter id="k" value="0.2" units="per_second" constant="true"/></listOfParameters>
<listOfReactions><reaction id="decay_A" reversible="false">
<listOfReactants><speciesReference species="A" stoichiometry="1" constant="true"/></listOfReactants>
<kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML"><apply><times/><ci>k</ci><ci>A</ci></apply></math></kineticLaw>
</reaction></listOfReactions></model></sbml>"""

add(
    "sbml",
    "sbml-declaration sbml-execution",
    """
from chemworld.data.sbml import SBMLModel
from chemworld.data.scientific_inputs import InputSource
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.sbml_system import SBMLObservable
from chemworld.world.sbml_world import sbml_world
from chemworld.runtime import ChemWorldRuntime
""",
    {"xml": SBML, "duration_s": 2},
    """
source = InputSource(DataSourceProvenance("demo", 0, "Synthetic formal pool", "1", "CC0"), "synthetic")
model = SBMLModel(inputs["xml"], source)
world = sbml_world(model, maximum_step_s=0.1, observables=(SBMLObservable("amount", "A"),), readout_relative_sd=0)
session = ChemWorldRuntime(composition=world)
session.reset(seed=7)
try:
    advanced = session.execute({"operation": "wait", "duration_s": inputs["duration_s"]})
    measured = session.execute({"operation": "measure", "instrument": "formal_quantity_meter"})
    assert advanced.info["transaction_status"] == measured.info["transaction_status"] == "committed"
    output = {"xml_preserved": model.to_xml() == inputs["xml"],
              "status": measured.info["transaction_status"],
              "channels": [c.to_dict() for c in measured.observation.channels]}
finally:
    session.close()
""",
    [
        "chemworld.data.sbml:SBMLModel",
        "chemworld.world.sbml_world:sbml_world",
        "chemworld.physchem.sbml_system:SBMLObservable",
    ],
    "完整的自定义 SBML 输入：形式实体 A 以 0.2 s⁻¹ 衰减。未声明化学式与反应热，因此不据此声称元素或能量守恒。",
    "Complete self-authored SBML input: formal entity A decays at 0.2 s⁻¹. Formula and reaction heat are undeclared, so this is not an elemental or energy conservation claim.",
)

for key in ("uvvis", "hplc"):
    function = "uvvis_spectrum" if key == "uvvis" else "hplc_chromatogram"
    add(
        key,
        "spectra " + ("spectra-optical" if key == "uvvis" else "spectra-chromatography"),
        f"from chemworld.world.spectra import {function}",
        {"values": {"conversion": 0.4, "yield": 0.3, "impurity_signal": 0.1}, "seed": 7},
        f"""
packet = {function}(inputs["values"], seed=inputs["seed"])
output = {{key: packet[key] for key in ("kind", "axis", "raw_signal", "processed_estimates", "uncertainty")}}
output["array_shapes"] = {{key: len(value) for key, value in packet.items() if isinstance(value, list)}}
""",
        [f"chemworld.world.spectra:{function}"],
        "把公开标量反馈转为合成报告信号。示例不预测未知分子结构或真实谱图；输出选择在代码中列明。",
        "Turn public scalar feedback into a synthetic reporter signal, without predicting unknown molecular structure or measured spectra. Output selection is explicit in the code.",
    )

add(
    "nmr",
    "spectra spectra-structure",
    """
import numpy as np
from chemworld.physchem.configurable_spectra import acquire_spectrum
""",
    {
        "instrument": "nmr",
        "concentrations_mol_L": [0.1, 0.2, 0.05],
        "settings": {"resolution_factor": 1, "dilution_factor": 1, "scan_count": 4},
        "seed": 7,
    },
    """
packet = acquire_spectrum(inputs["instrument"], tuple(inputs["concentrations_mol_L"]),
                          inputs["settings"], np.random.default_rng(inputs["seed"]))
output = {"axis_unit": packet["axis"]["unit"], "points": len(packet["axis"]["values"]),
          "noise": packet["noise"], "processed_estimates": packet["processed_estimates"],
          "boundary": packet["metadata"]["boundary"]}
""",
    ["chemworld.physchem.configurable_spectra:acquire_spectrum"],
    "三个匿名合成报告组分的 NMR 采集与反演。浓度为 mol/L；扫描次数影响噪声。此处显式省略大数组并显示点数。",
    "Acquire and fit three anonymous synthetic NMR reporters. Concentration is in mol/L; scan count changes noise. The code omits large arrays and shows their size.",
)

add(
    "programmed",
    "programmed spectra-chromatography",
    """
from pathlib import Path
from tempfile import TemporaryDirectory
from chemworld.examples.programmed_chromatography import run_case
""",
    {"instrument": "hplc", "program": 0},
    """
with TemporaryDirectory() as directory:
    output = run_case(inputs["instrument"], inputs["program"], Path(directory) / "record.jsonl")
    assert output["passed"]
""",
    [
        "chemworld.examples.programmed_chromatography:run_case",
        "chemworld.world.chromatography_world:chromatography_world_spec",
    ],
    "程序 0 的 HPLC：标准、样品、空白各两次，保留柱内残留、流出物与回放。program 可选 0、1、2；instrument 可选 hplc 或 gc。",
    "HPLC program 0: two standard/sample/blank cycles with retained inventory, effluent and replay. program accepts 0, 1 or 2; instrument accepts hplc or gc.",
)

add(
    "particle-readout",
    "spectra-particle",
    """
from chemworld.runtime import ChemWorldRuntime
""",
    {"task_id": "reaction-to-crystallization", "seed": 7},
    """
session = ChemWorldRuntime(task_id=inputs["task_id"])
session.reset(seed=inputs["seed"])
try:
    schema = session.action_schema("measure")
    output = {"task_instruments": session.task_info()["allowed_instruments"],
              "measure_schema": schema,
              "particle_request": session.validate_action({"operation": "measure", "instrument": "particle_size"})}
finally:
    session.close()
""",
    [
        "chemworld.runtime:ChemWorldRuntime.action_schema",
        "chemworld.runtime:ChemWorldRuntime.validate_action",
    ],
    "粒径接口是可选安装项。本例展示任务级许可检查：该基准任务只允许 HPLC 与终检，直接请求粒径测量会被拒绝；拒绝原因保留在输出中。",
    "Particle sizing is optional. This example demonstrates task permission checking: the selected benchmark allows HPLC and final assay, so a direct particle-size request is rejected and the reason is retained.",
)

add(
    "cli",
    "interface-cli",
    """
from contextlib import redirect_stdout
from io import StringIO
from chemworld.cli import main
""",
    {"argv": ["tasks", "show", "partition-discovery"]},
    """
stream = StringIO()
with redirect_stdout(stream):
    main(inputs["argv"])
task = json.loads(stream.getvalue())
output = {key: task[key] for key in ("task_id", "objective", "budget", "allowed_operations")}
""",
    ["chemworld.cli:main"],
    "等价命令：chemworld tasks show partition-discovery。这里从 Python 调用同一 CLI 并提取 JSON 的几个字段；没有启动实验或服务。",
    "Equivalent shell command: chemworld tasks show partition-discovery. This Python call invokes the same CLI and selects JSON fields without starting a run or server.",
)

TDB = """ELEMENT FE BCC_A2 55.845 0 0 !
ELEMENT NI FCC_A1 58.69 0 0 !
TYPE_DEF % SEQ * !
PHASE ALPHA % 1 1 !
CONSTITUENT ALPHA :FE,NI: !
PARAMETER G(ALPHA,FE;0) 100 -1000+10*T-20*T*LN(T); 2000 N !
PARAMETER G(ALPHA,NI;0) 100 -2000+5*T-25*T*LN(T); 2000 N !
PARAMETER L(ALPHA,FE,NI;0) 100 1000+T; 2000 N !"""
DAT = """Synthetic mixed-valence Fe-O declarations
2 2 0 3 0
FE O
55.845 15.999
6 1 2 3 4 5 6
1 1
SLAG
SUBQ
2 3
FEO
1 1 1 1
2000 0 0 0 0 0 0
1 1 0 0 0
2.4
FE2O3
1 1 2 3
2000 0 0 0 0 0 0
2 3 0 0 0
2.4
2 1
FE FE
O
2 3
1 2
2
1
1 2
1 1
1 1 3 3 4 4 4 4
1 2 3 3 4 6 4 4
2 2 3 3 6 6 4 4
0
"""
for key, module, database, model, phase, source_text in (
    ("cef", "tdb", "TdbDatabase", "CefPhaseModel", "ALPHA", TDB),
    ("mqmqa", "chem_sage", "ChemSageDatabase", "MQMQAPhaseModel", "SLAG", DAT),
):
    model_module = "calphad" if key == "cef" else "mqmqa"
    add(
        key,
        "calphad-" + key,
        f"""
from chemworld.data.{module} import {database}
from chemworld.physchem.{model_module} import {model}
from chemworld.data.scientific_inputs import InputSource
from chemworld.physchem.data_conflicts import DataSourceProvenance
""",
        {"database_text": source_text, "phase": phase, "temperature_K": 1000},
        f"""
source = InputSource(DataSourceProvenance("demo", 0, "Self-authored synthetic phase", "1", "CC0"), "synthetic")
database = {database}.from_string(inputs["database_text"], source=source)
model = {model}(database, inputs["phase"])
fractions = [dict.fromkeys(sublattice, 1/len(sublattice)) for sublattice in model.constituents]
result = model.evaluate(fractions, inputs["temperature_K"])
output = {{"site_fractions": fractions, "gibbs_J_mol": result.gibbs_J_mol,
          "enthalpy_J_mol": result.enthalpy_J_mol, "entropy_J_mol_K": result.entropy_J_mol_K,
          "contributions_J_mol": dict(result.contributions_J_mol)}}
""",
        [
            f"chemworld.data.{module}:{database}.from_string",
            f"chemworld.physchem.{model_module}:{model}",
            f"chemworld.physchem.{model_module}:{model}.evaluate",
        ],
        "完整合成相声明，使用各亚点阵均匀分数计算局部单相响应。G/H 为每摩尔非空位原子的 J；不执行多相平衡搜索。",
        "Complete synthetic phase declaration, evaluated at uniform sublattice fractions. G/H are in J per mole of nonvacant atoms; this does not perform multiphase equilibrium search.",
    )

add(
    "calibration-fit",
    "calibration",
    """
from chemworld.examples.native_calibration import generate_data, parameters
from chemworld.data.calibration import NativeCalibration
""",
    {"model": "thermal"},
    """
table, experiments, truth, runs = generate_data(inputs["model"])
result = NativeCalibration(table, parameters(inputs["model"]), experiments).fit()
output = {"known_parameters": truth, "status": result["status"], "estimate": result["estimate"],
          "training": result["training"], "validation": result["validation"],
          "identifiability": result["identifiability"]}
assert result["status"] == "fitted"
""",
    [
        "chemworld.data.calibration:NativeCalibration",
        "chemworld.data.calibration:NativeCalibration.fit",
    ],
    "对自带合成压力数据拟合热损失系数（W/K），训练与验证组固定。结果用于接口演示，不属于真实材料校准。",
    "Fit heat-loss conductance (W/K) to packaged synthetic pressure data with fixed training/validation groups. This is an API demonstration, not empirical material calibration.",
)

add(
    "battery-impedance",
    "impedance",
    """
from chemworld.physchem.battery_models import BatterySPMSpec
from chemworld.physchem.battery_impedance import battery_impedance
""",
    {"frequencies_Hz": [0.1, 1, 10]},
    """
result = battery_impedance(BatterySPMSpec(), inputs["frequencies_Hz"])
output = result.to_dict()
assert result.success
""",
    ["chemworld.physchem.battery_impedance:battery_impedance"],
    "静止等温 SPM 的只读频响，Z = −δV/δI，电流正方向为放电。它不推进装置时钟，也不是计费测量。",
    "Read-only impedance of a resting isothermal SPM: Z = −δV/δI with positive discharge current. It neither advances the apparatus clock nor acquires a charged measurement.",
)

add(
    "particle-acquisition",
    "spectra-particle",
    "from chemworld.runtime import ChemWorldRuntime",
    {
        "seed": 7,
        "actions": [
            {"operation": "add_solvent", "volume_L": 0.025, "solvent": 0},
            {"operation": "add_reagent", "amount_mol": 0.012},
            {"operation": "add_catalyst", "catalyst_amount_mol": 0.0002, "catalyst": 0},
            {
                "operation": "heat",
                "target_temperature_K": 380,
                "duration_s": 1800,
                "stirring_speed_rpm": 600,
            },
            {"operation": "quench"},
            {"operation": "cool_crystallize", "target_temperature_K": 320, "duration_s": 3600},
            {"operation": "measure", "instrument": "particle_size"},
        ],
    },
    """
session = ChemWorldRuntime(task_id="reaction-to-crystallization",
    full_process_contract_id="phase-resolved-process-v5", budget_override=40,
    episode_mode_override="single_experiment")
session.reset(seed=inputs["seed"])
try:
    for action in inputs["actions"]:
        result = session.execute(action)
        if result.info["transaction_status"] != "committed":
            raise RuntimeError(result.info["error_message"])
    output = {"status": result.info["measurement_status"],
              "channels": [c.to_dict() for c in result.observation.channels if c.observed]}
finally:
    session.close()
""",
    ["chemworld.runtime:ChemWorldRuntime", "chemworld.runtime:ChemWorldRuntime.execute"],
    "显式选择已提供粒径测量的 phase-resolved-process-v5 合同。完成反应、淬灭与结晶后采集粒径；预算覆盖仅用于本教学流程。",
    "Explicitly select phase-resolved-process-v5, which provides particle sizing. React, quench and crystallize before acquisition; the budget override is for this teaching workflow.",
)
