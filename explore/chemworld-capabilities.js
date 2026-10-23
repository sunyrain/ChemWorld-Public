// Editorial navigation over the existing capability snapshot; not a runtime registry.
window.CHEMWORLD_CAPABILITIES = {
  "snapshot": "2026-10-23",
  "registry": {
    "modules": 27,
    "components": 32,
    "operations": 51,
    "measurements": 26
  },
  "tree": [
    {
      "id": "physics",
      "name": "物理世界",
      "kind": "layer",
      "preview": "",
      "children": [
        {
          "id": "foundation",
          "name": "共同科学基础",
          "kind": "domain",
          "preview": "身份、量纲、数值与科学数据",
          "children": [
            {
              "id": "identity-family",
              "name": "物质身份与来源",
              "kind": "family",
              "preview": "明确计算对象及其来源",
              "children": [
                {
                  "id": "identity",
                  "name": "元素、同位素与组分身份",
                  "state": "done",
                  "ability": "分子式、CAS、组分标识与版本化外部来源。",
                  "boundary": "数据查询可用性与具体物性的有效域分别检查。",
                  "example": "按真实组分身份装入参数与来源。",
                  "source": "docs/world-foundations.md",
                  "kind": "capability",
                  "type": "数据与标识",
                  "children": [
                    {
                      "id": "identity-elements",
                      "name": "元素与同位素",
                      "kind": "method",
                      "type": "数据与标识",
                      "state": "done",
                      "ability": "查询元素与同位素身份。",
                      "boundary": "数据查询可用性与具体物性的有效域分别检查。",
                      "source": "docs/world-foundations.md",
                      "derivedFrom": "identity"
                    },
                    {
                      "id": "identity-formula",
                      "name": "分子式与 CAS",
                      "kind": "method",
                      "type": "数据与标识",
                      "state": "done",
                      "ability": "分子式解析与 CAS 标识。",
                      "boundary": "数据查询可用性与具体物性的有效域分别检查。",
                      "source": "docs/world-foundations.md",
                      "derivedFrom": "identity"
                    },
                    {
                      "id": "identity-sources",
                      "name": "版本化来源",
                      "kind": "method",
                      "type": "数据与标识",
                      "state": "done",
                      "ability": "保留组分标识和外部来源版本。",
                      "boundary": "数据查询可用性与具体物性的有效域分别检查。",
                      "source": "docs/world-foundations.md",
                      "derivedFrom": "identity"
                    }
                  ]
                }
              ]
            },
            {
              "id": "quantities-family",
              "name": "量纲、参考与守恒",
              "kind": "family",
              "preview": "单位和历史温标属于共同科学输入",
              "children": [
                {
                  "id": "conservation",
                  "name": "单位、库存与守恒",
                  "state": "done",
                  "ability": "物料／相／能量归属，真实迁移与显式参考态。",
                  "boundary": "未知热量模型不能自动获得能量闭合。",
                  "example": "加料、转移、采样后核对剩余物料。",
                  "source": "docs/world-foundations.md",
                  "kind": "capability",
                  "type": "物理约束"
                },
                {
                  "id": "temperature",
                  "name": "历史温标转换",
                  "state": "done",
                  "ability": "五温标、500 节点、全方向互换；原量／输入／观测及跨功能消费。",
                  "boundary": "公开文献与参考版本的差值约定显式区分；真实端点限定有效域。",
                  "example": "历史温度读数换算后进入 Henry 相分配或水电光物性。",
                  "source": "docs/native_foundation.md",
                  "kind": "capability",
                  "type": "校准方法"
                }
              ]
            },
            {
              "id": "numerics-family",
              "name": "数值建模与求解",
              "kind": "family",
              "preview": "共同推进、代数约束与离散",
              "children": [
                {
                  "id": "numerics",
                  "name": "ODE、DAE、残差与空间离散",
                  "state": "done",
                  "ability": "共同根求解、index-1 DAE、导数、守恒离散及事件。",
                  "boundary": "已声明模型与可解域；模型需显式接入。",
                  "example": "同一求解基础用于电池、输运和热力学。",
                  "source": "docs/world-foundations.md",
                  "kind": "capability",
                  "type": "数值方法",
                  "children": [
                    {
                      "id": "numerics-ode",
                      "name": "ODE 与事件",
                      "kind": "method",
                      "type": "数值方法",
                      "state": "done",
                      "ability": "声明模型的常微分推进与事件。",
                      "boundary": "已声明模型与可解域；模型需显式接入。",
                      "source": "docs/world-foundations.md",
                      "derivedFrom": "numerics"
                    },
                    {
                      "id": "numerics-dae",
                      "name": "残差与 index-1 DAE",
                      "kind": "method",
                      "type": "数值方法",
                      "state": "done",
                      "ability": "共同残差、根求解与 index-1 代数约束推进。",
                      "boundary": "已声明模型与可解域；模型需显式接入。",
                      "source": "docs/world-foundations.md",
                      "derivedFrom": "numerics"
                    },
                    {
                      "id": "numerics-space",
                      "name": "守恒空间离散与导数",
                      "kind": "method",
                      "type": "数值方法",
                      "state": "done",
                      "ability": "共同守恒离散与模型导数。",
                      "boundary": "已声明模型与可解域；模型需显式接入。",
                      "source": "docs/world-foundations.md",
                      "derivedFrom": "numerics"
                    }
                  ]
                }
              ]
            },
            {
              "id": "calibration-family",
              "name": "科学输入与参数标定",
              "kind": "family",
              "preview": "数据质量、有限拟合与留出验证",
              "children": [
                {
                  "id": "calibration",
                  "name": "科学输入与有限参数标定",
                  "state": "done",
                  "ability": "带来源科学输入、有限参数拟合、温度／参数导数及可辨识性诊断。",
                  "boundary": "显式参数白名单与固定数据分组。",
                  "example": "用冷却曲线估计热损失参数。",
                  "source": "docs/world-foundations.md",
                  "kind": "capability",
                  "type": "数据与标定"
                }
              ]
            }
          ]
        },
        {
          "id": "properties",
          "name": "物性与相平衡",
          "kind": "domain",
          "preview": "物性方法与热力学闭合",
          "children": [
            {
              "id": "caloric-family",
              "name": "热量与相变性质",
              "kind": "family",
              "preview": "热容、生成量、蒸气压与潜热",
              "children": [
                {
                  "id": "caloric",
                  "name": "热容、生成热与相变",
                  "state": "done",
                  "ability": "热容方法族、H/S 积分与反求、蒸气／升华压力、潜热及来源消费。",
                  "boundary": "具体方法、相参考与温域显式选择。",
                  "example": "由温度和组成计算相变热量。",
                  "source": "docs/native_foundation.md",
                  "kind": "capability",
                  "type": "物性方法"
                }
              ]
            },
            {
              "id": "bulk-properties",
              "name": "体积与输运性质",
              "kind": "family",
              "preview": "密度、黏度和导热分别建模",
              "children": [
                {
                  "id": "density",
                  "name": "临界性质、密度与维里",
                  "state": "partial",
                  "ability": "外部密度／体积来源、7 类 B 与 2 类 C 维里方法、交叉参数和混合规则已接入；纯气体 PVT／热量性质连接封闭定容热过程。",
                  "boundary": "调用者显式提供来源与适用范围；混合物通用热力学和开放非理想流路不在当前消费范围。",
                  "source": "docs/native_foundation.md",
                  "kind": "capability",
                  "type": "物性方法"
                },
                {
                  "id": "viscosity",
                  "name": "分子参数与黏度",
                  "state": "done",
                  "ability": "LJ／碰撞积分、纯物质与混合黏度、稠密修正及黏度尺度。",
                  "boundary": "经验方法适用域与所需参数各自保留。",
                  "example": "组成变化进入气体混合黏度。",
                  "source": "docs/native_foundation.md",
                  "kind": "capability",
                  "type": "物性方法"
                },
                {
                  "id": "conductivity",
                  "name": "导热方法族",
                  "state": "done",
                  "ability": "纯物质、混合物与稠密流体关联式，水／空气方法、导数及来源输入。",
                  "boundary": "各方法要求独立参数和有效域，来源身份审查仍在继续。",
                  "example": "由实际状态与导热系数计算换热响应。",
                  "source": "docs/native_foundation.md",
                  "kind": "capability",
                  "type": "物性方法"
                }
              ]
            },
            {
              "id": "interfacial-properties",
              "name": "界面与溶解性质",
              "kind": "family",
              "preview": "表面张力、溶解与有限相分配",
              "children": [
                {
                  "id": "interfaces",
                  "name": "表面与界面张力",
                  "state": "done",
                  "ability": "纯液气、混合及水油相关式；毛细力与 Laplace 压力消费。",
                  "boundary": "水油经验式、两相密度与组成约束。",
                  "example": "表面张力变化影响界面压差。",
                  "source": "docs/native_foundation.md",
                  "kind": "capability",
                  "type": "物性方法"
                },
                {
                  "id": "solubility",
                  "name": "Henry、溶解度与溶剂比较",
                  "state": "done",
                  "ability": "Henry 全基准换算与混合导数、固液溶解度、Hansen 参数与有限分配。",
                  "boundary": "明确溶剂、基准与中性物种适用范围。",
                  "example": "气液分配与有限水／有机相分离。",
                  "source": "docs/native_foundation.md",
                  "kind": "capability",
                  "type": "物性方法"
                }
              ]
            },
            {
              "id": "optical-properties",
              "name": "电光性质",
              "kind": "family",
              "preview": "介电、折射、极化与浓度换算",
              "children": [
                {
                  "id": "optics",
                  "name": "介电、折射、极化与 Brix",
                  "state": "done",
                  "ability": "水相关式、来源曲线／单点、偶极、Lorentz–Lorenz 与 Brix。",
                  "boundary": "未知测量温度保留；285 行自动组分映射仍待补齐。",
                  "example": "来源折射率进入 Brix，介电值进入电容。",
                  "source": "docs/native_foundation.md",
                  "kind": "capability",
                  "type": "物性方法"
                }
              ]
            },
            {
              "id": "fluid-equilibria",
              "name": "流体状态与相平衡",
              "kind": "family",
              "preview": "EOS、相稳定性、闪蒸与相包络",
              "children": [
                {
                  "id": "advanced-thermo",
                  "name": "纯流体 Helmholtz 热力学",
                  "state": "done",
                  "ability": "原生导数、热力学性质、密度求根、饱和、闪蒸及有限热过程消费。",
                  "boundary": "显式提供来源系数和有效域；均相局部稳定不等同于完整相平衡。",
                  "source": "docs/native_foundation.md",
                  "kind": "capability",
                  "type": "热力学模型"
                },
                {
                  "id": "activity-thermo",
                  "name": "活度模型与立方混合规则",
                  "state": "done",
                  "ability": "超额与偏摩尔性质、六种 UNIFAC 变体、Twu／PPR78、GE 混合及共同相态求值。",
                  "boundary": "基团分配、相互作用及来源参数须显式提供；模型一致性不等同于材料验证。",
                  "source": "docs/native_foundation.md",
                  "kind": "capability",
                  "type": "科学模型"
                },
                {
                  "id": "general-fluid-eos",
                  "name": "缔合与多流体状态方程",
                  "state": "planned",
                  "ability": "未来扩展通用缔合／多流体模型与无约束相包络。",
                  "boundary": "不在当前纯流体及选定立方／活度模型范围内。",
                  "source": "docs/native_foundation.md",
                  "kind": "capability",
                  "type": "科学模型"
                }
              ]
            }
          ]
        },
        {
          "id": "fuel-assessment",
          "name": "燃料与性质评估",
          "kind": "domain",
          "preview": "燃烧热化学、安全与环境指标",
          "children": [
            {
              "id": "combustion-family",
              "name": "燃烧热化学",
              "kind": "family",
              "preview": "产物、热值及流量规格",
              "children": [
                {
                  "id": "combustion",
                  "name": "燃烧计量与热值",
                  "state": "done",
                  "ability": "完整产物计量、HHV/LHV、相参考热量及两／三流规格。",
                  "boundary": "完全燃烧平衡；不含火焰或产物温度求解。",
                  "example": "给总出口量与空燃比，求进料与尾气。",
                  "source": "docs/native_foundation.md",
                  "kind": "capability",
                  "type": "热化学计算",
                  "children": [
                    {
                      "id": "combustion-products",
                      "name": "产物计量",
                      "kind": "method",
                      "type": "热化学计算",
                      "state": "done",
                      "ability": "由明确的元素组成求完全燃烧产物与耗氧量。",
                      "boundary": "完全燃烧平衡；不含火焰或产物温度求解。",
                      "source": "docs/native_foundation.md",
                      "derivedFrom": "combustion"
                    },
                    {
                      "id": "combustion-heat",
                      "name": "HHV / LHV",
                      "kind": "method",
                      "type": "热化学计算",
                      "state": "done",
                      "ability": "按明确相参考换算高、低位热值并计算有限热量。",
                      "boundary": "完全燃烧平衡；不含火焰或产物温度求解。",
                      "source": "docs/native_foundation.md",
                      "derivedFrom": "combustion"
                    },
                    {
                      "id": "combustion-streams",
                      "name": "两流与三流规格",
                      "kind": "method",
                      "type": "热化学计算",
                      "state": "done",
                      "ability": "有限燃料、空气及第三流的独立规格求解。",
                      "boundary": "完全燃烧平衡；不含火焰或产物温度求解。",
                      "source": "docs/native_foundation.md",
                      "derivedFrom": "combustion"
                    }
                  ]
                }
              ]
            },
            {
              "id": "assessment-family",
              "name": "性质与风险指标",
              "kind": "family",
              "preview": "安全、环境及燃料品质分别评价",
              "children": [
                {
                  "id": "assessment",
                  "name": "安全与暴露评估",
                  "state": "done",
                  "ability": "暴露限值、分类与可燃混合规则。",
                  "boundary": "来源、浓度基准与适用域显式保留；不是新的操作许可。",
                  "example": "在明确浓度基准与来源下比较暴露限值和可燃混合规则。",
                  "source": "docs/native_foundation.md",
                  "kind": "capability",
                  "type": "性质评估"
                },
                {
                  "id": "environment-assessment",
                  "name": "环境当量",
                  "state": "done",
                  "ability": "按排放质量与明确时间跨度计算 GWP / GTP / ODP 当量。",
                  "boundary": "历史来源、时间跨度及预测标签分别保留。",
                  "source": "docs/native_foundation.md",
                  "kind": "capability",
                  "type": "性质评估",
                  "derivedFrom": "assessment"
                },
                {
                  "id": "fuel-quality",
                  "name": "燃料品质指标",
                  "state": "done",
                  "ability": "RON / MON 与来源点火延迟品质比较。",
                  "boundary": "来源与估算方法分别记录；不推断发动机真实表现。",
                  "source": "docs/native_foundation.md",
                  "kind": "capability",
                  "type": "性质评估",
                  "derivedFrom": "assessment"
                }
              ]
            }
          ]
        },
        {
          "id": "reaction",
          "name": "反应与催化",
          "kind": "domain",
          "preview": "动力学模型与反应系统",
          "children": [
            {
              "id": "reaction-models",
              "name": "反应网络与热耦合",
              "kind": "family",
              "preview": "声明反应网络及其热反馈",
              "children": [
                {
                  "id": "thermal-reaction",
                  "name": "有限热反应与小型气相机制",
                  "state": "finite",
                  "ability": "显式反应网络、温度反馈与声明的压力相关速率。",
                  "boundary": "已写明的机制和参数；不是任意反应预测。",
                  "example": "加料、加热、等待、测量反应进程。",
                  "source": "docs/native_mechanisms.md",
                  "kind": "capability",
                  "type": "反应模型"
                }
              ]
            },
            {
              "id": "catalysis-models",
              "name": "表面催化模型",
              "kind": "family",
              "preview": "位点机制和描述符方法",
              "children": [
                {
                  "id": "surface",
                  "name": "单／多位点平均场催化",
                  "state": "finite",
                  "ability": "有限位点、吸附／反应、跨位点交换、相互作用与热量。",
                  "boundary": "声明的平均场机制；非空间 kMC 或真实催化剂拟合。",
                  "example": "CO/O₂ 表面反应及覆盖度干预。",
                  "source": "docs/native_mechanisms.md",
                  "kind": "capability",
                  "type": "催化模型"
                },
                {
                  "id": "descriptor-catalysis",
                  "name": "描述符催化图与控制度",
                  "state": "planned",
                  "ability": "描述符／标度、稳态催化图与控制度。",
                  "boundary": "后续输运、机制与催化工作流交付。",
                  "source": "docs/native_mechanisms.md",
                  "kind": "capability",
                  "type": "催化模型",
                  "derivedFrom": "flame-catalysis"
                }
              ]
            },
            {
              "id": "reaction-generation",
              "name": "机制与反应流扩展",
              "kind": "family",
              "preview": "完整机制与开放反应系统",
              "children": [
                {
                  "id": "kinetic-expansion",
                  "name": "机制生成与反应器网络",
                  "state": "planned",
                  "ability": "完整热化学／机制、开放反应器网络、灵敏度与压力依赖生成。",
                  "boundary": "Cantera／RMG 类完整工作流待深化。",
                  "source": "docs/native_mechanisms.md",
                  "kind": "capability",
                  "type": "反应系统"
                },
                {
                  "id": "flame-catalysis",
                  "name": "一维火焰与多组分反应流",
                  "state": "planned",
                  "ability": "多组分反应流和一维火焰。",
                  "boundary": "后续输运、机制与催化工作流交付。",
                  "source": "docs/native_mechanisms.md",
                  "kind": "capability",
                  "type": "反应流模型"
                }
              ]
            }
          ]
        },
        {
          "id": "separation",
          "name": "分离与颗粒过程",
          "kind": "domain",
          "preview": "分离单元与有限库存变化",
          "children": [
            {
              "id": "phase-separation",
              "name": "相平衡分离",
              "kind": "family",
              "preview": "闪蒸、液液萃取与蒸馏单元",
              "children": [
                {
                  "id": "flash",
                  "name": "有限 TP/PH 闪蒸",
                  "state": "finite",
                  "ability": "显式汽液相与热量参考下的有限相库存闭合。",
                  "boundary": "声明组分与模型；非通用多相全局求解。",
                  "example": "改变进料热量，观察汽液量。",
                  "source": "docs/native_foundation.md",
                  "kind": "capability",
                  "type": "分离模型"
                },
                {
                  "id": "extraction",
                  "name": "液液平衡、萃取与洗涤",
                  "state": "finite",
                  "ability": "液相稳定性、有限两相分配、混合／静置／分相／洗涤。",
                  "boundary": "支持的组分、活度模型与组成域。",
                  "example": "分配规律发现与有限溶剂洗涤。",
                  "source": "docs/native_foundation.md",
                  "kind": "capability",
                  "type": "分离模型"
                },
                {
                  "id": "distillation",
                  "name": "动态蒸馏与真空浓缩",
                  "state": "finite",
                  "ability": "塔板／接收器持液、回流、定时切割及变组成浓缩。",
                  "boundary": "小型显式装置；非任意工业塔设计。",
                  "example": "回流干预影响馏分与回收量。",
                  "source": "docs/native_foundation.md",
                  "kind": "capability",
                  "type": "分离模型"
                }
              ]
            },
            {
              "id": "particle-separation",
              "name": "结晶与吸附",
              "kind": "family",
              "preview": "粒子群、过滤与界面库存",
              "children": [
                {
                  "id": "crystal",
                  "name": "结晶、过滤与重溶",
                  "state": "finite",
                  "ability": "晶种、有限冷却生长／溶解、粒子群、过滤与再悬浮。",
                  "boundary": "已定义的粒子群与物性模型。",
                  "example": "冷却 → 生长 → 过滤 → 升温重溶。",
                  "source": "docs/native_foundation.md",
                  "kind": "capability",
                  "type": "分离模型"
                },
                {
                  "id": "adsorption",
                  "name": "吸附接触与再生",
                  "state": "finite",
                  "ability": "竞争吸附动力学、有限载体、清洗与保留库存。",
                  "boundary": "明确吸附体系、温域与载体热量。",
                  "example": "同一吸附剂多次接触与洗脱。",
                  "source": "docs/native_foundation.md",
                  "kind": "capability",
                  "type": "分离模型"
                }
              ]
            }
          ]
        },
        {
          "id": "process-systems",
          "name": "输运与流程系统",
          "kind": "domain",
          "preview": "流路、传热和流程组织",
          "children": [
            {
              "id": "transport-units",
              "name": "有限流路与传热",
              "kind": "family",
              "preview": "泵、管、罐和换热",
              "children": [
                {
                  "id": "transport",
                  "name": "连续流、泵管与换热",
                  "state": "finite",
                  "ability": "有限源罐／泵／管／接收器、压降、轴功与显式换热。",
                  "boundary": "有限工作点与已声明开放流路。",
                  "example": "流量／停留时间／温度共同干预。",
                  "source": "docs/native_foundation.md",
                  "kind": "capability",
                  "type": "输运模型"
                },
                {
                  "id": "fluid-engineering",
                  "name": "工程流体计算",
                  "state": "done",
                  "ability": "流量计、阀门、摩阻、管件、泵、气体／两相流、填料床／塔、明渠与设备几何。",
                  "boundary": "独立关联式与有限设备消费，不是无约束流程网络求解器。",
                  "source": "docs/native_foundation.md",
                  "kind": "capability",
                  "type": "科学模型"
                }
              ]
            },
            {
              "id": "flowsheet-models",
              "name": "通用流程建模",
              "kind": "family",
              "preview": "严格单元、回路与优化",
              "children": [
                {
                  "id": "flowsheet",
                  "name": "通用流程与严格单元",
                  "state": "planned",
                  "ability": "可压缩／两相流、通用控制体、严格分离、回路与流程优化。",
                  "boundary": "fluids／IDAES 类完整工作流尚待交付。",
                  "source": "docs/native_foundation.md",
                  "kind": "capability",
                  "type": "流程模型"
                }
              ]
            }
          ]
        },
        {
          "id": "materials",
          "name": "材料与电化学",
          "kind": "domain",
          "preview": "电极、电池和固相模型",
          "children": [
            {
              "id": "electrode-models",
              "name": "电极动力学",
              "kind": "family",
              "preview": "有限电极与双电层响应",
              "children": [
                {
                  "id": "electrode",
                  "name": "有限电化学与瞬态电极",
                  "state": "finite",
                  "ability": "电位／电流驱动、转化、有限双电层与瞬态读数。",
                  "boundary": "声明的电极动力学与有限体系。",
                  "example": "设定电位并观察转化与电响应。",
                  "source": "docs/native_domains.md",
                  "kind": "capability",
                  "type": "电极模型"
                }
              ]
            },
            {
              "id": "battery-models",
              "name": "电池模型与诊断",
              "kind": "family",
              "preview": "颗粒、电解液、空间耦合与阻抗",
              "children": [
                {
                  "id": "spm",
                  "name": "SPM、热循环与有限 SEI",
                  "state": "finite",
                  "ability": "颗粒扩散、循环、合成热模型及保留衰退。",
                  "boundary": "有限合成参数；不代表任意真实电池。",
                  "example": "充放电、温升及多循环状态延续。",
                  "source": "docs/native_domains.md",
                  "kind": "capability",
                  "type": "电池模型",
                  "children": [
                    {
                      "id": "battery-spm-model",
                      "name": "SPM 颗粒扩散",
                      "kind": "method",
                      "type": "电池模型",
                      "state": "finite",
                      "ability": "合成单颗粒电池的扩散与端电压响应。",
                      "boundary": "有限合成参数；不代表任意真实电池。",
                      "source": "docs/native_domains.md",
                      "derivedFrom": "spm"
                    },
                    {
                      "id": "battery-thermal-cycle",
                      "name": "热模型与循环",
                      "kind": "method",
                      "type": "电池模型",
                      "state": "finite",
                      "ability": "在声明的合成热模型下推进循环与温升。",
                      "boundary": "有限合成参数；不代表任意真实电池。",
                      "source": "docs/native_domains.md",
                      "derivedFrom": "spm"
                    },
                    {
                      "id": "battery-sei",
                      "name": "有限 SEI 衰退",
                      "kind": "method",
                      "type": "电池模型",
                      "state": "finite",
                      "ability": "保留有限 SEI、物料与循环衰退状态。",
                      "boundary": "有限合成参数；不代表任意真实电池。",
                      "source": "docs/native_domains.md",
                      "derivedFrom": "spm"
                    }
                  ]
                },
                {
                  "id": "spme-dfn",
                  "name": "SPMe 与 DFN",
                  "state": "done",
                  "ability": "电解液输运与空间耦合反应，接入原生电池推进。",
                  "boundary": "显式等温模型；DFN 具有固定合成参数／网格资格。",
                  "example": "比较电流变化下的浓度与端电压。",
                  "source": "docs/native_domains.md",
                  "kind": "capability",
                  "type": "电池模型",
                  "children": [
                    {
                      "id": "battery-spme-model",
                      "name": "SPMe",
                      "kind": "method",
                      "type": "电池模型",
                      "state": "done",
                      "ability": "显式等温电解液输运与原生推进。",
                      "boundary": "显式等温模型；DFN 具有固定合成参数／网格资格。",
                      "source": "docs/native_domains.md",
                      "derivedFrom": "spme-dfn"
                    },
                    {
                      "id": "battery-dfn-model",
                      "name": "DFN",
                      "kind": "method",
                      "type": "电池模型",
                      "state": "done",
                      "ability": "耦合空间反应与电解液输运；固定合成参数及网格资格。",
                      "boundary": "显式等温模型；DFN 具有固定合成参数／网格资格。",
                      "source": "docs/native_domains.md",
                      "derivedFrom": "spme-dfn"
                    }
                  ]
                },
                {
                  "id": "impedance",
                  "name": "阻抗与扩展电池族",
                  "state": "partial",
                  "ability": "共同频率响应与 SPM/SPMe/DFN 只读阻抗已接入；Thevenin RC／双节点热模型支持持久设备、共享时钟、原位测量和回放。",
                  "boundary": "SPM 独立频域资格已完成；其他模型的资格范围分别声明。Thevenin 使用经验参数与独立电荷／热收支，不推断化学物种。",
                  "source": "docs/native_domains.md",
                  "kind": "capability",
                  "type": "电池诊断"
                }
              ]
            },
            {
              "id": "battery-aging-family",
              "name": "电池寿命与老化",
              "kind": "family",
              "preview": "耦合老化和长循环扩展",
              "children": [
                {
                  "id": "battery-aging",
                  "name": "复杂老化与长循环",
                  "state": "planned",
                  "ability": "电池耦合老化与长循环。",
                  "boundary": "沿电池模型、热耦合及老化依赖分别扩展。",
                  "source": "docs/native_domains.md",
                  "kind": "capability",
                  "type": "老化模型",
                  "derivedFrom": "phase-diagram"
                }
              ]
            },
            {
              "id": "solid-models",
              "name": "固相热力学",
              "kind": "family",
              "preview": "局部相响应与全局相选择",
              "children": [
                {
                  "id": "calphad",
                  "name": "固溶体、TDB／CEF 与 MQMQA",
                  "state": "done",
                  "ability": "多亚点阵、ChemSage、相内组成、导数及有限热历程。",
                  "boundary": "单相局部响应；三处原资料矛盾明确拒绝。",
                  "example": "加热／冷却材料并观测相内变化。",
                  "source": "docs/native_domains.md",
                  "kind": "capability",
                  "type": "固相热力学模型",
                  "children": [
                    {
                      "id": "calphad-cef",
                      "name": "TDB / CEF",
                      "kind": "method",
                      "type": "固相热力学模型",
                      "state": "done",
                      "ability": "声明亚点阵、相内组成、导数与局部热响应。",
                      "boundary": "单相局部响应；三处原资料矛盾明确拒绝。",
                      "source": "docs/native_domains.md",
                      "derivedFrom": "calphad"
                    },
                    {
                      "id": "calphad-mqmqa",
                      "name": "ChemSage / MQMQA",
                      "kind": "method",
                      "type": "固相热力学模型",
                      "state": "done",
                      "ability": "明确资料与有限参数范围内的局部相模型。",
                      "boundary": "单相局部响应；三处原资料矛盾明确拒绝。",
                      "source": "docs/native_domains.md",
                      "derivedFrom": "calphad"
                    }
                  ]
                },
                {
                  "id": "phase-diagram",
                  "name": "多组分全局多相平衡与相图",
                  "state": "planned",
                  "ability": "多组分全局多相平衡与相图。",
                  "boundary": "依赖全局相选择与多相热力学交付。",
                  "source": "docs/native_domains.md",
                  "kind": "capability",
                  "type": "固相热力学模型"
                }
              ]
            }
          ]
        },
        {
          "id": "water",
          "name": "水化学与膜过程",
          "kind": "domain",
          "preview": "水相、矿物和膜分离",
          "children": [
            {
              "id": "aqueous-models",
              "name": "水相与矿物反应",
              "kind": "family",
              "preview": "平衡与有限表面积动力学",
              "children": [
                {
                  "id": "aqueous",
                  "name": "有限水相与矿物平衡",
                  "state": "finite",
                  "ability": "声明常数下的酸碱／络合、碳酸盐与方解石平衡。",
                  "boundary": "有限物种及离子强度范围；不是完整地化数据库。",
                  "example": "加入有限酸并观察 pH 与矿物量。",
                  "source": "docs/native_foundation.md",
                  "kind": "capability",
                  "type": "水相模型"
                },
                {
                  "id": "mineral",
                  "name": "方解石溶解／沉淀动力学",
                  "state": "finite",
                  "ability": "快水相平衡、慢矿物速率与真实表面积库存。",
                  "boundary": "固定 298.15 K；无自发矿物成核。",
                  "example": "表面积与饱和度改变反应历程。",
                  "source": "docs/native_foundation.md",
                  "kind": "capability",
                  "type": "矿物动力学"
                }
              ]
            },
            {
              "id": "membrane-models",
              "name": "膜分离与水处理",
              "kind": "family",
              "preview": "RO、污染、清洗及流程扩展",
              "children": [
                {
                  "id": "membrane",
                  "name": "有限 RO、膜污染与冲洗",
                  "state": "finite",
                  "ability": "稀 NaCl 反渗透、盐／水通量、沉积阻力与有限洗液。",
                  "boundary": "298.15 K 稀溶液、正向驱动；简化污染模型。",
                  "example": "重复过滤 → 污染 → 有限冲洗。",
                  "source": "docs/native_foundation.md",
                  "kind": "capability",
                  "type": "膜模型"
                },
                {
                  "id": "water-expansion",
                  "name": "完整电解质与水处理流程",
                  "state": "planned",
                  "ability": "数据库水相、反应输运、分布式 RO、NF／电渗析及流程经济性。",
                  "boundary": "Reaktoro／WaterTAP 类完整工作流待交付。",
                  "source": "docs/native_foundation.md",
                  "kind": "capability",
                  "type": "水处理模型"
                }
              ]
            }
          ]
        },
        {
          "id": "bio",
          "name": "生化与随机网络",
          "kind": "domain",
          "preview": "网络声明、历程和分析",
          "children": [
            {
              "id": "stochastic-models",
              "name": "随机动力学",
              "kind": "family",
              "preview": "低拷贝整数反应网络",
              "children": [
                {
                  "id": "ssa",
                  "name": "整数 Gillespie SSA",
                  "state": "finite",
                  "ability": "质量作用 propensity、整数计量、持久事件与独立 RNG。",
                  "boundary": "恒定体积／温度；无自动 ODE/SSA 混合切换。",
                  "example": "低拷贝 A⇌B 或酶结合／催化。",
                  "source": "docs/native_biochemical.md",
                  "kind": "capability",
                  "type": "随机模型"
                }
              ]
            },
            {
              "id": "biochemical-format",
              "name": "模型声明与交换",
              "kind": "family",
              "preview": "安全声明、数学表达与往返",
              "children": [
                {
                  "id": "sbml-declaration",
                  "name": "SBML 安全声明与往返",
                  "state": "done",
                  "ability": "安全数学表达、单位与声明编解码已完成。",
                  "boundary": "声明可读写与模型实际执行分别验收。",
                  "source": "docs/native_biochemical.md",
                  "kind": "capability",
                  "type": "模型格式"
                }
              ]
            },
            {
              "id": "biochemical-execution",
              "name": "生化执行与分析",
              "kind": "family",
              "preview": "多隔室、事件及系统分析",
              "children": [
                {
                  "id": "sbml-execution",
                  "name": "多隔室、规则与事件执行",
                  "state": "partial",
                  "ability": "SBML 网络、优先／延迟事件向共同推进接入。",
                  "boundary": "执行分支开发中；SED-ML／COMBINE 与完整资格待交付。",
                  "source": "docs/native_biochemical.md",
                  "kind": "capability",
                  "type": "生化执行"
                },
                {
                  "id": "bio-analysis",
                  "name": "完整生化动力学与系统分析",
                  "state": "planned",
                  "ability": "确定性／随机／混合历程、稳态、控制分析、噪声与估参。",
                  "boundary": "COPASI 类完整工作流尚待交付。",
                  "source": "docs/native_biochemical.md",
                  "kind": "capability",
                  "type": "系统分析"
                }
              ]
            }
          ]
        }
      ]
    },
    {
      "id": "runtime",
      "name": "实验运行时",
      "kind": "layer",
      "preview": "",
      "children": [
        {
          "id": "actions",
          "name": "实验操作与控制",
          "kind": "domain",
          "preview": "对物料和装置施加操作",
          "children": [
            {
              "id": "material-actions",
              "name": "物料操作",
              "kind": "family",
              "preview": "容器、加料、路由与采样",
              "children": [
                {
                  "id": "handling",
                  "name": "基本实验动作",
                  "state": "done",
                  "ability": "容器选择／创建、加料、加溶剂／催化剂、路由、转移、采样与终止。",
                  "boundary": "已安装、任务允许、当前可执行分别查询。",
                  "example": "物料从源瓶进入反应器，再分到样品瓶。",
                  "source": "docs/operations.md",
                  "kind": "capability",
                  "type": "实验操作"
                }
              ]
            },
            {
              "id": "device-actions",
              "name": "装置设定与控制",
              "kind": "family",
              "preview": "已声明设备的控制合同",
              "children": [
                {
                  "id": "devices",
                  "name": "有限设备控制",
                  "state": "finite",
                  "ability": "加热器、流路／泵、冷却／结晶、电极／电源及声明的分离操作。",
                  "boundary": "模拟装置合同；不等于真实硬件驱动。",
                  "example": "同一加热器依次服务不同体系。",
                  "source": "docs/operations.md",
                  "kind": "capability",
                  "type": "装置控制"
                }
              ]
            }
          ]
        },
        {
          "id": "instruments",
          "name": "仪器与测量",
          "kind": "domain",
          "preview": "仪器响应与可观测通道",
          "children": [
            {
              "id": "analytical-instruments",
              "name": "分析仪器",
              "kind": "family",
              "preview": "谱学、色谱、粒径与程序分析",
              "children": [
                {
                  "id": "spectra",
                  "name": "光谱、色谱与粒径",
                  "state": "finite",
                  "ability": "HPLC、GC、UV–Vis、NMR、IR、MS、粒径接口。",
                  "boundary": "声明的信号／峰形与反演模型；不是通用谱库鉴定。",
                  "source": "docs/native_foundation.md",
                  "kind": "capability",
                  "type": "仪器响应",
                  "children": [
                    {
                      "id": "spectra-optical",
                      "name": "UV–Vis / IR",
                      "kind": "method",
                      "type": "仪器响应",
                      "state": "finite",
                      "ability": "声明的光谱信号与反演模型。",
                      "boundary": "声明的信号／峰形与反演模型；不是通用谱库鉴定。",
                      "source": "docs/native_foundation.md",
                      "derivedFrom": "spectra"
                    },
                    {
                      "id": "spectra-structure",
                      "name": "NMR / MS",
                      "kind": "method",
                      "type": "仪器响应",
                      "state": "finite",
                      "ability": "声明的磁共振与质谱响应。",
                      "boundary": "声明的信号／峰形与反演模型；不是通用谱库鉴定。",
                      "source": "docs/native_foundation.md",
                      "derivedFrom": "spectra"
                    },
                    {
                      "id": "spectra-chromatography",
                      "name": "HPLC / GC",
                      "kind": "method",
                      "type": "仪器响应",
                      "state": "finite",
                      "ability": "声明峰形与已知组分的色谱响应。",
                      "boundary": "声明的信号／峰形与反演模型；不是通用谱库鉴定。",
                      "source": "docs/native_foundation.md",
                      "derivedFrom": "spectra"
                    },
                    {
                      "id": "spectra-particle",
                      "name": "粒径测量",
                      "kind": "method",
                      "type": "仪器响应",
                      "state": "finite",
                      "ability": "粒子群状态中的有限粒径响应。",
                      "boundary": "声明的信号／峰形与反演模型；不是通用谱库鉴定。",
                      "source": "docs/native_foundation.md",
                      "derivedFrom": "spectra"
                    }
                  ]
                },
                {
                  "id": "programmed",
                  "name": "程序色谱",
                  "state": "finite",
                  "ability": "程序 HPLC／GC、标准校准、样品／空白、残留与漂移。",
                  "boundary": "三种合成分析物、有限程序与已知峰模板。",
                  "example": "标准 → 样品 → 空白，观察残留。",
                  "source": "docs/native_foundation.md",
                  "kind": "capability",
                  "type": "仪器响应"
                }
              ]
            },
            {
              "id": "state-instruments",
              "name": "状态测量",
              "kind": "family",
              "preview": "物理相态、电学与离散计数",
              "children": [
                {
                  "id": "physical-readouts",
                  "name": "物理与相态读数",
                  "state": "finite",
                  "ability": "pH、气压／气体组成、相态、流量、水相 pH、膜、固相与 CALPHAD 相。",
                  "boundary": "对应世界与有效物料状态决定可用性。",
                  "source": "docs/native_foundation.md",
                  "kind": "capability",
                  "type": "测量接口"
                },
                {
                  "id": "electrical-readouts",
                  "name": "电学与电池读数",
                  "state": "finite",
                  "ability": "电极电学表、电池表。",
                  "boundary": "已声明的电极／电池模型与测量渠道。",
                  "source": "docs/native_foundation.md",
                  "kind": "capability",
                  "type": "测量接口"
                },
                {
                  "id": "discrete-readouts",
                  "name": "分子计数与捕获",
                  "state": "finite",
                  "ability": "离散分子计数、有限捕获与延迟分析。",
                  "boundary": "低拷贝计数／抽样合同；不是连续液体取样替身。",
                  "source": "docs/native_foundation.md",
                  "kind": "capability",
                  "type": "测量接口"
                }
              ]
            }
          ]
        },
        {
          "id": "lifecycle",
          "name": "调度与状态生命周期",
          "kind": "domain",
          "preview": "实验时间、设备和续算",
          "children": [
            {
              "id": "scheduling-family",
              "name": "时间与设备调度",
              "kind": "family",
              "preview": "共享推进和设备占用",
              "children": [
                {
                  "id": "scheduler",
                  "name": "共享实验室时间与设备队列",
                  "state": "done",
                  "ability": "声明系统同步推进、多设备身份、占用、FIFO 与延迟结果。",
                  "boundary": "显式有限系统；组合失败按共同提交边界回滚。",
                  "example": "两个样品竞争同一台离线分析仪。",
                  "source": "docs/native_shared_lab.md",
                  "kind": "capability",
                  "type": "调度机制"
                }
              ]
            },
            {
              "id": "persistent-state",
              "name": "持久状态与恢复",
              "kind": "family",
              "preview": "换料、维护和断点恢复",
              "children": [
                {
                  "id": "persistence",
                  "name": "持久设备与局部换料",
                  "state": "done",
                  "ability": "标定、污染、负载、老化与真实维护；局部换料／换设备。",
                  "boundary": "各状态有唯一所有者，历史样品不被后续修改。",
                  "example": "换一批料，保留膜污染或仪器残留。",
                  "source": "docs/native_shared_lab.md",
                  "kind": "capability",
                  "type": "生命周期机制"
                },
                {
                  "id": "continuation",
                  "name": "快照与状态续算",
                  "state": "done",
                  "ability": "物理状态、事务、RNG 和队列的快照与续算。",
                  "boundary": "快照与执行语义兼容范围显式检查。",
                  "example": "中途保存，恢复后延续事件与设备状态。",
                  "source": "docs/native_shared_lab.md",
                  "kind": "capability",
                  "type": "续算机制"
                }
              ]
            }
          ]
        },
        {
          "id": "records",
          "name": "观测与资源记录",
          "kind": "domain",
          "preview": "实验结果和真实消耗",
          "children": [
            {
              "id": "observations-family",
              "name": "原生观测与终检",
              "kind": "family",
              "preview": "单位、质量、采集时间与终检",
              "children": [
                {
                  "id": "assay",
                  "name": "终检与有质量标记的观测",
                  "state": "done",
                  "ability": "液体／气体终检、原生单位、采集时间、不确定度、缺值与质量标记。",
                  "boundary": "未观测与无效值保留；读取缓存不等于重新采样。",
                  "source": "docs/native_foundation.md",
                  "kind": "capability",
                  "type": "观测合同"
                }
              ]
            },
            {
              "id": "resource-ledger",
              "name": "资源结算",
              "kind": "family",
              "preview": "时间、费用、样品和能量",
              "children": [
                {
                  "id": "resource-accounting",
                  "name": "物理与资源账本",
                  "state": "done",
                  "ability": "结算物理时间、费用、样品与能量消耗。",
                  "boundary": "资源事件与实际执行事务一致，不用缓存读取替代新采集。",
                  "source": "docs/native_shared_lab.md",
                  "kind": "capability",
                  "type": "资源记录",
                  "derivedFrom": "continuation"
                }
              ]
            }
          ]
        },
        {
          "id": "access",
          "name": "访问接口与工作台",
          "kind": "domain",
          "preview": "程序访问、观察界面及外部接入",
          "children": [
            {
              "id": "programmatic-access",
              "name": "程序访问接口",
              "kind": "family",
              "preview": "不同调用方式共享同一环境",
              "children": [
                {
                  "id": "entrypoints",
                  "name": "Runtime／Gym／CLI／Agent",
                  "state": "done",
                  "ability": "同一实验状态、动作合同、原生观测和资源记录。",
                  "boundary": "Agent 权重、训练器及私有记忆在环境之外。",
                  "source": "docs/agents.md",
                  "kind": "capability",
                  "type": "程序接口",
                  "children": [
                    {
                      "id": "interface-runtime",
                      "name": "Runtime",
                      "kind": "method",
                      "type": "程序接口",
                      "state": "done",
                      "ability": "直接调用同一实验会话与原生观测。",
                      "boundary": "Agent 权重、训练器及私有记忆在环境之外。",
                      "source": "docs/agents.md",
                      "derivedFrom": "entrypoints"
                    },
                    {
                      "id": "interface-gym",
                      "name": "Gym",
                      "kind": "method",
                      "type": "程序接口",
                      "state": "done",
                      "ability": "通过 reset / step 使用同一实验状态与动作合同。",
                      "boundary": "Agent 权重、训练器及私有记忆在环境之外。",
                      "source": "docs/agents.md",
                      "derivedFrom": "entrypoints"
                    },
                    {
                      "id": "interface-cli",
                      "name": "CLI",
                      "kind": "method",
                      "type": "程序接口",
                      "state": "done",
                      "ability": "通过命令行执行动作、保存和检查轨迹。",
                      "boundary": "Agent 权重、训练器及私有记忆在环境之外。",
                      "source": "docs/agents.md",
                      "derivedFrom": "entrypoints"
                    },
                    {
                      "id": "interface-agent",
                      "name": "Agent",
                      "kind": "method",
                      "type": "程序接口",
                      "state": "done",
                      "ability": "通过 Agent 执行入口使用同一环境与资源记录。",
                      "boundary": "Agent 权重、训练器及私有记忆在环境之外。",
                      "source": "docs/agents.md",
                      "derivedFrom": "entrypoints"
                    }
                  ]
                }
              ]
            },
            {
              "id": "interactive-tools",
              "name": "交互与观察工具",
              "kind": "family",
              "preview": "本地实验和只读轨迹浏览",
              "children": [
                {
                  "id": "observatory",
                  "name": "Student Lab 与 Agent Observatory",
                  "state": "done",
                  "ability": "本地交互实验；只读轨迹浏览、资源检查与回放。",
                  "boundary": "界面运行与模型、任务的支持域一致。",
                  "source": "docs/student-lab.md",
                  "kind": "capability",
                  "type": "工作台"
                }
              ]
            },
            {
              "id": "external-integration",
              "name": "外部数据与硬件接入",
              "kind": "family",
              "preview": "独立模型与有限物理系统迁移",
              "children": [
                {
                  "id": "bridge",
                  "name": "真实数据、模型与硬件桥接",
                  "state": "planned",
                  "ability": "独立模型、真实数据、狭窄物理系统的迁移与验证。",
                  "boundary": "整体桥接路线；不把已有接口当成真实实验室接管。",
                  "source": "docs/student-lab.md",
                  "kind": "capability",
                  "type": "外部接入"
                }
              ]
            }
          ]
        }
      ]
    },
    {
      "id": "evaluation",
      "name": "任务与评测",
      "kind": "layer",
      "preview": "",
      "children": [
        {
          "id": "research-tasks",
          "name": "研究任务",
          "kind": "domain",
          "preview": "科学问题与干预目标",
          "children": [
            {
              "id": "discovery-tasks",
              "name": "规律发现与表征",
              "kind": "family",
              "preview": "分配规律和非理想平衡",
              "children": [
                {
                  "id": "partition-task",
                  "name": "分配规律发现",
                  "state": "finite",
                  "ability": "相条件干预、公开测量与隐藏世界响应比较。",
                  "boundary": "固定任务合同与有限相模型。",
                  "source": "docs/quickstart.md",
                  "kind": "capability",
                  "type": "研究任务"
                },
                {
                  "id": "equilibrium-task",
                  "name": "非理想平衡表征",
                  "state": "finite",
                  "ability": "浓度探针、pH／UV–Vis 反馈与平衡规律。",
                  "boundary": "任务声明的非理想模型与组分。",
                  "source": "docs/quickstart.md",
                  "kind": "capability",
                  "type": "研究任务"
                }
              ]
            },
            {
              "id": "coupled-process-tasks",
              "name": "反应与分离任务",
              "kind": "family",
              "preview": "结晶与蒸馏各有独立目标",
              "children": [
                {
                  "id": "coupled-tasks",
                  "name": "反应 → 结晶",
                  "state": "finite",
                  "ability": "反应动力学传播到产率、纯度与粒度。",
                  "boundary": "独立结晶研究任务及其模型范围。",
                  "source": "docs/quickstart.md",
                  "kind": "capability",
                  "type": "研究任务"
                },
                {
                  "id": "distillation-task",
                  "name": "反应 → 蒸馏",
                  "state": "finite",
                  "ability": "反应动力学传播到产率、纯度与馏分。",
                  "boundary": "独立蒸馏研究任务及其模型范围。",
                  "source": "docs/quickstart.md",
                  "kind": "capability",
                  "type": "研究任务",
                  "derivedFrom": "coupled-tasks"
                }
              ]
            },
            {
              "id": "control-tasks",
              "name": "过程控制与优化",
              "kind": "family",
              "preview": "流动反应与电化学干预",
              "children": [
                {
                  "id": "flow-task",
                  "name": "流动反应优化",
                  "state": "finite",
                  "ability": "流量、停留时间、温度与隐藏速率律。",
                  "boundary": "明确的有限流动反应世界。",
                  "source": "docs/quickstart.md",
                  "kind": "capability",
                  "type": "研究任务"
                },
                {
                  "id": "electro-task",
                  "name": "电化学转化",
                  "state": "finite",
                  "ability": "电化学构成律、诊断反馈与设定值干预。",
                  "boundary": "任务允许的电极与观测接口。",
                  "source": "docs/quickstart.md",
                  "kind": "capability",
                  "type": "研究任务"
                }
              ]
            }
          ]
        },
        {
          "id": "protocol",
          "name": "评测与复现",
          "kind": "domain",
          "preview": "研究循环、回放和计分",
          "children": [
            {
              "id": "research-protocol",
              "name": "研究流程",
              "kind": "family",
              "preview": "设计、预测和留出检验",
              "children": [
                {
                  "id": "research-loop",
                  "name": "设计、预测与盲测研究循环",
                  "state": "finite",
                  "ability": "实验设计、研究笔记、执行前预测、留出测试与事实账本。",
                  "boundary": "已有开发教学范例；不代表 Agent 科学能力已证实。",
                  "source": "docs/agents.md",
                  "kind": "capability",
                  "type": "评测协议"
                }
              ]
            },
            {
              "id": "replay-protocol",
              "name": "轨迹复现与计分",
              "kind": "family",
              "preview": "重放已记录的执行语义",
              "children": [
                {
                  "id": "replay",
                  "name": "回放与重新计分",
                  "state": "done",
                  "ability": "记录动作与结果，重放轨迹并重新计算指标。",
                  "boundary": "符合原执行语义；缺终检不能补造最终分数。",
                  "source": "docs/agents.md",
                  "kind": "capability",
                  "type": "复现功能"
                }
              ]
            }
          ]
        },
        {
          "id": "learning-resources",
          "name": "教学与任务资源",
          "kind": "domain",
          "preview": "可检视的研究范例",
          "children": [
            {
              "id": "notebook-resources",
              "name": "任务 Notebook",
              "kind": "family",
              "preview": "配对世界、干预和反馈",
              "children": [
                {
                  "id": "notebooks",
                  "name": "六类任务 Notebook",
                  "state": "done",
                  "ability": "候选干预、三组反馈、完整轨迹与配对隐藏世界。",
                  "boundary": "教学配对与正式 benchmark 结果分别表达。",
                  "source": "docs/student-lab.md",
                  "kind": "capability",
                  "type": "教学资源"
                }
              ]
            }
          ]
        }
      ]
    }
  ],
  "delivery": {
    "id": "release",
    "name": "最终跨平台与发布收尾",
    "state": "partial",
    "ability": "已有安装／示例验收；最终平台兼容工作仍待完成。",
    "boundary": "Linux／macOS 资格待补；开发能力与正式发布分开。",
    "source": "docs/current-foundation.md"
  },
  "taxonomy": {
    "version": 2,
    "levels": [
      "项目",
      "架构层",
      "能力领域",
      "功能组",
      "具体能力",
      "方法 / 变体"
    ],
    "counting": "仅统计具体能力；方法变体和交付事项不重复计数。"
  }
};
