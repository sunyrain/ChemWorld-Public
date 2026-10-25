# 设计自己的实验环境

从已支持的物理领域选择设备和模型，声明可用操作、仪器与资源，然后把环境交给自己的Agent。
实验问题、探索策略、奖励与评价由使用者决定。当前入口是`WorldCompositionSpec`与
`ChemWorldRuntime`；能力及参数范围由实际安装的模块声明决定。

## 修改一个可运行的环境

下面复用原生表面反应环境，设置两个研究容器的温控范围、单次操作时长和总操作预算。
容器、有限气源、催化剂与热模型来自现有构造器；这是合成参数示例，不代表真实催化剂预测。
代码只创建设计和读取接口，没有为Agent预排实验动作。

```python
import json
from pathlib import Path

from chemworld.runtime import ChemWorldRuntime
from chemworld.world.composition import WorldCompositionSpec
from chemworld.world.surface_world import surface_world_spec

design = surface_world_spec(terminal_assay=False).to_dict()
design["composition_id"] = "my-two-vessel-study"
design["task"]["objective"] = "Investigate how heating changes measured gas composition"
design["task"]["budget"] = 24
for component in design["components"]:
    if component["kind"] == "thermal":
        component["parameters"]["temperature_range_K"] = [330, 410]
        component["parameters"]["duration_range_s"] = [1, 60]

spec = WorldCompositionSpec.from_dict(design)
Path("world.json").write_text(
    json.dumps(spec.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
)
session = ChemWorldRuntime(composition=spec, observation_noise_mode="keyed")
try:
    session.reset(seed=12)
    print(session.task_prompt()["text"])
    print(session.available_actions())
    print(session.action_schema("heat"))
finally:
    session.close()
```

在自己的工作目录保存脚本及`world.json`。声明解析与运行时编译会检查部件兼容性和参数；
填写一个模块名并不会安装新物理能力。换用其他领域时，先查看[领域工作流](native_domains.md)
和[安装能力](native_capabilities.md)，再使用对应的已支持构造器／模型声明。

## 让Agent操作环境

用户宿主加载`world.json`并创建会话，将研究问题、`task_prompt()`、动作规格和已取得的原生观测
交给Agent。Agent自行决定下一步，宿主调用`validate_action(action)`和`execute(action)`，
保存执行记录后返回公开结果。读取`observe()`不会重新测量，也不会推进时钟。

测量值取自`result.observation.to_dict()["native"]["channels"]`，连同单位、质量、采样时间和
不确定度一起保留。未测量不等于零，上一轮缓存的读数不等于新实验结果。世界声明、完整
`result.info`和会话快照属于宿主；其中可能包含Agent不应看到的物理参数或内部诊断。

无评分环境的提示采用用户声明的`objective`，不要求优化内置分数或提交Agent清单。
结构化字段中的`trajectory_validity`用于记录／回放检查，不是Agent表现指标。
是否设置奖励可参考[普通Python奖励示例](environment.md)。
终结规则按当前世界声明生效；上面的环境不要求终检，其他环境可以显式声明终检仪器。

## 记录、恢复与展示

- 保存动作和原生测量，使用现有轨迹记录与回放接口检查执行是否可复现。
- `session.snapshot()`保存私有状态，`ChemWorldRuntime.from_snapshot(...)`用于延续兼容会话。
- 模型调用预算由用户宿主管理，与环境操作预算分别计账；需要模型总结时提前保留额度。
- Research Workbench与Blender Lab暂缓发布，当前通过[环境API及CLI](environment.md)操作。

## 与旧研究协议的区别

[历史应用归档](legacy.md)中的Work I world-fork协议规定受控对照实验的单变量修改与冻结证据。
它不是当前自定义环境的通用作者流程。复现该研究时仍使用其绑定版本；创建设备环境从本页开始。
