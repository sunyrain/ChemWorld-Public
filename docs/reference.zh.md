# API 参考

[能力索引](https://sunyrain.github.io/ChemWorld-Public/explore/?lang=zh#index/combustion)
是当前公开基座的交互式 API 文档。每项已实现能力都提供页内 Python 接口、参数与默认值、
完整示例、明确输入以及由该代码实际运行产生的输出。无需打开实现源码即可复制使用。

在“使用文档”中阅读完整用法，在“接口与参数”中查看函数签名与动作字段，在“示例与输出”
中查看可运行代码及结果。同一条目的其他示例可以展开。规划项明确标记，尚无可调用接口。

输入与输出默认按“字段 / 值 / 单位”展示，嵌套分组可以展开。切换至 **JSON** 可查看完整结构；
两种视图均可用“复制 JSON”获取原始内容。

示例使用合成输入或随包提供的预设。部分示例演示初始化、配置或请求拒绝，具体用途写在
代码旁。它们是代表性入口，并非全部 Python 符号的穷举，也不是新增的科学资格结果。
不同平台或求解库版本可能导致数值末位差异；交付边界见[当前基座](current-foundation.md)。

## 运行示例

按[安装说明](getting_started.md)安装当前源码，将复制的 Python 代码保存为 `example.py`，
执行 `uv run --no-sync python example.py`。例如，使用公开的物理量接口转换温度：

```python
from chemworld.foundation.units import Quantity

result = Quantity(26.85, "degC").to("K")
print(result.value, result.unit)
```

```text
300.0 K
```

## 运行时与世界构建

装置示例提供初始化后实际的 `action_schema()` 字段，包括单位、必填项、范围与可选值。
范围可能随状态改变，执行前可用 `validate_action()` 检查。操作回执区分已提交动作与拒绝请求。
`observe()` 只读取已有公开观测；重新采集测量需调用 `execute({"operation": "measure", ...})`。

| 主题 | 参考文档 |
| --- | --- |
| 安装与第一个环境 | [Getting started](getting_started.md) |
| 有类型操作 payload | [Action schema](action_schema.md) |
| 支持的过程操作 | [Operations](operations.md) |
| 世界组件兼容性 | [Composition contract](world-composition-contract.md) |
| 世界组合示例 | [Composition examples](world-composition-examples.md) |
| 能力覆盖 | [World capability map](world-capability-map.md) |

Gymnasium 适配器的 `env.unwrapped` 也提供核心 Agent 接口：

```python
env.unwrapped.task_prompt()
env.unwrapped.available_actions()
env.unwrapped.action_schema("heat")
env.unwrapped.validate_action(action)
env.unwrapped.agent_view_bundle(observation, info)
```
