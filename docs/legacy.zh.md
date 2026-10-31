<!-- Generated from https://github.com/sunyrain/ChemWorld/blob/main/docs/legacy.md; edit the upstream guide. -->

# 历史应用归档

ChemWorld当前定位是可编程物理环境。旧排行榜、官方基线工厂和逐操作LLM适配器已退出默认入口。
当前使用入口见[可编程环境](environment.md)。

设计新环境请使用[当前作者指南](environment-authoring.md)。旧Work I的受控fork协议
描述冻结研究，其单变量修改、证书和冻结流程不约束一般环境开发；
[v1组合合同](world-composition-contract.md)也按其首篇论文研究范围阅读。

旧 Work I／Work II、S0／G2、机理适应及 Experiment 1 应用已整组退出当前产品，
包括研究运行器、恢复入口、专属评测适配器和证据流水线。当前 NCS 的组织与分析位于论文 notebook 旁。
冻结说明、配置和原实现作为历史证据保存，不再作为当前开发入口。

| 历史功能 | 显式归档入口 |
| --- | --- |
| run、evaluate、leaderboard、suite、baselines、submission | `python -m chemworld.legacy.cli --help` |
| 旧排行榜汇总 | `chemworld.legacy.leaderboard` |
| 预设策略工厂 | `chemworld.legacy.agent_registry.make_agent` |
| 11个旧基线／适配模块 | `chemworld.legacy.agents`：bo、greedy、lhs、random、event、rl、llm、live_llm、diagnostic_live_llm、documented_codex_g2、structured_g2 |

上表兼容接口仍为既有消费者分发；已退役的研究运行器不再随产品分发。
这些兼容接口不作为环境推荐用法或默认评分。
旧路径不提供静默重导出。通用`run_agent`、文档工作区、完整公开历史与版本化笔记继续保留；
研究专用 MCP 与类型化信念评测协议随原研究应用归档。
内核／轨迹中的历史字段尚有消费者，本次没有用重命名改变物理、资源或已有结果。

复现冻结实验仍应使用实验原绑定版本与数据；迁移后的归档入口不是旧物理模拟器。
本次没有重跑、更新或重新认证历史实验结果。
