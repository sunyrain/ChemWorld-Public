# 历史应用归档

ChemWorld当前定位是可编程物理环境。旧排行榜、官方基线工厂和逐操作LLM适配器已退出默认入口。
当前使用入口见[可编程环境](environment.md)。

设计新环境请使用[当前作者指南](environment-authoring.md)。保留的
[Work I world-authoring contract](https://github.com/sunyrain/ChemWorld/blob/main/docs/world-authoring-contract.md)是冻结研究的受控fork协议，
其单变量修改、证书和冻结流程不约束一般环境开发。原文路径及内容因历史证据绑定保持不变；
[v1组合合同](world-composition-contract.md)也按其首篇论文研究范围阅读。

| 历史功能 | 显式归档入口 |
| --- | --- |
| run、evaluate、leaderboard、suite、baselines、submission及旧发布流程 | `python -m chemworld.legacy.cli --help` |
| 旧排行榜汇总 | `chemworld.legacy.leaderboard` |
| 预设策略工厂 | `chemworld.legacy.agent_registry.make_agent` |
| 11个旧基线／适配模块 | `chemworld.legacy.agents`：bo、greedy、lhs、random、event、rl、llm、live_llm、diagnostic_live_llm、documented_codex_g2、structured_g2 |

归档应用保留现有研究辅助消费者的明确导入，仍随当前源码分发，但不再作为环境推荐用法或默认评分。
旧路径不提供静默重导出。共享`run_agent`、MCP自由探索、笔记本和实际使用的研究实现继续存在。
内核／轨迹中的历史字段尚有消费者，本次没有用重命名改变物理、资源或已有结果。

旧文档和迁移前源码保存在Git提交`3e5fe32598d96c4ab2b83fcfb26b48939c26c967`。
复现冻结实验仍应使用实验原绑定版本与数据；迁移后的归档入口不是旧物理模拟器。
本次没有重跑、更新或重新认证历史实验结果。
