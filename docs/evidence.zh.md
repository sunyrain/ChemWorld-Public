# 证据

> 当前基座见[10月更新](current-foundation.md)。下文已有图示与数值证据对应8月冻结运行时。

ChemWorld v0.3.0 完整保留 v0.2.0 的四组冻结净化证据。新增展示层不会引入新的 benchmark 实验。

| 证据组 | 冻结结果 | 支持的判断 |
| --- | ---: | --- |
| Composition qualification | 64/64 个单元；1,786/1,786 条 recipe | 注册世界组合与参考执行 |
| Deterministic use cases | 8/8 个案例；89 个提交动作 | 生命周期、失败、资源和精确回放语义 |
| Controlled world forks | 6 对世界；24 条轨迹 | 固定公开实验下的单组件世界变化 |
| Agent instrument use | 1 条生命周期；15 个提交动作 | 持续使用公开动作与仪器接口 |

## 为什么图示可审计

`scripts/build_readme_visuals.py` 只读取净化后的压缩报告，并重新生成 README SVG、网站交互数据和详细静态表。任何输出过期时，`--check` 都会失败。

## 来源与净化

每份压缩报告都会绑定原报告 SHA-256、净化器版本、公开协议，以及被移除元数据的摘要哈希。规范化 JSON 与 gzip `mtime=0` 使净化过程可重复。

参见[完整报告地图](https://github.com/sunyrain/ChemWorld-Public/blob/main/evidence/README.md)、[代表性行为表](representative-behavior.md)和[发布 manifest](https://github.com/sunyrain/ChemWorld-Public/blob/main/release/manifest.json)。

!!! warning "有限声明边界"
    通过这些检查不代表普适化学保真度、通用 Agent 智能或能够安全迁移到真实实验室。
