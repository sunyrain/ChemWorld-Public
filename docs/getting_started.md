# 安装并跑通第一次实验

这一页只做一件事：**从空环境走到一条通过回放验证的 ChemWorld 轨迹**。第一次运行建议使用
`reaction-to-assay`，整个过程通常只需要几分钟。

## 1. 安装项目

需要 Python 3.11 或更高版本。

```bash
git clone https://github.com/sunyrain/ChemWorld-Public.git
cd ChemWorld-Public
python -m pip install -e ".[dev]"
```

确认命令行入口已经安装：

```bash
chemworld --help
chemworld tasks list
```

=== "Windows"

    如果 `python` 被解析成 Microsoft Store 别名，可以直接使用仓库虚拟环境：

    ```powershell
    .\.venv\Scripts\python.exe -m pip install -e ".[dev]"
    ```

=== "文档与参考验证"

    只有在构建站点或运行扩展物理参考检查时，才需要额外依赖：

    ```bash
    python -m pip install -e ".[dev,docs,physchem-ref]"
    ```

## 2. 运行一个完整回合

先让内置 Agent 完成一次实验：

```bash
chemworld run --task reaction-to-assay --agent random --seed 0
```

命令会把 trajectory JSONL 和 manifest 写入 `runs/`，并在终端打印实际路径。这里的 random Agent
不是性能基线；它只是最短的端到端连通性检查。

## 3. 验证结果

把上一步输出的轨迹路径代入下面两条命令：

```bash
chemworld verify --constitution --submission runs/<trajectory>.jsonl
chemworld evaluate --submission runs/<trajectory>.jsonl
```

- `verify` 检查文件结构、合同摘要、状态守恒和确定性回放。
- `evaluate` 从轨迹重新计算结果，不采用提交文件自带的分数。

看到验证通过后，你已经跑通了 ChemWorld 最核心的链路：

```text
任务 → Agent → Action → 环境 → 轨迹 → 回放 → 评分
```

## 4. 用 Python 查看环境

```python
import gymnasium as gym
import chemworld  # 注册 ChemWorld 环境

env = gym.make("ChemWorld", task_id="reaction-to-assay", seed=0)
observation, info = env.reset(seed=0)

print(info["task_id"])
print(info["physics_maturity"])
print(env.unwrapped.available_actions())
```

`available_actions()` 返回当前状态下真正可执行的操作。它比维护一套固定动作模板更可靠，因为不同
任务、阶段和物料状态会开放不同操作。

## 5. 手动执行一个 Action

先读取 schema，再在不改变环境状态的情况下校验动作：

```python
schema = env.unwrapped.action_schema("add_reagent")
action = {"operation": "add_reagent", "amount_mol": 0.01}
check = env.unwrapped.validate_action(action)

if check["valid"]:
    observation, reward, terminated, truncated, info = env.step(action)
else:
    print(check["invalid_reasons"])
```

无效动作会带回 `invalid_reasons` 和前置条件提示，方便 Agent 修改后重试。任务允许哪些操作、
仪器、预算与终止方式，以 reset 信息和任务卡为准。

!!! tip "正确处理未测量值"
    数组里的未测量值是 `NaN`，写入 JSONL 后是 `null`。不要把它当成零；读取数值时同时检查
    `observed_mask` 或 `observed_keys`。

## 6. 打开可视化实验室

```bash
python -m apps.task_lab.server --port 8876
```

启动后可以打开：

- [Agent Observatory](http://127.0.0.1:8876/agent/)：观看 Agent 的逐步决策、谱图与资源消耗。
- [环境API与CLI](environment.md)：执行自己的操作并读取观测；旧Student Lab入口已退役。

经典算法和本地实验不需要在线模型密钥。Provider 凭据和私有评测配置不属于本公开版本。

## 接下来做什么

| 如果你想…… | 继续阅读 |
| --- | --- |
| 换一个更复杂的任务 | [查看世界能力地图](world-capability-map.md) |
| 编写自己的 Agent | [从动作 schema 开始](action_schema.md) |
| 理解每种操作的含义 | [认识操作语言](operations.md) |
| 做可复现的方法比较 | [阅读公开协议](https://github.com/sunyrain/ChemWorld-Public/tree/main/protocols) |
