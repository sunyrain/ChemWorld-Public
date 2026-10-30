# 五分钟开始

安装环境，运行包含材料延续和终检的一次实验。示例使用合成模型，不需要模型密钥或可选上游求解器。

```bash
git clone https://github.com/sunyrain/ChemWorld-Public.git
cd ChemWorld-Public
uv sync --locked --extra dev
uv run --no-sync python -m chemworld.examples.native_surface_research --bulk --mobile --terminal-assay
```

命令输出 JSON 执行记录。接下来可以查看可用能力，或尝试自定义奖励函数的小例子：

```bash
uv run --no-sync chemworld capabilities --format summary
uv run --no-sync python -m chemworld.examples.user_reward
```

然后[设计自己的环境](environment-authoring.md)、[接入Agent](environment.md)，或查看
[原生工作流](native_domains.md)。文档安装目标为Python 3.11与3.12；当前交付为开发源码版本。

历史教学Notebook保留固定版本及原始输出，运行方法见[Notebook指南](notebooks.md)。
