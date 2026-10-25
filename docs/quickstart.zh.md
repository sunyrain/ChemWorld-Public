# 五分钟开始

安装环境，执行三次操作并取得仪器读数。示例使用合成模型，不需要模型密钥或可选上游求解器。

```bash
git clone https://github.com/sunyrain/ChemWorld-Public.git
cd ChemWorld-Public
uv sync --locked --extra dev
uv run --no-sync chemworld capabilities --format json
uv run --no-sync python -m chemworld.examples.user_reward
```

最后一个命令执行两次投料和一次压力测量。奖励是可选的普通Python函数，更改奖励不改变取得的观测。

运行包含材料延续、转移和终检的较长实验：

```bash
uv run --no-sync python -m chemworld.examples.native_surface_research --bulk --mobile --terminal-assay
```

然后[设计自己的环境](environment-authoring.md)、[接入Agent](environment.md)，或查看
[原生工作流](native_domains.md)。文档安装目标为Python 3.11与3.12；当前交付为开发源码版本。

历史教学Notebook保留固定版本及原始输出，运行方法见[Notebook指南](notebooks.md)。
