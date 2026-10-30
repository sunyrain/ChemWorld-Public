<!-- Generated from https://github.com/sunyrain/ChemWorld/blob/main/docs/offline_research.md; edit the upstream guide. -->

# 无密钥完整研究范例 / Provider-free research example

这是当前运行时的教学开发范例，不是论文实验、模型能力评测或真实化学验证。
代码见 `examples/demo_offline_research.py`；不需要网络或模型密钥。

## 执行前固定的设计

问题：同一虚拟世界中，基于三个夹套设定温度的终检 yield，分段线性预测能否迁移到一个内插和一个外推条件？
固定 seed=0；每次重新创建同一任务的独立单实验容器，其他操作、剂量、加热时间、取样和等待均相同。
这不是三个独立世界，也不是三次随机重复；设定温度不等于物料实际温度。

- 训练：330、350、370 K。记录一次预期的空釜 HPLC 拒绝，不跳过其失败记录。
- 盲测：360、400 K。只用已完成训练的可观测 yield 作分段线性估计，边界外沿最近一段外推并裁剪到 [0,1]；同时比较训练均值基线。
- 两组预测均在执行任何盲测操作前写入 `predictions.json`，执行后不修改。
- 另保留一个主动停止、一个预算=2 截断的两步加料实验；均没有终检，yield 与终检分数不得补为 0。
- 固定分母：7 个独立 episode / 7 个记录批次，预计 45 次动作、5 次终检、1 次预期拒绝、1 个开放批次、1 个截断批次。

验收是记录完整、预测时间顺序正确、宿主账本与研究笔记分离、原生导出及每条轨迹在新进程零容差回放通过；**不以预测准确或得分高为通过条件**。
任何意外拒绝/运行异常均保留并使程序非零退出，不改参数重选有利结果。实例输出目录已存在时拒绝覆盖。

## 运行 / Run

在开发 checkout 安装锁定环境后：

```bash
uv run --no-sync python examples/demo_offline_research.py --output /tmp/chemworld-offline-example
uv run --no-sync chemworld verify --submission /tmp/chemworld-offline-example/trajectories/train-01.jsonl --tolerance 0
```

在安装了 wheel 的独立虚拟环境中，使用同一脚本的绝对路径和该环境的 `python` 即可；脚本不依赖 checkout 导入。Windows 请用自己的新输出路径代替 `/tmp/...`。

产物：`design.json`、实验前/后的研究笔记、宿主只追加事实账本、七条原生轨迹、`predictions.json`、合并 JSONL、`dataset_card.json`、逐条新进程 replay、含实际误差与失败分母的 `summary.json`。
合并导出用于分析；按单条轨迹回放，不把不同 episode 当成同一釜。

## What the example demonstrates

Seven fixed, independently reset episodes in one virtual world: three training
setpoints, two held-out setpoints, one deliberately open record and one truncated
record. The host writes the design before execution, seals both interpolation
and observed-mean predictions before either held-out run, and records their actual
absolute errors. It preserves the intended rejection and every unexpected failure.
No result is selected for being favorable. Missing assays remain null.

The notebook is writable through the document API; the host owns the append-only
fact ledger. Native trajectories include evaluator replay provenance and are not
fed to a model as public observations. This separation is an API convention, not
an OS sandbox. Reports explicitly retain uncertainty about kinetics, thermal lag,
measurement effects, new media, and other worlds. Two forecast errors cannot
establish a general law or a calibrated prediction interval.
