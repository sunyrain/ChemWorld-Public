# 公共分发

本仓库选择性发布ChemWorld开发源码。使用`uv sync --locked --extra dev`安装，
或用`uv build`构建wheel与源码包。源码版本仍为`0.5.0.dev0`，仓库更新不代表发布新PyPI版本。

分发包含运行时、已批准的模拟器资源、环境CLI、Python示例及可选CPU构建源码。
研究输入、私有评测集、凭据、提供方原始响应、内部界面原型与论文素材不进入分发。
源码包可脱离研究仓库重新构建。

唯一控制台命令是`chemworld`；历史评测应用通过[显式归档入口](legacy.md)使用。
原生工作流不需要上游化学求解器，可选参考库用于另外的独立比较。
具体检查范围见[当前基座](current-foundation.md)。

旧论文结果使用[冻结源码和依赖](https://github.com/sunyrain/ChemWorld-Public/tree/03e8026301c185fd6ba5bdbda7460765d9b3e724)，当前物理不保证重现所有历史轨迹。
