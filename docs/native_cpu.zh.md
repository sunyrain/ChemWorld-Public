<!-- Generated from https://github.com/sunyrain/ChemWorld/blob/main/docs/native_cpu.md; edit the upstream guide. -->

# 可选 CPU 构建边界

ChemWorld 的 Python/reference 路径不需要 C++ 编译器。当前主线提供一个隔离的 C++20 CPU ABI
骨架，供后续原生执行任务接入；它只演示受控句柄、单世界操作和批量操作的构建与绑定，不把
现有 Python 科学模块宣称为已经迁移到 C++。

## 构建与 smoke test

在仓库根目录执行：

```powershell
uv run --no-sync python scripts/build_cpu.py --test
```

脚本会把中间文件和安装结果放在 `build/chemworld-cpu/`。需要指定 Windows 生成器时可以写：

```powershell
uv run --no-sync python scripts/build_cpu.py --generator "Visual Studio 17 2022" --test
```

构建结果的 ABI 是 `chemworld-cpu-abi-1`，导出的是 C 形状接口：`reset`、`observe`、`step` 和
`step_batch`。当前 fixture 操作码为 `0`（observe）、`1`（add）和 `2`（scale）；它们只用于验证
绑定边界，不代表化学模型。接口不跨越 STL 容器或 C++ 对象布局，适合后续生成薄绑定；native 库的安装目录
包含 `chemworld_cpu_build.json`，记录 ABI、配置和安装位置。

## Python 选择规则

```python
from chemworld.cpu import CpuBackendUnavailable, CpuWorld, select_backend

backend = select_backend("auto")       # 缺少库时返回 "reference"
try:
    cpu_backend = select_backend("cpu")  # 显式请求时缺失会抛出异常
except CpuBackendUnavailable:
    cpu_backend = "reference"
```

也可以通过 `CHEMWORLD_CPU_LIBRARY` 指定共享库路径。`auto` 只在初始化阶段选择一次完整执行器；
原生步骤失败不会静默切换到另一套执行器。基础 wheel 不携带 C++ 工具链或运行时库，预编译的
CPU 构建由相同 CMake 入口安装，科学 capability 声明仍来自唯一的 Python `DomainModuleRegistry`。
