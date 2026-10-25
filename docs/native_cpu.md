# Optional CPU build

The Python backend runs without a C++ compiler. The optional C++20 build currently
provides an ABI and thin bindings, with fixture operations for checking handles and
single/batch calls. It does not execute the existing chemistry models in C++.

```bash
uv run --no-sync python scripts/build_cpu.py --test
```

Build products live in `build/chemworld-cpu/`. The fixture ABI supports `reset`,
`observe`, `step` and `step_batch`; observe/add/scale are binding demonstrations.
Base wheels do not bundle a native compiler or CPU runtime.

```python
from chemworld.cpu import CpuBackendUnavailable, select_backend

backend = select_backend("auto")
try:
    requested = select_backend("cpu")
except CpuBackendUnavailable:
    requested = "reference"
```

`auto` chooses an executor at initialization. An execution failure does not silently
switch backends. Set `CHEMWORLD_CPU_LIBRARY` to load an explicitly built library.
Scientific capabilities still come from the shared Python module declarations;
full numerical execution, state migration and performance qualification remain in development.
