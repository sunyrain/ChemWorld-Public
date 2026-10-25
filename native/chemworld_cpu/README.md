# ChemWorld optional CPU library

This directory contains the optional C++20 execution boundary for ChemWorld.
It is a small ABI fixture for the build and binding contract, not a claim that
the scientific Python modules have already been migrated to C++. The library
uses an opaque world handle and C-shaped arguments so the exported ABI contains
no STL types or C++ object layout.

Build from the repository root with:

```powershell
uv run --no-sync python scripts/build_cpu.py --test
```

The base Python package remains installable without a C++ compiler. The
`chemworld.cpu` adapter chooses this library only when it is present and the
caller requests `cpu`; `auto` selects the maintained Python reference when it
is absent.
