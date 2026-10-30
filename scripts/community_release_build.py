"""Select version-controlled public files, not everything in a research checkout.

A generated source index travels inside the sdist, so rebuilding it needs no Git.
This is a packaging boundary, not a claim that installed Python hides its physics.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path, PurePosixPath

SOURCE_INDEX = "scripts/community_release_source_files.json"
BUILD_HOOK = "scripts/community_release_build.py"
INTERNAL_UI_PREFIXES = ("apps/", "src/chemworld/lab/", "src/chemworld/workbench/")
RUNTIME_CONFIGS = frozenset(
    {
        "configs/scenarios/mechanism_scenarios.yaml",
        "configs/benchmark/resource_limits.json",
        "configs/benchmark/risk_cost_vnext.json",
        "configs/public_dev.json",
        "configs/public_test.json",
    }
)
SOURCE_EXTRAS = frozenset(
    {
        # Both repositories carry the same guides with different locale suffixes.
        *{
            f"docs/{name}{suffix}.md"
            for name in (
                "agents", "environment", "environment-authoring", "getting_started",
                "operations", "action_schema", "legacy", "offline_research",
                "native_foundation", "native_domains", "native_capabilities",
                "native_biochemical", "native_shared_lab", "native_mechanisms",
                "native_stochastic", "native_cpu",
            )
            for suffix in ("", ".en", ".zh")
        },
        BUILD_HOOK,
        "scripts/build_cpu.py",
        "scripts/smoke_test_wheel.py",
        "pyproject.toml",
        "uv.lock",
        "README.md",
        "README.zh-CN.md",
        "LICENSE",
        "CONTRIBUTING.md",
        "DEVELOPMENT.md",
        "SECURITY.md",
        "examples/demo_manual_event_sequence.py",
        "examples/demo_agent_facing_api.py",
        "examples/demo_offline_research.py",
        "examples/demo_minimal_agent.py",
        "examples/native_surface_research.py",
        "examples/reuse_native_equipment.py",
        "docs/native_foundation.md",
        "docs/environment.md",
        "docs/environment.en.md",
        "docs/environment-authoring.md",
        "docs/environment-authoring.en.md",
        "docs/legacy.md",
        "docs/legacy.en.md",
        "docs/native_domains.md",
        "docs/native_capabilities.md",
        "docs/native_cpu.md",
        "docs/native_biochemical.md",
        "docs/native_shared_lab.md",
        "docs/community_release.md",
        "docs/community_release.en.md",
        "docs/getting_started.md",
        "docs/getting_started.en.md",
        "docs/operations.md",
        "docs/operations.en.md",
        "docs/action_schema.md",
        "docs/offline_research.md",
        "docs/offline_research.en.md",
    }
)


def public_path(name: str) -> bool:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "__pycache__" in path.parts:
        return False
    if name.startswith(INTERNAL_UI_PREFIXES):
        return False
    if name.startswith("native/chemworld_cpu/"):
        return path.suffix in {".cpp", ".h", ".hpp", ".txt", ".md", ".cmake"} or name.endswith(
            "CMakeLists.txt"
        )
    if name in SOURCE_EXTRAS or name in RUNTIME_CONFIGS:
        return True
    if name.startswith("configs/mechanisms/") and path.suffix == ".yaml":
        return True
    if name.startswith("src/chemworld/"):
        return (
            path.suffix == ".py"
            or name in {"src/chemworld/data/parameters/incompressible.json",
                        "src/chemworld/data/parameters/CoolProp-MIT.txt",
                        "src/chemworld/data/parameters/battery_chen2020_pybamm_25_4_2.json"}
            or (name.startswith("src/chemworld/schemas/") and path.suffix == ".json")
            or (name.startswith("src/chemworld/physchem/atomic_data/")
                and path.suffix in {".csv", ".json"})
        )
    return False


def selected_files(root: Path) -> list[str]:
    index = root / SOURCE_INDEX
    # Do not use a parent repository accidentally (e.g. an unpacked sdist in /tmp).
    if (root / ".git").exists():
        names = subprocess.check_output(
            ["git", "ls-files", "-z"], cwd=root, text=True, encoding="utf-8",
        ).split("\0")
    elif index.is_file():
        names = json.loads(index.read_text(encoding="utf-8"))
        if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
            raise ValueError("Invalid community source index")
        if any(not public_path(name) for name in names):
            raise ValueError("Source index contains a non-public path")
    else:
        raise ValueError("Build from a Git checkout or generated sdist, not a GitHub source ZIP.")
    selected = sorted({name for name in names if public_path(name)})
    for name in selected:
        path = root / name
        if (
            not path.is_file()
            or path.is_symlink()
            or not path.resolve().is_relative_to(root.resolve())
        ):
            raise ValueError(f"Public distribution file missing or unsafe: {name}")
    required = {BUILD_HOOK, "src/chemworld/__init__.py", *RUNTIME_CONFIGS}
    if not required.issubset(selected):
        raise ValueError(f"Missing required distribution files: {sorted(required - set(selected))}")
    return selected


def wheel_destination(name: str) -> str | None:
    if name.startswith("src/chemworld/"):
        return name.removeprefix("src/")
    if name.startswith("configs/"):
        return "chemworld/resources/" + name
    return None


def get_build_hook():
    from hatchling.builders.hooks.plugin.interface import BuildHookInterface

    class CommunityBuildHook(BuildHookInterface):
        def initialize(self, version, build_data):
            # Editable development uses source discovery; no generated resources
            # should shadow changes made by the core team in the same checkout.
            if version == "editable":
                return
            root = Path(self.root)
            names = selected_files(root)
            forced = build_data.setdefault("force_include", {})
            if self.target_name == "wheel":
                for name in names:
                    destination = wheel_destination(name)
                    if destination:
                        forced[str(root / name)] = destination
            elif self.target_name == "sdist":
                for name in names:
                    forced[str(root / name)] = name
                self._index_directory = tempfile.TemporaryDirectory(prefix="chemworld-sdist-")
                index = Path(self._index_directory.name) / "source-files.json"
                index.write_text(json.dumps(names, indent=2) + "\n", encoding="utf-8")
                forced[str(index)] = SOURCE_INDEX

        def finalize(self, version, build_data, artifact_path):
            if hasattr(self, "_index_directory"):
                self._index_directory.cleanup()

    return CommunityBuildHook
