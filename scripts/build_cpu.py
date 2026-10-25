"""Configure, build and optionally test the isolated ChemWorld CPU library."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def run(command: list[str], *, cwd: Path) -> None:
    print("[chemworld-cpu]", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--build-dir", type=Path, default=None)
    parser.add_argument("--install-dir", type=Path, default=None)
    parser.add_argument(
        "--config", choices=("Debug", "Release", "RelWithDebInfo"), default="Release"
    )
    parser.add_argument(
        "--generator", default=None, help="CMake generator, e.g. Visual Studio 17 2022"
    )
    parser.add_argument(
        "--test", action="store_true", help="Build and run the native ABI smoke test"
    )
    args = parser.parse_args()

    root = args.source_root.resolve()
    source = root / "native" / "chemworld_cpu"
    build = (args.build_dir or root / "build" / "chemworld-cpu").resolve()
    install = (args.install_dir or build / "install").resolve()
    if not source.is_dir():
        raise SystemExit(f"native CPU source directory does not exist: {source}")
    configure = [
        "cmake",
        "-S",
        str(source),
        "-B",
        str(build),
        f"-DCMAKE_INSTALL_PREFIX={install}",
        f"-DCMAKE_BUILD_TYPE={args.config}",
        f"-DCHEMWORLD_CPU_BUILD_TESTS={'ON' if args.test else 'OFF'}",
    ]
    if args.generator:
        configure.extend(["-G", args.generator])
    run(configure, cwd=root)
    run(["cmake", "--build", str(build), "--config", args.config, "--target", "install"], cwd=root)
    if args.test:
        run(["ctest", "--test-dir", str(build), "-C", args.config, "--output-on-failure"], cwd=root)
    manifest = {
        "schema": "chemworld-cpu-build-v1",
        "abi": "chemworld-cpu-abi-1",
        "config": args.config,
        "source": str(source),
        "build": str(build),
        "install": str(install),
        "tests": args.test,
    }
    install.mkdir(parents=True, exist_ok=True)
    (install / "chemworld_cpu_build.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
