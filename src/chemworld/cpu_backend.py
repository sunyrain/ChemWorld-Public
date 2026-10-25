"""Optional C ABI CPU boundary with an explicit reference fallback.

The shared library is deliberately loaded only on request.  Importing the
base package never requires a compiler, a DLL or a native runtime dependency.
"""

from __future__ import annotations

import ctypes
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

EXPECTED_ABI = "chemworld-cpu-abi-1"
OPERATION_OBSERVE = 0
OPERATION_ADD = 1
OPERATION_SCALE = 2
_LIBRARY_CACHE: dict[Path, ctypes.CDLL] = {}


class CpuBackendUnavailableError(RuntimeError):
    """Raised when callers explicitly request a CPU library that is absent."""


CpuBackendUnavailable = CpuBackendUnavailableError


class CpuBackendError(RuntimeError):
    """Raised for a native operation failure after the library is loaded."""


def _library_names() -> tuple[str, ...]:
    if sys.platform == "win32":
        return ("chemworld_cpu.dll", "libchemworld_cpu.dll")
    if sys.platform == "darwin":
        return ("libchemworld_cpu.dylib", "chemworld_cpu.dylib")
    return ("libchemworld_cpu.so", "chemworld_cpu.so")


def _candidate_paths() -> tuple[Path, ...]:
    candidates: list[Path] = []
    configured = os.environ.get("CHEMWORLD_CPU_LIBRARY")
    if configured:
        candidates.append(Path(configured).expanduser())
    package = Path(__file__).resolve().parent
    candidates.extend(package / name for name in _library_names())
    install_roots = (
        package.parent.parent / "build" / "chemworld-cpu" / "install",
        package.parent.parent / "native" / "chemworld_cpu" / "install",
    )
    for root in install_roots:
        candidates.extend(
            root / part / name
            for part in ("bin", "lib", "")
            for name in _library_names()
        )
    return tuple(dict.fromkeys(path.resolve() for path in candidates))


def _load_library(path: str | os.PathLike[str] | None = None) -> tuple[ctypes.CDLL, Path]:
    paths = (Path(path).expanduser().resolve(),) if path is not None else _candidate_paths()
    errors = []
    for candidate in paths:
        if not candidate.is_file():
            continue
        cached = _LIBRARY_CACHE.get(candidate)
        if cached is not None:
            return cached, candidate
        try:
            library = ctypes.CDLL(str(candidate))
        except OSError as exc:
            errors.append(f"{candidate}: {exc}")
            continue
        library.cw_cpu_abi_version.argtypes = []
        library.cw_cpu_abi_version.restype = ctypes.c_char_p
        abi = library.cw_cpu_abi_version()
        if abi is None or abi.decode("ascii") != EXPECTED_ABI:
            errors.append(f"{candidate}: unsupported ABI {abi!r}")
            continue
        _configure_functions(library)
        _LIBRARY_CACHE[candidate] = library
        return library, candidate
    details = "; ".join(errors) if errors else "no compiled chemworld_cpu library was found"
    raise CpuBackendUnavailable(
        f"CPU backend unavailable ({details}); build it with scripts/build_cpu.py"
    )


def _configure_functions(library: ctypes.CDLL) -> None:
    world = ctypes.c_void_p
    library.cw_cpu_last_error.argtypes = []
    library.cw_cpu_last_error.restype = ctypes.c_char_p
    library.cw_world_create.argtypes = [ctypes.POINTER(world)]
    library.cw_world_create.restype = ctypes.c_int
    library.cw_world_destroy.argtypes = [world]
    library.cw_world_destroy.restype = None
    library.cw_world_reset.argtypes = [world, ctypes.c_uint64]
    library.cw_world_reset.restype = ctypes.c_int
    library.cw_world_observe.argtypes = [
        world,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
    ]
    library.cw_world_observe.restype = ctypes.c_int
    library.cw_world_step.argtypes = [
        world,
        ctypes.c_uint32,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_double),
    ]
    library.cw_world_step.restype = ctypes.c_int
    library.cw_world_step_batch.argtypes = [
        ctypes.POINTER(world),
        ctypes.c_size_t,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    library.cw_world_step_batch.restype = ctypes.c_int


@dataclass(frozen=True)
class CpuObservation:
    value: float
    steps: int
    seed: int


def _error(library: ctypes.CDLL, status: int) -> CpuBackendError:
    message = library.cw_cpu_last_error()
    detail = "" if message is None else f": {message.decode('utf-8', errors='replace')}"
    return CpuBackendError(f"CPU operation failed with status {status}{detail}")


class CpuWorld:
    """Thin owner of one opaque native world handle."""

    def __init__(self, library: ctypes.CDLL, handle: ctypes.c_void_p):
        self._library = library
        self._handle = handle

    @classmethod
    def create(cls, path: str | os.PathLike[str] | None = None) -> CpuWorld:
        library, _ = _load_library(path)
        handle = ctypes.c_void_p()
        status = library.cw_world_create(ctypes.byref(handle))
        if status != 0:
            raise _error(library, status)
        return cls(library, handle)

    def close(self) -> None:
        if self._handle:
            self._library.cw_world_destroy(self._handle)
            self._handle = ctypes.c_void_p()

    def __enter__(self) -> CpuWorld:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __del__(self):  # pragma: no cover - interpreter shutdown ordering is platform-specific
        self.close()

    def reset(self, seed: int = 0) -> None:
        status = self._library.cw_world_reset(self._handle, ctypes.c_uint64(seed))
        if status != 0:
            raise _error(self._library, status)

    def observe(self) -> CpuObservation:
        value = ctypes.c_double()
        steps = ctypes.c_uint64()
        seed = ctypes.c_uint64()
        status = self._library.cw_world_observe(
            self._handle, ctypes.byref(value), ctypes.byref(steps), ctypes.byref(seed)
        )
        if status != 0:
            raise _error(self._library, status)
        return CpuObservation(value.value, steps.value, seed.value)

    def step(self, operation: int, input_value: float) -> float:
        value = ctypes.c_double()
        status = self._library.cw_world_step(
            self._handle,
            ctypes.c_uint32(operation),
            ctypes.c_double(input_value),
            ctypes.byref(value),
        )
        if status != 0:
            raise _error(self._library, status)
        return value.value

    @staticmethod
    def step_batch(
        worlds: tuple[CpuWorld, ...] | list[CpuWorld],
        operation: int,
        inputs: tuple[float, ...] | list[float],
    ) -> tuple[float, ...]:
        if len(worlds) != len(inputs):
            raise ValueError("batch worlds and inputs must have matching lengths")
        if not worlds:
            return ()
        library = worlds[0]._library
        if any(world._library is not library for world in worlds):
            raise ValueError("batch worlds must use the same loaded library")
        handles = (ctypes.c_void_p * len(worlds))(*(world._handle for world in worlds))
        values_in = (ctypes.c_double * len(inputs))(*inputs)
        values_out = (ctypes.c_double * len(worlds))()
        status = library.cw_world_step_batch(
            handles,
            len(worlds),
            ctypes.c_uint32(operation),
            values_in,
            values_out,
        )
        if status != 0:
            raise _error(library, status)
        return tuple(values_out)


def cpu_status(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    try:
        _, loaded = _load_library(path)
    except CpuBackendUnavailable as exc:
        return {"available": False, "backend": "reference", "reason": str(exc)}
    return {"available": True, "backend": "cpu", "abi": EXPECTED_ABI, "library": str(loaded)}


def select_backend(
    preference: str = "auto", path: str | os.PathLike[str] | None = None
) -> str:
    """Select one whole executor; never fall back after a CPU step fails."""
    if preference not in {"auto", "cpu", "reference"}:
        raise ValueError("backend preference must be auto, cpu or reference")
    if preference == "reference":
        return "reference"
    try:
        _load_library(path)
    except CpuBackendUnavailable:
        if preference == "cpu":
            raise
        return "reference"
    return "cpu"


__all__ = [
    "EXPECTED_ABI",
    "OPERATION_ADD",
    "OPERATION_OBSERVE",
    "OPERATION_SCALE",
    "CpuBackendError",
    "CpuBackendUnavailable",
    "CpuBackendUnavailableError",
    "CpuObservation",
    "CpuWorld",
    "cpu_status",
    "select_backend",
]
