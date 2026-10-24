"""Finite ternary CEF sections; geometry never interpolates across failed nodes."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from itertools import pairwise
from math import fsum, isfinite
from time import monotonic
from typing import Any, cast

from chemworld.physchem.calphad_equilibrium import (
    CalphadEquilibriumResult,
    CalphadEquilibriumSpec,
    calphad_tp_equilibrium,
)

Composition = tuple[float, float, float]
_SCOPE = "finite_declared_nodes; brackets_not_continuous_boundaries; no_global_certificate"


def _composition(value: Any) -> Composition:
    values = tuple(value)
    if (
        len(values) != 3
        or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not isfinite(v) or v < 0
            for v in values
        )
        or abs(fsum(values) - 1) > 1e-12
    ):
        raise ValueError(
            "ternary coordinates require three finite nonnegative atom fractions summing to one"
        )
    return cast(Composition, tuple(float(v) for v in values))


@dataclass(frozen=True)
class CalphadTernaryPath:
    name: str
    compositions: tuple[Composition, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "compositions", tuple(_composition(x) for x in self.compositions))
        if not isinstance(self.name, str) or not self.name or len(self.compositions) < 2:
            raise ValueError("ternary path needs a name and at least two explicit compositions")


@dataclass(frozen=True)
class CalphadTernarySpec:
    equilibrium: CalphadEquilibriumSpec
    components: tuple[str, str, str]
    temperature_K: float
    pressure_Pa: float
    grid_intervals: int = 6
    requested_compositions: tuple[Composition, ...] = ()
    paths: tuple[CalphadTernaryPath, ...] = ()
    maximum_evaluations: int = 256

    def __post_init__(self) -> None:
        object.__setattr__(self, "components", tuple(self.components))
        object.__setattr__(
            self,
            "requested_compositions",
            tuple(_composition(x) for x in self.requested_compositions),
        )
        object.__setattr__(self, "paths", tuple(self.paths))
        if (
            len(self.components) != 3
            or len(set(self.components)) != 3
            or any(not isinstance(e, str) or not e for e in self.components)
            or not isfinite(self.temperature_K)
            or self.temperature_K <= 0
            or not isfinite(self.pressure_Pa)
            or self.pressure_Pa <= 0
            or type(self.grid_intervals) is not int
            or not 1 <= self.grid_intervals <= 100
            or type(self.maximum_evaluations) is not int
            or self.maximum_evaluations < 1
            or any(not isinstance(p, CalphadTernaryPath) for p in self.paths)
            or len({p.name for p in self.paths}) != len(self.paths)
        ):
            raise ValueError(
                "ternary section requires three components, finite T/P and finite grid/budget"
            )
        elements: set[str] = set()
        for model in self.equilibrium.models:
            species = cast(Mapping[str, Any], model.database.species)
            elements.update(
                e
                for sites in model.constituents
                for s in sites
                for e, n in species[s].constituents
                if n > 0
            )
        if elements != set(self.components):
            raise ValueError(
                "ternary components must equal the allowed CEF models' elemental support"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": "calphad_ternary_section_v1",
            "equilibrium": self.equilibrium.to_dict(),
            "components": list(self.components),
            "temperature_K": self.temperature_K,
            "pressure_Pa": self.pressure_Pa,
            "grid_intervals": self.grid_intervals,
            "requested_compositions": [list(x) for x in self.requested_compositions],
            "paths": [
                {"name": p.name, "compositions": [list(x) for x in p.compositions]}
                for p in self.paths
            ],
            "maximum_evaluations": self.maximum_evaluations,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CalphadTernarySpec:
        if (
            set(data) != {"model_type", *cls.__dataclass_fields__}
            or data["model_type"] != "calphad_ternary_section_v1"
        ):
            raise ValueError("invalid ternary section declaration")
        values = {k: v for k, v in data.items() if k != "model_type"}
        values["equilibrium"] = CalphadEquilibriumSpec.from_dict(data["equilibrium"])
        values["paths"] = tuple(CalphadTernaryPath(**p) for p in data["paths"])
        return cls(**values)


def _layout(
    spec: CalphadTernarySpec,
) -> tuple[
    list[Composition],
    list[list[str]],
    list[tuple[int, int]],
    list[tuple[int, int, int]],
    list[dict[str, Any]],
]:
    points: list[Composition] = []
    origins: list[list[str]] = []
    lookup: dict[Composition, int] = {}

    def add(x: Composition, origin: str) -> int:
        # Exact tuple equality only: near-but-different user points remain distinct.
        if x not in lookup:
            lookup[x] = len(points)
            points.append(x)
            origins.append([])
        idx = lookup[x]
        if origin not in origins[idx]:
            origins[idx].append(origin)
        return idx

    n = spec.grid_intervals
    lattice = {
        (i, j): add((i / n, j / n, (n - i - j) / n), "grid")
        for i in range(n + 1)
        for j in range(n + 1 - i)
    }
    edges: set[tuple[int, int]] = set()
    cells: list[tuple[int, int, int]] = []
    for (i, j), idx in lattice.items():
        for step in ((i + 1, j), (i, j + 1), (i + 1, j - 1)):
            if step in lattice:
                edges.add((min(idx, lattice[step]), max(idx, lattice[step])))
        if (i + 1, j) in lattice and (i, j + 1) in lattice:
            cells.append((idx, lattice[i + 1, j], lattice[i, j + 1]))
        if (i + 1, j) in lattice and (i, j + 1) in lattice and (i + 1, j + 1) in lattice:
            cells.append((lattice[i + 1, j], lattice[i, j + 1], lattice[i + 1, j + 1]))
    for index, x in enumerate(spec.requested_compositions):
        add(x, f"requested:{index}")
    paths = [
        {"name": p.name, "point_indices": [add(x, f"path:{p.name}") for x in p.compositions]}
        for p in spec.paths
    ]
    return points, origins, sorted(edges), cells, paths


def _edge(points: tuple[dict[str, Any], ...], indices: tuple[int, int]) -> dict[str, Any]:
    a, b = (points[i] for i in indices)
    resolved = a["success"] and b["success"]
    return {
        "point_indices": list(indices),
        "success": resolved,
        "kind": (
            "assemblage_bracket"
            if a["phase_signature"] != b["phase_signature"]
            else "same_sampled_assemblage"
        )
        if resolved
        else "unresolved_edge",
        "failed_point_indices": [i for i in indices if not points[i]["success"]],
        "continuous_segment_certified": False,
    }


def _geometry(spec: CalphadTernarySpec, points: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    _, _, links, cells, paths = _layout(spec)
    edges = [_edge(points, edge) for edge in links]
    for path in paths:
        ids = path["point_indices"]
        path["edges"] = [_edge(points, (a, b)) for a, b in pairwise(ids)]
        path["complete"] = all(points[i]["success"] for i in ids)
    ties: list[dict[str, Any]] = []
    for index, point in enumerate(points):
        phases = point["equilibrium"]["phases"]
        if point["success"] and len(phases) in (2, 3):
            ties.append(
                {
                    "point_index": index,
                    "kind": "tie_line" if len(phases) == 2 else "tie_triangle",
                    "vertices": [
                        {
                            "phase_name": p["phase_name"],
                            "instance": j,
                            "composition": [
                                p["element_fractions"].get(e, 0.0) for e in spec.components
                            ],
                            "fraction": p["fraction"],
                        }
                        for j, p in enumerate(phases)
                    ],
                }
            )
    return {
        "grid_edges": edges,
        "grid_cells": [
            {
                "point_indices": list(c),
                "success": all(points[i]["success"] for i in c),
                "continuous_interior_certified": False,
            }
            for c in cells
        ],
        "boundary_brackets": [e for e in edges if e["kind"] == "assemblage_bracket"],
        "tie_geometry": ties,
        "paths": paths,
    }


@dataclass(frozen=True)
class CalphadTernaryResult:
    specification: CalphadTernarySpec
    points: tuple[dict[str, Any], ...]

    @property
    def complete(self) -> bool:
        """All declared finite nodes solved; this is not a global map certificate."""
        return bool(self.points) and all(p["success"] for p in self.points)

    @property
    def geometry(self) -> dict[str, Any]:
        return _geometry(self.specification, self.points)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": "calphad_ternary_section_result_v1",
            "complete": self.complete,
            "specification": self.specification.to_dict(),
            "points": list(self.points),
            **self.geometry,
            "scope": _SCOPE,
            "physical_time_s": None,
            "heat_duty_J": None,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CalphadTernaryResult:
        expected = {
            "model_type",
            "complete",
            "specification",
            "points",
            "grid_edges",
            "grid_cells",
            "boundary_brackets",
            "tie_geometry",
            "paths",
            "scope",
            "physical_time_s",
            "heat_duty_J",
        }
        if set(data) != expected or data["model_type"] != "calphad_ternary_section_result_v1":
            raise ValueError("invalid ternary result encoding")
        json.dumps(dict(data), allow_nan=False)
        spec = CalphadTernarySpec.from_dict(data["specification"])
        coordinates, origins, _, _, _ = _layout(spec)
        points = tuple(data["points"])
        if len(coordinates) != len(points):
            raise ValueError("ternary result must retain every declared point")
        for i, (point, x) in enumerate(zip(points, coordinates, strict=True)):
            if (
                set(point)
                != {"index", "composition", "origins", "success", "phase_signature", "equilibrium"}
                or point["index"] != i
                or tuple(point["composition"]) != x
                or point["origins"] != origins[i]
                or type(point["success"]) is not bool
                or point["success"] != point["equilibrium"]["success"]
                or point["phase_signature"]
                != sorted(p["phase_name"] for p in point["equilibrium"]["phases"])
            ):
                raise ValueError("ternary point disagrees with declared composition or equilibrium")
        result = cls(spec, points)
        if (
            data["complete"] != result.complete
            or data["scope"] != _SCOPE
            or data["physical_time_s"] is not None
            or data["heat_duty_J"] is not None
            or any(data[k] != v for k, v in result.geometry.items())
        ):
            raise ValueError(
                "ternary geometry must reflect solved nodes, without invented heat/time"
            )
        return result


def scan_calphad_ternary_diagram(
    specification: CalphadTernarySpec,
    *,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> CalphadTernaryResult:
    coordinates, origins, _, _, _ = _layout(specification)
    points: list[dict[str, Any]] = []
    start = monotonic()
    for i, x in enumerate(coordinates):
        if i >= specification.maximum_evaluations:
            equilibrium = CalphadEquilibriumResult(
                False, failure_reason="ternary_TP_budget_exhausted"
            )
        else:
            try:
                equilibrium = calphad_tp_equilibrium(
                    specification.equilibrium,
                    specification.temperature_K,
                    specification.pressure_Pa,
                    dict(zip(specification.components, x, strict=True)),
                )
            except (ValueError, ArithmeticError) as exc:
                equilibrium = CalphadEquilibriumResult(
                    False, failure_reason=f"{type(exc).__name__}: {exc}"
                )
        # Normalize tuples for a JSON-native, finite public representation.
        encoded = json.loads(json.dumps(equilibrium.to_dict(), allow_nan=False))
        points.append(
            {
                "index": i,
                "composition": list(x),
                "origins": origins[i],
                "success": equilibrium.success,
                "phase_signature": sorted(p.phase_name for p in equilibrium.phases),
                "equilibrium": encoded,
            }
        )
        if progress is not None:
            elapsed = max(monotonic() - start, 1e-12)
            progress(
                {
                    "stage": "ternary_TP",
                    "completed": i + 1,
                    "total": len(coordinates),
                    "points_per_second": (i + 1) / elapsed,
                    "eta_s": elapsed * (len(coordinates) - i - 1) / (i + 1),
                    "point_success": equilibrium.success,
                }
            )
    return CalphadTernaryResult(specification, tuple(points))
