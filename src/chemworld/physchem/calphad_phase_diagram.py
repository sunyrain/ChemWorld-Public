"""Finite binary CEF boundary analysis on the original equilibrium solver."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from math import isfinite
from typing import Any, cast

from chemworld.physchem.calphad_equilibrium import (
    CalphadEquilibriumPhase,
    CalphadEquilibriumResult,
    CalphadEquilibriumSpec,
    calphad_tp_equilibrium,
)


@dataclass(frozen=True)
class CalphadTracePolicy:
    maximum_temperature_step_K: float = 50.0
    minimum_temperature_step_K: float = 0.005
    maximum_evaluations: int = 512
    endpoint_probe_K: float = 2.0

    def __post_init__(self) -> None:
        if (
            not 0 < self.minimum_temperature_step_K < self.maximum_temperature_step_K
            or not isfinite(self.maximum_temperature_step_K)
            or not isfinite(self.endpoint_probe_K)
            or self.endpoint_probe_K <= 0
            or type(self.maximum_evaluations) is not int
            or self.maximum_evaluations < 1
        ):
            raise ValueError("phase trace requires finite steps and a positive TP budget")


@dataclass(frozen=True)
class CriticalPointRequest:
    phase_name: str
    temperature_K: float
    composition: float
    temperature_bounds_K: tuple[float, float]
    composition_bounds: tuple[float, float]


@dataclass(frozen=True)
class PhaseCrossingRequest:
    phase_names: tuple[str, str]
    composition: float
    temperature_K: float
    temperature_bounds_K: tuple[float, float]


@dataclass(frozen=True)
class CalphadDiagramSpec:
    equilibrium: CalphadEquilibriumSpec
    components: tuple[str, str]
    pressure_Pa: float
    temperature_bounds_K: tuple[float, float]
    scan_temperatures_K: tuple[float, ...]
    scan_compositions: tuple[float, ...]
    boundary_temperatures_K: tuple[float, ...] = ()
    coexistence_seeds: tuple[tuple[float, float], ...] = ()
    critical_requests: tuple[CriticalPointRequest, ...] = ()
    crossing_requests: tuple[PhaseCrossingRequest, ...] = ()
    policy: CalphadTracePolicy = field(default_factory=CalphadTracePolicy)
    window_compositions: tuple[float, ...] = (0.25,)

    def __post_init__(self) -> None:
        for name in (
            "components",
            "temperature_bounds_K",
            "scan_temperatures_K",
            "scan_compositions",
            "boundary_temperatures_K",
            "coexistence_seeds",
            "critical_requests",
            "crossing_requests",
            "window_compositions",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        low, high = self.temperature_bounds_K
        if (
            len(self.components) != 2
            or len(set(self.components)) != 2
            or not 0 < low < high
            or not isfinite(high)
            or not isfinite(self.pressure_Pa)
            or self.pressure_Pa <= 0
            or not self.scan_temperatures_K
            or not self.scan_compositions
        ):
            raise ValueError("binary phase diagram requires explicit components and finite domain")
        for t in (
            *self.scan_temperatures_K,
            *self.boundary_temperatures_K,
            *(p[0] for p in self.coexistence_seeds),
        ):
            if not isfinite(t) or not low <= t <= high:
                raise ValueError("diagram temperature outside declared domain")
        for x in (
            *self.scan_compositions,
            *self.window_compositions,
            *(p[1] for p in self.coexistence_seeds),
        ):
            if not isfinite(x) or not 0 <= x <= 1:
                raise ValueError("diagram composition must be an actual atom fraction")
        names = {m.phase_name for m in self.equilibrium.models}
        for model in self.equilibrium.models:
            if len(model.constituents) != 1 or len(model.constituents[0]) != 2:
                raise ValueError("this diagram requires a single binary substitutional sublattice")
            species = cast(Mapping[str, Any], model.database.species)
            if any(
                tuple(species[s].constituents) != ((s, 1.0),) for s in model.constituents[0]
            ) or set(model.constituents[0]) != set(self.components):
                raise ValueError("diagram coordinates require matching monatomic elemental sites")
        requests: tuple[CriticalPointRequest | PhaseCrossingRequest, ...] = (
            *self.critical_requests,
            *self.crossing_requests,
        )
        for request in requests:
            a, b = request.temperature_bounds_K
            if not low <= a < request.temperature_K < b <= high or not 0 < request.composition < 1:
                raise ValueError("endpoint seed requires interior declared T/x bounds")
            if isinstance(request, CriticalPointRequest):
                c, d = request.composition_bounds
                if request.phase_name not in names or not 0 < c < request.composition < d < 1:
                    raise ValueError(
                        "critical request requires allowed phase and interior composition"
                    )
            elif len(set(request.phase_names)) != 2 or set(request.phase_names) - names:
                raise ValueError("crossing requires two distinct allowed phases")

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": "calphad_binary_diagram_v1",
            **{
                k: getattr(self, k)
                for k in self.__dataclass_fields__
                if k not in {"equilibrium", "policy", "critical_requests", "crossing_requests"}
            },
            "equilibrium": self.equilibrium.to_dict(),
            "policy": asdict(self.policy),
            "critical_requests": [asdict(r) for r in self.critical_requests],
            "crossing_requests": [asdict(r) for r in self.crossing_requests],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CalphadDiagramSpec:
        expected = {"model_type", *cls.__dataclass_fields__}
        if set(data) != expected or data["model_type"] != "calphad_binary_diagram_v1":
            raise ValueError("unknown or incomplete binary diagram declaration")
        values = {k: v for k, v in data.items() if k != "model_type"}
        values.update(
            equilibrium=CalphadEquilibriumSpec.from_dict(values["equilibrium"]),
            critical_requests=tuple(CriticalPointRequest(**r) for r in values["critical_requests"]),
            crossing_requests=tuple(PhaseCrossingRequest(**r) for r in values["crossing_requests"]),
            policy=CalphadTracePolicy(**values["policy"]),
        )
        return cls(**values)


@dataclass(frozen=True)
class CalphadDiagramResult:
    specification: CalphadDiagramSpec
    scan_points: tuple[dict[str, Any], ...]
    boundaries: tuple[dict[str, Any], ...]
    endpoints: tuple[dict[str, Any], ...]
    branches: tuple[dict[str, Any], ...]
    windows: tuple[dict[str, Any], ...]
    attempts: tuple[dict[str, Any], ...]

    @property
    def complete(self) -> bool:
        return all(
            p["success"]
            for p in (*self.scan_points, *self.boundaries, *self.endpoints, *self.windows)
        ) and all(b["complete"] for b in self.branches)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": "calphad_binary_diagram_result_v1",
            "complete": self.complete,
            "specification": self.specification.to_dict(),
            **{
                k: getattr(self, k)
                for k in (
                    "scan_points",
                    "boundaries",
                    "endpoints",
                    "branches",
                    "windows",
                    "attempts",
                )
            },
            "scope": "finite_declared_points_starts_and_branches; no_global_certificate",
            "physical_time_s": None,
            "heat_duty_J": None,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CalphadDiagramResult:
        expected = {
            "model_type",
            "complete",
            "specification",
            "scan_points",
            "boundaries",
            "endpoints",
            "branches",
            "windows",
            "attempts",
            "scope",
            "physical_time_s",
            "heat_duty_J",
        }
        if set(data) != expected or data["model_type"] != "calphad_binary_diagram_result_v1":
            raise ValueError("invalid diagram result encoding")
        result = cls(
            CalphadDiagramSpec.from_dict(data["specification"]),
            *(
                tuple(data[k])
                for k in (
                    "scan_points",
                    "boundaries",
                    "endpoints",
                    "branches",
                    "windows",
                    "attempts",
                )
            ),
        )
        if (
            data["complete"] != result.complete
            or data["physical_time_s"] is not None
            or data["heat_duty_J"] is not None
        ):
            raise ValueError("diagram is a numerical boundary tool, not a physical history")
        return result


class _DiagramEvaluator:
    def __init__(
        self, spec: CalphadDiagramSpec, progress: Callable[[dict[str, Any]], None] | None = None
    ) -> None:
        self.spec, self.progress = spec, progress
        self.attempts: list[dict[str, Any]] = []
        self.evaluations = 0

    def at(
        self,
        temperature: float,
        x: float,
        *,
        warm: tuple[CalphadEquilibriumPhase, ...] = (),
        purpose: str = "scan",
        equilibrium: CalphadEquilibriumSpec | None = None,
    ) -> CalphadEquilibriumResult:
        if self.evaluations >= self.spec.policy.maximum_evaluations:
            result = CalphadEquilibriumResult(False, failure_reason="diagram_TP_budget_exhausted")
        else:
            self.evaluations += 1
            try:
                result = calphad_tp_equilibrium(
                    equilibrium or self.spec.equilibrium,
                    temperature,
                    self.spec.pressure_Pa,
                    dict(zip(self.spec.components, (x, 1 - x), strict=True)),
                    initial_phases=warm or None,
                )
            except (ValueError, ArithmeticError) as exc:
                result = CalphadEquilibriumResult(False, failure_reason=str(exc))
        self.attempts.append(
            {
                "temperature_K": temperature,
                "composition": x,
                "purpose": purpose,
                "equilibrium": result.to_dict(),
            }
        )
        if self.progress is not None:
            self.progress(
                {"TP_evaluations": self.evaluations, "purpose": purpose, "success": result.success}
            )
        return result


def scan_calphad_phase_diagram(
    spec: CalphadDiagramSpec,
    *,
    direction: int = 1,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> CalphadDiagramResult:
    from chemworld.physchem.calphad_phase_trace import (
        _probe_composition,
        boundary_point,
        locate_critical,
        locate_crossing,
        split_windows,
        trace_branch,
    )

    if type(direction) is not int or direction not in (-1, 1):
        raise ValueError("diagram direction must be +1 or -1")
    evaluate = _DiagramEvaluator(spec, progress)
    rows: list[dict[str, Any]] = []
    warm: tuple[CalphadEquilibriumPhase, ...] = ()
    for t in sorted(spec.scan_temperatures_K, reverse=direction < 0):
        discovery = {
            x: evaluate.at(t, x, warm=warm, purpose="declared_composition_discovery")
            for x in dict.fromkeys(p[1] for p in spec.coexistence_seeds)
        }
        seeds = tuple(p for r in discovery.values() if r.success for p in r.phases)
        for x in sorted(spec.scan_compositions, reverse=direction < 0):
            result = discovery[x] if x in discovery else evaluate.at(t, x, warm=seeds + warm)
            rows.append(
                {
                    "temperature_K": t,
                    "composition": x,
                    "success": result.success,
                    "equilibrium": result.to_dict(),
                }
            )
            if result.success:
                warm = result.phases
    endpoints = [locate_critical(evaluate, r, i) for i, r in enumerate(spec.critical_requests)]
    endpoints += [locate_crossing(evaluate, r, i) for i, r in enumerate(spec.crossing_requests)]
    branches = [
        trace_branch(evaluate, seed, endpoints, direction, i)
        for i, seed in enumerate(spec.coexistence_seeds)
    ]
    boundaries = []
    for t in spec.boundary_temperatures_K:
        points = [p for b in branches for path in b["paths"] for p in path["points"]]
        if not points:
            result = CalphadEquilibriumResult(False, failure_reason="no_resolved_boundary_seed")
        else:
            result = evaluate.at(
                t, _probe_composition(points, t), purpose="requested_boundary_anchor")
        boundaries.append(boundary_point(t, result, spec.components[0]))
    windows = [split_windows(evaluate, branches, x) for x in spec.window_compositions]
    return CalphadDiagramResult(
        spec,
        tuple(rows),
        tuple(boundaries),
        tuple(endpoints),
        tuple(branches),
        tuple(windows),
        tuple(evaluate.attempts),
    )
