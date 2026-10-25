"""Local molecular coexistence boundaries and genuine pseudo-arclength curves.

Bubble fixes the liquid composition; dew fixes the vapor composition. Both
differentiate the same residual Helmholtz model. Finite TPD qualification is
explicit and distinct from following a locally stable coexistence branch.
No envelope is identified with a global phase diagram or an inventory split.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from math import exp, isfinite, log
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.special import softmax

from chemworld.physchem.molecular_consumers import (
    molecular_model_from_dict,
    molecular_model_to_dict,
)
from chemworld.physchem.molecular_critical import (
    molecular_composition_response,
    solve_molecular_critical,
)
from chemworld.physchem.molecular_eos import (
    MolecularEOSState,
    ResidualHelmholtzModel,
    molecular_eos_state,
    molecular_residual_chemical_potentials,
)
from chemworld.physchem.molecular_flash import MolecularDensityBasin, molecular_tp_stability
from chemworld.physchem.residual_system import (
    ResidualDomainError,
    ResidualVariable,
    RootPolicy,
    finite_jacobian,
    solve_root,
)
from chemworld.physchem.second_order_pair import SecondOrderPair

FloatArray = NDArray[np.float64]
BOUNDARY_POLICY = RootPolicy(tolerance=2e-10, maximum_iterations=80)


def _strict(data: Mapping[str, Any], fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(data, Mapping) or set(data) != fields:
        raise ValueError(f"{label} requires exactly {sorted(fields)}")
    return dict(data)


def _positive(value: float) -> bool:
    return not isinstance(value, bool) and isfinite(value) and value > 0


def _fractions(values: Sequence[float], size: int) -> tuple[float, ...]:
    xs = tuple(values)
    if (
        len(xs) != size
        or any(isinstance(x, bool) or not isfinite(x) or x < 0 for x in xs)
        or abs(sum(xs) - 1) > 1e-12
    ):
        raise ValueError("boundary composition must be explicitly normalized and nonnegative")
    return xs


@dataclass(frozen=True)
class MolecularPhaseBoundarySpec:
    model: ResidualHelmholtzModel
    fixed_composition: tuple[float, ...]
    kind: str
    liquid_basin: MolecularDensityBasin
    vapor_basin: MolecularDensityBasin
    temperature_bounds_K: tuple[float, float]
    pressure_bounds_Pa: tuple[float, float]
    stability_mode: str = "finite_tpd"
    composition_starts: tuple[tuple[float, ...], ...] | None = None
    stability_max_iterations: int = 200
    tpd_tolerance: float = 1e-8
    coexistence_tolerance: float = 2e-7
    minimum_phase_separation: float = 1e-6

    def __post_init__(self) -> None:
        ids = self.model.component_ids
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("boundary requires distinct model component identities")
        object.__setattr__(self, "fixed_composition", _fractions(self.fixed_composition, len(ids)))
        for name in ("temperature_bounds_K", "pressure_bounds_Pa"):
            pair = tuple(getattr(self, name))
            if len(pair) != 2 or not all(_positive(v) for v in pair) or pair[0] >= pair[1]:
                raise ValueError("boundary requires positive ordered T and P bounds")
            object.__setattr__(self, name, pair)
        if self.kind not in {"bubble", "dew"}:
            raise ValueError("boundary kind must be bubble or dew")
        if self.liquid_basin.name == self.vapor_basin.name:
            raise ValueError("liquid and vapor require distinct declared basin identities")
        if self.stability_mode not in {"finite_tpd", "local_coexistence"}:
            raise ValueError("explicit finite_tpd or local_coexistence stability mode required")
        if (
            type(self.stability_max_iterations) is not int
            or self.stability_max_iterations < 1
            or not all(
                _positive(v)
                for v in (
                    self.tpd_tolerance,
                    self.coexistence_tolerance,
                    self.minimum_phase_separation,
                )
            )
        ):
            raise ValueError("boundary tolerances and stability iteration budget must be positive")
        if self.composition_starts is not None:
            starts = tuple(_fractions(x, len(ids)) for x in self.composition_starts)
            if not starts or any(
                any((z == 0) != (x == 0) for z, x in zip(self.fixed_composition, w, strict=True))
                for w in starts
            ):
                raise ValueError("TPD starts must preserve the exact positive active support")
            object.__setattr__(self, "composition_starts", starts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "chemworld_molecular_phase_boundary_v1",
            **{
                k: getattr(self, k)
                for k in self.__dataclass_fields__
                if k not in {"model", "liquid_basin", "vapor_basin"}
            },
            "model": molecular_model_to_dict(self.model),
            "liquid_basin": asdict(self.liquid_basin),
            "vapor_basin": asdict(self.vapor_basin),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> MolecularPhaseBoundarySpec:
        data = _strict(payload, set(cls.__dataclass_fields__) | {"schema"}, "boundary spec")
        if data.pop("schema") != "chemworld_molecular_phase_boundary_v1":
            raise ValueError("unknown boundary spec encoding")
        data["model"] = molecular_model_from_dict(data["model"])
        for key in ("liquid_basin", "vapor_basin"):
            data[key] = MolecularDensityBasin(
                **_strict(
                    data[key], set(MolecularDensityBasin.__dataclass_fields__), "density basin"
                )
            )
        return cls(**data)


@dataclass(frozen=True)
class MolecularBoundarySeed:
    temperature_K: float
    liquid_density_mol_m3: float
    vapor_density_mol_m3: float
    incipient_composition: tuple[float, ...]

    def __post_init__(self) -> None:
        if not all(
            _positive(v)
            for v in (self.temperature_K, self.liquid_density_mol_m3, self.vapor_density_mol_m3)
        ):
            raise ValueError("boundary seed T and densities must be positive finite values")
        object.__setattr__(
            self,
            "incipient_composition",
            _fractions(self.incipient_composition, len(self.incipient_composition)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> MolecularBoundarySeed:
        return cls(**_strict(payload, set(cls.__dataclass_fields__), "boundary seed"))


@dataclass(frozen=True)
class MolecularBoundaryResult:
    success: bool
    kind: str
    liquid: MolecularEOSState | None = None
    vapor: MolecularEOSState | None = None
    coordinates: tuple[float, ...] = ()
    chemical_potential_error_RT: float | None = None
    pressure_relative_error: float | None = None
    phase_separation: float | None = None
    stability_mode: str = "local_coexistence"
    stability_reports: tuple[dict[str, Any], ...] = ()
    failure_reason: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "coordinates", tuple(self.coordinates))
        object.__setattr__(self, "stability_reports", tuple(self.stability_reports))
        if (
            type(self.success) is not bool
            or self.kind not in {"bubble", "dew", "azeotrope"}
            or self.stability_mode not in {"finite_tpd", "local_coexistence"}
            or any(not isfinite(v) for v in self.coordinates)
            or any(
                v is not None and (not isfinite(v) or v < 0)
                for v in (
                    self.chemical_potential_error_RT,
                    self.pressure_relative_error,
                    self.phase_separation,
                )
            )
        ):
            raise ValueError("invalid boundary result identity, status or finite diagnostics")
        if self.success and (self.liquid is None or self.vapor is None or self.failure_reason):
            raise ValueError("successful boundary result requires both phases and no failure")

    def to_dict(self) -> dict[str, Any]:
        return {"schema": "chemworld_molecular_boundary_result_v1", **asdict(self)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> MolecularBoundaryResult:
        data = _strict(payload, set(cls.__dataclass_fields__) | {"schema"}, "boundary result")
        if data.pop("schema") != "chemworld_molecular_boundary_result_v1":
            raise ValueError("unknown boundary result encoding")
        for name in ("liquid", "vapor"):
            if data[name] is not None:
                state = _strict(
                    data[name], set(MolecularEOSState.__dataclass_fields__), "phase state"
                )
                state["mole_fractions"] = tuple(state["mole_fractions"])
                if state["log_fugacity_coefficients"] is not None:
                    state["log_fugacity_coefficients"] = tuple(state["log_fugacity_coefficients"])
                data[name] = MolecularEOSState(**state)
        data["coordinates"] = tuple(data["coordinates"])
        data["stability_reports"] = tuple(data["stability_reports"])
        return cls(**data)


@dataclass(frozen=True)
class MolecularEnvelopeResult:
    success: bool
    points: tuple[MolecularBoundaryResult, ...]
    requested_steps: int
    completed_steps: int
    termination_reason: str
    attempts: tuple[dict[str, Any], ...] = ()
    tangents: tuple[tuple[float, ...], ...] = ()
    turning_intervals: tuple[dict[str, Any], ...] = ()
    critical_result: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if (
            type(self.success) is not bool
            or type(self.requested_steps) is not int
            or type(self.completed_steps) is not int
            or self.requested_steps < 1
            or not 0 <= self.completed_steps <= self.requested_steps
            or self.completed_steps != len(self.points) - 1
            or not self.termination_reason
        ):
            raise ValueError("envelope result needs exact point/step counts and termination")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "chemworld_molecular_envelope_v1",
            **asdict(self),
            "points": [p.to_dict() for p in self.points],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> MolecularEnvelopeResult:
        data = _strict(payload, set(cls.__dataclass_fields__) | {"schema"}, "envelope result")
        if data.pop("schema") != "chemworld_molecular_envelope_v1":
            raise ValueError("unknown envelope result encoding")
        data["points"] = tuple(MolecularBoundaryResult.from_dict(p) for p in data["points"])
        for name in ("attempts", "turning_intervals"):
            data[name] = tuple(data[name])
        data["tangents"] = tuple(tuple(t) for t in data["tangents"])
        return cls(**data)


class _BoundaryEquations:
    def __init__(
        self,
        spec: MolecularPhaseBoundarySpec,
        seed: MolecularBoundarySeed,
        *,
        azeotrope: bool = False,
    ) -> None:
        self.spec, self.azeotrope = spec, azeotrope
        self.z = np.array(spec.fixed_composition)
        anchor = int(np.argmax(self.z))
        self.active = np.array([i for i in np.flatnonzero(self.z > 0) if i != anchor] + [anchor])
        if azeotrope and len(self.active) != 2:
            raise ValueError("isolated azeotrope solver requires exactly two active components")
        w = np.array(_fractions(seed.incipient_composition, len(self.z)))
        if np.any((w == 0) != (self.z == 0)):
            raise ValueError("incipient composition must preserve exact positive feed support")
        self.initial = np.r_[
            np.log([seed.temperature_K, seed.liquid_density_mol_m3, seed.vapor_density_mol_m3]),
            np.log(w[self.active][:-1]) - log(w[self.active][-1]),
        ]
        pairs = (
            spec.temperature_bounds_K,
            spec.liquid_basin.density_bounds_mol_m3,
            spec.vapor_basin.density_bounds_mol_m3,
        )
        self.lower = np.r_[np.log([p[0] for p in pairs]), [-700.0] * (len(self.active) - 1)]
        self.upper = np.r_[np.log([p[1] for p in pairs]), [700.0] * (len(self.active) - 1)]
        self.variables = tuple(
            ResidualVariable(f"boundary_coordinate_{i}", "1", lower=a, upper=b)
            for i, (a, b) in enumerate(zip(self.lower, self.upper, strict=True))
        )
        self.pressure_scale = (
            spec.model.gas_constant_J_mol_K * seed.temperature_K * seed.vapor_density_mol_m3
        )
        self.calls = 0

    def decode(
        self, q: Sequence[float] | FloatArray
    ) -> tuple[float, float, float, FloatArray, FloatArray]:
        t, liquid_rho, vapor_rho = (exp(v) for v in q[:3])
        w = np.zeros_like(self.z)
        w[self.active] = softmax(np.r_[q[3:], 0.0])
        if np.any(w[self.active] <= 0):
            raise ValueError("incipient active composition underflow")
        liquid, vapor = (self.z, w) if self.spec.kind == "bubble" else (w, self.z)
        return (
            t,
            liquid_rho,
            vapor_rho,
            w if self.azeotrope else liquid,
            w if self.azeotrope else vapor,
        )

    def phase(self, temperature: float, rho: float, xs: FloatArray) -> tuple[float, FloatArray]:
        model = self.spec.model
        # Numerical trials evaluate the same potential, not a full accepted
        # state. Source pressure and stability validation follow convergence.
        ar = SecondOrderPair(0) + model.alphar(temperature, SecondOrderPair(rho, dx=1), tuple(xs))
        rt = model.gas_constant_J_mol_K * temperature
        pressure = rho * rt * (1 + rho * ar.dx)
        mu = (
            np.asarray(
                molecular_residual_chemical_potentials(
                    model, temperature, rho, tuple(float(x) for x in xs)
                ),
                dtype=float,
            )[self.active]
            / rt
        )
        mu += log(rho) + np.log(xs[self.active])
        if not isfinite(pressure) or not np.all(np.isfinite(mu)):
            raise ValueError("nonfinite boundary equation response")
        return pressure, mu

    def residual(self, q: FloatArray) -> FloatArray:
        self.calls += 1
        try:
            t, rl, rv, xl, yv = self.decode(q)
            pl, ml = self.phase(t, rl, xl)
            pv, mv = self.phase(t, rv, yv)
            return np.r_[ml - mv, (pl - pv) / self.pressure_scale]
        except (ValueError, ArithmeticError) as exc:
            raise ResidualDomainError(str(exc)) from exc

    def pressure(self, q: FloatArray) -> float:
        try:
            t, rl, _, xl, _ = self.decode(q)
            return self.phase(t, rl, xl)[0]
        except (ValueError, ArithmeticError) as exc:
            raise ResidualDomainError(str(exc)) from exc

    def jacobian(self, function: Any, q: FloatArray) -> FloatArray:
        return finite_jacobian(function, q, np.ones(len(q)), lower=self.lower, upper=self.upper)


def _finish(
    equations: _BoundaryEquations, q: FloatArray, diagnostics: dict[str, Any]
) -> MolecularBoundaryResult:
    spec = equations.spec
    kind = "azeotrope" if equations.azeotrope else spec.kind
    details = {
        **diagnostics,
        "jacobian": "bounded_finite_difference_of_shared_potential",
        "scope": "finite_declared_tpd"
        if spec.stability_mode == "finite_tpd"
        else "local_coexistence_not_global_phase_selection",
    }
    try:
        t, rl, rv, xl, yv = equations.decode(q)
        separation = max(abs(log(rl / rv)), float(np.max(np.abs(xl - yv))))
        if rl <= rv or separation <= spec.minimum_phase_separation:
            raise ValueError("trivial_or_merged_phases")
        liquid = molecular_eos_state(spec.model, t, rl, tuple(float(x) for x in xl))
        vapor = molecular_eos_state(spec.model, t, rv, tuple(float(x) for x in yv))
        pressure = (liquid.pressure_Pa + vapor.pressure_Pa) / 2
        if not spec.pressure_bounds_Pa[0] <= pressure <= spec.pressure_bounds_Pa[1]:
            raise ValueError("boundary_pressure_outside_declared_domain")
        error = equations.residual(q)
        mu_error = float(np.max(np.abs(error[:-1])))
        p_error = abs(liquid.pressure_Pa - vapor.pressure_Pa) / pressure
        if max(mu_error, p_error) > spec.coexistence_tolerance:
            raise ValueError("unresolved_physical_coexistence_equations")
        curvature = [
            molecular_composition_response(spec.model, t, rho * x)
            for rho, x in ((rl, xl), (rv, yv))
        ]
        details["phase_scaled_minimum_curvature"] = [r["scaled_eigenvalues"][0] for r in curvature]
        if any(r["scaled_eigenvalues"][0] <= 0 for r in curvature):
            raise ValueError("locally_unstable_coexisting_phase")
        stability = []
        if spec.stability_mode == "finite_tpd":
            for x, basin in ((xl, spec.liquid_basin), (yv, spec.vapor_basin)):
                report = molecular_tp_stability(
                    spec.model,
                    t,
                    pressure,
                    tuple(float(v) for v in x),
                    basins=(spec.liquid_basin, spec.vapor_basin),
                    composition_starts=spec.composition_starts,
                    max_iterations=spec.stability_max_iterations,
                    tpd_tolerance=spec.tpd_tolerance,
                    reference_basin=basin.name,
                )
                stability.append(report.to_dict())
            details["stability_complete"] = all(r["complete"] for r in stability)
            if any(not r["complete"] or r["unstable"] for r in stability):
                return MolecularBoundaryResult(
                    False,
                    kind,
                    liquid,
                    vapor,
                    tuple(q),
                    mu_error,
                    p_error,
                    separation,
                    spec.stability_mode,
                    tuple(stability),
                    "remaining_or_unresolved_boundary_stability",
                    details,
                )
        return MolecularBoundaryResult(
            True,
            kind,
            liquid,
            vapor,
            tuple(float(v) for v in q),
            mu_error,
            p_error,
            separation,
            spec.stability_mode,
            tuple(stability),
            diagnostics=details,
        )
    except (ValueError, ArithmeticError) as exc:
        return MolecularBoundaryResult(
            False,
            kind,
            coordinates=tuple(float(v) for v in q),
            stability_mode=spec.stability_mode,
            failure_reason=str(exc),
            diagnostics=details,
        )


def _solve(
    spec: MolecularPhaseBoundarySpec,
    seed: MolecularBoundarySeed,
    *,
    temperature_K: float | None,
    pressure_Pa: float | None,
    policy: RootPolicy,
    azeotrope: bool,
) -> MolecularBoundaryResult:
    if (temperature_K is None) == (pressure_Pa is None):
        raise ValueError("supply exactly one boundary temperature or pressure")
    target = temperature_K if temperature_K is not None else pressure_Pa
    assert target is not None
    bounds = spec.temperature_bounds_K if temperature_K is not None else spec.pressure_bounds_Pa
    if not _positive(target) or not bounds[0] <= target <= bounds[1]:
        raise ValueError("boundary target outside declared domain")
    eq = _BoundaryEquations(spec, seed, azeotrope=azeotrope)

    def residual(q: FloatArray) -> FloatArray:
        anchor = (
            q[0] - log(target) if temperature_K is not None else (eq.pressure(q) - target) / target
        )
        return np.r_[eq.residual(q), anchor]

    root = solve_root(
        residual,
        eq.initial,
        jacobian=lambda q: eq.jacobian(residual, q),
        variables=eq.variables,
        residual_scales=np.ones(len(eq.initial)),
        policy=policy,
    )
    diagnostics = {
        "root": root.diagnostics,
        "condition_evaluations": eq.calls,
        "constraint": {"temperature_K" if temperature_K is not None else "pressure_Pa": target},
    }
    if not root.success:
        return MolecularBoundaryResult(
            False,
            "azeotrope" if azeotrope else spec.kind,
            stability_mode=spec.stability_mode,
            failure_reason=root.failure_reason,
            diagnostics=diagnostics,
        )
    if azeotrope:
        singular = np.linalg.svd(eq.jacobian(residual, root.value), compute_uv=False)
        diagnostics["root_singular_values"] = singular.tolist()
        if singular[-1] <= 1e-9 * max(1.0, singular[0]):
            return MolecularBoundaryResult(
                False,
                "azeotrope",
                coordinates=tuple(root.value),
                stability_mode=spec.stability_mode,
                failure_reason="nonisolated_or_degenerate_azeotrope",
                diagnostics=diagnostics,
            )
    return _finish(eq, root.value, diagnostics)


def solve_molecular_phase_boundary(
    specification: MolecularPhaseBoundarySpec,
    seed: MolecularBoundarySeed,
    *,
    temperature_K: float | None = None,
    pressure_Pa: float | None = None,
    policy: RootPolicy = BOUNDARY_POLICY,
) -> MolecularBoundaryResult:
    """Resolve bubble/dew chemical and mechanical coexistence at a specified T or P."""
    return _solve(
        specification,
        seed,
        temperature_K=temperature_K,
        pressure_Pa=pressure_Pa,
        policy=policy,
        azeotrope=False,
    )


def solve_molecular_azeotrope(
    specification: MolecularPhaseBoundarySpec,
    seed: MolecularBoundarySeed,
    *,
    temperature_K: float | None = None,
    pressure_Pa: float | None = None,
    policy: RootPolicy = BOUNDARY_POLICY,
) -> MolecularBoundaryResult:
    """An isolated binary azeotrope: equal compositions, unequal stable densities.

    The spec composition selects the active binary face; the seed's incipient
    composition initializes the unknown common azeotropic composition.
    Identical-component continua and merged one-phase roots are rejected.
    """
    return _solve(
        specification,
        seed,
        temperature_K=temperature_K,
        pressure_Pa=pressure_Pa,
        policy=policy,
        azeotrope=True,
    )


def _tangent(
    eq: _BoundaryEquations, q: FloatArray, previous: FloatArray | None, direction: int
) -> tuple[FloatArray, dict[str, Any]]:
    jac = eq.jacobian(eq.residual, q)
    _, singular, vectors = np.linalg.svd(jac, full_matrices=True)
    if singular[-1] <= 1e-10 * max(1.0, singular[0]):
        raise ValueError("coexistence_curve_rank_loss")
    tangent = vectors[-1]
    if previous is None:
        pivot = 0 if abs(tangent[0]) > 1e-8 else int(np.argmax(np.abs(tangent)))
        tangent *= direction * (1 if tangent[pivot] > 0 else -1)
    elif tangent @ previous < 0:
        tangent *= -1
    pressure_jac = eq.jacobian(lambda v: np.array([eq.pressure(v)]), q)[0]
    return tangent, {
        "logT_slope": float(tangent[0]),
        "pressure_slope_Pa": float(pressure_jac @ tangent),
        "singular_values": singular.tolist(),
    }


def _critical(spec: MolecularPhaseBoundarySpec, point: MolecularBoundaryResult) -> dict[str, Any]:
    assert point.liquid is not None and point.vapor is not None
    low = max(spec.liquid_basin.density_bounds_mol_m3[0], spec.vapor_basin.density_bounds_mol_m3[0])
    high = min(
        spec.liquid_basin.density_bounds_mol_m3[1], spec.vapor_basin.density_bounds_mol_m3[1]
    )
    if low >= high:
        return {"success": False, "failure_reason": "declared_phase_basins_do_not_overlap"}
    try:
        result = solve_molecular_critical(
            spec.model,
            spec.fixed_composition,
            initial_temperature_K=point.liquid.temperature_K,
            initial_density_mol_m3=(
                point.liquid.molar_density_mol_m3 * point.vapor.molar_density_mol_m3
            )
            ** 0.5,
            temperature_bounds_K=spec.temperature_bounds_K,
            density_bounds_mol_m3=(low, high),
        )
        data = result.to_dict()
        if result.success and result.response is not None:
            response = result.response
            distance = max(
                abs(log(response["temperature_K"] / point.liquid.temperature_K)),
                abs(log(response["molar_density_mol_m3"] / point.liquid.molar_density_mol_m3)),
                abs(log(response["molar_density_mol_m3"] / point.vapor.molar_density_mol_m3)),
            )
            within_pressure = (
                spec.pressure_bounds_Pa[0] <= response["pressure_Pa"] <= spec.pressure_bounds_Pa[1]
            )
            data["matches_envelope_endpoint"] = distance < 0.05 and within_pressure
            data["within_declared_pressure_domain"] = within_pressure
            data["endpoint_log_distance"] = distance
        return data
    except (ValueError, ArithmeticError) as exc:
        return {"success": False, "failure_reason": str(exc)}


def trace_molecular_phase_envelope(
    specification: MolecularPhaseBoundarySpec,
    initial: MolecularBoundaryResult,
    *,
    steps: int,
    arclength_step: float,
    direction: int = 1,
    maximum_step_reductions: int = 6,
    critical_separation: float = 2e-3,
    policy: RootPolicy = BOUNDARY_POLICY,
) -> MolecularEnvelopeResult:
    """Predict/correct in dimensionless log coordinates, including T/P folds.

    Every corrector adds t_prev dot(q-q_predictor)=0, not a T/P increment.
    Failed corrections and step reductions are retained; a failure terminates
    the curve and never connects across a missing point. Turning intervals are
    sign changes of directional dT/ds or dP/ds, not claimed exact extrema.
    """
    if (
        type(steps) is not int
        or steps < 1
        or not _positive(arclength_step)
        or type(direction) is not int
        or direction not in (-1, 1)
        or type(maximum_step_reductions) is not int
        or maximum_step_reductions < 0
        or not _positive(critical_separation)
        or critical_separation <= specification.minimum_phase_separation
    ):
        raise ValueError("invalid envelope step count, arclength, direction or stopping bounds")
    if (
        not initial.success
        or initial.kind != specification.kind
        or initial.liquid is None
        or initial.vapor is None
    ):
        raise ValueError("envelope needs a successful matching bubble/dew start")
    incipient = initial.vapor if specification.kind == "bubble" else initial.liquid
    seed = MolecularBoundarySeed(
        initial.liquid.temperature_K,
        initial.liquid.molar_density_mol_m3,
        initial.vapor.molar_density_mol_m3,
        incipient.mole_fractions,
    )
    eq = _BoundaryEquations(specification, seed)
    q = eq.initial
    start = _finish(eq, q, {"initial_point_revalidated": True})
    if not start.success:
        return MolecularEnvelopeResult(False, (start,), steps, 0, "invalid_initial_boundary")
    points, attempts, tangents, turns = [start], [], [], []
    previous, previous_slopes = None, None
    reason, critical = "requested_steps_completed", None
    for station in range(steps + 1):
        if (
            points[-1].phase_separation is not None
            and points[-1].phase_separation < critical_separation
        ):
            critical = _critical(specification, points[-1])
            reason = (
                "near_critical_verified"
                if critical.get("success") and critical.get("matches_envelope_endpoint")
                else "near_critical_unresolved"
            )
            break
        try:
            tangent, slopes = _tangent(eq, q, previous, direction)
        except (ValueError, ArithmeticError, ResidualDomainError) as exc:
            reason = str(exc)
            break
        tangents.append(tuple(float(v) for v in tangent))
        if previous_slopes is not None:
            for key in ("logT_slope", "pressure_slope_Pa"):
                if previous_slopes[key] * slopes[key] < 0:
                    turns.append(
                        {
                            "coordinate": "temperature_K" if key == "logT_slope" else "pressure_Pa",
                            "left_point": station - 1,
                            "right_point": station,
                            "left_slope": previous_slopes[key],
                            "right_slope": slopes[key],
                        }
                    )
        if station == steps:
            break
        accepted = None
        for reduction in range(maximum_step_reductions + 1):
            step = arclength_step / 2**reduction
            predictor = q + step * tangent
            record: dict[str, Any] = {
                "station": station + 1,
                "reduction": reduction,
                "arclength_step": step,
            }
            if np.any(predictor <= eq.lower) or np.any(predictor >= eq.upper):
                record.update(success=False, failure_reason="predictor_outside_declared_bounds")
                attempts.append(record)
                continue

            def residual(
                values: FloatArray, tangent: FloatArray = tangent, predictor: FloatArray = predictor
            ) -> FloatArray:
                return np.r_[eq.residual(values), tangent @ (values - predictor)]

            root = solve_root(
                residual,
                predictor,
                jacobian=lambda v: eq.jacobian(residual, v),
                variables=eq.variables,
                residual_scales=np.ones(len(q)),
                policy=policy,
            )
            record.update(root=root.diagnostics)
            if root.success:
                distance = float(np.linalg.norm(root.value - predictor))
                point = _finish(
                    eq,
                    root.value,
                    {
                        "root": root.diagnostics,
                        "arclength_step": step,
                        "corrector_distance": distance,
                        "arclength_residual": float(tangent @ (root.value - predictor)),
                    },
                )
                record.update(success=point.success, failure_reason=point.failure_reason)
                # A corrected root far from its predictor is a branch jump,
                # even if all local coexistence equations happen to vanish.
                if distance > step:
                    record.update(success=False, failure_reason="corrector_left_local_branch")
                if record["success"]:
                    accepted = point
            else:
                record.update(success=False, failure_reason=root.failure_reason)
            attempts.append(record)
            if accepted is not None:
                break
        if accepted is None:
            reason = "continuation_unresolved: " + attempts[-1]["failure_reason"]
            break
        q = np.asarray(accepted.coordinates)
        points.append(accepted)
        previous, previous_slopes = tangent, slopes
    success = reason in {"requested_steps_completed", "near_critical_verified"}
    return MolecularEnvelopeResult(
        success,
        tuple(points),
        steps,
        len(points) - 1,
        reason,
        tuple(attempts),
        tuple(tangents),
        tuple(turns),
        critical,
    )
