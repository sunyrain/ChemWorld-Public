"""Boundary orchestration only: original G, TP/KKT and common point roots."""

from __future__ import annotations

from dataclasses import replace
from itertools import pairwise
from typing import Any

import numpy as np

from chemworld.physchem.caloric_math import derivative_values
from chemworld.physchem.calphad_response import branch_diagnostics
from chemworld.physchem.residual_system import ResidualVariable, finite_jacobian, solve_root
from chemworld.physchem.tdb_expressions import TDB_GAS_CONSTANT_J_MOL_K as R


def _g_derivatives(
    model: Any, components: Any, temperature: float, x: float, pressure: float
) -> Any:
    sites = ({components[0]: x, components[1]: 1 - x},)
    if branch_diagnostics(model, sites, temperature, pressure):
        raise ValueError("critical derivative crosses an undeclared TDB/magnetic branch")
    return derivative_values(
        lambda q: sum(
            model._energies(
                ({components[0]: q, components[1]: 1 - q},), temperature, pressure
            ).values()
        ),
        x,
        order=4,
    )


def locate_critical(evaluate: Any, request: Any, index: int) -> dict[str, Any]:
    spec = evaluate.spec
    model = next(m for m in spec.equilibrium.models if m.phase_name == request.phase_name)
    row: dict[str, Any] = {
        "id": f"critical:{index}",
        "kind": "critical",
        "success": False,
        "phase": request.phase_name,
        "request": vars(request),
    }
    lower = np.array([request.temperature_bounds_K[0], request.composition_bounds[0]])
    upper = np.array([request.temperature_bounds_K[1], request.composition_bounds[1]])
    try:

        def residual(v: Any) -> Any:
            _, _, second, third, _ = _g_derivatives(
                model, spec.components, v[0], v[1], spec.pressure_Pa
            )
            return np.array([second, third]) / (R * v[0])

        root = solve_root(
            residual,
            [request.temperature_K, request.composition],
            jacobian=lambda v: finite_jacobian(residual, v, [100.0, 1.0], lower=lower, upper=upper),
            variables=(
                ResidualVariable("T", "K", scale=100.0, lower=lower[0], upper=upper[0]),
                ResidualVariable("x", "1", lower=lower[1], upper=upper[1]),
            ),
            residual_scales=np.ones(2),
            policy=spec.equilibrium.policy.root,
        )
        row["root"] = {
            "success": root.success,
            "failure_reason": root.failure_reason,
            **root.diagnostics,
        }
        if not root.success:
            row["failure_reason"] = root.failure_reason
            return row
        t, x = map(float, root.value)
        row.update(
            temperature_K=t,
            composition=x,
            derivatives_J_mol=list(_g_derivatives(model, spec.components, t, x, spec.pressure_Pa)),
            derivative_method="shared_nested_analytic_G_jets_through_order_four",
        )
        if row["derivatives_J_mol"][4] <= 0:
            row["failure_reason"] = "critical_requires_positive_fourth_derivative"
            return row
        center = evaluate.at(t, x, purpose="critical_candidate_global_TP")
        row["candidate_equilibrium"] = center.to_dict()
        g = row["derivatives_J_mol"][0]
        gx = row["derivatives_J_mol"][1]
        chemical = dict(zip(spec.components, (g + (1-x)*gx, g-x*gx), strict=True))
        if (
            not center.success
            or abs(center.total_gibbs_J - g) > spec.equilibrium.policy.tangent_tolerance_J_mol
            or any(p.phase_name != request.phase_name for p in center.phases)
            or max(abs(center.element_potentials_J_mol[e] - chemical[e]) for e in spec.components)
            > spec.equilibrium.policy.tangent_tolerance_J_mol
        ):
            row["failure_reason"] = "critical_candidate_common_tangent_unconfirmed"
            return row
        probe = spec.policy.endpoint_probe_K
        points = [evaluate.at(t + d, x, purpose="critical_two_sided_TP") for d in (-probe, probe)]
        row["sides"] = [p.to_dict() for p in points]
        counts = [len(p.phases) if p.success else 0 for p in points]
        if sorted(counts) != [1, 2] or any(
            phase.phase_name != request.phase_name for point in points for phase in point.phases
        ):
            row["failure_reason"] = "critical_two_sided_stable_topology_unresolved"
            return row
        # The critical point itself is a limit, not a uniquely divided stock.
        # Candidate and both sides retain the original finite allowed-phase search.
        row.update(
            success=True,
            phase_fractions=None,
            side_phase_counts=counts,
            split_side=-1 if counts[0] == 2 else 1,
        )
    except (ValueError, ArithmeticError) as exc:
        row["failure_reason"] = str(exc)
    return row


def locate_crossing(evaluate: Any, request: Any, index: int) -> dict[str, Any]:
    spec = evaluate.spec
    restricted = [
        replace(
            spec.equilibrium,
            models=tuple(m for m in spec.equilibrium.models if m.phase_name == name),
        )
        for name in request.phase_names
    ]
    cache: dict[float, Any] = {}
    row: dict[str, Any] = {
        "id": f"crossing:{index}",
        "kind": "nonunique_first_order",
        "success": False,
        "request": vars(request),
    }
    try:

        def pair(t: float) -> Any:
            t = float(t)
            if t not in cache:
                cache[t] = [
                    evaluate.at(
                        t, request.composition, equilibrium=s, purpose="restricted_phase_crossing"
                    )
                    for s in restricted
                ]
            if not all(p.success and len(p.phases) == 1 for p in cache[t]):
                raise ValueError("restricted homogeneous crossing is unresolved")
            return cache[t]

        def residual(v: Any) -> Any:
            a, b = pair(v[0])
            return np.array([(a.total_gibbs_J - b.total_gibbs_J) / (R * v[0])])

        def jacobian(v: Any) -> Any:
            a, b = pair(v[0])
            return np.array([[-(a.total_enthalpy_J - b.total_enthalpy_J) / (R * v[0] ** 2)]])

        root = solve_root(
            residual,
            [request.temperature_K],
            jacobian=jacobian,
            variables=(
                ResidualVariable(
                    "T",
                    "K",
                    scale=100.0,
                    lower=request.temperature_bounds_K[0],
                    upper=request.temperature_bounds_K[1],
                ),
            ),
            residual_scales=np.ones(1),
            policy=spec.equilibrium.policy.root,
        )
        row["root"] = {
            "success": root.success,
            "failure_reason": root.failure_reason,
            **root.diagnostics,
        }
        if not root.success:
            row["failure_reason"] = root.failure_reason
            return row
        t = float(root.value[0])
        a, b = pair(t)
        tolerance = spec.equilibrium.policy.tangent_tolerance_J_mol
        if abs(a.total_enthalpy_J - b.total_enthalpy_J) <= tolerance:
            row["failure_reason"] = "crossing_has_no_resolved_first_order_enthalpy_jump"
            return row
        potential_error = max(
            abs(a.element_potentials_J_mol[e] - b.element_potentials_J_mol[e])
            for e in spec.components
        )
        row["element_potential_error_J_mol"] = potential_error
        if potential_error > tolerance:
            row["failure_reason"] = "crossing_candidates_do_not_share_common_tangent"
            return row
        center = evaluate.at(t, request.composition, purpose="crossing_candidate_global_TP")
        row["candidate_equilibrium"] = center.to_dict()
        if (
            not center.success
            or abs(center.total_gibbs_J - a.total_gibbs_J) > tolerance
            or max(abs(center.element_potentials_J_mol[e] - a.element_potentials_J_mol[e])
                   for e in spec.components) > tolerance
        ):
            row["failure_reason"] = "crossing_candidate_global_common_tangent_unconfirmed"
            return row
        p = spec.policy.endpoint_probe_K
        sides = [
            evaluate.at(t + d, request.composition, purpose="crossing_two_sided_TP")
            for d in (-p, p)
        ]
        row.update(
            temperature_K=t,
            composition=request.composition,
            sides=[s.to_dict() for s in sides],
            alternatives=[a.to_dict(), b.to_dict()],
        )
        same = (
            max(
                abs(a.phases[0].element_fractions[e] - b.phases[0].element_fractions[e])
                for e in spec.components
            )
            < 1e-7
        )
        identities = [{p.phase_name for p in s.phases} for s in sides]
        if (
            not same
            or not all(s.success for s in sides)
            or identities[0] == identities[1]
            or set.union(*identities) != set(request.phase_names)
        ):
            row["failure_reason"] = "same_composition_first_order_crossing_not_confirmed"
            return row
        row.update(
            success=True,
            phase_fractions=None,
            possible_phases=list(request.phase_names),
            enthalpy_interval_J_mol=sorted((a.total_enthalpy_J, b.total_enthalpy_J)),
        )
    except (ValueError, ArithmeticError) as exc:
        row["failure_reason"] = str(exc)
    return row


def boundary_point(temperature: float, result: Any, component: str) -> dict[str, Any]:
    row = {
        "temperature_K": float(temperature),
        "success": result.success and len(result.phases) == 2,
        "equilibrium": result.to_dict(),
    }
    if row["success"]:
        phases = sorted(result.phases, key=lambda p: p.element_fractions[component])
        row.update(
            compositions=[p.element_fractions[component] for p in phases],
            phase_names=[p.phase_name for p in phases],
        )
    else:
        row["failure_reason"] = result.failure_reason or "no_distinct_two_phase_boundary"
    return row


def _cuts(spec: Any) -> list[float]:
    curves = list(spec.equilibrium.models[0].database.functions.values())
    curves += [p.source.function for m in spec.equilibrium.models for p in m._parameters]
    return sorted({segment.lower_K for curve in curves for segment in curve.segments[1:]})


def _prediction(
    previous: Any, current: Any, old_t: float, t: float, next_t: float, components: Any
) -> Any:
    if previous is None or len(previous.phases) != len(current.phases) or old_t == t:
        return current.phases
    output = []
    for a, b in zip(previous.phases, current.phases, strict=True):
        if a.phase_name != b.phase_name:
            return current.phases
        x = b.element_fractions[components[0]] + (next_t - t) / (t - old_t) * (
            b.element_fractions[components[0]] - a.element_fractions[components[0]]
        )
        if not 0 < x < 1:
            return current.phases
        output.append(replace(b, site_fractions=({components[0]: x, components[1]: 1 - x},)))
    return tuple(output)


def trace_branch(
    evaluate: Any, seed: Any, endpoints: Any, initial_direction: int, index: int
) -> dict[str, Any]:
    spec = evaluate.spec
    first = evaluate.at(seed[0], seed[1], purpose="branch_seed")
    row: dict[str, Any] = {
        "id": f"branch:{index}",
        "seed": seed,
        "complete": False,
        "paths": [],
        "breaks": [],
    }
    if not first.success or len(first.phases) != 2:
        row["failure_reason"] = first.failure_reason or "seed_does_not_resolve_two_phases"
        return row
    seed_point = boundary_point(seed[0], first, spec.components[0])
    names = seed_point["phase_names"]
    criticals = [
        e
        for e in endpoints
        if e["success"] and e["kind"] == "critical" and names == [e["phase"], e["phase"]]
    ]
    cuts = _cuts(spec)
    for direction in (initial_direction, -initial_direction):
        t, current, previous, old_t = float(seed[0]), first, None, float(seed[0])
        path: dict[str, Any] = {"direction": direction, "points": [seed_point], "edges": []}
        row["paths"].append(path)
        bound = spec.temperature_bounds_K[1 if direction > 0 else 0]
        possible = [e for e in criticals if (e["temperature_K"] - t) * direction > 0]
        tip = min(possible, key=lambda e: abs(e["temperature_K"] - t)) if possible else None
        stop = tip["temperature_K"] - direction * spec.policy.endpoint_probe_K if tip else bound
        obstacles: set[float] = set()
        while abs(t - stop) > 1e-9:
            next_t = t + direction * min(spec.policy.maximum_temperature_step_K, abs(stop - t))
            barriers = [v for v in obstacles if direction * (v - t) > 0]
            if barriers:
                obstacle = min(barriers, key=lambda v: abs(v - t))
                if direction * (next_t - obstacle) > 0:
                    next_t = obstacle
            cut = next((c for c in cuts if min(t, next_t) < c <= max(t, next_t)), None)
            if cut is not None:
                row["breaks"].append(
                    {
                        "interval_K": sorted((t, next_t)),
                        "reason": "TDB_segment_boundary",
                        "cut_K": cut,
                    }
                )
                break
            while True:
                warm = _prediction(previous, current, old_t, t, next_t, spec.components)
                xs = [p.element_fractions[spec.components[0]] for p in current.phases]
                trial = evaluate.at(
                    next_t, sum(xs) / 2, warm=warm, purpose="boundary_predictor_corrector"
                )
                point = boundary_point(next_t, trial, spec.components[0])
                if point["success"] and point["phase_names"] == names:
                    obstacles.discard(next_t)
                    path["edges"].append([t, next_t])
                    path["points"].append(point)
                    old_t, t, previous, current = t, next_t, current, trial
                    break
                obstacles.add(next_t)
                if (
                    abs(next_t - t) <= spec.policy.minimum_temperature_step_K
                    or evaluate.evaluations >= spec.policy.maximum_evaluations
                ):
                    row["breaks"].append(
                        {
                            "interval_K": sorted((t, next_t)),
                            "reason": trial.failure_reason or "phase_set_change_or_disappearance",
                            "last_trial": point,
                        }
                    )
                    break
                next_t = (next_t + t) / 2
            if (
                row["breaks"]
                and row["breaks"][-1]["interval_K"][0] <= t <= row["breaks"][-1]["interval_K"][1]
            ):
                break
        if abs(t - stop) <= 1e-9:
            cut = next((c for c in cuts if tip and min(t, tip["temperature_K"]) < c
                        <= max(t, tip["temperature_K"])), None)
            if cut is not None and tip is not None:
                row["breaks"].append({"interval_K": sorted((t, tip["temperature_K"])),
                                      "reason": "TDB_segment_boundary", "cut_K": cut})
                continue
            path["termination"] = {
                "kind": "critical" if tip else "declared_domain_end",
                "temperature_K": tip["temperature_K"] if tip else bound,
                "endpoint_id": tip["id"] if tip else None,
            }
            if tip:
                path["points"].append(
                    {
                        "temperature_K": tip["temperature_K"],
                        "success": True,
                        "compositions": [tip["composition"], tip["composition"]],
                        "phase_names": names,
                        "limiting_endpoint": True,
                    }
                )
                path["edges"].append([t, tip["temperature_K"]])
    row["complete"] = not row["breaks"] and all("termination" in p for p in row["paths"])
    return row


def split_windows(evaluate: Any, branches: Any, composition: float) -> dict[str, Any]:
    spec = evaluate.spec
    out: dict[str, Any] = {
        "composition": composition,
        "success": True,
        "intervals": [],
        "attempts": [],
    }
    for branch in branches:
        if not branch["complete"]:
            out.update(success=False, failure_reason="incomplete_branch_cannot_define_split_window")
            continue
        points = {p["temperature_K"]: p for path in branch["paths"] for p in path["points"]}
        ordered = sorted(points.values(), key=lambda p: p["temperature_K"])

        def margin(point: Any) -> float:
            left, right = point["compositions"]
            return min(composition - left, right - composition)

        transitions: list[float] = []
        for a, b in pairwise(ordered):
            fa, fb = margin(a), margin(b)
            if fa == 0:
                transitions.append(a["temperature_K"])
            if fb == 0:
                transitions.append(b["temperature_K"])
            if fa * fb >= 0:
                continue
            low, high = a["temperature_K"], b["temperature_K"]
            cache: dict[float, float] = {low: fa, high: fb}
            try:

                def residual(v: Any, cache: Any = cache, a: Any = a, b: Any = b) -> Any:
                    t = float(v[0])
                    if t not in cache:
                        point = boundary_point(
                            t,
                            evaluate.at(t, _probe_composition((a, b), t),
                                        purpose="composition_window_root"),
                            spec.components[0],
                        )
                        if not point["success"]:
                            raise ValueError("window boundary TP unresolved")
                        cache[t] = margin(point)
                    return np.array([cache[t]])

                def residual_jacobian(
                    v: np.ndarray, low: float = low, high: float = high,
                ) -> np.ndarray:
                    return finite_jacobian(residual, v, [100.0], lower=[low], upper=[high])

                root = solve_root(
                    residual,
                    [(low + high) / 2],
                    jacobian=residual_jacobian,
                    variables=(ResidualVariable("T", "K", scale=100.0, lower=low, upper=high),),
                    residual_scales=np.ones(1),
                    policy=spec.equilibrium.policy.root,
                )
                out["attempts"].append(
                    {
                        "bracket_K": [low, high],
                        "success": root.success,
                        "failure_reason": root.failure_reason,
                        **root.diagnostics,
                    }
                )
                if not root.success:
                    out["success"] = False
                else:
                    transitions.append(float(root.value[0]))
            except (ValueError, ArithmeticError) as exc:
                out["success"] = False
                out["attempts"].append(
                    {"bracket_K": [low, high], "success": False, "failure_reason": str(exc)}
                )
        if not out["success"]:
            continue
        bounds = sorted({ordered[0]["temperature_K"], *transitions,
                         ordered[-1]["temperature_K"]})
        for low, high in pairwise(bounds):
            t = (low + high) / 2
            point = boundary_point(
                t, evaluate.at(t, _probe_composition(ordered, t),
                               purpose="window_interval_confirmation"), spec.components[0]
            )
            if not point["success"]:
                out.update(success=False, failure_reason="window_interval_confirmation_failed")
                continue
            if margin(point) > 0:
                out["intervals"].append(
                    {"branch_id": branch["id"], "temperature_bounds_K": [low, high]}
                )
    return out


def _probe_composition(points: Any, temperature: float) -> float:
    """Interpolate only the numerical feed inside an already traced branch."""
    ordered = sorted(points, key=lambda p: p["temperature_K"])
    for a, b in pairwise(ordered):
        low, high = a["temperature_K"], b["temperature_K"]
        if low <= temperature <= high:
            fraction = (temperature - low) / (high - low)
            return float((1 - fraction) * sum(a["compositions"]) / 2
                         + fraction * sum(b["compositions"]) / 2)
    nearest = min(ordered, key=lambda p: abs(p["temperature_K"] - temperature))
    return float(sum(nearest["compositions"]) / 2)
