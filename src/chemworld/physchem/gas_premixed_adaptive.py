"""Opt-in profile-driven meshes for the original steady premixed BVP.

These are numerical guesses and local steady queries, never a runtime regrid,
world transfer or time advance. Jump/curvature indicators are resolution
heuristics, not a certified spatial error estimate or flame identification.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from inspect import signature
from itertools import pairwise
from math import fsum, isfinite
from numbers import Real
from traceback import format_exc
from typing import Any

import numpy as np

from chemworld.physchem.finite_volume import DiffusionMesh
from chemworld.physchem.gas_polynomial_transport import GasTransportPolynomialSpec
from chemworld.physchem.gas_premixed_flow import (
    GasPremixedFlowSpec,
    premixed_flow_initial_profile,
    solve_premixed_gas_flow,
)
from chemworld.physchem.reaction_network import R_J_PER_MOL_K as R
from chemworld.physchem.reaction_network import ReactionNetworkSpec
from chemworld.physchem.specs import ComponentSpec

_ROOT_UNKNOWN_DEFAULT = signature(solve_premixed_gas_flow).parameters["maximum_unknowns"].default
_PHYSICAL_CHECKS = {
    "all_species_including_omitted_balance",
    "all_incoming_species_including_omitted",
    "EOS",
    "inlet_trace_normalization",
    "energy",
    "constant_face_mass_flow",
    "chemical_mass_conservation",
    "low_Mach",
}


def _number(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite numeric")
    result = float(value)
    if not isfinite(result) or (positive and result <= 0):
        raise ValueError(f"{name} must be finite" + (" and positive" if positive else ""))
    return result


@dataclass(frozen=True)
class GasPremixedMeshPolicy:
    """Explicit mesh budgets; floors normalize indicators and never alter states.

    A cycle assesses one fresh original root. Pruning is disabled at zero;
    otherwise its jump and curvature threshold must be below one quarter of
    both refinement thresholds. Faces ever inserted by refinement are never
    pruned during this query, so coarsening cannot undo a refinement decision.
    """

    maximum_cycles: int
    maximum_cells: int
    maximum_root_calls: int
    minimum_cell_width_m: float
    slope_threshold: float = 0.15
    curvature_threshold: float = 0.15
    maximum_neighbor_ratio: float = 3.0
    temperature_indicator_floor_K: float = 1.0
    mass_fraction_indicator_floor: float = 1e-8
    prune_threshold: float = 0.0
    minimum_cells: int = 2

    def __post_init__(self) -> None:
        for name in ("maximum_cycles", "maximum_cells", "maximum_root_calls", "minimum_cells"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.minimum_cells < 2 or self.maximum_cells < self.minimum_cells:
            raise ValueError("mesh cell budgets must allow at least two cells")
        for name in (
            "minimum_cell_width_m",
            "slope_threshold",
            "curvature_threshold",
            "temperature_indicator_floor_K",
            "mass_fraction_indicator_floor",
        ):
            object.__setattr__(self, name, _number(getattr(self, name), name, positive=True))
        ratio = _number(self.maximum_neighbor_ratio, "maximum_neighbor_ratio")
        if ratio < 2:
            raise ValueError("dyadic ratio balancing requires maximum_neighbor_ratio >= 2")
        object.__setattr__(self, "maximum_neighbor_ratio", ratio)
        prune = _number(self.prune_threshold, "prune_threshold")
        if prune < 0 or prune >= 0.25 * min(self.slope_threshold, self.curvature_threshold):
            raise ValueError("prune_threshold must be nonnegative and below refinement hysteresis")
        object.__setattr__(self, "prune_threshold", prune)

    def to_dict(self):
        return asdict(self)


def _rows(value, count, ids, name):
    if not isinstance(value, list | tuple) or len(value) != count:
        raise ValueError(f"{name} must cover every original cell")
    rows = []
    for row in value:
        if not isinstance(row, Mapping) or set(row) != set(ids):
            raise ValueError(f"{name} must cover every original species")
        row = {s: _number(row[s], name) for s in ids}
        if any(v < 0 for v in row.values()) or _number(fsum(row.values()), name) <= 0:
            raise ValueError(f"{name} must be nonnegative with a positive total")
        rows.append(row)
    return rows


def _profile_fields(spec, network, profile):
    if not isinstance(profile, Mapping):
        raise ValueError("root must supply an original profile")
    ids = network.species_ids
    amounts = _rows(profile["amounts_mol"], spec.mesh.cells, ids, "amounts")
    temperatures = profile["temperatures_K"]
    if not isinstance(temperatures, list | tuple) or len(temperatures) != spec.mesh.cells:
        raise ValueError("temperature profile must cover every cell")
    temperatures = [_number(t, "temperature", positive=True) for t in temperatures]
    if any(
        not spec.temperature_bounds_K[0] <= t <= spec.temperature_bounds_K[1] for t in temperatures
    ):
        raise ValueError("profile temperature outside the original domain")
    trace = _rows([profile["inlet_trace_mole_fractions"]], 1, ids, "inlet trace")[0]
    if abs(fsum(trace.values()) - 1) > 1e-12:
        raise ValueError("physical inlet trace must be normalized")
    mass = _number(profile["mass_flux_kg_m2_s"], "mass flux", positive=True)
    if spec.mass_flux_kg_m2_s is not None and mass != spec.mass_flux_kg_m2_s:
        raise ValueError("profile changed the original fixed mass flux")
    x = [{s: row[s] / fsum(row.values()) for s in ids} for row in amounts]
    return amounts, temperatures, x, trace, mass


def _masses(spec, network):
    if isinstance(spec.transport, GasTransportPolynomialSpec):
        return spec.transport.molecular_weights_kg_mol
    return {
        s.species_id: ComponentSpec(s.species_id, s.formula, charge=s.charge).molecular_weight_g_mol
        / 1000
        for s in network.species
    }


def premixed_mesh_indicators(spec, network, profile, policy):
    """T and every original Y on actual coordinates, including boundary traces.

    For q, S=max(range(q), declared floor); J=|delta q|/S and
    C=|delta gradient| * mean(neighbor center distances)/S. Boundary values
    are the original Tin/physical inlet trace and zero-gradient outlet.
    A zero species remains exactly zero; no concentration floor is introduced.
    """
    _, temperatures, mole, trace, _ = _profile_fields(spec, network, profile)
    masses = _masses(spec, network)
    fractions = [trace, *mole, mole[-1]]
    mass = [fsum(row[s] * masses[s] for s in network.species_ids) for row in fractions]
    if any(not isfinite(v) or v <= 0 for v in mass):
        raise ValueError("mass fractions exceeded their finite numeric domain")
    coordinates = np.asarray([0.0, *spec.mesh.centers_m, spec.mesh.faces_m[-1]])
    distances = np.diff(coordinates)
    fields = {"T": [spec.inlet_temperature_K, *temperatures, temperatures[-1]]}
    fields.update(
        {
            "Y:" + s: [
                row[s] * masses[s] / total for row, total in zip(fractions, mass, strict=True)
            ]
            for s in network.species_ids
        }
    )
    jump, curve = np.zeros(spec.mesh.cells), np.zeros(spec.mesh.cells)
    details = {}
    for name, values in fields.items():
        values = np.asarray(values)
        span = float(np.max(values) - np.min(values))
        floor = (
            policy.temperature_indicator_floor_K
            if name == "T"
            else policy.mass_fraction_indicator_floor
        )
        scale = max(span, floor)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            gradients = np.diff(values) / distances
            slopes = np.abs(np.diff(values)) / scale
            curves = np.abs(np.diff(gradients)) * (distances[:-1] + distances[1:]) / (2 * scale)
        if not np.all(np.isfinite(slopes)) or not np.all(np.isfinite(curves)):
            raise ValueError("profile indicators exceeded their finite numeric domain")
        jump = np.maximum(jump, np.maximum(slopes[:-1], slopes[1:]))
        curve = np.maximum(curve, curves)
        details[name] = {
            "normalizer": scale,
            "range": span,
            "jump_by_interval": slopes.tolist(),
            "curvature_by_cell": curves.tolist(),
        }
    return {
        "fields": details,
        "jump_by_cell": jump.tolist(),
        "curvature_by_cell": curve.tolist(),
        "refine_cells": [
            i
            for i in range(spec.mesh.cells)
            if jump[i] > policy.slope_threshold or curve[i] > policy.curvature_threshold
        ],
        "spatial_error_certified": False,
    }


class _MeshLimitError(ValueError):
    pass


def _ratio_ok(faces, policy):
    widths = np.diff(faces)
    return all(max(a, b) <= policy.maximum_neighbor_ratio * min(a, b) for a, b in pairwise(widths))


def _insert_midpoint(faces, cell, policy):
    if len(faces) - 1 >= policy.maximum_cells:
        raise _MeshLimitError("maximum_cells_exceeded")
    left, right = faces[cell : cell + 2]
    midpoint = left + (right - left) / 2
    if not left < midpoint < right:
        raise _MeshLimitError("mesh_resolution_exhausted")
    if min(midpoint - left, right - midpoint) < policy.minimum_cell_width_m:
        raise _MeshLimitError("minimum_cell_width_exhausted")
    faces.insert(cell + 1, midpoint)
    return midpoint


def _propose_mesh(spec, indicators, policy, inserted_before):
    old = list(spec.mesh.faces_m)
    faces, inserted, removed = old.copy(), [], []
    # Descending indices make the original marks independent of insertion order.
    for cell in reversed(indicators["refine_cells"]):
        inserted.append(_insert_midpoint(faces, cell, policy))
    balanced = []
    while not _ratio_ok(faces, policy):
        widths = np.diff(faces)
        for i, (left, right) in enumerate(pairwise(widths)):
            if max(left, right) > policy.maximum_neighbor_ratio * min(left, right):
                cell = i if left > right else i + 1
                point = _insert_midpoint(faces, cell, policy)
                inserted.append(point)
                balanced.append(point)
                break
    # Refinement always wins. Coarsening never removes an inserted face, never
    # merges adjacent pairs, and must already meet ratio and physical-phase rules.
    if not inserted and policy.prune_threshold > 0:
        last_removed = -3
        for face in range(1, len(old) - 1):
            if face - last_removed < 3 or old[face] in inserted_before:
                continue
            if len(faces) - 2 < policy.minimum_cells:
                break
            cells = (face - 1, face)
            if any(
                indicators[key][i] >= policy.prune_threshold
                for key in ("jump_by_cell", "curvature_by_cell")
                for i in cells
            ):
                continue
            candidate = [point for point in faces if point != old[face]]
            if not _ratio_ok(candidate, policy):
                continue
            mesh = DiffusionMesh("planar", tuple(candidate), area_m2=spec.mesh.area_m2)
            if spec.phase is not None and not (
                mesh.centers_m[0] < spec.phase.position_m < mesh.centers_m[-1]
            ):
                continue
            faces = candidate
            removed.append(old[face])
            last_removed = face
    mesh = DiffusionMesh("planar", tuple(faces), area_m2=spec.mesh.area_m2)
    return mesh, {
        "inserted_faces_m": sorted(inserted),
        "ratio_balance_insertions_m": sorted(balanced),
        "removed_faces_m": removed,
        "changed": tuple(faces) != tuple(old),
    }


def _inventory(network, amounts, temperatures):
    ids = network.species_ids
    totals = {s: fsum(row[s] for row in amounts) for s in ids}
    thermo = network.imported_gas_thermochemistry
    energy = fsum(
        row[s] * (thermo[s].evaluate(t).enthalpy_J_mol - R * t)
        for row, t in zip(amounts, temperatures, strict=True)
        for s in ids
    )
    if any(not isfinite(v) for v in (*totals.values(), energy)):
        raise ValueError("guess inventory receipt exceeded its finite numeric domain")
    return {"amounts_mol": totals, "internal_energy_J": energy}


def _initial_guess(old_spec, new_spec, network, profile):
    amounts, temperatures, fractions, trace, mass = _profile_fields(old_spec, network, profile)
    old_x = [0.0, *old_spec.mesh.centers_m, old_spec.mesh.faces_m[-1]]
    t = np.interp(
        new_spec.mesh.centers_m,
        old_x,
        [old_spec.inlet_temperature_K, *temperatures, temperatures[-1]],
    ).tolist()
    rows = [trace, *fractions, fractions[-1]]
    interpolated = {
        s: np.interp(new_spec.mesh.centers_m, old_x, [row[s] for row in rows])
        for s in network.species_ids
    }
    x = [
        {s: float(interpolated[s][i]) for s in network.species_ids}
        for i in range(new_spec.mesh.cells)
    ]
    guess = premixed_flow_initial_profile(
        new_spec,
        network,
        temperatures_K=t,
        mole_fractions=x,
        inlet_trace_mole_fractions=trace,
        mass_flux_kg_m2_s=mass,
    )
    before = _inventory(network, amounts, temperatures)
    after = _inventory(network, guess["amounts_mol"], guess["temperatures_K"])
    return guess, {
        "method": "convex-T-X-interpolation-and-original-EOS-initial-profile",
        "scope": "caller-owned steady-BVP numerical guess; not accepted runtime inventory",
        "conservative_inventory_remap": False,
        "before": before,
        "after": after,
        "amount_change_mol": {
            s: after["amounts_mol"][s] - before["amounts_mol"][s] for s in network.species_ids
        },
        "internal_energy_change_J": after["internal_energy_J"] - before["internal_energy_J"],
        "inlet_trace_and_mass_flux_preserved": True,
        "world_transfer_executed": False,
    }


def solve_adaptive_premixed_gas_flow(
    spec: GasPremixedFlowSpec,
    network: ReactionNetworkSpec,
    initial_profile: Mapping[str, Any],
    *,
    mesh_policy: GasPremixedMeshPolicy | None = None,
    **solver_options,
):
    """Run the same original BVP after deterministic profile-driven mesh changes.

    None delegates exactly, including return identity and the original defaults.
    Explicit policies return only one accepted final profile. Full numerical
    root snapshots in history preserve all mesh/guess evidence; they are not
    runtime physical states. A failed mesh/root/budget never falls back to an
    earlier successful profile. Ordinary root exceptions are recorded with
    their current input and traceback; cancellation still propagates. All
    declaration preflight validation remains outside the root exception handler.
    """
    if mesh_policy is None:
        return solve_premixed_gas_flow(spec, network, initial_profile, **solver_options)
    if not isinstance(mesh_policy, GasPremixedMeshPolicy):
        raise ValueError("declare a GasPremixedMeshPolicy or None")
    if not isinstance(spec, GasPremixedFlowSpec) or not isinstance(network, ReactionNetworkSpec):
        raise ValueError("adaptive premixed flow requires the original spec and network")
    if not mesh_policy.minimum_cells <= spec.mesh.cells <= mesh_policy.maximum_cells:
        raise ValueError("initial mesh exceeds the explicit cell budgets")
    if min(np.diff(spec.mesh.faces_m)) < mesh_policy.minimum_cell_width_m:
        raise ValueError("initial mesh is narrower than the explicit minimum width")
    unknown_cap = solver_options.get("maximum_unknowns", _ROOT_UNKNOWN_DEFAULT)
    if type(unknown_cap) is not int or unknown_cap < 1:
        raise ValueError("maximum_unknowns must be a positive integer")
    # Bind both declarations for the entire multi-root query, including remaps.
    current = GasPremixedFlowSpec.from_dict(deepcopy(spec.to_dict()))
    bound = ReactionNetworkSpec.from_dict(deepcopy(network.to_dict()))
    guess = deepcopy(initial_profile)
    report = {
        "schema": "adaptive-premixed-gas-flow-query-v1",
        "success": False,
        "failure_reason": "",
        "physical_root_success": False,
        "mesh_indicator_converged": False,
        "spatial_error_certified": False,
        "scope": "profile-driven local steady BVP; no automatic flame or spatial qualification",
        "policy": mesh_policy.to_dict(),
        "profile": None,
        "spec": current.to_dict(),
        "root_calls": 0,
        "history": [],
        "live_operations_executed": 0,
        "experimental_elapsed_s": 0.0,
        "accepted_states_projected": False,
    }
    seen, inserted = {current.mesh.faces_m}, set()
    for cycle in range(mesh_policy.maximum_cycles):
        count = current.mesh.cells * (len(bound.species_ids) + 1) + len(bound.species_ids)
        count += current.mass_flux_kg_m2_s is None
        if count > unknown_cap:
            report["failure_reason"] = "maximum_unknowns_exceeded"
            return report
        report["root_calls"] += 1
        report["physical_root_success"] = False
        report["spec"] = current.to_dict()
        entry = {
            "cycle": cycle + 1,
            "faces_m": list(current.mesh.faces_m),
            "spec": current.to_dict(),
            "role": "numerical-root-history-not-world-state",
            "initial_profile": deepcopy(guess),
            "root_returned": False,
            "root": None,
        }
        report["history"].append(entry)
        try:
            result = solve_premixed_gas_flow(current, bound, guess, **solver_options)
        except Exception as exc:
            exception = {
                "type": f"{type(exc).__module__}.{type(exc).__qualname__}",
                "message": str(exc),
                "traceback": format_exc(),
            }
            entry["exception"] = exception
            report.update(failure_reason="mesh_root_exception", root_exception=exception)
            return report
        report["last_root_result"] = result
        entry.update(root_returned=True, root=deepcopy(result))
        if result.get("success") is not True:
            report.update(failure_reason="mesh_root_failed", failed_root_result=result)
            return report
        checks = result.get("physical_checks")
        if (
            not isinstance(checks, Mapping)
            or not _PHYSICAL_CHECKS.issubset(checks)
            or any(checks[k] is not True for k in _PHYSICAL_CHECKS)
        ):
            report.update(failure_reason="missing_fresh_physical_checks", failed_root_result=result)
            return report
        report["physical_root_success"] = True
        try:
            indicators = premixed_mesh_indicators(current, bound, result["profile"], mesh_policy)
            entry["indicators"] = indicators
            mesh, proposal = _propose_mesh(current, indicators, mesh_policy, inserted)
        except (ValueError, KeyError, ArithmeticError) as exc:
            report.update(failure_reason=str(exc), failed_root_result=result)
            return report
        entry["proposal"] = proposal
        if not proposal["changed"]:
            report.update(success=True, mesh_indicator_converged=True, profile=result["profile"])
            return report
        entry["proposed_faces_m"] = list(mesh.faces_m)
        if report["root_calls"] >= mesh_policy.maximum_root_calls:
            report["failure_reason"] = "maximum_root_calls_exceeded"
            return report
        if cycle + 1 >= mesh_policy.maximum_cycles:
            report["failure_reason"] = "maximum_cycles_exceeded"
            return report
        if mesh.faces_m in seen:
            report["failure_reason"] = "mesh_cycle_detected"
            return report
        try:
            following = replace(current, mesh=mesh)
            guess, receipt = _initial_guess(current, following, bound, result["profile"])
        except (ValueError, KeyError, ArithmeticError) as exc:
            report["failure_reason"] = "guess_validation_failed: " + str(exc)
            return report
        entry["next_guess"] = receipt
        seen.add(mesh.faces_m)
        inserted.update(proposal["inserted_faces_m"])
        current = following
    raise AssertionError("positive cycle budget must terminate explicitly")
