"""Prospective half-cell DFN tooling. Printing a plan never executes a history.

The coordinator must bind the experiment note and explicitly authorize execution.
Every route has an isolated POSIX process-group budget and immutable archives.
Full Xu2019 and raw Basic metal kinetics remain explicit model discrepancies.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, fields
from functools import lru_cache
from itertools import pairwise
from pathlib import Path
from unittest.mock import patch

import numpy as np
import run_battery_parameters45 as transport
from battery_half_cell_dfn45_cases import (
    CASE_IDS,
    GRIDS,
    PROFILE_COORDINATES,
    RESPONSE_ORDER,
    SOURCE_VARIANTS,
    THRESHOLD_CANDIDATES,
    XU2019_PROJECTION,
    F,
    case_data,
    native_initial_state,
    native_spec,
    planned_denominators,
    routes,
    stage_times,
)

# Reuse the delivered exclusive/verified archive and process-tree primitives.
archive_new, read_archive, write_new = (
    transport.archive_new,
    transport.read_archive,
    transport.write_new,
)
json_value, raw_completed_segments = transport.json_value, transport.raw_completed_segments
raw_saved_counts = transport.raw_saved_counts
THRESHOLDS = deepcopy(THRESHOLD_CANDIDATES)
IMPLEMENTATIONS = ("native", *SOURCE_VARIANTS)
LEVELS = tuple(GRIDS)
ROUTE_CHECKS = (
    "available",
    "execution",
    "identity",
    "mesh",
    "initial-state",
    "stage-count",
    "conditioning-policy",
    "source-conventions",
)
STAGE_CHECKS = (
    "present",
    "schema",
    "boundary-continuity",
    "requested-observations",
    "stop-identity",
    "stop-value",
    "hardware-current",
    "imposed-control",
    "metal-faradaic",
    "positive-faradaic",
    "salt-inventory",
    "solid-lithium",
    "physical-domain",
    "raw-state-retention",
    "all-original-times-retained",
    "physical-current-closure",
    "metal-BV",
    "positive-BV",
    "left-salt-boundary",
    "charge-work-rates",
)
PROFILE_METRICS = (
    "surface_fraction",
    "radial_fraction",
    "salt_mol_m3",
    "electrolyte_potential_V",
    "solid_potential_V",
    "positive_ocp_V",
    "positive_eta_V",
    "positive_j0_A_m2",
    "local_current_A",
    "ionic_face_current_A",
    "solid_face_current_A",
    "diffusive_salt_equivalent_A",
    "cation_flux_equivalent_A",
)
METRICS = (*RESPONSE_ORDER, *PROFILE_METRICS)
PAIR_CHECKS = (
    "both-present",
    "common-time-support",
    "termination-kind",
    "stop-time",
    *(prefix + metric for prefix in ("common:", "t0:", "endpoint:") for metric in METRICS),
)
AREA = XU2019_PROJECTION["area_m2"]
LS, LP = XU2019_PROJECTION["lengths_m"]
CMAX, RADIUS = XU2019_PROJECTION["cmax_mol_m3"], XU2019_PROJECTION["radius_m"]


def route_id(route):
    return "/".join(route[key] for key in ("unit", "implementation", "grid_level"))


def comparison_edges(identity):
    """Declare every edge before observing any outcome; no transitive replacement."""
    implementations = IMPLEMENTATIONS if identity != "H04" else IMPLEMENTATIONS[:3]
    edges = []
    for level in LEVELS:
        pairs = [("native", "modular-constant-transport")]
        if identity != "H04":
            pairs += [
                ("modular-full-Xu2019", "basic-full-pure-metal"),
                ("modular-constant-transport", "basic-constant-transport-pure-metal"),
            ]
        for left, right in pairs:
            edges.append(("cross", (left, level), (right, level), 1.0, True))
        differences = [
            ("modular-full-Xu2019", "native"),
            ("modular-full-Xu2019", "modular-constant-transport"),
        ]
        if identity != "H04":
            differences.append(("basic-full-raw-metal", "basic-full-pure-metal"))
        for left, right in differences:
            edges.append(("model-discrepancy", (left, level), (right, level), 1.0, False))
    for implementation in implementations:
        for lower, upper in pairwise(LEVELS):
            edges.append(
                (
                    "self-refinement",
                    (implementation, lower),
                    (implementation, upper),
                    THRESHOLDS["finest_pair_refinement_factor"],
                    True,
                )
            )
    return tuple(edges)


def plan():
    edges = [(identity, edge) for identity in CASE_IDS for edge in comparison_edges(identity)]
    strict = [(identity, edge) for identity, edge in edges if edge[-1]]
    descriptive = [(identity, edge) for identity, edge in edges if not edge[-1]]
    strict_stages = sum(len(case_data(identity)["stages"]) for identity, _ in strict)
    discrepancy_stages = sum(len(case_data(identity)["stages"]) for identity, _ in descriptive)
    counts = planned_denominators()
    return {
        "schema": "half-cell-dfn45-plan-v1",
        **counts,
        "routes": routes(),
        "cases": [case_data(identity) for identity in CASE_IDS],
        "strict_comparisons": len(strict),
        "strict_comparison_stages": strict_stages,
        "strict_cross_edges": sum(edge[0] == "cross" for _, edge in strict),
        "strict_self_refinement_edges": sum(edge[0] == "self-refinement" for _, edge in strict),
        "model_discrepancy_edges": len(descriptive),
        "model_discrepancy_stages": discrepancy_stages,
        "comparison_edges": [
            {
                "unit": identity,
                "kind": edge[0],
                "left": edge[1],
                "right": edge[2],
                "factor": edge[3],
                "strict": edge[4],
            }
            for identity, edge in edges
        ],
        "route_check_names": ROUTE_CHECKS,
        "stage_check_names": STAGE_CHECKS,
        "pair_check_names": PAIR_CHECKS,
        "check_denominator": counts["histories"] * len(ROUTE_CHECKS)
        + counts["requested_stages"] * len(STAGE_CHECKS)
        + strict_stages * len(PAIR_CHECKS),
        "thresholds": deepcopy(THRESHOLDS),
        "refinement_semantics": "Every adjacent self-refinement absolute+relative gate is "
        "half the cross gate. No error-shrinkage-ratio criterion is imposed. A fine-grid or "
        "refinement pass never cancels a failed coarse cross check.",
        "observation_semantics": "All exact t0 roots, original returned y/t, true endpoints "
        "and unpaired tails are retained. Requested slots beyond a true stop are not required. "
        "Missing/failed stages still retain their entire assessment denominator.",
        "profile_semantics": "Interior profiles use common x/r fractions0.1..0.9 without "
        "extrapolation. q is conservatively integrated into ten common positive-x bins [A]. "
        "Face currents include both physical boundaries. Salt fluxes also report F*A flux "
        "equivalents [A], with diffusive flux divided by1-t+.",
        "inventory_metric_semantics": "Paired solid Li errors use F*(n-n_t0) [C] and the "
        "existing charge gate; paired physical salt uses mean ce=n/Vpore and the salt gate. "
        "Raw mol inventories and independent absolute Faraday/salt/solid-Li gates remain.",
        "execution_authorized_by_this_plan": False,
    }


def _snapshot(state):
    return {field.name: json_value(getattr(state, field.name)) for field in fields(state)}


def _native_contract(spec, data, grid):
    e = spec.electrolyte
    mesh = {}
    for name, sl in (
        ("separator", slice(0, grid["x"][0])),
        ("positive electrode", slice(grid["x"][0], sum(grid["x"]))),
    ):
        faces = np.asarray(e.mesh.faces_m)[sl.start : sl.stop + 1]
        nodes = np.asarray(e.mesh.centers_m)[sl]
        mesh[name] = {
            "cells": len(nodes),
            "class": type(e.mesh).__name__,
            "local_nodes_m": nodes.tolist(),
            "local_edges_m": faces.tolist(),
            "original_nodes_m": nodes.tolist(),
            "original_edges_m": faces.tolist(),
        }
    p = spec.positive
    mesh["positive particle"] = {
        "cells": p.cells,
        "class": type(p.mesh).__name__,
        "local_nodes_m": list(p.mesh.centers_m),
        "local_edges_m": list(p.mesh.faces_m),
        "original_nodes_m": list(p.mesh.centers_m),
        "original_edges_m": list(p.mesh.faces_m),
    }
    return {
        "implementation": "native",
        "variant": "native",
        "spec": spec.to_dict(),
        "case": deepcopy(data),
        "grid": deepcopy(grid),
        "mesh": mesh,
        "F_C_mol": F,
        "R_J_mol_K": 8.31446261815324,
        "response_order": RESPONSE_ORDER,
        "inventory_geometry_contract": {
            "area_m2": AREA,
            "actual_salt_pore_volumes_m3": e.diffusion.storage_volumes_m3.tolist(),
        },
        "residual_observation": "Original common solve_residual report is returned "
        "unchanged; all exposed original y/ydot/events retained, hidden steps unavailable",
    }


def _native_record(spec, state, program, system):
    from chemworld.physchem.battery_half_cell import METAL_KEY
    from chemworld.physchem.battery_half_cell_dfn import half_cell_dfn_fields

    f = half_cell_dfn_fields(spec, state, program=program)
    e = spec.electrolyte
    cs, amounts = [], []
    for _, _, particle, keys, _ in spec.particle_bins:
        cell = np.asarray([state.amounts_mol[key] for key in keys])
        amounts.append(cell)
        cs.append(cell / particle.mesh.cell_volumes_m3)
    cs, amounts = np.asarray(cs), np.asarray(amounts)
    salt_amounts = np.asarray([state.amounts_mol[key] for key in e.keys])
    diffusion = e.diffusion.evaluate(salt_amounts).face_rates_mol_s.copy() / AREA
    diffusion[0] += (1 - e.cation_transference) * f["current_A"] / (F * AREA)
    ionic, solid = f["electrolyte_face_current_A_m2"], f["positive_solid_face_current_A_m2"]
    q = f["reaction_current_A"]
    closure = np.r_[
        AREA * np.diff(ionic) - np.r_[np.zeros(e.cells[0]), q],
        AREA * np.diff(solid) + q,
        AREA * ionic[0] - f["current_A"],
        AREA * ionic[-1],
        AREA * solid[0],
        AREA * solid[-1] - f["current_A"],
        sum(q) + f["current_A"],
    ]
    scalar = {
        "voltage_V": f["voltage_V"],
        "current_A": f["current_A"],
        "charge_C": state.coordinates["discharged_charge_C"],
        "work_J": state.coordinates["delivered_work_J"],
        "metal_Li_mol": state.amounts_mol[METAL_KEY],
        "positive_Li_mol": float(amounts.sum()),
        "electrolyte_salt_mol": float(salt_amounts.sum()),
        "metal_boundary_ce_mol_m3": f["metal_boundary_concentration_mol_m3"],
        "metal_j0_A_m2": f["metal_exchange_current_density_A_m2"],
        "metal_eta_V": f["metal_overpotential_V"],
        "metal_ohmic_V": f["metal_ohmic_V"],
    }
    return {
        "time_s": state.time_s,
        "row": [scalar[name] for name in RESPONSE_ORDER],
        "native_state": _snapshot(state),
        "native_events": {event.event_id: float(event.value(state)) for event in system.all_events},
        "physical_current_closure": {"maximum_absolute_A": float(np.max(abs(closure)))},
        "fields": {
            "ce_mol_m3": e.concentrations(state).tolist(),
            "cs_mol_m3_x_r": cs.tolist(),
            "theta_x_r": (cs / CMAX).tolist(),
            "theta_surface": f["surface_fraction"].tolist(),
            "phi_e": f["electrolyte_potential_V"].tolist(),
            "phi_s": f["solid_potential_V"].tolist(),
            "positive_j_A_m2": f["interfacial_current_A_m2"].tolist(),
            "positive_j0_A_m2": f["exchange_current_density_A_m2"].tolist(),
            "positive_eta_V": f["kinetic_overpotential_V"].tolist(),
            "positive_ocp_V": f["open_circuit_potential_V"].tolist(),
            "positive_q_cell_A": q.tolist(),
            "ionic_current_A_m2": ionic.tolist(),
            "solid_current_A_m2": solid.tolist(),
            "diffusive_salt_flux_mol_m2_s": diffusion.tolist(),
            "total_cation_flux_mol_m2_s": (diffusion + e.cation_transference * ionic / F).tolist(),
            "electrolyte_salt_amounts_mol": salt_amounts.tolist(),
            "positive_shell_amounts_mol_x_r": amounts.tolist(),
            "metal_phi_e_V": f["metal_electrolyte_potential_V"],
        },
    }


def native_time(data, grid, *, progress_callback=None):
    import chemworld.physchem.coupled_residual as residual_module
    from chemworld.physchem.battery_half_cell_dfn import half_cell_dfn_system
    from chemworld.physchem.reactor_shared import ScalarControlProgram

    spec = native_spec(data, grid)
    state, segments = native_initial_state(data, spec), []
    contract = _native_contract(spec, data, grid)
    progress = {"contract": contract, "case": deepcopy(data), "segments": segments}

    def publish(**extra):
        progress.update(extra)
        if progress_callback is not None:
            progress_callback(progress)

    for index, stage in enumerate(data["stages"]):
        boundary, captured = None, []
        try:
            program = ScalarControlProgram(((0.0, stage["target"]),), stage["unit"])
            system = half_cell_dfn_system(spec, program)
            boundary = _native_record(spec, state, program, system)
            publish(
                active_stage=index,
                active_request=deepcopy(stage),
                boundary_t0=boundary,
                phase="before-native-integration",
            )
            original_solver = residual_module.solve_residual

            def observe(
                residual, *args, original_solver=original_solver, captured=captured, **kwargs
            ):
                report = original_solver(residual, *args, **kwargs)
                captured.append(
                    {
                        "variables": [asdict(v) for v in residual.variables],
                        "equations": [asdict(e) for e in residual.equations],
                        "differential_indices": residual.differential.tolist(),
                        "algebraic_indices": residual.algebraic.tolist(),
                        "points": [
                            {
                                "relative_time_s": p.time_s,
                                "y": p.state.tolist(),
                                "ydot": p.rate.tolist(),
                            }
                            for p in report.points
                        ],
                        "events": [
                            {
                                "name": name,
                                "relative_time_s": p.time_s,
                                "y": p.state.tolist(),
                                "ydot": p.rate.tolist(),
                            }
                            for name, p in report.events
                        ],
                        "success": report.success,
                        "failure_reason": report.failure_reason,
                        "stopping_event": report.stopping_event,
                        "diagnostics": json_value(report.diagnostics),
                    }
                )
                publish(phase="native-residual-report-returned", pending_residual_reports=captured)
                return report

            with patch.object(residual_module, "solve_residual", observe):
                result = system.advance(
                    state, stage["duration_s"], evaluation_times_s=stage_times(stage)
                )
            states = list(result.trajectory)
            if not states or states[-1].time_s != result.final_state.time_s:
                states.append(result.final_state)
            publish(
                phase="native-solution-returned",
                pending_native_states=[_snapshot(s) for s in states],
            )
            records = [_native_record(spec, s, program, system) for s in states]
            segments.append(
                {
                    "index": index,
                    "request": deepcopy(stage),
                    "boundary_t0": boundary,
                    "requested_start_time_s": state.time_s,
                    "requested_end_time_s": state.time_s + stage["duration_s"],
                    "requested_relative_times_s": stage_times(stage),
                    "records": records,
                    "original_reported_times_s": [s.time_s for s in states],
                    "original_residual_reports": captured,
                    "actual_start_time_s": state.time_s,
                    "actual_end_time_s": result.final_state.time_s,
                    "termination": "final time"
                    if result.outcome == "success"
                    else result.stopping_event,
                    "outcome": result.outcome,
                    "solver": json_value(result.solver_diagnostic),
                    "native_solver_policy": asdict(system.policy),
                    "conservation_residuals": dict(result.conservation_residuals),
                    "roundoff_corrections_mol": dict(result.roundoff_corrections_mol),
                    "energy_balance_residual_J": result.energy_balance_residual_J,
                    "actual_exchanges": json_value(result.actual_exchanges),
                }
            )
            state = result.final_state
            progress.pop("pending_native_states", None)
            progress.pop("pending_residual_reports", None)
            publish(phase="native-stage-captured")
            if result.outcome != "success":
                break
        except Exception as exc:
            progress.update(
                failed_stage=index,
                failed_request=deepcopy(stage),
                failed_boundary_t0=boundary,
                last_completed_time_s=state.time_s,
            )
            exc.partial_history = progress
            raise
    return {
        "contract": contract,
        "case": deepcopy(data),
        "segments": segments,
        "actual_stages": len(segments),
        "requested_stages": len(data["stages"]),
    }


def stop_kind(value):
    text = str(value).lower().replace("_", "-")
    if "final time" in text:
        return "complete"
    for fragments, kind in (
        (("finite-metal-depleted", "metal:depleted"), "finite-metal-depleted"),
        (("minimum voltage", "minimum-voltage"), "minimum-voltage"),
        (("maximum voltage", "maximum-voltage"), "maximum-voltage"),
        (("minimum-surface", "working:depleted", "minimum positive"), "working-minimum-surface"),
        (("maximum-surface", "working:saturated", "maximum positive"), "working-maximum-surface"),
        (("salt-floor", "electrolyte:depleted", "electrolyte concentration"), "salt-floor"),
    ):
        if any(fragment in text for fragment in fragments):
            return kind
    return str(value)


def _array(value, shape):
    if not transport._finite_array(value, shape):
        raise ValueError(f"incomplete or nonfinite numeric array, expected {shape}")
    return np.asarray(value, dtype=float)


def _mesh(contract):
    result = {}
    for name in ("separator", "positive electrode", "positive particle"):
        mesh = contract["mesh"][name]
        count = mesh["cells"]
        if type(count) is not int or count < 2:
            raise ValueError("invalid actual cell count")
        nodes = _array(mesh["local_nodes_m"], (count,))
        edges = _array(mesh["local_edges_m"], (count + 1,))
        if not (
            np.all(np.diff(nodes) > 0)
            and np.all(np.diff(edges) > 0)
            and np.all(nodes > edges[:-1])
            and np.all(nodes < edges[1:])
        ):
            raise ValueError("actual mesh nodes must be strictly inside physical cells")
        result[name] = (nodes, edges)
    return result


def _layout_values(record, contract, name, *, rate=False):
    entry = next(item for item in contract["state_layout"] if item["name"] == name)
    raw = np.asarray(record["source_rhs"] if rate else record["source_y"], dtype=float)
    values = np.concatenate([raw[a:b] for a, b in entry["slices"]]) * entry["scale"]
    return values if rate else values + entry["reference"]


@lru_cache(maxsize=2)
def _source_evaluator(
    variant,
    ns,
    np_,
    nr,
    metal_length,
    case_json,
    numerical_policy="legacy",
    gradient_policy=None,
    radial_exponent=None,
):
    """Fresh actual source expression DAG; no root, solver set-up or integration."""
    import battery_half_cell_dfn45_pybamm as source

    grid = {"x": (ns, np_), "r": nr}
    if radial_exponent is not None:
        from battery_half_cell_dfn45_meshes import development_mesh_bounds

        grid["radial_exponent"] = radial_exponent
        development_mesh_bounds(grid)
    if (
        variant not in SOURCE_VARIANTS
        or min(ns, np_) < 3
        or nr < 3
        or (radial_exponent is None and (max(ns, np_) > 32 or nr > 64))
        or not math.isfinite(metal_length)
        or metal_length <= 0
    ):
        raise ValueError("source evaluator outside the declared bounded mesh/geometry domain")
    data = json.loads(case_json)
    data["metal_thickness_m"] = metal_length
    built = source.build_reference(
        data, grid, variant, numerical_policy=numerical_policy, gradient_policy=gradient_policy
    )
    return built, source.source_contract(built)


def _source_complete(record, contract, *, boundary=False, inputs=None, data=None):
    """No state or event is manufactured to repair incomplete source evidence."""
    if not isinstance(record, Mapping) or not isinstance(contract, Mapping):
        return False
    try:
        time_s = record["time_s"]
        if (
            isinstance(time_s, bool)
            or not isinstance(time_s, (int, float))
            or not math.isfinite(time_s)
            or time_s < 0
        ):
            return False
        rhs, alg = contract["rhs_size"], contract["algebraic_size"]
        if type(rhs) is not int or type(alg) is not int or min(rhs, alg) <= 0:
            return False
        _array(record["source_y"], (rhs + alg,))
        _array(record["source_rhs"], (rhs,))
        _array(record["source_algebraic"], (alg,))
        _array(record["source_algebraic_original"], (alg,))
        if boundary and (
            record.get("differential_state_unchanged_by_boundary_root") is not True
            or record.get("boundary_operation")
            != "actual source algebraic consistency only; no integration"
        ):
            return False
        events = record["source_events"]
        if len(events) != len(contract["all_events"]):
            return False
        for actual, expected in zip(events, contract["all_events"], strict=True):
            if any(actual[key] != expected[key] for key in ("ordinal", "name", "type")):
                return False
            values = actual["value"]
            _array(values, (len(values),))
            if not values:
                return False
        f, row = record["fields"], _array(record["row"], (len(RESPONSE_ORDER),))
        _assert_close(
            _layout_values(record, contract, "Positive particle concentration [mol.m-3]").reshape(
                np.asarray(f["cs_mol_m3_x_r"]).shape
            ),
            f["cs_mol_m3_x_r"],
            1e-9,
        )
        ce_name = (
            "Electrolyte concentration [mol.m-3]"
            if contract["variant"].startswith("basic")
            else "Porosity times concentration [mol.m-3]"
        )
        ce = _layout_values(record, contract, ce_name)
        if not contract["variant"].startswith("basic"):
            ns = contract["mesh"]["separator"]["cells"]
            ce = ce / np.r_[np.full(ns, 0.39), np.full(len(ce) - ns, 0.331)]
        _assert_close(ce, f["ce_mol_m3"], 1e-9)
        for name, key in (
            ("Electrolyte potential [V]", "phi_e"),
            ("Positive electrode potential [V]", "phi_s"),
        ):
            _assert_close(_layout_values(record, contract, name), f[key], 1e-12)
        _assert_close(
            _layout_values(record, contract, "Discharge capacity [A.h]") * 3600, [row[2]], 1e-12
        )
        work_name = (
            "half-cell-dfn45:Basic signed work [J]"
            if ce_name.startswith("Electrolyte")
            else "Discharge energy [W.h]"
        )
        _assert_close(
            _layout_values(record, contract, work_name)
            * (1 if ce_name.startswith("Electrolyte") else 3600),
            [row[3]],
            1e-12,
        )
        if not contract["variant"].startswith("basic"):
            _assert_close(_layout_values(record, contract, "Current variable [A]"), [row[1]], 1e-12)
        # Reconstruct the released expression DAG from pinned source. Merely
        # checking finiteness or multiplying two saved, possibly fake residual
        # vectors cannot authenticate them. This never integrates or recloses y.
        import battery_half_cell_dfn45_pybamm as source

        ns = contract["mesh"]["separator"]["cells"]
        np_ = contract["mesh"]["positive electrode"]["cells"]
        nr = contract["mesh"]["positive particle"]["cells"]
        metal_length = contract["inventory_geometry_contract"]["metal_length_m"]
        evaluation_case = deepcopy(data) if isinstance(data, Mapping) else case_data("H02")
        evaluation_case["metal_thickness_m"] = metal_length
        built, expected_contract = _source_evaluator(
            contract["variant"],
            ns,
            np_,
            nr,
            metal_length,
            json.dumps(json_value(evaluation_case), sort_keys=True),
            contract.get("numerical_policy", "legacy"),
            contract.get("gradient_policy", {}).get("name"),
            contract.get("declared_development_grid", {}).get("radial_exponent"),
        )
        if json_value(contract["state_layout"]) != json_value(expected_contract["state_layout"]):
            return False
        if any(
            json_value(contract.get(key)) != json_value(expected_contract.get(key))
            for key in ("gradient_policy", "declared_development_grid", "mesh")
        ):
            return False
        if json_value(contract["algebraic_conditioning"]) != json_value(
            expected_contract["algebraic_conditioning"]
        ):
            return False
        for key in (
            "source_parameter_manifest",
            "released_parameter_factory_source",
            "material_parameter_updates",
            "control_parameter_updates",
            "endpoint_contract",
            "metal_argument_contract",
            "original_events",
            "all_events",
        ):
            if json_value(contract[key]) != json_value(expected_contract[key]):
                return False
        if inputs is None:
            inputs = source.stage_inputs({"unit": "A", "target": row[1]}, contract["variant"])
        expected = source.state_record(built, time_s, record["source_y"], inputs)
        # The unscaled vector is authenticated against its original expression,
        # never subjected to the conditioned1e-8 gate.
        if np.max(abs(np.asarray(record["source_algebraic"]))) > 1e-8:
            return False
        for key in ("source_rhs", "source_algebraic", "source_algebraic_original", "row"):
            _assert_close(record[key], expected[key], 1e-12)
        for key in expected["fields"]:
            _assert_close(record["fields"][key], expected["fields"][key], 1e-12)
        for actual, fresh in zip(record["source_events"], expected["source_events"], strict=True):
            _assert_close(actual["value"], fresh["value"], 1e-12)
        if json_value(record["original_source_reported_inventory"]) != json_value(
            expected["original_source_reported_inventory"]
        ):
            return False
    except Exception:
        return False
    return True


def _assert_close(left, right, tolerance):
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if left.shape != right.shape or not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("shape or finite-data mismatch")
    if np.max(abs(left - right)) > tolerance:
        raise ValueError("raw state and derived physical fields disagree")


def _source_stage_inputs(stage, contract):
    from battery_half_cell_dfn45_pybamm import stage_inputs

    return stage_inputs(stage, contract["variant"])


def _initial_source_stop(trace, partial):
    if not isinstance(trace, Mapping) or not isinstance(partial, Mapping):
        return None
    error = trace.get("exception")
    if not isinstance(error, Mapping) or not isinstance(error.get("message"), str):
        return None
    try:
        error, boundary, contract = (
            trace["exception"],
            partial["failed_boundary_t0"],
            partial["contract"],
        )
        message = error["message"].lower()
        index, request = partial["failed_stage"], partial["failed_request"]
        if (
            error["type"] != "SolverError"
            or "initial conditions" not in message
            or "events" not in message
            or type(index) is not int
            or index < 0
            or not isinstance(request, Mapping)
            or set(request) != {"duration_s", "target", "unit", "samples", "expected"}
            or request["unit"] not in ("A", "V")
            or not transport._finite_array([request["duration_s"], request["target"]], (2,))
            or request["duration_s"] <= 0
            or type(request["samples"]) is not int
            or request["samples"] < 2
            or not isinstance(request["expected"], str)
            or not _source_complete(
                boundary,
                contract,
                boundary=True,
                inputs=_source_stage_inputs(request, contract),
                data=partial.get("case"),
            )
        ):
            return None
        completed = raw_completed_segments(trace)
        if index != len(completed):
            return None
        if completed:
            previous = completed[-1]["records"][-1]
            if (
                not _source_complete(previous, contract)
                or previous["time_s"] != boundary["time_s"]
                or previous["source_y"][: contract["rhs_size"]]
                != boundary["source_y"][: contract["rhs_size"]]
            ):
                return None
        elif boundary["time_s"] != 0:
            return None
        for event in boundary["source_events"]:
            if (
                event["type"] == "TERMINATION"
                and event["name"].lower() in message
                and min(event["value"]) <= 0
            ):
                return {
                    "index": index,
                    "request": deepcopy(request),
                    "boundary_t0": boundary,
                    "records": [boundary],
                    "original_reported_times_s": [],
                    "actual_start_time_s": boundary["time_s"],
                    "actual_end_time_s": boundary["time_s"],
                    "termination": event["name"],
                    "physical_initial_stop": True,
                    "original_exception": error,
                }
    except Exception:
        pass
    return None


def _view(trace):
    trace = trace if isinstance(trace, Mapping) else {}
    partial = trace.get("partial_history") or {}
    partial = partial if isinstance(partial, Mapping) else {}
    contract = trace.get("contract", partial.get("contract", {}))
    contract = contract if isinstance(contract, Mapping) else {}
    segments = raw_completed_segments(trace)
    initial_stop = _initial_source_stop(trace, partial)
    if initial_stop is not None:
        segments.append(initial_stop)
    return contract, segments, initial_stop is not None


def _interp(nodes, values, target):
    if target[0] < nodes[0] - 1e-14 or target[-1] > nodes[-1] + 1e-14:
        raise ValueError("common profile would require extrapolation")
    return np.interp(target, nodes, values)


def _profile_values(record, contract):
    mesh, f = _mesh(contract), record["fields"]
    ns, np_ = (len(mesh[name][0]) for name in ("separator", "positive electrode"))
    nr = len(mesh["positive particle"][0])
    z = np.asarray(PROFILE_COORDINATES)
    px = mesh["positive electrode"][0]
    target_x = LS + LP * z
    target_r = RADIUS * z
    radial = _array(f["theta_x_r"], (np_, nr))
    radial_r = np.asarray([_interp(mesh["positive particle"][0], row, target_r) for row in radial])
    profiles = {
        "radial_fraction": np.column_stack(
            [_interp(px, radial_r[:, i], target_x) for i in range(len(z))]
        ),
        "surface_fraction": _interp(px, _array(f["theta_surface"], (np_,)), target_x),
    }
    for key, metric in (("ce_mol_m3", "salt_mol_m3"), ("phi_e", "electrolyte_potential_V")):
        values = _array(f[key], (ns + np_,))
        profiles[metric] = np.r_[
            _interp(mesh["separator"][0], values[:ns], LS * z), _interp(px, values[ns:], target_x)
        ]
    for key, metric in (
        ("phi_s", "solid_potential_V"),
        ("positive_ocp_V", "positive_ocp_V"),
        ("positive_eta_V", "positive_eta_V"),
        ("positive_j0_A_m2", "positive_j0_A_m2"),
    ):
        profiles[metric] = _interp(px, _array(f[key], (np_,)), target_x)
    qe = mesh["positive electrode"][1]
    q = _array(f["positive_q_cell_A"], (np_,))
    edges = np.linspace(LS, LS + LP, 11)
    overlap = np.maximum(
        0, np.minimum(qe[1:, None], edges[None, 1:]) - np.maximum(qe[:-1, None], edges[None, :-1])
    )
    profiles["local_current_A"] = (q / np.diff(qe)) @ overlap
    salt_edges = np.r_[mesh["separator"][1], mesh["positive electrode"][1][1:]]
    common_faces = np.r_[np.linspace(0, LS, 11), np.linspace(LS, LS + LP, 11)[1:]]
    for key, metric, coefficient in (
        ("ionic_current_A_m2", "ionic_face_current_A", AREA),
        (
            "diffusive_salt_flux_mol_m2_s",
            "diffusive_salt_equivalent_A",
            AREA * contract["F_C_mol"] / (1 - 0.38),
        ),
        ("total_cation_flux_mol_m2_s", "cation_flux_equivalent_A", AREA * contract["F_C_mol"]),
    ):
        profiles[metric] = (
            _interp(salt_edges, _array(f[key], (ns + np_ + 1,)), common_faces) * coefficient
        )
    profiles["solid_face_current_A"] = (
        _interp(qe, _array(f["solid_current_A_m2"], (np_ + 1,)), edges) * AREA
    )
    return profiles


def _stage_view(segment, contract):
    boundary = segment["boundary_t0"]
    records = [boundary, *(r for r in segment["records"] if r["time_s"] > boundary["time_s"])]
    times = _array([r["time_s"] for r in records], (len(records),))
    rows = _array([r["row"] for r in records], (len(records), len(RESPONSE_ORDER)))
    if not len(times) or np.any(np.diff(times) <= 0):
        raise ValueError("non-monotone original stage times")
    projected = [_profile_values(record, contract) for record in records]
    return {
        "times": times,
        "rows": rows,
        "profiles": {
            key: np.asarray([value[key] for value in projected]) for key in PROFILE_METRICS
        },
        "records": records,
        "raw": segment,
        "stop": stop_kind(segment["termination"]),
        "contract": contract,
    }


def _physical_checks(record, contract, data):
    f, row, mesh = record["fields"], np.asarray(record["row"], dtype=float), _mesh(contract)
    ns, np_ = (len(mesh[name][0]) for name in ("separator", "positive electrode"))
    widths = np.r_[np.diff(mesh["separator"][1]), np.diff(mesh["positive electrode"][1])]
    pore = AREA * widths * np.r_[np.full(ns, 0.39), np.full(np_, 0.331)]
    active = AREA * widths[ns:] * 0.518
    shells = np.diff((mesh["positive particle"][1] / RADIUS) ** 3)
    ce, cs = np.asarray(f["ce_mol_m3"]), np.asarray(f["cs_mol_m3_x_r"])
    q = np.asarray(f["positive_j_A_m2"]) * 3 * active / RADIUS
    ie, solid = (
        np.asarray(f["ionic_current_A_m2"]) * AREA,
        np.asarray(f["solid_current_A_m2"]) * AREA,
    )
    closure = np.r_[
        np.diff(ie) - np.r_[np.zeros(ns), q],
        np.diff(solid) + q,
        ie[0] - row[1],
        ie[-1],
        solid[0],
        solid[-1] - row[1],
        sum(q) + row[1],
    ]
    thermal = 2 * contract["R_J_mol_K"] * data["temperature_K"] / contract["F_C_mol"]
    material_j0 = (
        3.5e-8
        * 96485.33212331001
        * (48230 if contract.get("variant") == "basic-full-raw-metal" else 1 / 1.3e-5) ** 0.7
        * row[7] ** 0.3
    )
    metal_eta = thermal * np.arcsinh(row[1] / (2 * AREA * row[8]))
    positive_eta = thermal * np.arcsinh(
        np.asarray(f["positive_j_A_m2"]) / (2 * np.asarray(f["positive_j0_A_m2"]))
    )
    bv = (
        np.asarray(f["phi_s"])
        - np.asarray(f["phi_e"])[ns:]
        - np.asarray(f["positive_ocp_V"])
        - np.asarray(f["positive_eta_V"])
    )
    theta = np.asarray(f["theta_surface"])
    expected_j0 = (
        5.76e-11 * 96485.33212331001 * np.sqrt(ce[ns:] * (theta * CMAX) * (CMAX * (1 - theta)))
    )
    clipped = np.clip(theta, 1e-10, 1 - 1e-10)
    amplitude, exponent, offset = XU2019_PROJECTION["ocv_exponential"]
    expected_ocp = (
        np.polynomial.polynomial.polyval(clipped, XU2019_PROJECTION["ocv_polynomial_V"])
        + amplitude * np.exp(exponent * clipped - offset)
        + 1e-6 * (1 / clipped + 1 / (clipped - 1))
    )
    diffusive = np.asarray(f["diffusive_salt_flux_mol_m2_s"])
    concentration_floor = min(float(np.min(ce)), row[7])
    _assert_close(f["electrolyte_salt_amounts_mol"], ce * pore, 1e-14)
    _assert_close(f["positive_shell_amounts_mol_x_r"], cs * active[:, None] * shells, 1e-14)
    _assert_close(
        [row[5], row[6]], [np.sum(cs * active[:, None] * shells), np.sum(ce * pore)], 1e-14
    )
    _assert_close(q, f["positive_q_cell_A"], 1e-12)
    _assert_close(f["theta_x_r"], cs / CMAX, 1e-12)
    return {
        "physical-current-closure": np.max(abs(closure)) <= 1e-8,
        "metal-BV": (
            abs(row[8] - material_j0) <= 1e-8
            and abs(row[9] - metal_eta) <= 1e-8
            and abs(row[10] - row[1] * data["metal_thickness_m"] / (AREA * 10776000)) <= 1e-12
            and abs(f["metal_phi_e_V"] + row[9] + row[10]) <= 1e-8
        ),
        "positive-BV": np.max(abs(bv)) <= 1e-6
        and np.max(abs(np.asarray(f["positive_eta_V"]) - positive_eta)) <= 1e-8
        and np.max(abs(np.asarray(f["positive_j0_A_m2"]) - expected_j0)) <= 1e-8
        and np.max(abs(np.asarray(f["positive_ocp_V"]) - expected_ocp)) <= 1e-8,
        "left-salt-boundary": abs(diffusive[0] * AREA * contract["F_C_mol"] - (1 - 0.38) * row[1])
        <= 1e-8
        and abs(diffusive[-1]) <= 1e-12,
        "physical-domain": (
            concentration_floor >= data["minimum_concentration_mol_m3"] - 1e-6
            and row[4] >= -1e-10
            and np.min(cs / CMAX) >= -1e-10
            and np.max(cs / CMAX) <= 1 + 1e-10
            and np.min(f["theta_surface"]) >= 0.01 - 1e-6
            and np.max(f["theta_surface"]) <= 0.99 + 1e-6
            and data["minimum_voltage_V"] - 1e-6 <= row[0] <= data["maximum_voltage_V"] + 1e-6
        ),
    }


@lru_cache(maxsize=2)
def _native_evaluator(spec_json, stage_json, development_grid_json=None):
    from chemworld.physchem.battery_half_cell_dfn import (
        BatteryHalfCellDFNSpec,
        half_cell_dfn_system,
    )
    from chemworld.physchem.reactor_shared import ScalarControlProgram
    from chemworld.physchem.residual_system import ResidualEquation, ResidualVariable

    encoded, stage = json.loads(spec_json), json.loads(stage_json)
    ns, np_ = encoded["electrolyte"]["cells"]
    nr = encoded["positive"]["cells"]
    if development_grid_json is None:
        if not 3 <= min(ns, np_) <= max(ns, np_) <= 32 or not 3 <= nr <= 64:
            raise ValueError("native raw layout outside bounded declared mesh domain")
    else:
        from battery_half_cell_dfn45_meshes import development_mesh_bounds

        declaration = json.loads(development_grid_json)
        if not isinstance(declaration, Mapping) or "radial_exponent" not in declaration:
            raise ValueError("native development mesh requires explicit radial_exponent")
        declared = development_mesh_bounds(declaration)
        exponent = encoded["positive"].get("mesh_exponent", 1.0)
        if (
            any(type(value) is not int for value in (ns, np_, nr))
            or min(ns, np_, nr) < 3
            or tuple(declared["x"]) != (ns, np_)
            or declared["r"] != nr
            or isinstance(exponent, bool)
            or not isinstance(exponent, (int, float))
            or not math.isfinite(exponent)
            or float(declared["radial_exponent"]) != float(exponent)
        ):
            raise ValueError("native development mesh differs from the actual encoded spec")
    spec = BatteryHalfCellDFNSpec.from_dict(encoded)
    program = ScalarControlProgram(((0.0, stage["target"]),), stage["unit"])
    system = half_cell_dfn_system(spec, program)
    if system.ports or system.energy_ports:
        raise ValueError("this declared half-cell family has no external material/energy ports")
    material = tuple(system.thermodynamics.inventory_keys)
    names = (
        *material,
        "temperature",
        "boundary_heat",
        "chemical_energy",
        "heat_loss",
        "work",
        *(coordinate.name for coordinate in system.coordinates),
    )
    units = (
        *("mol" for _ in material),
        "K",
        *("J" for _ in range(4)),
        *(coordinate.unit for coordinate in system.coordinates),
    )
    scales = (
        *(system.inventory_scales_mol[key] for key in material),
        system.temperature_scale_K,
        *(system.energy_scale_J for _ in range(4)),
        *(coordinate.scale for coordinate in system.coordinates),
    )
    variables = [
        asdict(ResidualVariable(name, unit, scale))
        for name, unit, scale in zip(names, units, scales, strict=True)
    ]
    equations = [
        asdict(ResidualEquation(name, unit + "/s", scale))
        for name, unit, scale in zip(names, units, scales, strict=True)
    ]
    variables += [asdict(variable) for variable in system.algebraic.variables]
    equations += [asdict(equation) for equation in system.algebraic.equations]
    return spec, system, material, variables, equations, len(names)


def _native_raw_complete(view, contract):
    from chemworld.physchem.coupled_system import CoupledStateView, assemble_coupled_rates
    from chemworld.physchem.reactor_shared import ScalarControlProgram

    raw = view["raw"]
    arguments = (
        json.dumps(json_value(contract["spec"]), sort_keys=True),
        json.dumps(json_value(raw["request"]), sort_keys=True),
    )
    if "radial_exponent" in contract.get("grid", {}):
        arguments += (json.dumps(json_value(contract["grid"]), sort_keys=True),)
    spec, system, material, variables, equations, differential = _native_evaluator(*arguments)
    reports = raw["original_residual_reports"]
    if not reports:
        return False
    index = {variable["name"]: i for i, variable in enumerate(variables)}
    program = ScalarControlProgram(((0.0, raw["request"]["target"]),), raw["request"]["unit"])
    volume = spec.initial_state().volume_L
    points, terminal_events = [], []
    for report in reports:
        if (
            json_value(report["variables"]) != json_value(variables)
            or json_value(report["equations"]) != json_value(equations)
            or report["differential_indices"] != list(range(differential))
            or report["algebraic_indices"] != list(range(differential, len(variables)))
            or not report["points"]
            or report["success"] is not True
        ):
            return False
        for point in [*report["points"], *report["events"]]:
            y, rate = (
                _array(point["y"], (len(variables),)),
                _array(point["ydot"], (len(variables),)),
            )
            relative = point["relative_time_s"]
            if not math.isfinite(relative) or relative < 0:
                return False
            coordinates = {
                coordinate.name: y[index[coordinate.name]] for coordinate in system.coordinates
            }
            candidate = CoupledStateView(
                {name: max(0.0, y[index[name]]) for name in material},
                spec.temperature_K,
                volume,
                raw["actual_start_time_s"] + relative,
                coordinates=coordinates,
                algebraic_values={
                    variable.name: y[index[variable.name]]
                    for variable in system.algebraic.variables
                },
            )
            rates = assemble_coupled_rates(
                candidate,
                processes=system.processes,
                thermodynamics=system.thermodynamics,
                heat_flux=system.heat_flux,
                ports=system.ports,
                isothermal=system.isothermal,
                energy_ports=system.energy_ports,
            )
            expected_rates = np.r_[
                [rates.amount_rates_mol_s[name] for name in material],
                rates.temperature_rate_K_s,
                rates.heater_W,
                rates.chemical_energy_W,
                rates.loss_W,
                rates.work_W,
                [rates.coordinate_rates[coordinate.name] for coordinate in system.coordinates],
            ]
            tolerances = np.r_[np.full(len(material), 1e-12), 1e-8, np.full(4, 1e-7), 1e-8, 1e-7]
            if (
                np.any(abs(rate[:differential] - expected_rates) > tolerances)
                or np.max(abs(system.algebraic.residual(candidate))) > 1e-8
            ):
                return False
            points.append((candidate.time_s, y, candidate))
        terminal_events.extend(report["events"])
    boundary_state = raw["boundary_t0"]["native_state"]
    for record in view["records"]:
        state = record["native_state"]
        if state["temperature_K"] != spec.temperature_K or state["volume_L"] != volume:
            return False
        if set(state["amounts_mol"]) != set(material) or set(state["coordinates"]) != {
            coordinate.name for coordinate in system.coordinates
        }:
            return False
        matched = [
            (y, candidate)
            for time_s, y, candidate in points
            if abs(time_s - record["time_s"]) <= 1e-8
        ]
        if not matched or state["time_s"] != record["time_s"]:
            return False
        expected = {
            **state["amounts_mol"],
            **state["coordinates"],
            "temperature": state["temperature_K"],
            "boundary_heat": state["boundary_heat_J"] - boundary_state["boundary_heat_J"],
            "chemical_energy": state["chemical_energy_change_J"]
            - boundary_state["chemical_energy_change_J"],
            "heat_loss": state["heat_loss_J"] - boundary_state["heat_loss_J"],
            "work": state["work_J"] - boundary_state["work_J"],
            "battery:current_A": record["row"][1],
            "battery:voltage_V": record["row"][0],
            "half-cell-dfn:metal-electrolyte-potential_V": record["fields"]["metal_phi_e_V"],
        }
        expected.update(
            zip(spec.reaction_keys, record["fields"]["positive_q_cell_A"][:-1], strict=True)
        )
        valid = False
        for y, candidate in matched:
            valid = True
            for name, value in expected.items():
                tolerance = (
                    1e-8
                    if name not in material
                    else max(1e-24, 1e-10 * system.inventory_scales_mol[name])
                )
                if abs(y[index[name]] - value) > tolerance:
                    valid = False
                    break
            if valid:
                fresh = _native_record(spec, candidate, program, system)
                for name, expected_field in fresh["fields"].items():
                    tolerance = 1e-6 if name in ("ce_mol_m3", "cs_mol_m3_x_r") else 1e-8
                    try:
                        _assert_close(record["fields"][name], expected_field, tolerance)
                    except ValueError:
                        valid = False
                        break
            if valid:
                break
        if not valid:
            return False
    return view["stop"] == "complete" or any(
        stop_kind(point["name"]) == view["stop"]
        and abs(raw["actual_start_time_s"] + point["relative_time_s"] - view["times"][-1]) <= 1e-8
        for point in terminal_events
    )


def _source_boundary_identity(raw, contract, previous):
    """Compare unchanged differential states only at the same original time."""
    boundary, size = raw["boundary_t0"], contract["rhs_size"]
    if type(raw["index"]) is not int or raw["index"] < 0:
        return False
    if previous is None:
        return raw["index"] == 0 and boundary["time_s"] == 0
    prior = previous["raw"]
    times, values = prior["original_all_ts"][-1], np.asarray(prior["original_all_ys"][-1])
    return (
        raw["index"] == prior["index"] + 1
        and boundary["time_s"] == times[-1] == prior["records"][-1]["time_s"]
        and np.array_equal(values[:, -1], prior["records"][-1]["source_y"])
        and np.array_equal(boundary["source_y"][:size], values[:size, -1])
    )


def _source_first_state_continuity(raw, contract):
    boundary, first = raw["boundary_t0"], raw["records"][0]
    t0, first_time = boundary["time_s"], first["time_s"]
    if raw["requested_start_time_s"] != t0 or raw["actual_start_time_s"] != first_time:
        return False
    size = contract["rhs_size"]
    initial = np.asarray(boundary["source_y"][:size])
    returned = np.asarray(first["source_y"][:size])
    if first_time == t0:
        return np.array_equal(initial, returned)
    # Pinned BaseSolver.step shifts a nonzero stage start by exactly one float
    # to avoid duplicate times. Preserve that actual state advancement rather
    # than demanding bit equality between states at different times.
    if t0 == 0 or first_time != np.nextafter(t0, np.inf):
        return False
    rate = np.maximum(abs(np.asarray(boundary["source_rhs"])), abs(np.asarray(first["source_rhs"])))
    # Only a one-ULP time interval is admitted. Bound its differential change
    # by the retained, freshly authenticated endpoint RHS plus arithmetic
    # roundoff; this is independent of all physical and residual error gates.
    roundoff = 8 * np.finfo(float).eps * np.maximum(1, np.maximum(abs(initial), abs(returned)))
    return bool(np.all(abs(returned - initial) <= (first_time - t0) * rate + roundoff))


def _raw_retained(view, contract, native, data=None, *, previous=None):
    raw = view["raw"]
    if native:
        if not _native_raw_complete(view, contract):
            return False
        reports = raw["original_residual_reports"]
        return (
            bool(reports)
            and all(
                report["points"]
                and len(report["variables"]) > 0
                and all(
                    transport._finite_array(p["y"], (len(report["variables"]),))
                    and transport._finite_array(p["ydot"], (len(report["variables"]),))
                    for p in [*report["points"], *report["events"]]
                )
                for report in reports
            )
            and all(isinstance(r.get("native_state"), Mapping) for r in view["records"])
        )
    if not (
        _source_complete(
            raw["boundary_t0"], contract, boundary=True, inputs=raw.get("inputs"), data=data
        )
        and all(
            _source_complete(record, contract, inputs=raw.get("inputs"), data=data)
            for record in raw["records"]
        )
    ):
        return False
    if not _source_boundary_identity(raw, contract, previous):
        return False
    if raw.get("physical_initial_stop"):
        return True
    event_t, event_y = raw["original_t_event"], raw["original_y_event"]
    # Pinned IDAKLU exposes endpoint t_event/y_event even at 'final time'.
    # Its actual Solution constructor always supplies this pair.
    event_t = np.asarray(event_t, dtype=float).ravel()
    event_y = np.asarray(event_y, dtype=float)
    size = contract["rhs_size"] + contract["algebraic_size"]
    if (
        not len(event_t)
        or not transport._finite_array(event_t, (len(event_t),))
        or not transport._finite_array(event_y, (size, len(event_t)))
        or abs(event_t[-1] - view["times"][-1]) > 1e-8
        or not np.array_equal(event_y[:, -1], raw["records"][-1]["source_y"])
    ):
        return False
    size = contract["rhs_size"] + contract["algebraic_size"]
    all_ts, all_ys = raw["original_all_ts"], raw["original_all_ys"]
    if not all_ts or len(all_ts) != len(all_ys):
        return False
    for times, values in zip(all_ts, all_ys, strict=True):
        _array(times, (len(times),))
        _array(values, (size, len(times)))
    flat_t = np.concatenate(all_ts)
    flat_y = np.concatenate(all_ys, axis=1)
    records = raw["records"]
    return (
        np.array_equal(flat_t, [r["time_s"] for r in records])
        and np.array_equal(flat_y.T, [r["source_y"] for r in records])
        and np.all(np.diff(flat_t) > 0)
        and _source_first_state_continuity(raw, contract)
    )


def _charge_work_rates(view, contract, stage):
    rows, raw = view["rows"], view["raw"]
    if contract.get("variant") == "native":
        if not _native_raw_complete(view, contract):
            return False
        # Native report y/ydot are physical units (the common packer documents
        # this). Authenticate rates at every exposed point and event, including
        # CV; a nonempty report alone is not evidence of the port equations.
        reports = raw["original_residual_reports"]
        if not reports:
            return False
        for report in reports:
            names = {variable["name"]: i for i, variable in enumerate(report["variables"])}
            for point in [*report["points"], *report["events"]]:
                y, rate = np.asarray(point["y"]), np.asarray(point["ydot"])
                current, voltage = y[names["battery:current_A"]], y[names["battery:voltage_V"]]
                if (
                    abs(rate[names["discharged_charge_C"]] - current) > 1e-8
                    or abs(rate[names["delivered_work_J"]] - current * voltage) > 1e-7
                ):
                    return False
                if abs(rate[names["metal:Li"]] * F + current) > 1e-8:
                    return False
                positive = [i for name, i in names.items() if name.startswith("positive:x:")]
                salt = [i for name, i in names.items() if name.startswith("electrolyte:salt:")]
                if (
                    not positive
                    or not salt
                    or abs(np.sum(rate[positive]) * F - current) > 1e-8
                    or abs(np.sum(rate[salt])) * F > 1e-8
                ):
                    return False
        if stage["unit"] == "A":
            return (
                np.max(
                    abs(
                        rows[:, 2]
                        - rows[0, 2]
                        - stage["target"] * (view["times"] - view["times"][0])
                    )
                )
                <= 1e-6
            )
        return True
    for record in view["records"]:
        row = record["row"]
        qdot = _layout_values(record, contract, "Discharge capacity [A.h]", rate=True)[0] * 3600
        basic = contract["variant"].startswith("basic")
        name = "half-cell-dfn45:Basic signed work [J]" if basic else "Discharge energy [W.h]"
        wdot = _layout_values(record, contract, name, rate=True)[0] * (1 if basic else 3600)
        if abs(qdot - row[1]) > 1e-8 or abs(wdot - row[0] * row[1]) > 1e-7:
            return False
    return True


def _stage_checks(view, previous, original, stage, start, contract, data):
    result = dict.fromkeys(STAGE_CHECKS, False)
    if view is None:
        return result
    rows, times, raw = view["rows"], view["times"], view["raw"]
    result.update(present=True, schema=True)
    result["boundary-continuity"] = abs(times[0] - start) <= 1e-8
    if previous is not None:
        result["boundary-continuity"] &= (
            abs(times[0] - previous["times"][-1]) <= 1e-8
            and np.array_equal(rows[0, 2:7], previous["rows"][-1, 2:7])
            and all(
                np.array_equal(
                    view["records"][0]["fields"][key], previous["records"][-1]["fields"][key]
                )
                for key in ("ce_mol_m3", "cs_mol_m3_x_r")
            )
        )
    wanted = np.asarray(stage_times(stage)) + start
    required = wanted[wanted <= times[-1] + 1e-8]
    result["requested-observations"] = (
        bool(len(required))
        and all(np.min(abs(times - t)) <= 1e-8 for t in required)
        and times[-1] <= wanted[-1] + 1e-8
        and abs(times[-1] - raw["actual_end_time_s"]) <= 1e-8
    )
    result["stop-identity"] = view["stop"] == stage["expected"]
    if view["stop"] == "complete":
        result["stop-value"] = abs(times[-1] - wanted[-1]) <= 1e-8
    elif view["stop"] == "finite-metal-depleted":
        result["stop-value"] = abs(rows[-1, 4]) <= 1e-10 and rows[-1, 1] > 0
    elif view["stop"] in ("minimum-voltage", "maximum-voltage"):
        cutoff = (
            data["minimum_voltage_V"]
            if view["stop"].startswith("minimum")
            else data["maximum_voltage_V"]
        )
        result["stop-value"] = abs(rows[-1, 0] - cutoff) <= 1e-6
    result["hardware-current"] = np.max(abs(rows[:, 1])) <= data["maximum_current_A"] + 1e-8
    result["imposed-control"] = (
        np.max(abs(rows[:, 1] - stage["target"])) <= 1e-8
        if stage["unit"] == "A"
        else all(
            abs(row[0] - stage["target"]) <= 1e-6
            if abs(row[1]) < data["maximum_current_A"] - 1e-8
            else (
                row[0] >= stage["target"] - 1e-6 if row[1] > 0 else row[0] <= stage["target"] + 1e-6
            )
            for row in rows
        )
    )
    faraday = contract["F_C_mol"]
    result["metal-faradaic"] = (
        np.max(abs(faraday * (rows[:, 4] - original[4]) + rows[:, 2] - original[2])) <= 1e-6
    )
    result["positive-faradaic"] = (
        np.max(abs(faraday * (rows[:, 5] - original[5]) - rows[:, 2] + original[2])) <= 1e-6
    )
    result["salt-inventory"] = np.max(abs(rows[:, 6] - original[6])) <= 1e-10
    result["solid-lithium"] = (
        np.max(abs(rows[:, 4] + rows[:, 5] - original[4] - original[5])) <= 1e-10
    )
    physical = [_physical_checks(record, contract, data) for record in view["records"]]
    for name in physical[0]:
        result[name] = all(check[name] for check in physical)
    result["raw-state-retention"] = _raw_retained(
        view, contract, contract.get("variant") == "native", data, previous=previous
    )
    result["all-original-times-retained"] = raw.get("physical_initial_stop", False) or raw[
        "original_reported_times_s"
    ] == [record["time_s"] for record in raw["records"]]
    result["charge-work-rates"] = _charge_work_rates(view, contract, stage)
    return result


def _check(name, passed, **details):
    return {"name": name, "passed": bool(passed), **json_value(details)}


def _route_assessment(route, trace):
    key, data = route_id(route), case_data(route["unit"])
    contract, segments, initial_stop = _view(trace)
    checks = dict.fromkeys(ROUTE_CHECKS, False)
    checks["available"] = isinstance(trace, Mapping)
    checks["execution"] = (
        isinstance(trace, Mapping) and bool(trace) and ("exception" not in trace or initial_stop)
    )
    checks["stage-count"] = len(segments) == len(data["stages"])
    native = route["implementation"] == "native"
    errors, views = [], []
    try:
        mesh = _mesh(contract)
        ns, np_ = route["grid"]["x"]
        from battery_half_cell_dfn45_meshes import radial_edges

        particle_edges = radial_edges(route["grid"], RADIUS)
        checks["mesh"] = all(
            len(mesh[name][0]) == count and np.max(abs(mesh[name][1] - expected)) <= 1e-16
            for name, count, expected in (
                ("separator", ns, np.linspace(0, LS, ns + 1)),
                ("positive electrode", np_, np.linspace(LS, LS + LP, np_ + 1)),
                (
                    "positive particle",
                    route["grid"]["r"],
                    particle_edges,
                ),
            )
        )
        if native:
            checks["identity"] = json_value(contract["spec"]) == json_value(
                native_spec(data, route["grid"]).to_dict()
            )
            checks["conditioning-policy"] = checks["source-conventions"] = True
        else:
            manifest = contract["source_parameter_manifest"]
            checks["identity"] = (
                contract["variant"] == route["implementation"]
                and contract["pybamm_version"] == "25.4.2"
                and contract["casadi_version"] == "3.6.7"
                and len(manifest) == 66
            )
            blocks = contract["algebraic_conditioning"]["blocks"]
            factor = ((LS + LP) / (data["metal_thickness_m"] + LS + LP)) ** 2
            checks["conditioning-policy"] = (
                contract["algebraic_conditioning"]["root_method"] == "casadi"
                and sum(block["rows"] for block in blocks) == contract["algebraic_size"]
                and all(
                    block["factor"] > 0
                    and abs(block["factor"] - (factor if block["charge_equation"] else 1)) <= 1e-15
                    and block["actual_row_factors"] == [block["factor"]] * block["rows"]
                    and block["original_discretized_expression"]
                    and block["conditioned_discretized_expression"]
                    for block in blocks
                )
            )
            originals = {(e["name"], e["type"]) for e in contract["original_events"]}
            checks["source-conventions"] = (
                originals <= {(e["name"], e["type"]) for e in contract["all_events"]}
                and len(contract["all_events"]) == len(contract["original_events"]) + 4
                and contract["original_ocp_and_barrier_preserved"]
                and contract["endpoint_contract"]["clip"] == 1e-10
                and contract["endpoint_contract"]["barrier_V"] == 1e-6
                and "Positive electrode OCP [V]" not in contract["material_parameter_updates"]
                and contract["metal_argument_contract"]["route_uses_raw_basic_argument"]
                == (route["implementation"] == "basic-full-raw-metal")
            )
    except Exception as exc:
        errors.append({"phase": "contract", "type": type(exc).__name__, "message": str(exc)})
    original = None
    all_checks = []
    for index, stage in enumerate(data["stages"]):
        view = None
        try:
            if index < len(segments):
                view = _stage_view(segments[index], contract)
                original = view["rows"][0] if original is None else original
        except Exception as exc:
            errors.append({"phase": "stage-schema", "stage": index, "message": str(exc)})
        views.append(view)
        try:
            stage_checks = _stage_checks(
                view,
                views[index - 1] if index else None,
                original,
                stage,
                sum(s["duration_s"] for s in data["stages"][:index]),
                contract,
                data,
            )
        except Exception as exc:
            stage_checks = dict.fromkeys(STAGE_CHECKS, False)
            errors.append({"phase": "stage-check", "stage": index, "message": str(exc)})
        all_checks += [
            _check(f"{key}/stage-{index}/{name}", stage_checks[name]) for name in STAGE_CHECKS
        ]
    if original is not None and views and views[0] is not None:
        spec = native_spec(data, route["grid"])
        state = native_initial_state(data, spec)
        checks["initial-state"] = (
            views[0]["times"][0] == 0
            and np.max(abs(original[2:4])) <= 1e-12
            and abs(original[4] - spec.metal.initial_amount_mol) <= 1e-14
            and abs(
                original[5]
                - sum(state.amounts_mol[k] for _, _, _, keys, _ in spec.particle_bins for k in keys)
            )
            <= 1e-14
            and abs(original[6] - sum(state.amounts_mol[k] for k in spec.electrolyte.keys)) <= 1e-14
            and np.max(abs(np.asarray(views[0]["records"][0]["fields"]["cs_mol_m3_x_r"]) - 4631.0))
            <= 1e-9
            and np.max(
                abs(
                    np.asarray(views[0]["records"][0]["fields"]["ce_mol_m3"])
                    - spec.electrolyte.concentrations(state)
                )
            )
            <= 1e-8
        )
    all_checks[:0] = [_check(key + "/" + name, checks[name]) for name in ROUTE_CHECKS]
    return {
        "route": route,
        "checks": all_checks,
        "passed": all(c["passed"] for c in all_checks),
        "errors": errors,
        "physical_initial_source_stop": initial_stop,
        "raw_completed_stages": len(raw_completed_segments(trace)),
        "available_physical_stages": sum(view is not None for view in views),
    }, views


def _sample(view, times):
    rows = view["rows"].copy()
    # Transform inventory comparison units transparently; raw mol rows remain in archives/tails.
    contract = view["contract"]
    metal_initial = (
        AREA * contract["case"]["metal_thickness_m"] / 1.3e-5
        if contract.get("variant") == "native"
        else contract["finite_metal_contract"]["initial_mol"]
    )
    positive_initial = AREA * LP * 0.518 * 4631
    for index, initial in ((4, metal_initial), (5, positive_initial)):
        rows[:, index] = (rows[:, index] - initial) * contract["F_C_mol"]
    rows[:, 6] /= sum(
        view["contract"]["inventory_geometry_contract"]["actual_salt_pore_volumes_m3"]
    )
    values = {
        name: np.interp(times, view["times"], rows[:, i]) for i, name in enumerate(RESPONSE_ORDER)
    }
    for key, array in view["profiles"].items():
        flat = array.reshape(len(view["times"]), -1)
        values[key] = np.column_stack(
            [np.interp(times, view["times"], flat[:, i]) for i in range(flat.shape[1])]
        )
    return values


def _metric_gate(metric):
    aliases = {
        "metal_Li_mol": "charge_C",
        "positive_Li_mol": "charge_C",
        "electrolyte_salt_mol": "salt_mol_m3",
        "metal_boundary_ce_mol_m3": "salt_mol_m3",
        "metal_ohmic_V": "potential_V",
        "metal_eta_V": "metal_eta_V",
        "electrolyte_potential_V": "potential_V",
        "solid_potential_V": "potential_V",
        "positive_ocp_V": "potential_V",
        "positive_eta_V": "metal_eta_V",
        "positive_j0_A_m2": "metal_j0_A_m2",
        "ionic_face_current_A": "local_current_A",
        "solid_face_current_A": "local_current_A",
        "diffusive_salt_equivalent_A": "local_current_A",
        "cation_flux_equivalent_A": "local_current_A",
    }
    return THRESHOLDS["same_equation_absolute_plus_relative"][aliases.get(metric, metric)]


def _norm(left, right, metric, factor):
    left, right = np.asarray(left), np.asarray(right)
    if (
        left.shape != right.shape
        or not left.size
        or not np.isfinite(left).all()
        or not np.isfinite(right).all()
    ):
        raise ValueError("invalid paired physical arrays")
    absolute, relative = _metric_gate(metric)
    limit = factor * (absolute + relative * np.maximum(abs(left), abs(right)))
    signed = left - right
    difference = abs(signed)
    return {
        "passed": bool(np.all(difference <= limit)),
        "maximum_absolute": float(difference.max()),
        "maximum_ratio": float(np.max(difference / limit)),
        "points": left.size,
        "absolute_gate": factor * absolute,
        "relative_gate": factor * relative,
        "signed_difference_min": float(np.min(signed)),
        "signed_difference_max": float(np.max(signed)),
        "signed_difference_mean": float(np.mean(signed)),
        "direction": "left-minus-right in the declared comparison edge order",
        "comparison_unit": "C"
        if metric in ("metal_Li_mol", "positive_Li_mol")
        else "mol.m-3"
        if metric == "electrolyte_salt_mol"
        else "named metric SI unit",
    }


def _compare_stage(name, left, right, stage, start, factor, strict):
    results = {key: {"passed": False} for key in PAIR_CHECKS}
    results["both-present"]["passed"] = left is not None and right is not None
    tail, errors = None, []
    if left is not None and right is not None:
        try:
            low = max(left["times"][0], right["times"][0])
            high = min(left["times"][-1], right["times"][-1])
            wanted = np.asarray(stage_times(stage)) + start
            common = wanted[(wanted >= low - 1e-8) & (wanted <= high + 1e-8)]
            results["common-time-support"] = {
                "passed": len(common) > 0,
                "count": len(common),
                "start_s": low,
                "end_s": high,
                "times_s": common.tolist(),
            }
            results["termination-kind"] = {
                "passed": left["stop"] == right["stop"],
                "left": left["stop"],
                "right": right["stop"],
            }
            limit = factor * (1e-3 + 1e-4 * max(left["times"][-1], right["times"][-1]))
            error = abs(left["times"][-1] - right["times"][-1])
            results["stop-time"] = {
                "passed": error <= limit,
                "maximum_absolute": error,
                "limit": limit,
            }
            pairs = {
                "common:": (_sample(left, common), _sample(right, common)),
                "t0:": (_sample(left, [left["times"][0]]), _sample(right, [right["times"][0]])),
                "endpoint:": (
                    _sample(left, [left["times"][-1]]),
                    _sample(right, [right["times"][-1]]),
                ),
            }
            for prefix, (a, b) in pairs.items():
                for metric in METRICS:
                    results[prefix + metric] = _norm(a[metric], b[metric], metric, factor)
            tail = {
                side: {
                    "actual_end_s": view["times"][-1],
                    "points_beyond_paired_interval": int(np.sum(view["times"] > high)),
                    "unpaired_duration_s": view["times"][-1] - high,
                    "actual_endpoint_row": view["rows"][-1].tolist(),
                    "raw_termination": view["raw"]["termination"],
                }
                for side, view in (("left", left), ("right", right))
            }
        except Exception as exc:
            errors.append({"type": type(exc).__name__, "message": str(exc)})
    checks = [_check(name + "/" + key, **results[key]) for key in PAIR_CHECKS]
    return {
        "checks": checks if strict else [],
        "descriptive_metrics": None if strict else results,
        "passed": all(check["passed"] for check in checks) if strict else None,
        "strict": strict,
        "tail_audit": json_value(tail),
        "errors": errors,
        "sample_metadata": None
        if left is None or right is None
        else {
            "common_profile_coordinates": PROFILE_COORDINATES,
            "t0_times_s": [float(left["times"][0]), float(right["times"][0])],
            "endpoint_times_s": [float(left["times"][-1]), float(right["times"][-1])],
            "inventory_delta_reference": "original whole-history inventory; never stage reset",
            "signed_direction": "left-minus-right",
        },
    }


def assess_histories(histories):
    expected = plan()
    result = {
        "schema": "half-cell-dfn45-assessment-v1",
        "routes": [],
        "comparisons": [],
        "model_discrepancies": [],
        "checks": [],
        "thresholds": deepcopy(THRESHOLDS),
    }
    for identity in CASE_IDS:
        views = {}
        for route in (route for route in routes() if route["unit"] == identity):
            report, normalized = _route_assessment(route, histories.get(route_id(route)))
            result["routes"].append(report)
            result["checks"].extend(report["checks"])
            views[(route["implementation"], route["grid_level"])] = normalized
        for kind, left, right, factor, strict in comparison_edges(identity):
            label = "/".join((identity, kind, *left, "versus", *right))
            comparison = {
                "unit": identity,
                "kind": kind,
                "left": left,
                "right": right,
                "strict": strict,
                "factor": factor,
                "stages": [],
            }
            start = 0.0
            for index, stage in enumerate(case_data(identity)["stages"]):
                paired = _compare_stage(
                    f"{label}/stage-{index}",
                    views[left][index],
                    views[right][index],
                    stage,
                    start,
                    factor,
                    strict,
                )
                comparison["stages"].append(paired)
                result["checks"].extend(paired["checks"])
                start += stage["duration_s"]
            comparison["passed"] = (
                all(stage["passed"] for stage in comparison["stages"]) if strict else None
            )
            result["comparisons" if strict else "model_discrepancies"].append(comparison)
    result.update(
        check_denominator=expected["check_denominator"],
        evaluated_checks=len(result["checks"]),
        failed_checks=sum(not check["passed"] for check in result["checks"]),
        history_denominator=99,
        strict_comparison_denominator=114,
        strict_comparison_stage_denominator=216,
        available_histories=sum(
            isinstance(histories.get(route_id(route)), Mapping) for route in routes()
        ),
    )
    result["all_passed"] = (
        result["failed_checks"] == 0 and result["evaluated_checks"] == result["check_denominator"]
    )
    result["units"] = [
        {
            "unit": identity,
            "passed": all(
                check["passed"]
                for check in result["checks"]
                if check["name"].startswith(identity + "/")
            ),
        }
        for identity in CASE_IDS
    ]
    return result


def _worker(request_path, directory):
    request = json.loads(request_path.read_text())
    route, data, checkpoints = request["route"], request["case"], []
    context = {"case": deepcopy(data), "segments": []}

    def checkpoint(value):
        # The frozen source adapter publishes raw returns before extraction;
        # add its already-built contract without editing that adapter.
        enriched = {**context, **value}
        path = directory / f"progress-{len(checkpoints):04d}.json.gz"
        checkpoints.append(archive_new(path, enriched))
        print(
            json.dumps(
                {"route": route_id(route), "phase": enriched.get("phase"), "checkpoint": str(path)}
            ),
            flush=True,
        )

    started = time.monotonic()
    try:
        if route["implementation"] == "native":
            trace = native_time(data, route["grid"], progress_callback=checkpoint)
        else:
            import battery_half_cell_dfn45_pybamm as source

            original_build = source.build_reference

            def observe_build(*args, **kwargs):
                built = original_build(*args, **kwargs)
                context["contract"] = source.source_contract(built)
                checkpoint({"phase": "source-built"})
                return built

            with patch.object(source, "build_reference", observe_build):
                numerical_options = (
                    {"numerical_policy": route["numerical_policy"]}
                    if "numerical_policy" in route
                    else {}
                )
                if "solver_options" in route:
                    numerical_options["solver_options"] = deepcopy(route["solver_options"])
                if "linear_policy" in route:
                    numerical_options["linear_policy"] = deepcopy(route["linear_policy"])
                if "observation_policy" in route:
                    numerical_options["observation_policy"] = route["observation_policy"]
                if "gradient_policy" in route:
                    numerical_options["gradient_policy"] = route["gradient_policy"]
                trace = source.reference_time(
                    data,
                    route["grid"],
                    route["implementation"],
                    progress_callback=checkpoint,
                    **numerical_options,
                )
    except Exception as exc:
        trace = {
            "exception": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
            "partial_history": getattr(exc, "partial_history", context),
        }
    trace["execution"] = {
        "route": route,
        "elapsed_s": time.monotonic() - started,
        "checkpoints": checkpoints,
    }
    archive_new(directory / "final.json.gz", trace)
    return 0 if "exception" not in trace else 1


def execute_route(route, directory, budget_s, *, progress, worker_entrypoint=None):
    """Own worker binding; reuse the delivered full process-tree termination policy."""
    if os.name != "posix":
        raise RuntimeError("route process-group budgets require POSIX")
    directory.mkdir()
    request_path = directory / "request.json"
    write_new(request_path, {"route": route, "case": case_data(route["unit"])})
    command = [
        sys.executable,
        "-u",
        str(Path(__file__ if worker_entrypoint is None else worker_entrypoint).resolve()),
        "--worker-request",
        str(request_path),
        "--worker-directory",
        str(directory),
    ]
    started, previous_count, heartbeat, timed_out = time.monotonic(), -1, 0.0, False
    with (directory / "stdout.log").open("x") as stdout:
        process = subprocess.Popen(
            command, stdout=stdout, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            while process.poll() is None:
                elapsed = time.monotonic() - started
                count = len(list(directory.glob("progress-*.json.gz")))
                if count != previous_count or elapsed - heartbeat >= 20:
                    progress(
                        {
                            "event": "route-progress",
                            "route": route_id(route),
                            "elapsed_s": elapsed,
                            "checkpoint_count": count,
                            "budget_s": budget_s,
                        }
                    )
                    heartbeat, previous_count = elapsed, count
                if elapsed >= budget_s:
                    timed_out = True
                    break
                time.sleep(0.2)
        finally:
            transport._terminate_route_group(process)
    final_path = directory / "final.json.gz"
    try:
        if timed_out or not final_path.exists():
            raise RuntimeError(
                "route exceeded its independent budget"
                if timed_out
                else "worker exited without its final artifact"
            )
        trace = read_archive(final_path)
        artifact = {
            "path": str(final_path),
            "archive_sha256": hashlib.sha256(final_path.read_bytes()).hexdigest(),
        }
    except Exception as exc:
        partial, unreadable = transport._latest_partial(directory)
        trace = {
            "exception": {
                "type": "RouteTimeout" if timed_out else type(exc).__name__,
                "message": str(exc),
            },
            "partial_history": partial,
            "worker_returncode": process.returncode,
            "unreadable_checkpoints": unreadable,
        }
        artifact = archive_new(directory / "supervisor-failure.json.gz", trace)
    artifact.update(
        route=route,
        worker_returncode=process.returncode,
        timed_out=timed_out,
        elapsed_s=time.monotonic() - started,
        budget_s=budget_s,
        process_group_id=process.pid,
        budget_scope="POSIX isolated process group; TERM then KILL even after leader exit",
        exception=trace.get("exception"),
        all_artifacts=[str(path) for path in sorted(directory.iterdir())],
    )
    return trace, artifact


def _provenance(protocol, execution_label="prospective-half-cell-dfn45"):
    import casadi
    import pybamm

    if (pybamm.__version__, casadi.__version__) != ("25.4.2", "3.6.7"):
        raise RuntimeError("actual pinned PyBaMM25.4.2/CasADi3.6.7 required")
    repo = Path(__file__).resolve().parents[2]
    source = Path(pybamm.__file__).parent
    paths = [Path(__file__), Path(transport.__file__), protocol]
    paths += [
        Path(__file__).with_name(name)
        for name in (
            "battery_half_cell_dfn45_cases.py",
            "battery_half_cell_dfn45_pybamm.py",
            "battery_half_cell_dfn45_numerics.py",
            "battery_half_cell_dfn45_linear.py",
            "battery_half_cell_dfn45_observation.py",
            "battery_half_cell_dfn45_centered.py",
            "battery_half_cell_dfn45_meshes.py",
        )
    ]
    paths += [
        repo / "src/chemworld/physchem" / name
        for name in (
            "battery_half_cell_dfn.py",
            "battery_materials.py",
            "battery_models.py",
            "finite_volume.py",
            "coupled_residual.py",
            "residual_system.py",
        )
    ]
    paths += [
        source / name
        for name in (
            "input/parameters/lithium_ion/Xu2019.py",
            "parameters/lithium_ion_parameters.py",
            "models/full_battery_models/lithium_metal/dfn.py",
            "models/full_battery_models/lithium_ion/basic_dfn_half_cell.py",
            "models/submodels/electrolyte_diffusion/full_diffusion.py",
            "models/submodels/electrolyte_conductivity/full_conductivity.py",
            "models/submodels/electrode/ohm/full_ohm.py",
            "spatial_methods/finite_volume.py",
            "solvers/base_solver.py",
            "solvers/idaklu_solver.py",
        )
    ]
    return {
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "protocol": str(protocol),
        "execution_label": execution_label,
        "python": sys.version,
        "platform": platform.platform(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("pybamm", "pybammsolvers", "casadi", "numpy", "scipy")
        },
        "files_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--attempt", default="1")
    parser.add_argument("--route", action="append", default=[])
    parser.add_argument("--route-budget-s", type=float, default=900.0)
    parser.add_argument("--worker-request", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-directory", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker_request is not None:
        return _worker(args.worker_request, args.worker_directory)
    selected = [route for route in routes() if not args.route or route_id(route) in args.route]
    if len(set(args.route)) != len(args.route) or set(args.route) - {route_id(r) for r in routes()}:
        raise ValueError("unknown or duplicate route selection")
    if not math.isfinite(args.route_budget_s) or args.route_budget_s <= 0:
        raise ValueError("finite positive process-group budget required")
    prospective = plan()
    if not args.execute:
        print(json.dumps(json_value({**prospective, "selected_routes": selected}), indent=2))
        return 0
    if args.output_dir is None or args.protocol is None:
        raise ValueError(
            "execution needs a new external output directory and fixed experiment note"
        )
    if os.name != "posix":
        raise RuntimeError("route process-group budgets require POSIX")
    output, protocol = args.output_dir.resolve(), args.protocol.resolve(strict=True)
    if output.is_relative_to(Path(__file__).resolve().parents[2]):
        raise ValueError("raw scientific histories must stay outside Git checkout")
    provenance = _provenance(
        protocol, "qualification" if len(selected) == 99 else "development-subset"
    )
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / "plan.json", prospective)
    write_new(output / "provenance.json", provenance)
    histories, artifacts, started = {}, [], time.monotonic()
    with (output / "progress.jsonl").open("x") as journal:

        def progress(event):
            record = {"total_elapsed_s": time.monotonic() - started, **event}
            line = json.dumps(json_value(record), separators=(",", ":"))
            journal.write(line + "\n")
            journal.flush()
            print(line, flush=True)

        for index, route in enumerate(selected):
            progress(
                {
                    "event": "route-start",
                    "route": route_id(route),
                    "attempted": index,
                    "selected_denominator": len(selected),
                    "full_denominator": 99,
                }
            )
            directory = output / route_id(route).replace("/", "-")
            try:
                trace, artifact = execute_route(
                    route, directory, args.route_budget_s, progress=progress
                )
            except Exception as exc:
                partial, unreadable = (
                    transport._latest_partial(directory) if directory.exists() else (None, [])
                )
                trace = {
                    "exception": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                    "partial_history": partial,
                    "unreadable_checkpoints": unreadable,
                }
                artifact = {
                    "route": route,
                    "exception": trace["exception"],
                    "timed_out": False,
                    "elapsed_s": None,
                    "supervisor_exception": True,
                }
                try:
                    artifact["retained_failure"] = archive_new(
                        output / (directory.name + "-supervisor-exception.json.gz"), trace
                    )
                except Exception as storage_error:
                    artifact["storage_exception"] = str(storage_error)
            histories[route_id(route)] = trace
            artifacts.append(artifact)
            progress(
                {
                    "event": "route-finish",
                    "route": route_id(route),
                    "attempted": index + 1,
                    "exception": artifact["exception"],
                    "timed_out": artifact["timed_out"],
                }
            )
    assessment = assess_histories(histories)
    write_new(output / "assessment.json", assessment)
    manifest = {
        "schema": "half-cell-dfn45-manifest-v1",
        "attempt": args.attempt,
        "mode": "qualification" if len(selected) == 99 else "development-subset",
        "protocol": str(protocol),
        "provenance": provenance,
        "plan": prospective,
        "artifacts": artifacts,
        "attempted_histories": len(selected),
        "returned_histories": sum("exception" not in trace for trace in histories.values()),
        "exceptions": sum("exception" in trace for trace in histories.values()),
        "timeouts": sum(artifact["timed_out"] for artifact in artifacts),
        "physical_initial_source_stops": sum(
            route["physical_initial_source_stop"] for route in assessment["routes"]
        ),
        **raw_saved_counts(histories.values()),
        "assessment": {
            key: assessment[key]
            for key in ("check_denominator", "evaluated_checks", "failed_checks", "units")
        },
        "elapsed_s": time.monotonic() - started,
        "all_passed": assessment["all_passed"] and len(selected) == 99,
    }
    write_new(output / "manifest.json", manifest)
    print(
        json.dumps(
            json_value(
                {
                    key: manifest[key]
                    for key in (
                        "attempted_histories",
                        "returned_histories",
                        "timeouts",
                        "exceptions",
                        "all_passed",
                    )
                }
            )
        ),
        flush=True,
    )
    return 0 if manifest["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
