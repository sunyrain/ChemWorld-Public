"""Evaluator contracts, including the fixed development-only short probes."""

import json
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
from battery_collector_spm_cases import native_spec
from battery_collector_spm_green import integrate_collectors
from battery_collector_spm_supplement_assess import (
    PHYSICAL_V,
    PHYSICAL_W,
    PREFIX,
    SOURCE_EVENTS,
    Checks,
    F,
    R,
    assess,
    canonical,
    compare_histories,
    control_checks,
    finite_tree,
    profile,
    refusal_checks,
    sample_indices,
    source_state_checks,
    source_state_layout,
    stop_checks,
    validate_trace,
)
from battery_collector_spm_supplement_cases import (
    GRIDS,
    LAYOUTS,
    THRESHOLDS,
    case_data,
    route_plan,
)
from battery_collector_spm_supplement_native import state_snapshot


def synthetic_native():
    """Construct a tiny, algebraic near-equilibrium history, without an ODE."""
    data = case_data("S01")
    data["waveform"] = ((1.0, 1e-10, "A"),)
    data["samples_per_segment"] = 3
    grid, layout = {"z": 2, "r": 2}, LAYOUTS["opposite"]
    spec = native_spec(data, grid, layout)
    initial = spec.initial_state()
    time = np.array([0.0, 0.5, 1.0])
    current = 1e-10
    q = np.full((3, 2), current / 2)
    voltage = (
        data["positive"]["ocv"][0]
        + data["positive"]["ocv"][1] * data["positive"]["initial_fraction"]
        - data["negative"]["ocv"][0]
        - data["negative"]["ocv"][1] * data["negative"]["initial_fraction"]
    )
    values = np.column_stack(
        (np.full(3, voltage), np.full(3, current), current * time, voltage * current * time)
    )
    profiles, all_amounts = {}, [{} for _ in time]
    for name, sign in (("negative", -1), ("positive", 1)):
        p = data[name]
        redges = np.linspace(0, p["radius_m"], 3)
        volumes = np.diff(redges**3) / p["radius_m"] ** 3 * p["active_volume_m3"] / 2
        theta = np.broadcast_to(
            p["initial_fraction"]
            + sign
            * current
            * time[:, None, None]
            / (F * p["maximum_concentration_mol_m3"] * p["active_volume_m3"]),
            (3, 2, 2),
        ).copy()
        amounts = theta * volumes * p["maximum_concentration_mol_m3"]
        reaction = -sign * q
        area = 3 * p["active_volume_m3"] / (2 * p["radius_m"])
        outer, inner = redges[-1], redges[-2]
        distance = (outer**2 - 3 / 5 * (outer**5 - inner**5) / (outer**3 - inner**3)) / (2 * outer)
        surface = theta[:, :, -1] - reaction * distance / (
            F * area * p["diffusivity_m2_s"] * p["maximum_concentration_mol_m3"]
        )
        j0 = p["exchange_current_scale_A_m2"] * np.sqrt(surface * (1 - surface))
        eta = 2 * R * data["temperature_K"] / F * np.arcsinh(reaction / (2 * area * j0))
        keys = [[f"{name}:z:{z}:Li:{r}" for r in range(2)] for z in range(2)]
        for i in range(3):
            all_amounts[i].update(
                {keys[z][r]: amounts[i, z, r] for z in range(2) for r in range(2)}
            )
        profiles[name] = {
            "theta": theta.tolist(),
            "shell_amounts_mol": amounts.tolist(),
            "inventory_keys": keys,
            "r_edges_m": redges.tolist(),
            "r_nodes_m": ((redges[:-1] + redges[1:]) / 2).tolist(),
            "z_edges_m": [0, data["height_m"] / 2, data["height_m"]],
            "z_nodes_m": [data["height_m"] / 4, data["height_m"] * 3 / 4],
            "average_fraction": theta[:, :, 0].tolist(),
            "surface_fraction": surface.tolist(),
            "reaction_current_A": reaction.tolist(),
            "reaction_current_density_A_m2": (reaction / area).tolist(),
            "exchange_current_density_A_m2": j0.tolist(),
            "kinetic_overpotential_V": eta.tolist(),
            "open_circuit_potential_V": (p["ocv"][0] + p["ocv"][1] * surface).tolist(),
        }
    conductance = data["sigma_cc_S_m"] * data["collector_thickness_m"] * data["width_m"]
    green = integrate_collectors(
        q[0],
        z_edges_m=profiles["negative"]["z_edges_m"],
        negative_conductance_S_m=conductance,
        positive_conductance_S_m=conductance,
        negative_tab="left",
        physical_tab_voltage_V=voltage,
    )
    native_fields = [
        {
            "collector_potential_negative_V": green["negative"]["fv_centre_V"].tolist(),
            "collector_potential_positive_V": green["positive"]["fv_centre_V"].tolist(),
            "negative_collector_face_current_A": green["negative"]["face_current_A"].tolist(),
            "positive_collector_face_current_A": green["positive"]["face_current_A"].tolist(),
            "local_voltage_V": (
                green["positive"]["fv_centre_V"] - green["negative"]["fv_centre_V"]
            ).tolist(),
        }
        for _ in time
    ]
    for name in profiles:
        profiles[name]["collector_potential_V"] = [
            f["collector_potential_" + name + "_V"] for f in native_fields
        ]
    for i, fields in enumerate(native_fields):
        fields.update(
            current_A=current,
            voltage_V=voltage,
            through_current_A=q[i].tolist(),
            reaction_current_A=[
                *profiles["negative"]["reaction_current_A"][i],
                *profiles["positive"]["reaction_current_A"][i],
            ],
            surface_fraction=[
                *profiles["negative"]["surface_fraction"][i],
                *profiles["positive"]["surface_fraction"][i],
            ],
            kinetic_overpotential_V=[
                *profiles["negative"]["kinetic_overpotential_V"][i],
                *profiles["positive"]["kinetic_overpotential_V"][i],
            ],
            electrolyte_potential_V=(
                np.asarray(fields["collector_potential_negative_V"])
                - profiles["negative"]["open_circuit_potential_V"][i]
                - np.asarray(profiles["negative"]["kinetic_overpotential_V"][i])
            ).tolist(),
            bv_residual_V=(
                np.asarray(fields["local_voltage_V"])
                - profiles["positive"]["open_circuit_potential_V"][i]
                - np.asarray(profiles["positive"]["kinetic_overpotential_V"][i])
                + profiles["negative"]["open_circuit_potential_V"][i]
                + np.asarray(profiles["negative"]["kinetic_overpotential_V"][i])
            ).tolist(),
        )
    snapshots = [
        state_snapshot(
            replace(
                initial,
                amounts_mol=all_amounts[i],
                time_s=float(t),
                coordinates={"discharged_charge_C": values[i, 2], "delivered_work_J": values[i, 3]},
            )
        )
        for i, t in enumerate(time)
    ]
    ids = ["battery:collector:minimum-voltage", "battery:collector:maximum-voltage"]
    ids += [
        f"battery:collector:{n}:z:{z}:{b}-surface"
        for n in profiles
        for z in range(2)
        for b in ("minimum", "maximum")
    ]
    segment = {
        "times_s": time.tolist(),
        "end_time_s": 1.0,
        "termination": None,
        "outcome": "success",
        "values": values.tolist(),
        "physical_values": values.tolist(),
        "physical_tab_voltage_V": values[:, 0].tolist(),
        "physical_work_J": values[:, 3].tolist(),
        "controlled_port": "current",
        "raw_profiles": profiles,
        "native_fields": native_fields,
        "source_boundary_voltage_V": [green["source_boundary_voltage_V"]] * 3,
        "raw_states": snapshots,
        "raw_amounts_mol": [s["amounts_mol"] for s in snapshots],
        "raw_coordinates": [s["coordinates"] for s in snapshots],
        "raw_algebraic_values": [{}, {}, {}],
        "initial_state": snapshots[0],
        "final_state": snapshots[-1],
        "event_definitions": [{"id": n, "direction": -1, "terminal": True} for n in ids],
        "native_event_values_at_endpoint": dict.fromkeys(ids, 1.0),
        "control": {
            "index": 0,
            "duration_s": 1.0,
            "target": current,
            "unit": "A",
            "current_limit_A": data["maximum_current_A"],
        },
        "control_modes": ["constant_current"] * 3,
    }
    trace = {
        "unit": data["id"],
        "inputs": deepcopy(data),
        "grid": grid,
        "actual_grid": grid,
        "requested_grid": grid,
        "layout": layout,
        "implementation": "native",
        "native_spec": spec.to_dict(),
        "native_spec_class": type(spec).__name__,
        "family": spec.model_family,
        "initial_state": {**state_snapshot(initial), "spec": spec.to_dict()},
        "final_state": {**snapshots[-1], "spec": spec.to_dict()},
        "segments": [segment],
        "faraday_C_mol": F,
    }
    return data, grid, layout, trace


def check_trace(trace, data, grid, layout):
    book = Checks()
    normalized = book.guarded(
        "trace-schema", validate_trace, book, "trace", trace, data, grid, layout
    )
    return book, normalized


def test_complete_synthetic_native_is_accepted_without_time_integration():
    data, grid, layout, trace = synthetic_native()
    book, normalized = check_trace(trace, data, grid, layout)
    assert normalized is not None
    assert all(c["passed"] for c in book.items), [c for c in book.items if not c["passed"]]


@pytest.mark.parametrize(
    "mutation",
    [
        "missing-profile",
        "nan",
        "truncate",
        "t0-only",
        "input",
        "QW",
        "state",
        "algebraic",
        "mode",
        "geometry",
        "endpoint",
    ],
)
def test_synthetic_trace_damage_cannot_pass(mutation):
    data, grid, layout, trace = synthetic_native()
    s = trace["segments"][0]
    if mutation == "missing-profile":
        del s["raw_profiles"]["positive"]["theta"]
    elif mutation == "nan":
        s["raw_states"][0]["boundary_heat_J"] = float("nan")
    elif mutation == "truncate":
        del s["times_s"][1]
    elif mutation == "t0-only":
        s["times_s"] = [0.0]
    elif mutation == "input":
        trace["inputs"]["sigma_cc_S_m"] *= 2
    elif mutation == "QW":
        s["physical_values"][-1][3] += 0.01
    elif mutation == "state":
        del s["raw_states"][0]["amounts_mol"]["negative:z:0:Li:0"]
    elif mutation == "algebraic":
        s["raw_algebraic_values"][0] = {"invented": 0}
    elif mutation == "mode":
        s["control_modes"] = ["constant_voltage"] * 3
    elif mutation == "geometry":
        s["raw_profiles"]["negative"]["z_edges_m"][1] *= 0.5
    elif mutation == "endpoint":
        del s["native_event_values_at_endpoint"]["battery:collector:minimum-voltage"]
    book, _ = check_trace(trace, data, grid, layout)
    assert any(not c["passed"] for c in book.items)


def test_all_48_logical_routes_are_required_and_failures_preserved():
    assert sum(len(route_plan(i)) for i in range(1, 11)) == 48
    item = {
        "id": "S01",
        "inputs": case_data("S01"),
        "routes": {"history_opposite_coarse": {"exception": "synthetic failure"}},
    }
    original = deepcopy(item["routes"])
    assess(item)
    assert not item["passed"]
    assert item["routes"] == original
    assert len([c for c in item["checks"] if c["name"].endswith(":route-present")]) == 3


@pytest.mark.parametrize("times", [[0, 2], [0, 1, 1, 2], [0, 1.01, 2]])
def test_common_samples_are_not_interpolated_or_silently_dropped(times):
    if times == [0, 1, 1, 2]:
        with pytest.raises(ValueError):
            sample_indices(times, [0, 1, 1, 2])
    else:
        with pytest.raises(ValueError):
            sample_indices(times, [0, 1, 2])


@pytest.mark.parametrize(
    "mode,target,current,voltage",
    [
        ("interior", 3.78, 0.1, 3.78),
        ("positive-limit", 3.4, 0.001, 3.8),
        ("negative-limit", 4.1, -0.001, 3.8),
    ],
)
def test_cv_uses_work_identity_only_on_interior(mode, target, current, voltage):
    data = case_data("S04" if mode == "interior" else "S05" if current > 0 else "S06")
    values = np.array([[voltage, current, 0, 0], [voltage, current, current, voltage * current]])
    book = Checks()
    control_checks(book, "cv", {"controlled_port": "physical-tab"}, values, (1, target, "V"), data)
    assert all(c["passed"] for c in book.items)
    assert any(c["name"].endswith("CV-work") for c in book.items) == (mode == "interior")
    values[:, 1] *= -1
    if mode != "interior":
        damaged = Checks()
        control_checks(
            damaged, "cv", {"controlled_port": "physical-tab"}, values, (1, target, "V"), data
        )
        assert any(not c["passed"] for c in damaged.items)


def test_common_z_comparison_aggregates_currents_without_point_interpolation():
    data, grid, layout, trace = synthetic_native()
    _, normalized = check_trace(trace, data, grid, layout)
    fine = deepcopy(normalized)
    fine[0]["grid"]["z"] = 4
    for n in ("negative", "positive"):
        fine[0]["profiles"][n]["q"] = np.repeat(normalized[0]["profiles"][n]["q"] / 2, 2, axis=1)
    compared = Checks()
    compare_histories(compared, "comparison", normalized, fine, data)
    assert all(c["passed"] for c in compared.items)
    coverage = next(c for c in compared.items if c["name"].endswith("spatial-coverage"))
    assert coverage["detail"]["intervals"] == 2
    fine[0]["profiles"]["negative"]["q"][:, 0] += 0.003
    damaged = Checks()
    compare_histories(damaged, "comparison", normalized, fine, data)
    assert any(not c["passed"] for c in damaged.items)


def test_high_sigma_finite_conductivity_has_approved_30_microvolt_limit():
    data, grid, layout, trace = synthetic_native()
    _, normalized = check_trace(trace, data, grid, layout)
    baseline = deepcopy(normalized)
    baseline[0]["values"][:, 0] += 24e-6
    book = Checks()
    compare_histories(book, "high-sigma", normalized, baseline, data, high_sigma=True)
    assert all(c["passed"] for c in book.items)
    baseline[0]["values"][:, 0] += 8e-6
    book = Checks()
    compare_histories(book, "high-sigma", normalized, baseline, data, high_sigma=True)
    assert any(not c["passed"] for c in book.items)


def test_source_and_native_surface_semantics_are_not_equated():
    data, grid, _, trace = synthetic_native()
    raw = deepcopy(trace["segments"][0]["raw_profiles"]["negative"])
    theta = np.asarray(raw["theta"])
    theta[:, :, -1] += 0.001
    # Rebuild a synthetic source profile using its own linear surface trace.
    p = data["negative"]
    volumes = (
        np.diff(np.asarray(raw["r_edges_m"]) ** 3) / p["radius_m"] ** 3 * p["active_volume_m3"] / 2
    )
    raw["theta"] = theta.transpose(0, 2, 1).tolist()
    raw["concentration_mol_m3"] = (
        (theta * p["maximum_concentration_mol_m3"]).transpose(0, 2, 1).tolist()
    )
    raw["shell_amounts_mol"] = (
        (theta * volumes * p["maximum_concentration_mol_m3"]).transpose(0, 2, 1).tolist()
    )
    raw["average_fraction"] = (theta * volumes).sum(axis=2) / (p["active_volume_m3"] / 2)
    surface = 1.5 * theta[:, :, -1] - 0.5 * theta[:, :, -2]
    raw["surface_fraction"] = surface.tolist()
    j0 = p["exchange_current_scale_A_m2"] * np.sqrt(surface * (1 - surface))
    raw["exchange_current_density_A_m2"] = j0.tolist()
    raw["open_circuit_potential_V"] = (p["ocv"][0] + p["ocv"][1] * surface).tolist()
    area = 3 * p["active_volume_m3"] / (p["radius_m"] * 2)
    raw["kinetic_overpotential_V"] = (
        2
        * R
        * data["temperature_K"]
        / F
        * np.arcsinh(np.asarray(raw["reaction_current_A"]) / (2 * area * j0))
    ).tolist()
    book = Checks()
    profile(book, "source", raw, data, "negative", grid, 3, True)
    assert all(c["passed"] for c in book.items)
    assert np.max(abs(surface - theta[:, :, -1])) > 1e-4


@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), {"__nonfinite__": "NaN"}, {"nonfinite": "nan"}]
)
def test_nonfinite_metadata_is_not_hidden_by_success(value):
    with pytest.raises(ValueError):
        finite_tree({"outcome": "success", "unconsumed_diagnostic": value})


def test_refusal_reason_and_state_are_canonical_not_self_reported():
    data, grid = case_data("S10"), GRIDS["coarse"]
    spec = native_spec(data, grid, LAYOUTS["opposite"])
    snapshot = {**state_snapshot(spec.initial_state()), "spec": spec.to_dict()}
    reason = "collector SPM impedance is unsupported"
    route = {
        "name": "EIS",
        "stage": "analysis",
        "entrypoint": "chemworld.physchem.battery_impedance.battery_impedance",
        "exception": {"type": "ValueError", "message": reason},
        "rejected": True,
        "before": snapshot,
        "after": deepcopy(snapshot),
        "state_unchanged": True,
        "request": spec.to_dict(),
    }
    book = Checks()
    refusal_checks(book, "refusal", route, "EIS", data, grid)
    assert all(c["passed"] for c in book.items)
    route["exception"]["message"] = route["expected_reason"] = "invented reason"
    book = Checks()
    refusal_checks(book, "refusal", route, "EIS", data, grid)
    assert any(not c["passed"] for c in book.items)


def synthetic_source_state():
    """Synthetic raw source vector; no PyBaMM build or integration is invoked."""
    data, grid, layout, native = synthetic_native()
    _, normalized = check_trace(native, data, grid, layout)
    profiles = normalized[0]["profiles"]
    values = normalized[0]["values"]
    states = source_state_layout(data, grid, False)
    rhs_size = 5 + 2 * grid["z"] * grid["r"]
    raw = np.zeros((3, rhs_size + 4 * grid["z"]))
    initial_rhs = np.zeros(rhs_size)
    fields = {
        PHYSICAL_V: values[:, 0].tolist(),
        PHYSICAL_W: values[:, 3].tolist(),
        "Negative current collector potential [V]": np.zeros((3, 2)).tolist(),
        "Current collector current density [A.m-2]": np.full(
            (3, 2), 1e-10 / (data["height_m"] * data["width_m"])
        ).tolist(),
    }
    for state in states:
        name = state["name"]
        start, stop = state["slices"][0]
        if "particle concentration" in name:
            electrode = "negative" if "negative" in name else "positive"
            physical = (
                profiles[electrode]["theta"].reshape(3, -1)
                * data[electrode]["maximum_concentration_mol_m3"]
            )
        elif "surface potential difference" in name:
            electrode = "negative" if "negative" in name else "positive"
            physical = profiles[electrode]["ocp"] + profiles[electrode]["eta"]
        elif name in ("Discharge capacity [A.h]", "Throughput capacity [A.h]"):
            physical = values[:, 2:3] / 3600
            initial_rhs[start] = values[0, 1] / 3600
        elif name in ("Discharge energy [W.h]", "Throughput energy [W.h]"):
            physical = values[:, 3:4] / 3600
            initial_rhs[start] = values[0, 0] * values[0, 1] / 3600
        elif name == PHYSICAL_W:
            physical = values[:, 3:4]
            initial_rhs[start] = values[0, 0] * values[0, 1]
        else:
            physical = np.asarray(fields[name])
        raw[:, start:stop] = (physical - state["reference"]) / state["scale"]
    events = [{"name": "original voltage event", "type": "TERMINATION"}] + [
        {"name": PREFIX + n, "type": "TERMINATION"} for n in SOURCE_EVENTS
    ]
    checked = {
        ("rhs:" if s["kind"] == "differential" else "algebraic:") + s["name"]: [] for s in states
    }
    checked.update({f"event:{i}:{e['type']}:{e['name']}": [] for i, e in enumerate(events)})
    checked.update(
        {
            "output:" + n: []
            for n in (
                "Voltage [V]",
                "Current [A]",
                "Discharge capacity [A.h]",
                "Discharge energy [W.h]",
                PHYSICAL_V,
            )
        }
    )
    work_state = next(s for s in states if s["name"] == PHYSICAL_W)
    build = {
        "states": states,
        "rhs_size": rhs_size,
        "algebraic_size": 4 * grid["z"],
        "uniform": False,
        "events": events,
        "original_events": ["original voltage event"],
        "original_event_definitions": [events[0]],
        "original_rhs_names": [
            s["name"] for s in states if s["kind"] == "differential" and s["name"] != PHYSICAL_W
        ],
        "physical_work_no_feedback": {
            "method": "processed expression DAG StateVector evaluation-array dependency inspection",
            "event_identities": [{"ordinal": i, **e} for i, e in enumerate(events)],
            "checked_expressions": checked,
            "physical_work_state_indices": [work_state["slices"][0][0]],
            "no_feedback": True,
        },
    }
    trace = {"inputs": data, "grid": grid, "source_build": build}
    segment = {
        "source_state_vectors": raw.tolist(),
        "source_initial_rhs": initial_rhs.tolist(),
        "source_initial_algebraic_residual": [0.0] * build["algebraic_size"],
        "values": values.tolist(),
        "source_fields": fields,
        "source_event_values_at_endpoint": {e["name"]: [1.0] for e in events},
    }
    return trace, segment, profiles, values


def test_source_raw_state_scales_and_real_observer_coverage_synthetic_positive():
    trace, segment, profiles, values = synthetic_source_state()
    book = Checks()
    source_state_checks(book, "source", trace, segment, profiles, values)
    assert all(c["passed"] for c in book.items), [c for c in book.items if not c["passed"]]


@pytest.mark.parametrize(
    "mutation",
    [
        "missing-state",
        "scale",
        "reference",
        "slice",
        "rhs-size",
        "observer-feedback",
        "observer-coverage",
        "observer-index",
        "raw-Q",
        "raw-concentration",
        "initial-work-RHS",
        "event",
        "nonfinite",
    ],
)
def test_source_state_damage_fails_closed(mutation):
    trace, segment, profiles, values = synthetic_source_state()
    build = trace["source_build"]
    if mutation == "missing-state":
        build["states"].pop()
    elif mutation == "scale":
        build["states"][0]["scale"] = 2.0
    elif mutation == "reference":
        build["states"][4]["reference"] = 1.0
    elif mutation == "slice":
        build["states"][0]["slices"] = [[1, 2]]
    elif mutation == "rhs-size":
        build["rhs_size"] -= 1
    elif mutation == "observer-feedback":
        build["physical_work_no_feedback"]["checked_expressions"]["rhs:" + PHYSICAL_W] = [
            build["rhs_size"] - 1
        ]
    elif mutation == "observer-coverage":
        del build["physical_work_no_feedback"]["checked_expressions"]["output:Voltage [V]"]
    elif mutation == "observer-index":
        build["physical_work_no_feedback"]["physical_work_state_indices"] = []
    elif mutation == "raw-Q":
        segment["source_state_vectors"][-1][0] = 0.1
    elif mutation == "raw-concentration":
        segment["source_state_vectors"][-1][4] += 0.1
    elif mutation == "initial-work-RHS":
        segment["source_initial_rhs"][-1] = 2
    elif mutation == "event":
        del segment["source_event_values_at_endpoint"][PREFIX + "minimum voltage"]
    else:
        segment["source_state_vectors"][-1][-1] = float("nan")
    book = Checks()
    book.guarded(
        "source-schema", source_state_checks, book, "source", trace, segment, profiles, values
    )
    assert any(not c["passed"] for c in book.items)


@pytest.mark.parametrize("uniform,mode", [(False, "A"), (False, "V"), (True, "A"), (True, "V")])
def test_source_layout_has_exact_release_model_dimensions(uniform, mode):
    data = case_data("S04" if mode == "V" else "S01")
    grid = {"z": 1 if uniform else 4, "r": 6}
    rows = source_state_layout(data, grid, uniform)
    rhs = sum(b - a for r in rows if r["kind"] == "differential" for a, b in r["slices"])
    algebraic = sum(b - a for r in rows if r["kind"] == "algebraic" for a, b in r["slices"])
    assert rhs == 5 + 2 * grid["z"] * grid["r"]
    assert algebraic == (2 if uniform else 4 * grid["z"]) + (mode == "V")
    for row in rows:
        if "surface potential difference" in row["name"]:
            assert row["domains"] == {
                "primary": ["current collector"],
                "secondary": [],
                "tertiary": [],
                "quaternary": [],
            }


@pytest.mark.parametrize("source", [False, True])
@pytest.mark.parametrize("which", ["S08", "S09"])
def test_finite_cutoff_requires_strictly_interior_initial_state(source, which):
    data = case_data(which)
    if which == "S08":
        wanted = PREFIX + "minimum voltage" if source else "battery:collector:minimum-voltage"
        values = np.array([[3.75, 0.03, 0, 0], [data["minimum_voltage_V"], 0.03, 0.1, 0.3]])
        profiles = {}
    else:
        wanted = (
            PREFIX + "negative minimum surface"
            if source
            else "battery:collector:negative:z:0:minimum-surface"
        )
        values = np.array([[3.75, 0.01, 0, 0], [3.72, 0.01, 0.1, 0.3]])
        profiles = {"negative": {"surface": np.array([[0.03, 0.04], [0.01, 0.02]])}}
    segment = {
        "termination": "event: " + wanted if source else wanted,
        "outcome": "stopped",
        "source_event_values_at_endpoint": {wanted: [0.0]},
        "native_event_values_at_endpoint": {wanted: 0.0},
    }
    normalized = {"values": values, "profiles": profiles}
    book = Checks()
    stop_checks(book, "stop", segment, normalized, data, source)
    assert all(c["passed"] for c in book.items)
    if which == "S08":
        values[0, 0] = data["minimum_voltage_V"]
    else:
        profiles["negative"]["surface"][0, 0] = data["surface_margin"]
    book = Checks()
    stop_checks(book, "stop", segment, normalized, data, source)
    assert any(not c["passed"] for c in book.items)


def test_incorrect_route_container_is_a_recorded_failure_not_an_evaluator_abort():
    for damaged in ([], {"history_opposite_coarse": []}):
        item = {"id": "S01", "inputs": case_data("S01"), "routes": damaged}
        assess(item)
        assert item["failures"] and not item["passed"]


def portable_scalar(value):
    """Independent test archive codec, deliberately separate from the evaluator."""
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


@pytest.mark.parametrize("implementation", ["native", "source"])
@pytest.mark.parametrize("case_id", ["S01", "S04", "S07", "S03"])
def test_actual_short_history_complete_live_and_reloaded_validation(
    implementation, case_id, tmp_path
):
    """Eight real histories / sixteen validations; never a qualification block."""
    from battery_collector_spm_supplement_native import native_time, native_uniform_time
    from battery_collector_spm_supplement_source import source_time, source_uniform_time

    data = case_data(case_id)
    data["waveform"] = tuple((1e-4, target, unit) for _, target, unit in data["waveform"])
    data["samples_per_segment"] = 3
    grid, layout = dict(GRIDS["coarse"]), dict(LAYOUTS["opposite"])
    source, uniform = implementation == "source", case_id == "S07"
    function = (
        (source_uniform_time if uniform else source_time)
        if source
        else (native_uniform_time if uniform else native_time)
    )
    trace = function(data, grid) if uniform else function(data, grid, layout)
    reports = {}

    def validate(value, representation):
        book = Checks()
        normalized = book.guarded(
            "trace-schema",
            validate_trace,
            book,
            representation,
            value,
            data,
            grid,
            layout,
            source,
            uniform,
        )
        reports[representation] = {
            "normalized": normalized is not None,
            "checks": book.items,
            "failures": [check for check in book.items if not check["passed"]],
        }

    # The live validation MUST precede serialization: archive codecs erase the
    # np.int64 source-build types that caused the complete second attempt to fail.
    validate(trace, "live")
    archive = tmp_path / f"{case_id}-{implementation}-trace.json"
    archive.write_text(json.dumps(trace, default=portable_scalar, allow_nan=False))
    validate(json.loads(archive.read_text()), "reload")
    (tmp_path / "validation.json").write_text(
        json.dumps(reports, default=portable_scalar, allow_nan=False)
    )
    print(f"{case_id} {implementation}: 1 history / 2 validations; archive={archive}", flush=True)
    assert all(r["normalized"] and not r["failures"] for r in reports.values()), reports


def test_canonical_numpy_scalars_preserve_nested_numeric_and_boolean_identity():
    actual = {"size": np.int64(197), "nested": [np.float64(0.125), np.bool_(False)]}
    assert canonical(actual) == canonical({"size": 197, "nested": [0.125, False]})


@pytest.mark.parametrize("value", [np.float64("nan"), np.float32("inf"), object()])
def test_canonical_does_not_coerce_nonfinite_or_unknown_types(value):
    with pytest.raises((TypeError, ValueError)):
        canonical({"nested": [value]})


def test_initial_snapshot_scale_roundtrip_is_representation_identity_only():
    data, grid, layout, trace = synthetic_native()
    segment = trace["segments"][0]
    original = deepcopy(segment["initial_state"])
    segment["initial_state"] = deepcopy(original)
    key = next(iter(original["amounts_mol"]))
    segment["initial_state"]["amounts_mol"][key] = np.nextafter(
        original["amounts_mol"][key], np.inf
    )
    book, normalized = check_trace(trace, data, grid, layout)
    assert normalized is not None
    assert all(c["passed"] for c in book.items), [c for c in book.items if not c["passed"]]
    assert segment["raw_states"][0] == original  # Preserve both original representations.
    assert segment["initial_state"] != original


@pytest.mark.parametrize("mutation", ["key", "numeric", "metadata", "bool", "nan"])
def test_initial_snapshot_identity_rejects_material_or_structural_tampering(mutation):
    data, grid, layout, trace = synthetic_native()
    segment = trace["segments"][0]
    segment["initial_state"] = deepcopy(segment["initial_state"])
    snapshot = segment["initial_state"]
    key = next(iter(snapshot["amounts_mol"]))
    if mutation == "key":
        snapshot["amounts_mol"][key + ":changed"] = snapshot["amounts_mol"].pop(key)
    elif mutation == "numeric":
        snapshot["amounts_mol"][key] *= 1 + 1e-10
    elif mutation == "metadata":
        snapshot["extra_metadata"] = "altered"
    elif mutation == "bool":
        snapshot["time_s"] = False
    else:
        snapshot["amounts_mol"][key] = np.float64("nan")
    book, _ = check_trace(trace, data, grid, layout)
    assert any(not c["passed"] for c in book.items)


def test_recorded_same_end_voltage_grid_error_remains_a_scientific_failure():
    data, grid, layout, trace = synthetic_native()
    _, coarse = check_trace(trace, data, grid, layout)
    fine = deepcopy(coarse)
    recorded_error = 0.0026958396010337005
    fine[0]["values"][:, 0] += recorded_error
    book = Checks()
    compare_histories(book, "same-end-grid", coarse, fine, data)
    failures = [c for c in book.items if not c["passed"]]
    assert THRESHOLDS["response_voltage_V"] == 0.002
    assert len(failures) == 1
    assert failures[0]["threshold"] == "response_voltage_V"
    assert failures[0]["max_abs"] == pytest.approx(recorded_error, abs=1e-15)


@pytest.mark.parametrize(
    "mutation",
    [
        None,
        "first-key",
        "second-key",
        "identity-order",
        "identity-type",
        "identity-name",
        "identity-missing",
        "second-feedback",
    ],
)
def test_same_named_source_events_require_distinct_ordered_dependency_checks(mutation):
    trace, segment, profiles, values = synthetic_source_state()
    build = trace["source_build"]
    proof = build["physical_work_no_feedback"]
    event = {
        "name": "Interpolant 'interpolating_function' lower bound",
        "type": "INTERPOLANT_EXTRAPOLATION",
    }
    keys = []
    for _ in range(2):
        ordinal = len(build["events"])
        build["events"].append(dict(event))
        build["original_events"].append(event["name"])
        build["original_event_definitions"].append(dict(event))
        proof["event_identities"].append({"ordinal": ordinal, **event})
        key = f"event:{ordinal}:{event['type']}:{event['name']}"
        keys.append(key)
        proof["checked_expressions"][key] = []
    if mutation in {"first-key", "second-key"}:
        del proof["checked_expressions"][keys[mutation == "second-key"]]
    elif mutation == "identity-order":
        proof["event_identities"][-2:] = reversed(proof["event_identities"][-2:])
    elif mutation == "identity-type":
        proof["event_identities"][-1]["type"] = "TERMINATION"
    elif mutation == "identity-name":
        proof["event_identities"][-1]["name"] += " altered"
    elif mutation == "identity-missing":
        del proof["event_identities"]
    elif mutation == "second-feedback":
        proof["checked_expressions"][keys[-1]] = proof["physical_work_state_indices"][:]
    book = Checks()
    book.guarded(
        "source-schema", source_state_checks, book, "source", trace, segment, profiles, values
    )
    failures = [c for c in book.items if not c["passed"]]
    assert bool(failures) is (mutation is not None), failures
