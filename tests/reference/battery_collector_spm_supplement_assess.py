"""Prospective, fail-closed qualification of the collector supplement.

No trajectory is integrated here. Recorded physical-tab V/W and the original
source V/W remain separate. Comparisons require every authored common sample;
stopping endpoints are assessed separately, never interpolated or extrapolated.
"""

from __future__ import annotations

import json
from math import gcd

import numpy as np
from battery_collector_spm_green import (
    integrate_collectors,
    solve_static_collector,
    static_inputs_from_case,
)
from battery_collector_spm_supplement_cases import GRIDS, LAYOUTS, THRESHOLDS, case_data, route_plan
from scipy import constants

F = 96485.33212
R = 8.31446261815324
ELECTRODES = ("negative", "positive")
PHYSICAL_W = "Supplement signed physical work [J]"
PHYSICAL_V = "Supplement physical tab voltage [V]"
PREFIX = "Supplement physical qualification: "
SOURCE_EVENTS = (
    "minimum voltage",
    "maximum voltage",
    "negative minimum surface",
    "negative maximum surface",
    "positive minimum surface",
    "positive maximum surface",
)
REFUSALS = {
    "thermal": ("codec", "collector SPM is fixed-T/fixed-electrolyte without side chemistry"),
    "electrolyte": ("codec", "collector SPM is fixed-T/fixed-electrolyte without side chemistry"),
    "SEI": ("codec", "collector SPM is fixed-T/fixed-electrolyte without side chemistry"),
    "plating": ("codec", "collector SPM does not support plating"),
    "invalid-tab": ("codec", "negative_tab must be left or right"),
    "negative-geometry": ("codec", "collector dimensions and conductivities must be positive"),
    "EIS": ("analysis", "collector SPM impedance is unsupported"),
    "SOH": (
        "analysis",
        "collector SPM chemical-energy and equilibrium-SOH closure is unsupported",
    ),
}


def canonical(value):
    def scalar(item):
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(f"unsupported canonical type: {type(item).__name__}")

    return json.dumps(value, default=scalar, sort_keys=True, allow_nan=False, separators=(",", ":"))


def finite_tree(value):
    if isinstance(value, dict):
        if "nonfinite" in value or any(str(key).startswith("__nonfinite") for key in value):
            raise ValueError("nonfinite archive marker")
        for child in value.values():
            finite_tree(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            finite_tree(child)
    elif isinstance(value, (float, np.floating)) and not np.isfinite(value):
        raise ValueError("nonfinite recorded number")


def array(value, shape=None):
    result = np.asarray(value, dtype=float)
    if (shape is not None and result.shape != shape) or not np.all(np.isfinite(result)):
        raise ValueError(f"expected finite shape {shape}, received {result.shape}")
    return result


class Checks:
    def __init__(self):
        self.items = []

    def truth(self, name, passed, detail=None):
        self.items.append({"name": name, "passed": bool(passed), "detail": detail})

    def equal(self, name, actual, expected):
        self.truth(name, canonical(actual) == canonical(expected))

    def bound(self, name, values, key):
        values = array(values)
        if not values.size:
            raise ValueError(f"{name}: empty measurement")
        maximum = float(np.max(np.abs(values)))
        self.items.append(
            {
                "name": name,
                "passed": maximum <= THRESHOLDS[key],
                "max_abs": maximum,
                "limit": THRESHOLDS[key],
                "threshold": key,
                "count": int(values.size),
            }
        )

    def identity(self, name, actual, expected):
        """Only floating representation tolerance, not a scientific pass limit."""
        actual, expected = array(actual), array(expected)
        if actual.shape != expected.shape:
            raise ValueError(f"{name}: representation shapes differ")
        scale = max(float(np.max(abs(expected))) if expected.size else 0.0, 1e-300)
        limit = 64 * np.finfo(float).eps * scale
        error = float(np.max(abs(actual - expected))) if expected.size else 0.0
        self.items.append(
            {
                "name": name,
                "passed": error <= limit,
                "max_abs": error,
                "limit": limit,
                "criterion": "64 machine eps times represented quantity scale",
            }
        )

    def snapshot_identity(self, name, actual, expected):
        """Exact structure/metadata, with the existing 64-epsilon rule per scalar.

        The solver's dimensional-scale round trip can change the first saved
        inventory by an ulp. Neither snapshot is rewritten, and a large field
        (such as temperature) never sets the tolerance for a small Li amount.
        """
        leaves = Checks()

        def visit(path, left, right):
            left = left.item() if isinstance(left, np.generic) else left
            right = right.item() if isinstance(right, np.generic) else right
            if isinstance(right, dict):
                matches = isinstance(left, dict) and left.keys() == right.keys()
                leaves.truth(path + ":keys", matches)
                if matches:
                    for key in right:
                        visit(path + ":" + str(key), left[key], right[key])
            elif isinstance(right, (tuple, list)):
                matches = isinstance(left, (tuple, list)) and len(left) == len(right)
                leaves.truth(path + ":length", matches)
                if matches:
                    for index, (a, b) in enumerate(zip(left, right, strict=True)):
                        visit(f"{path}:{index}", a, b)
            elif isinstance(left, bool) or isinstance(right, bool):
                leaves.truth(path, type(left) is type(right) and left == right)
            elif isinstance(left, (int, float)) and isinstance(right, (int, float)):
                leaves.identity(path, left, right)
            else:
                leaves.equal(path, left, right)

        visit(name, actual, expected)
        self.truth(
            name,
            all(check["passed"] for check in leaves.items),
            {
                "criterion": "exact structure and metadata; scalar representation identity",
                "checks": leaves.items,
            },
        )

    def guarded(self, name, function, *args):
        try:
            return function(*args)
        except Exception as error:
            self.truth(
                name,
                False,
                {"type": type(error).__name__, "message": str(error)},
            )
            return None


def sample_indices(times, expected):
    """Require a unique saved sample for every requested time; no interpolation."""
    times, expected = array(times), array(expected)
    if times.ndim != 1 or expected.ndim != 1 or not len(expected):
        raise ValueError("nonempty one-dimensional time arrays required")
    indices = np.argmin(abs(times[:, None] - expected[None, :]), axis=0)
    if len(set(indices)) != len(indices) or np.any(
        abs(times[indices] - expected) > THRESHOLDS["sample_time_s"]
    ):
        raise ValueError("missing or duplicated authored sample")
    return indices


def profile(book, label, raw, data, name, grid, nt, source):
    nz, nr = grid["z"], grid["r"]
    p = data[name]
    native_shape = (nt, nz, nr)
    stored_shape = (nt, nr, nz) if source else native_shape
    theta = array(raw["theta"], stored_shape)
    amounts = array(raw["shell_amounts_mol"], stored_shape)
    if source:
        theta, amounts = theta.transpose(0, 2, 1), amounts.transpose(0, 2, 1)
    redges = np.linspace(0, p["radius_m"], nr + 1)
    zedges = np.linspace(0, data["height_m"], nz + 1)
    book.identity(label + ":r-edges", raw["r_edges_m"], redges)
    book.identity(label + ":r-nodes", raw["r_nodes_m"], (redges[1:] + redges[:-1]) / 2)
    book.identity(label + ":z-edges", raw["z_edges_m"], zedges)
    book.identity(label + ":z-nodes", raw["z_nodes_m"], (zedges[1:] + zedges[:-1]) / 2)
    volumes = np.diff(redges**3) * p["active_volume_m3"] / (nz * p["radius_m"] ** 3)
    book.identity(
        label + ":shell-amounts", amounts, theta * volumes * p["maximum_concentration_mol_m3"]
    )
    book.truth(label + ":finite-inventory-domain", np.all((theta >= 0) & (theta <= 1)))
    average = array(raw["average_fraction"], (nt, nz))
    book.identity(
        label + ":volume-average",
        average,
        amounts.sum(axis=2) / (p["active_volume_m3"] / nz * p["maximum_concentration_mol_m3"]),
    )
    q = array(raw["reaction_current_A"], (nt, nz))
    area = 3 * p["active_volume_m3"] / (p["radius_m"] * nz)
    book.identity(label + ":interfacial-current", raw["reaction_current_density_A_m2"], q / area)
    surface = array(raw["surface_fraction"], (nt, nz))
    if source:
        concentration = array(raw["concentration_mol_m3"], stored_shape).transpose(0, 2, 1)
        book.identity(
            label + ":concentration", concentration, theta * p["maximum_concentration_mol_m3"]
        )
        reconstructed = 1.5 * theta[:, :, -1] - 0.5 * theta[:, :, -2]
    else:
        inner, outer = redges[-2:]
        moment = 3 / 5 * (outer**5 - inner**5) / (outer**3 - inner**3)
        distance = (outer**2 - moment) / (2 * outer)
        reconstructed = theta[:, :, -1] - q * distance / (
            area * F * p["diffusivity_m2_s"] * p["maximum_concentration_mol_m3"]
        )
    book.bound(label + ":surface-reconstruction", surface - reconstructed, "event_surface_fraction")
    book.truth(
        label + ":surface-domain",
        np.all(
            (surface >= data["surface_margin"] - THRESHOLDS["event_surface_fraction"])
            & (surface <= 1 - data["surface_margin"] + THRESHOLDS["event_surface_fraction"])
        ),
    )
    ocp = array(raw["open_circuit_potential_V"], (nt, nz))
    eta = array(raw["kinetic_overpotential_V"], (nt, nz))
    j0 = array(raw["exchange_current_density_A_m2"], (nt, nz))
    book.bound(label + ":ocp", ocp - (p["ocv"][0] + p["ocv"][1] * surface), "static_voltage_V")
    book.truth(label + ":positive-exchange-current", np.all(j0 > 0))
    evaluated = (
        np.clip(surface, data["surface_margin"] / 2, 1 - data["surface_margin"] / 2)
        if not source
        else surface
    )
    book.identity(
        label + ":exchange-current",
        j0,
        p["exchange_current_scale_A_m2"] * np.sqrt(evaluated * (1 - evaluated)),
    )
    expected_eta = 2 * R * data["temperature_K"] / F * np.arcsinh(q / (2 * area * j0))
    book.bound(label + ":BV", eta - expected_eta, "static_voltage_V")
    return {
        "theta": theta,
        "amounts": amounts,
        "total": amounts.sum(axis=(1, 2)),
        "q": q,
        "surface": surface,
        "ocp": ocp,
        "eta": eta,
        "z_edges": zedges,
    }


def control_checks(book, label, segment, values, stage, data):
    duration, target, mode = stage
    del duration
    book.equal(
        label + ":controlled-port",
        segment["controlled_port"],
        "current" if mode == "A" else "physical-tab",
    )
    voltage, current, charge, work = values.T
    book.truth(
        label + ":current-limit",
        np.all(abs(current) <= data["maximum_current_A"] + THRESHOLDS["current_A"]),
    )
    if mode == "A":
        book.bound(label + ":CC-current", current - target, "current_A")
        return
    expected = data["expected_mode"]
    if expected == "interior":
        book.truth(label + ":CV-interior", np.all(abs(current) < data["maximum_current_A"]))
        book.bound(label + ":CV-voltage", voltage - target, "CV_voltage_V")
        book.bound(label + ":CV-work", work - work[0] - target * (charge - charge[0]), "CV_work_J")
    else:
        sign = 1 if expected == "positive-limit" else -1
        book.bound(
            label + ":CV-saturated-current", current - sign * data["maximum_current_A"], "current_A"
        )
        book.truth(
            label + ":CV-saturated-voltage",
            np.all(sign * (voltage - target) >= -THRESHOLDS["CV_voltage_V"]),
        )


def electrical_checks(book, label, segment, profiles, values, data, grid, layout, source, stage):
    nt, nz = values.shape[0], grid["z"]
    if nz == 1:
        equilibrium = (
            profiles["positive"]["ocp"]
            + profiles["positive"]["eta"]
            - profiles["negative"]["ocp"]
            - profiles["negative"]["eta"]
        )
        book.bound(
            label + ":uniform-BV-port",
            equilibrium[:, 0] - values[:, 0] - values[:, 1] * data["series_resistance_ohm"],
            "static_voltage_V",
        )
        return
    if source:
        fields = segment["source_fields"]
        potentials = {
            n: array(fields[n.capitalize() + " current collector potential [V]"], (nt, nz))
            for n in ELECTRODES
        }
        local = array(fields["Local voltage [V]"], (nt, nz))
        book.bound(
            label + ":source-through-q",
            array(fields["through_current_A"], (nt, nz)) - profiles["negative"]["q"],
            "current_A",
        )
        for name in (
            "Current collector current density [A.m-2]",
            "Negative electrode current density [A.m-2]",
            "Positive electrode current density [A.m-2]",
        ):
            array(fields[name], (nt, nz))
    else:
        fields = segment["native_fields"]
        if len(fields) != nt:
            raise ValueError("missing native electrical sample")
        potentials = {
            n: array([f["collector_potential_" + n + "_V"] for f in fields], (nt, nz))
            for n in ELECTRODES
        }
        local = array([f["local_voltage_V"] for f in fields], (nt, nz))
        for name in ELECTRODES:
            book.identity(
                label + ":" + name + "-phi-copy",
                segment["raw_profiles"][name]["collector_potential_V"],
                potentials[name],
            )
        book.identity(label + ":native-current", [f["current_A"] for f in fields], values[:, 1])
        book.identity(label + ":native-voltage", [f["voltage_V"] for f in fields], values[:, 0])
        book.identity(
            label + ":native-through-q",
            [f["through_current_A"] for f in fields],
            profiles["negative"]["q"],
        )
        for field, profile_key in (
            ("reaction_current_A", "q"),
            ("surface_fraction", "surface"),
            ("kinetic_overpotential_V", "eta"),
        ):
            book.identity(
                label + ":native-" + field,
                [f[field] for f in fields],
                np.concatenate([profiles[n][profile_key] for n in ELECTRODES], axis=1),
            )
        book.bound(
            label + ":native-electrolyte-potential",
            array([f["electrolyte_potential_V"] for f in fields], (nt, nz))
            - potentials["negative"]
            + profiles["negative"]["ocp"]
            + profiles["negative"]["eta"],
            "static_voltage_V",
        )
        book.bound(
            label + ":native-recorded-BV-residual",
            array([f["bv_residual_V"] for f in fields], (nt, nz)),
            "static_voltage_V",
        )
    book.bound(
        label + ":local-voltage",
        local - potentials["positive"] + potentials["negative"],
        "green_voltage_V",
    )
    book.bound(
        label + ":local-BV",
        local
        - profiles["positive"]["ocp"]
        - profiles["positive"]["eta"]
        + profiles["negative"]["ocp"]
        + profiles["negative"]["eta"],
        "static_voltage_V",
    )
    conductance = data["sigma_cc_S_m"] * data["collector_thickness_m"] * data["width_m"]
    boundary = array(segment["source_boundary_voltage_V"], (nt,))
    metrics = {
        k: [] for k in ("phi", "power", "boundary", "static-q", "static-V", "static-residual")
    }
    for i in range(nt):
        q = profiles["negative"]["q"][i]
        physical = values[i, 0] + values[i, 1] * data["series_resistance_ohm"]
        green = integrate_collectors(
            q,
            z_edges_m=profiles["negative"]["z_edges"],
            negative_conductance_S_m=conductance,
            positive_conductance_S_m=conductance,
            negative_tab=layout["negative_tab"],
            physical_tab_voltage_V=physical,
        )
        for name in ELECTRODES:
            metrics["phi"].extend(potentials[name][i] - green[name]["fv_centre_V"])
            if not source:
                book.bound(
                    f"{label}:{i}:{name}-face-current",
                    array(fields[i][name + "_collector_face_current_A"], (nz + 1,))
                    - green[name]["face_current_A"],
                    "current_A",
                )
        metrics["power"].append(np.dot(q, local[i]) - values[i, 1] * physical - green["fv_joule_W"])
        metrics["boundary"].append(boundary[i] - green["source_boundary_voltage_V"])
        if not source:
            args = static_inputs_from_case(
                data,
                grid,
                layout,
                negative_outer_fraction=profiles["negative"]["theta"][i, :, -1],
                positive_outer_fraction=profiles["positive"]["theta"][i, :, -1],
            )
            answer = solve_static_collector(**args, control_mode=stage[2], target=stage[1])
            metrics["static-q"].extend(answer["through_current_A"] - q)
            metrics["static-q"].append(answer["current_A"] - values[i, 1])
            metrics["static-V"].append(answer["voltage_V"] - values[i, 0])
            metrics["static-residual"].extend(answer["local_residual_V"])
    for key, threshold in (
        ("phi", "green_voltage_V"),
        ("power", "green_power_W"),
        ("boundary", "green_voltage_V"),
    ):
        book.bound(label + ":Green-" + key, metrics[key], threshold)
    if not source:
        for key, threshold in (
            ("static-q", "static_current_A"),
            ("static-V", "static_voltage_V"),
            ("static-residual", "static_voltage_V"),
        ):
            book.bound(label + ":" + key, metrics[key], threshold)


def source_state_layout(data, grid, uniform):
    """Released 25.4.2 SPM variable/scaling contract, checked by actual builds."""
    nz, nr = grid["z"], grid["r"]
    empty = {"primary": [], "secondary": [], "tertiary": [], "quaternary": []}
    # Even a zero-dimensional uniform SPM retains the singleton collector
    # domain on each surface-potential variable; it is not a domainless scalar.
    collector = {**empty, "primary": ["current collector"]}
    items = [
        (name, 1, 1.0, 0.0, "differential", empty)
        for name in (
            "Discharge capacity [A.h]",
            "Throughput capacity [A.h]",
            "Discharge energy [W.h]",
            "Throughput energy [W.h]",
        )
    ]
    for name in ELECTRODES:
        items.append(
            (
                f"X-averaged {name} particle concentration [mol.m-3]",
                nz * nr,
                data[name]["maximum_concentration_mol_m3"],
                0.0,
                "differential",
                {**empty, "primary": [name + " particle"], "secondary": ["current collector"]},
            )
        )
    items.append((PHYSICAL_W, 1, 1.0, 0.0, "differential", empty))
    # Marquis2019 nominal capacity sets source variable scales, not our active
    # material inventory. It is left unchanged in the authored parameter mapping.
    nominal_Ah = 0.680616
    if data["waveform"][0][2] == "V":
        items.append(("Current variable [A]", 1, nominal_Ah, 0.0, "algebraic", empty))
    for name in ELECTRODES:
        fractions = data["initial_z_fractions"][name]
        initial = np.mean(
            np.interp((np.arange(nz) + 0.5) / nz, np.linspace(0, 1, len(fractions)), fractions)
        )
        potential = data[name]["ocv"][0] + data[name]["ocv"][1] * initial
        items.append(
            (
                f"X-averaged {name} electrode surface potential difference [V]",
                nz,
                1.0,
                potential,
                "algebraic",
                collector,
            )
        )
    if not uniform:
        items.extend(
            (
                ("Negative current collector potential [V]", nz, 1.0, 0.0, "algebraic", collector),
                (
                    "Current collector current density [A.m-2]",
                    nz,
                    nominal_Ah / (data["height_m"] * data["width_m"]),
                    0.0,
                    "algebraic",
                    collector,
                ),
            )
        )
    rows, start = [], 0
    for name, size, scale, reference, kind, domains in items:
        rows.append(
            {
                "name": name,
                "slices": [[start, start + size]],
                "scale": scale,
                "reference": reference,
                "kind": kind,
                "domains": domains,
            }
        )
        start += size
    return rows


def source_state_checks(book, label, trace, segment, profiles, values):
    build = trace["source_build"]
    states = build["states"]
    rhs_size, algebraic_size = build["rhs_size"], build["algebraic_size"]
    raw = array(segment["source_state_vectors"], (len(values), rhs_size + algebraic_size))
    array(segment["source_initial_rhs"], (rhs_size,))
    array(segment["source_initial_algebraic_residual"], (algebraic_size,))
    names = [s["name"] for s in states]
    expected_states = source_state_layout(trace["inputs"], trace["grid"], build["uniform"])
    book.equal(
        label + ":canonical-source-variable-order", names, [s["name"] for s in expected_states]
    )
    expected_rhs = sum(
        b - a for s in expected_states if s["kind"] == "differential" for a, b in s["slices"]
    )
    expected_alg = sum(
        b - a for s in expected_states if s["kind"] == "algebraic" for a, b in s["slices"]
    )
    book.equal(
        label + ":canonical-source-dimensions",
        [rhs_size, algebraic_size],
        [expected_rhs, expected_alg],
    )
    for state, expected_state in zip(states, expected_states, strict=True):
        for key in ("slices", "kind", "domains"):
            book.equal(label + ":" + state["name"] + ":" + key, state[key], expected_state[key])
        for key in ("scale", "reference"):
            book.identity(label + ":" + state["name"] + ":" + key, state[key], expected_state[key])
    book.truth(label + ":unique-state-names", len(names) == len(set(names)))
    occupied, decoded = [], {}
    for state in states:
        indices = [i for start, stop in state["slices"] for i in range(start, stop)]
        if not indices or min(indices) < 0 or max(indices) >= raw.shape[1]:
            raise ValueError("invalid source state slices")
        if not state["scale"] > 0 or state["kind"] not in ("differential", "algebraic"):
            raise ValueError("invalid source state scale/kind")
        book.truth(
            label + ":state-kind:" + state["name"],
            all(i < rhs_size for i in indices)
            if state["kind"] == "differential"
            else all(i >= rhs_size for i in indices),
        )
        if not isinstance(state["domains"], dict):
            raise ValueError("missing source state domains")
        occupied.extend(indices)
        decoded[state["name"]] = raw[:, indices] * state["scale"] + state["reference"]
    book.equal(label + ":complete-state-partition", sorted(occupied), list(range(raw.shape[1])))
    for name in ELECTRODES:
        variable = f"X-averaged {name} particle concentration [mol.m-3]"
        expected = profiles[name]["theta"] * trace["inputs"][name]["maximum_concentration_mol_m3"]
        book.identity(
            label + ":source-raw-concentration:" + name,
            decoded[variable],
            expected.reshape(len(values), -1),
        )
        declaration = states[names.index(variable)]
        book.identity(
            label + ":source-concentration-scale:" + name,
            declaration["scale"],
            trace["inputs"][name]["maximum_concentration_mol_m3"],
        )
        book.equal(label + ":source-concentration-reference:" + name, declaration["reference"], 0.0)
    original = array(segment["values"], values.shape)
    for name, expected in (
        ("Discharge capacity [A.h]", original[:, 2] / 3600),
        ("Discharge energy [W.h]", original[:, 3] / 3600),
        (PHYSICAL_W, values[:, 3]),
    ):
        book.identity(label + ":source-receipt:" + name, decoded[name], expected[:, None])
    initial_rhs = array(segment["source_initial_rhs"], (rhs_size,))
    for name, expected, factor, threshold in (
        ("Discharge capacity [A.h]", values[0, 1], 3600, "current_A"),
        ("Throughput capacity [A.h]", abs(values[0, 1]), 3600, "current_A"),
        ("Discharge energy [W.h]", original[0, 0] * values[0, 1], 3600, "green_power_W"),
        ("Throughput energy [W.h]", abs(original[0, 0] * values[0, 1]), 3600, "green_power_W"),
        (PHYSICAL_W, values[0, 0] * values[0, 1], 1, "green_power_W"),
    ):
        declaration = states[names.index(name)]
        position = declaration["slices"][0][0]
        book.bound(
            label + ":initial-receipt-RHS:" + name,
            factor * declaration["scale"] * initial_rhs[position] - expected,
            threshold,
        )
    for name in ELECTRODES:
        variable = f"X-averaged {name} electrode surface potential difference [V]"
        book.bound(
            label + ":raw-surface-potential:" + name,
            decoded[variable] - profiles[name]["ocp"] - profiles[name]["eta"],
            "static_voltage_V",
        )
    if not build["uniform"]:
        for variable in (
            "Negative current collector potential [V]",
            "Current collector current density [A.m-2]",
        ):
            book.identity(
                label + ":raw-electrical-variable:" + variable,
                decoded[variable],
                segment["source_fields"][variable],
            )
    if trace["inputs"]["waveform"][0][2] == "V":
        book.identity(
            label + ":raw-current-control", decoded["Current variable [A]"], values[:, 1:2]
        )
    events = build["events"]
    event_names = [e["name"] for e in events]
    book.truth(
        label + ":source-event-definitions",
        all(
            set(event) == {"name", "type"}
            and isinstance(event["name"], str)
            and bool(event["name"])
            and event["type"]
            in {"TERMINATION", "SWITCH", "DISCONTINUITY", "INTERPOLANT_EXTRAPOLATION"}
            for event in events
        ),
    )
    book.truth(
        label + ":physical-events-retained", all(PREFIX + n in event_names for n in SOURCE_EVENTS)
    )
    original_events = build["original_event_definitions"]
    book.equal(
        label + ":original-event-names",
        sorted(e["name"] for e in original_events),
        sorted(build["original_events"]),
    )
    book.truth(
        label + ":original-events-retained",
        all(e in events for e in original_events) and bool(original_events),
    )
    book.equal(
        label + ":complete-source-endpoint-events",
        sorted(segment["source_event_values_at_endpoint"]),
        sorted(e["name"] for e in events if e["type"] == "TERMINATION"),
    )
    for values_at_event in segment["source_event_values_at_endpoint"].values():
        array(values_at_event, (1,))
    book.equal(
        label + ":original-rhs-retained",
        sorted(build["original_rhs_names"]),
        sorted(
            s["name"] for s in states if s["kind"] == "differential" and s["name"] != PHYSICAL_W
        ),
    )
    check = build["physical_work_no_feedback"]
    # PyBaMM can emit same-named interpolant events for different expressions.
    # Preserve each event's position and type; never collapse them by name.
    event_identities = [{"ordinal": index, **event} for index, event in enumerate(events)]
    book.equal(label + ":observer-event-identities", check["event_identities"], event_identities)
    expected_labels = [
        ("rhs:" if s["kind"] == "differential" else "algebraic:") + s["name"] for s in states
    ]
    expected_labels += [
        f"event:{event['ordinal']}:{event['type']}:{event['name']}" for event in event_identities
    ]
    expected_labels += [
        "output:" + n
        for n in (
            "Voltage [V]",
            "Current [A]",
            "Discharge capacity [A.h]",
            "Discharge energy [W.h]",
            PHYSICAL_V,
        )
    ]
    book.equal(
        label + ":observer-inspection-coverage",
        sorted(check["checked_expressions"]),
        sorted(expected_labels),
    )
    work_state = states[names.index(PHYSICAL_W)]
    work_indices = [i for a, b in work_state["slices"] for i in range(a, b)]
    book.equal(
        label + ":observer-state-indices", check["physical_work_state_indices"], work_indices
    )
    book.truth(
        label + ":observer-no-feedback",
        check["no_feedback"] is True
        and all(v == [] for v in check["checked_expressions"].values()),
    )
    book.equal(
        label + ":observer-method",
        check["method"],
        "processed expression DAG StateVector evaluation-array dependency inspection",
    )
    fields = segment["source_fields"]
    book.identity(label + ":physical-V-source-output", fields[PHYSICAL_V], values[:, 0])
    book.identity(label + ":physical-W-source-output", fields[PHYSICAL_W], values[:, 3])


def native_state_checks(book, label, trace, segment, profiles, values, times):
    from dataclasses import fields

    from chemworld.physchem.coupled_system import CoupledState

    required = {f.name for f in fields(CoupledState)}
    snapshots = segment["raw_states"]
    for key in ("raw_states", "raw_amounts_mol", "raw_coordinates", "raw_algebraic_values"):
        if len(segment[key]) != len(values):
            raise ValueError("incomplete native raw state array")
    inventory = {}
    for name in ELECTRODES:
        raw = segment["raw_profiles"][name]
        keys = raw["inventory_keys"]
        nz, nr = profiles[name]["theta"].shape[1:]
        expected_keys = [
            [
                f"{name}:z:{z}:Li:{r}" if trace["grid"]["z"] != 1 else f"{name}:Li:{r}"
                for r in range(nr)
            ]
            for z in range(nz)
        ]
        book.equal(label + ":inventory-identities:" + name, keys, expected_keys)
        for z in range(nz):
            for r in range(nr):
                inventory[keys[z][r]] = profiles[name]["amounts"][:, z, r]
    for i, snapshot in enumerate(snapshots):
        if set(snapshot) != required:
            raise ValueError("missing or extra persistent native state field")
        book.equal(f"{label}:{i}:amount-keys", sorted(snapshot["amounts_mol"]), sorted(inventory))
        book.equal(
            f"{label}:{i}:coordinate-keys",
            sorted(snapshot["coordinates"]),
            ["delivered_work_J", "discharged_charge_C"],
        )
        book.equal(f"{label}:{i}:persistent-algebraic", snapshot["algebraic_values"], {})
        book.equal(
            f"{label}:{i}:amount-copy", segment["raw_amounts_mol"][i], snapshot["amounts_mol"]
        )
        book.equal(
            f"{label}:{i}:coordinate-copy", segment["raw_coordinates"][i], snapshot["coordinates"]
        )
        book.equal(f"{label}:{i}:algebraic-copy", segment["raw_algebraic_values"][i], {})
        book.identity(
            f"{label}:{i}:raw-amount",
            [snapshot["amounts_mol"][k] for k in inventory],
            [v[i] for v in inventory.values()],
        )
        book.identity(
            f"{label}:{i}:raw-QW",
            [
                snapshot["coordinates"]["discharged_charge_C"],
                snapshot["coordinates"]["delivered_work_J"],
            ],
            values[i, 2:],
        )
        book.identity(f"{label}:{i}:raw-time", snapshot["time_s"], times[i])
        book.identity(
            f"{label}:{i}:fixed-temperature",
            snapshot["temperature_K"],
            trace["inputs"]["temperature_K"],
        )
        book.identity(
            f"{label}:{i}:fixed-volume",
            snapshot["volume_L"],
            trace["initial_state"]["volume_L"],
        )
    book.snapshot_identity(
        label + ":segment-initial-snapshot", segment["initial_state"], snapshots[0]
    )
    book.equal(label + ":segment-final-snapshot", segment["final_state"], snapshots[-1])
    events = segment["event_definitions"]
    ids = [e["id"] for e in events]
    book.truth(label + ":event-identity-unique", len(ids) == len(set(ids)))
    book.equal(
        label + ":endpoint-events-complete",
        sorted(segment["native_event_values_at_endpoint"]),
        sorted(ids),
    )
    if trace["grid"]["z"] > 1:
        expected = ["battery:collector:minimum-voltage", "battery:collector:maximum-voltage"]
        expected += [
            f"battery:collector:{name}:z:{z}:{bound}-surface"
            for name in ELECTRODES
            for z in range(trace["grid"]["z"])
            for bound in ("minimum", "maximum")
        ]
        book.truth(label + ":native-domain-events", set(expected) <= set(ids))


def stop_checks(book, label, segment, normalized, data, source):
    expected = data.get("expected_event")
    termination = segment["termination"]
    if not expected:
        book.equal(label + ":outcome", segment["outcome"], "success")
        book.truth(label + ":no-premature-stop", termination in (None, "final time"))
        return
    book.equal(label + ":outcome", segment["outcome"], "stopped")
    if expected == "minimum-voltage":
        book.truth(
            label + ":initial-event-interior",
            normalized["values"][0, 0] > data["minimum_voltage_V"],
        )
        wanted = PREFIX + "minimum voltage" if source else "battery:collector:minimum-voltage"
        book.truth(
            label + ":stop-reason",
            isinstance(termination, str)
            and (termination == wanted or termination == "event: " + wanted),
        )
        residual = normalized["values"][-1, 0] - data["minimum_voltage_V"]
        threshold = "event_voltage_V"
    else:
        book.truth(
            label + ":initial-event-interior",
            np.min(normalized["profiles"]["negative"]["surface"][0]) > data["surface_margin"],
        )
        wanted = PREFIX + "negative minimum surface" if source else "battery:collector:negative:z:"
        book.truth(
            label + ":stop-reason",
            isinstance(termination, str)
            and (
                (termination in (wanted, "event: " + wanted))
                if source
                else (termination.startswith(wanted) and termination.endswith(":minimum-surface"))
            ),
        )
        residual = (
            np.min(normalized["profiles"]["negative"]["surface"][-1]) - data["surface_margin"]
        )
        threshold = "event_surface_fraction"
    book.bound(label + ":independent-stop-residual", residual, threshold)
    event_values = (
        segment["source_event_values_at_endpoint"]
        if source
        else segment["native_event_values_at_endpoint"]
    )
    event_name = termination.removeprefix("event: ")
    book.bound(label + ":recorded-stop-residual", event_values[event_name], threshold)


def validate_trace(book, label, trace, data, grid, layout, source=False, uniform=False):
    finite_tree(trace)
    book.equal(label + ":unit", trace["unit"], data["id"])
    book.equal(label + ":inputs", trace["inputs"], data)
    actual = {"z": 1 if uniform else grid["z"], "r": grid["r"]}
    book.equal(label + ":actual-grid", trace["grid"], actual)
    expected_impl = (
        ("source-uniform" if uniform else "source")
        if source
        else ("native-uniform" if uniform else "native")
    )
    book.equal(label + ":implementation", trace["implementation"], expected_impl)
    book.equal(label + ":requested-grid", trace["requested_grid"], grid)
    book.equal(label + ":actual-grid-alias", trace["actual_grid"], actual)
    if not uniform:
        book.equal(label + ":layout", trace["layout"], layout)
    book.equal(
        label + ":Faraday-constant",
        trace["faraday_C_mol"],
        constants.physical_constants["Faraday constant"][0] if source else F,
    )
    if source:
        build = trace["source_build"]
        book.equal(
            label + ":source-version", trace["source_version"], data["source_versions"]["pybamm"]
        )
        for key, expected_grid in (
            ("grid", actual),
            ("actual_grid", actual),
            ("requested_grid", grid),
            ("authored_grid", grid),
        ):
            book.equal(label + ":source-build-" + key, build[key], expected_grid)
        book.equal(
            label + ":source-control",
            build["control"],
            "CC" if data["waveform"][0][2] == "A" else "physical-CV-with-finite-current-limits",
        )
        book.equal(label + ":source-uniform", build["uniform"], uniform)
        book.equal(label + ":source-build-layout", build["layout"], layout)
        book.equal(
            label + ":source-value-labels",
            trace["value_columns"],
            ["source_voltage_V", "current_A", "signed_charge_C", "source_work_J"],
        )
        book.equal(
            label + ":physical-value-labels",
            trace["physical_value_columns"],
            ["physical_tab_voltage_V", "current_A", "signed_charge_C", "physical_work_J"],
        )
    else:
        from battery_collector_spm_cases import native_spec
        from battery_collector_spm_supplement_native import _uniform_spec, state_snapshot

        spec = _uniform_spec(data, grid) if uniform else native_spec(data, grid, layout)
        declaration = json.loads(canonical(spec.to_dict()))
        book.equal(label + ":native-spec-binding", trace["native_spec"], declaration)
        book.equal(label + ":native-spec-class", trace["native_spec_class"], type(spec).__name__)
        book.equal(
            label + ":native-family",
            trace["family"],
            "original-BatterySPMSpec" if uniform else spec.model_family,
        )
        book.equal(
            label + ":native-initial-state",
            trace["initial_state"],
            {**state_snapshot(spec.initial_state()), "spec": declaration},
        )
    if len(trace["segments"]) != len(data["waveform"]):
        raise ValueError("missing or extra waveform segment")
    result, start = [], 0.0
    for index, (segment, stage) in enumerate(zip(trace["segments"], data["waveform"], strict=True)):
        tag = f"{label}:stage{index}"
        times = array(segment["times_s"])
        if times.ndim != 1 or len(times) < 2 or np.any(np.diff(times) <= 0):
            raise ValueError("time history is absent, t0-only, or nonincreasing")
        nt = len(times)
        values = array(segment["physical_values"], (nt, 4))
        book.truth(
            tag + ":physical-voltage-domain",
            np.all(
                (values[:, 0] >= data["minimum_voltage_V"] - THRESHOLDS["event_voltage_V"])
                & (values[:, 0] <= data["maximum_voltage_V"] + THRESHOLDS["event_voltage_V"])
            ),
        )
        original = array(segment["values"], (nt, 4))
        book.identity(tag + ":physical-V-alias", segment["physical_tab_voltage_V"], values[:, 0])
        book.identity(tag + ":physical-W-alias", segment["physical_work_J"], values[:, 3])
        book.equal(tag + ":same-IQ", original[:, 1:3].tolist(), values[:, 1:3].tolist())
        if not source:
            book.equal(tag + ":native-original-is-physical", original.tolist(), values.tolist())
        book.bound(tag + ":start", times[0] - start, "time_s")
        book.bound(tag + ":end-alias", times[-1] - segment["end_time_s"], "time_s")
        duration, target, mode = stage
        stop = bool(data.get("expected_event"))
        book.truth(
            tag + ":duration-domain",
            times[-1] > start and times[-1] <= start + duration + THRESHOLDS["time_s"],
        )
        if not stop:
            book.bound(tag + ":full-duration", times[-1] - start - duration, "time_s")
        authored = np.linspace(start, start + duration, data["samples_per_segment"])
        expected_times = (
            authored[authored <= times[-1] + THRESHOLDS["sample_time_s"]] if stop else authored
        )
        indices = sample_indices(times, expected_times)
        book.truth(
            tag + ":all-authored-samples",
            len(indices) == len(expected_times),
            {"required": len(expected_times), "matched": len(indices), "saved": nt},
        )
        if any(np.min(abs(expected_times - t)) > THRESHOLDS["sample_time_s"] for t in times[:-1]):
            raise ValueError("unexpected non-authored interior sample")
        profiles = {
            n: profile(book, tag + ":" + n, segment["raw_profiles"][n], data, n, actual, nt, source)
            for n in ELECTRODES
        }
        if index == 0:
            book.identity(tag + ":initial-QW", values[0, 2:], [0, 0])
            for n in ELECTRODES:
                authored_profile = data["initial_z_fractions"][n]
                initial = np.interp(
                    (np.arange(actual["z"]) + 0.5) / actual["z"],
                    np.linspace(0, 1, len(authored_profile)),
                    authored_profile,
                )
                book.identity(
                    tag + ":initial-profile:" + n,
                    profiles[n]["theta"][0],
                    np.repeat(initial[:, None], actual["r"], axis=1),
                )
        totals = [profiles[n]["total"] for n in ELECTRODES]
        book.bound(tag + ":total-Li", totals[0] + totals[1] - totals[0][0] - totals[1][0], "Li_mol")
        for n, sign in (("negative", 1), ("positive", -1)):
            book.bound(
                tag + ":" + n + "-Faraday",
                trace["faraday_C_mol"] * (profiles[n]["total"] - profiles[n]["total"][0])
                + sign * (values[:, 2] - values[0, 2]),
                "Faraday_C",
            )
        book.bound(
            tag + ":sum-q", profiles["negative"]["q"].sum(axis=1) - values[:, 1], "current_A"
        )
        book.bound(
            tag + ":paired-q", profiles["negative"]["q"] + profiles["positive"]["q"], "current_A"
        )
        local_inventory = sum(profiles[n]["amounts"].sum(axis=2) for n in ELECTRODES)
        book.bound(tag + ":local-paired-Li", local_inventory - local_inventory[0], "Li_mol")
        control_checks(book, tag, segment, values, stage, data)
        if mode == "A":
            book.bound(
                tag + ":constant-current-charge",
                values[:, 2] - values[0, 2] - target * (times - times[0]),
                "Faraday_C",
            )
        if source:
            source_state_checks(book, tag, trace, segment, profiles, values)
        else:
            native_state_checks(book, tag, trace, segment, profiles, values, times)
            control = segment["control"]
            book.equal(
                tag + ":native-control-declaration",
                {
                    k: control[k]
                    for k in ("index", "duration_s", "target", "unit", "current_limit_A")
                },
                {
                    "index": index,
                    "duration_s": duration,
                    "target": target,
                    "unit": mode,
                    "current_limit_A": data["maximum_current_A"],
                },
            )
            if len(segment["control_modes"]) != nt:
                raise ValueError("missing native control-mode observations")
            expected_mode = (
                "constant_current"
                if mode == "A"
                else {
                    "interior": "constant_voltage",
                    "positive-limit": "positive_current_limit",
                    "negative-limit": "negative_current_limit",
                }[data["expected_mode"]]
            )
            book.equal(
                tag + ":native-control-modes", segment["control_modes"], [expected_mode] * nt
            )
        normalized = {"times": times, "values": values, "profiles": profiles, "grid": actual}
        book.guarded(
            tag + ":electrical-schema",
            electrical_checks,
            book,
            tag,
            segment,
            profiles,
            values,
            data,
            actual,
            layout,
            source,
            stage,
        )
        book.guarded(
            tag + ":stop-schema", stop_checks, book, tag, segment, normalized, data, source
        )
        if result:
            previous = result[-1]
            book.bound(
                tag + ":QW-continuity", values[0, 2:] - previous["values"][-1, 2:], "CV_work_J"
            )
            for name in ELECTRODES:
                book.identity(
                    tag + ":inventory-continuity:" + name,
                    profiles[name]["amounts"][0],
                    previous["profiles"][name]["amounts"][-1],
                )
            previous_raw = trace["segments"][index - 1]
            if source:
                differential_size = trace["source_build"]["rhs_size"]
                book.identity(
                    tag + ":all-source-differential-state-continuity",
                    array(segment["source_state_vectors"])[0, :differential_size],
                    array(previous_raw["source_state_vectors"])[-1, :differential_size],
                )
            else:
                book.equal(
                    tag + ":all-native-state-continuity",
                    segment["initial_state"],
                    previous_raw["final_state"],
                )
        result.append(normalized)
        start = times[-1]
    book.truth(
        label + ":real-evolution",
        any(
            np.any(segment["profiles"][n]["amounts"][-1] != segment["profiles"][n]["amounts"][0])
            for segment in result
            for n in ELECTRODES
        ),
    )
    if not source:
        book.equal(
            label + ":native-final-state",
            trace["final_state"],
            {**trace["segments"][-1]["raw_states"][-1], "spec": trace["native_spec"]},
        )
    return result


def compare_histories(book, label, left, right, data, *, high_sigma=False):
    if left is None or right is None:
        raise ValueError("comparison requires two complete, schema-readable histories")
    if len(left) != len(data["waveform"]) or len(right) != len(left):
        raise ValueError("incomplete comparison stages")
    prefix = "high_sigma_" if high_sigma else "response_"
    for index, (a, b, stage) in enumerate(zip(left, right, data["waveform"], strict=True)):
        tag = f"{label}:stage{index}"
        start = sum(s[0] for s in data["waveform"][:index])
        authored = np.linspace(start, start + stage[0], data["samples_per_segment"])
        if data.get("expected_event"):
            book.bound(tag + ":stop-time", a["times"][-1] - b["times"][-1], "event_time_s")
            authored = authored[
                authored <= min(a["times"][-1], b["times"][-1]) + THRESHOLDS["sample_time_s"]
            ]
        ia, ib = sample_indices(a["times"], authored), sample_indices(b["times"], authored)
        book.truth(
            tag + ":full-common-coverage",
            len(ia) == len(ib) == len(authored),
            {
                "expected": len(authored),
                "left": len(ia),
                "right": len(ib),
                "endpoints_compared_separately": bool(data.get("expected_event")),
            },
        )
        for column, key in enumerate(("voltage_V", "current_A", "charge_C", "work_J")):
            book.bound(
                tag + ":" + key, a["values"][ia, column] - b["values"][ib, column], prefix + key
            )
        for n in ELECTRODES:
            book.bound(
                tag + ":" + n + "-Li",
                a["profiles"][n]["total"][ia] - b["profiles"][n]["total"][ib],
                prefix + "Li_mol",
            )
        # These are sums of whole local reaction-current cells. Joint z/r
        # refinement has no claimed asymptotic order or pointwise-q projection.
        if not high_sigma:
            na, nb = a["grid"]["z"], b["grid"]["z"]
            bins = gcd(na, nb)
            qa = a["profiles"]["negative"]["q"][ia].reshape(len(ia), bins, na // bins).sum(axis=2)
            qb = b["profiles"]["negative"]["q"][ib].reshape(len(ib), bins, nb // bins).sum(axis=2)
            book.bound(tag + ":common-interval-q", qa - qb, "response_interval_current_A")
            book.truth(
                tag + ":spatial-coverage",
                True,
                {
                    "intervals": bins,
                    "left_cells": na,
                    "right_cells": nb,
                    "method": "whole cells on common uniform z intervals; no point interpolation",
                },
            )


def refusal_checks(book, label, route, name, data, grid):
    from battery_collector_spm_cases import native_spec
    from battery_collector_spm_supplement_native import state_snapshot

    finite_tree(route)
    stage, reason = REFUSALS[name]
    entrypoint = {
        "EIS": "chemworld.physchem.battery_impedance.battery_impedance",
        "SOH": "chemworld.physchem.battery_soh.soh_from_battery",
    }.get(name, "chemworld.physchem.battery_models.battery_spec_from_dict")
    book.equal(label + ":name", route["name"], name)
    book.equal(label + ":stage", route["stage"], stage)
    book.equal(label + ":entrypoint", route["entrypoint"], entrypoint)
    book.equal(label + ":exception-class", route["exception"]["type"], "ValueError")
    book.equal(label + ":exception-reason", route["exception"]["message"], reason)
    book.truth(label + ":actually-rejected", route["rejected"] is True)
    spec = native_spec(data, grid, LAYOUTS["opposite"])
    snapshot = json.loads(
        canonical({**state_snapshot(spec.initial_state()), "spec": spec.to_dict()})
    )
    book.equal(label + ":initial-state-binding", route["before"], snapshot)
    book.equal(label + ":unchanged-state", route["after"], route["before"])
    book.truth(label + ":unchanged-state-flag", route["state_unchanged"] is True)
    request = route["request"]
    baseline = snapshot["spec"]
    changed = {
        key
        for key in set(request) | set(baseline)
        if canonical(request.get(key)) != canonical(baseline.get(key))
    }
    field = {
        "SEI": "sei",
        "invalid-tab": "negative_tab",
        "negative-geometry": "negative_collector_thickness_m",
    }.get(name, name)
    book.equal(
        label + ":request-change-scope", sorted(changed), [] if stage == "analysis" else [field]
    )
    if name in ("thermal", "electrolyte", "SEI", "plating"):
        book.truth(
            label + ":real-unsupported-declaration",
            isinstance(request[field], dict) and bool(request[field]),
        )
    if name == "invalid-tab":
        book.equal(label + ":invalid-tab-input", request[field], "middle")
    if name == "negative-geometry":
        book.identity(label + ":negative-geometry-input", request[field], -abs(baseline[field]))


def assess(item):
    """Annotate one complete fixed unit, retaining every raw failure unchanged."""
    book = Checks()
    normalized = {}
    try:
        data = case_data(item["id"])
        book.equal("canonical-inputs", item["inputs"], data)
        plan = route_plan(item["id"])
        routes = item["routes"]
        if not isinstance(routes, dict):
            raise ValueError("routes must be a dictionary")
        book.equal("route-denominator", sorted(routes), sorted(p[0] for p in plan))
    except Exception as error:
        book.truth("unit-schema", False, {"type": type(error).__name__, "message": str(error)})
        plan, routes, data = [], {}, None
    for name, kind, layout_name, grid_name in plan:
        grid = GRIDS[grid_name]
        route = routes.get(name)
        if route is None:
            book.truth(name + ":route-present", False)
            continue
        if not isinstance(route, dict):
            book.truth(name + ":route-schema", False, "route must be a dictionary")
            continue
        if kind == "refusal":
            book.guarded(
                name + ":refusal-schema",
                refusal_checks,
                book,
                name,
                route,
                name.removeprefix("refusal_"),
                data,
                grid,
            )
            continue
        layout = LAYOUTS[layout_name] if layout_name is not None else LAYOUTS["opposite"]
        if kind == "history":

            def paired_header(route=route, grid=grid, layout=layout, name=name):
                finite_tree(route)
                for key, expected in (
                    ("implementation", "history"),
                    ("unit", item["id"]),
                    ("grid", grid),
                    ("layout", layout),
                ):
                    book.equal(name + ":" + key, route[key], expected)

            book.guarded(name + ":paired-schema", paired_header)
            for impl in ("native", "source"):
                normalized[(name, impl)] = book.guarded(
                    name + ":" + impl + ":trace-schema",
                    validate_trace,
                    book,
                    name + ":" + impl,
                    route.get(impl, {}),
                    data,
                    grid,
                    layout,
                    impl == "source",
                    False,
                )
            book.guarded(
                name + ":comparison-schema",
                compare_histories,
                book,
                name + ":same-physical-port",
                normalized[(name, "native")],
                normalized[(name, "source")],
                data,
            )
        else:
            impl = "source" if kind == "source-uniform" else "native"
            normalized[(name, impl)] = book.guarded(
                name + ":trace-schema",
                validate_trace,
                book,
                name,
                route,
                data,
                grid,
                layout,
                impl == "source",
                True,
            )
    if data is not None and item["id"] != "S10":
        for layout_name in LAYOUTS:
            for impl in ("native", "source"):
                left = normalized.get((f"history_{layout_name}_coarse", impl))
                right = normalized.get((f"history_{layout_name}_fine", impl))
                book.guarded(
                    f"{layout_name}:{impl}:grid-comparison",
                    compare_histories,
                    book,
                    f"{layout_name}:{impl}:coarse-fine",
                    left,
                    right,
                    data,
                )
        if item["id"] == "S07":
            for impl in ("native", "source"):
                book.guarded(
                    impl + ":uniform-grid-comparison",
                    compare_histories,
                    book,
                    impl + ":uniform-coarse-fine",
                    normalized.get((f"uniform_{impl}_coarse", impl)),
                    normalized.get((f"uniform_{impl}_fine", impl)),
                    data,
                )
                for grid_name in GRIDS:
                    for layout_name in LAYOUTS:

                        def baseline_comparison(
                            layout_name=layout_name, grid_name=grid_name, impl=impl
                        ):
                            compare_histories(
                                book,
                                f"{layout_name}:{grid_name}:{impl}:high-conductivity",
                                normalized.get((f"history_{layout_name}_{grid_name}", impl)),
                                normalized.get((f"uniform_{impl}_{grid_name}", impl)),
                                data,
                                high_sigma=True,
                            )

                        book.guarded(
                            f"{layout_name}:{grid_name}:{impl}:baseline-schema", baseline_comparison
                        )
    item["checks"] = book.items
    item["failures"] = [check for check in book.items if not check["passed"]]
    item["passed"] = bool(book.items) and not item["failures"]
    return item
