"""Fixed complete-history, independent inventory and finite-grid comparisons."""

import json
from functools import lru_cache

import numpy as np
from battery_dfn_composite_cases import (
    ATOL,
    GRIDS,
    OBSERVATION_COORDINATES,
    RTOL,
    UNITS,
    case_data,
    declared_phases,
    native_spec,
    route_plan,
)
from scipy.integrate import trapezoid


def same(a, b):
    return json.dumps(a, sort_keys=True, allow_nan=False) == json.dumps(
        b, sort_keys=True, allow_nan=False
    )


def finite(value):
    if isinstance(value, dict):
        return "nonfinite" not in value and all(finite(v) for v in value.values())
    if isinstance(value, (list, tuple, np.ndarray)):
        return all(finite(v) for v in value)
    return bool(np.isfinite(value)) if isinstance(value, (int, float, np.number)) else True


def arr(value, shape):
    out = np.asarray(value, dtype=float)
    if out.shape != shape or not np.all(np.isfinite(out)):
        raise ValueError(f"nonfinite/missing/incorrect raw shape {out.shape}, expected {shape}")
    return out


def identity(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return bool(
        a.shape == b.shape
        and np.all(np.isfinite(a))
        and np.all(np.isfinite(b))
        and np.all(abs(a - b) <= 1e-12 + 1e-10 * abs(b))
    )


@lru_cache(maxsize=128)
def source_schema(inputs, grid, limit):
    from battery_dfn_composite_pybamm import build_metadata, build_reference

    return build_metadata(build_reference(json.loads(inputs), json.loads(grid), limit=limit))


def event_kind(label):
    if label is None or label == "final time":
        return None
    label = str(label).removeprefix("event: ")
    if "Minimum voltage" in label or label == "battery:minimum-voltage":
        return "minimum-voltage"
    if "Maximum voltage" in label or label == "battery:maximum-voltage":
        return "maximum-voltage"
    if "phase:" in label:
        parts = label.split(":")
        return "phase:" + parts[parts.index("phase") + 1] + ":" + parts[-1]
    return label


def structural(trace, data, grid, kind, limit):
    """Raises on missing data; successful return never relies on optional fields."""
    for key, value in (
        ("unit", data["id"]),
        ("implementation", kind),
        ("inputs", data),
        ("grid", grid),
        ("limit", limit),
        ("response_units", UNITS),
        ("family", "SPM" if limit == "composite-spm" else "DFN"),
    ):
        if not same(trace[key], value):
            raise ValueError("trace input/schema binding: " + key)
    if "exception" in trace or not finite(trace):
        raise ValueError("exception/nonfinite trace")
    segments = trace["segments"]
    if len(segments) != len(data["waveform"]):
        raise ValueError("incomplete stage history")
    phases, spatial = declared_phases(data, limit), limit != "composite-spm"
    spec = native_spec(data, grid, limit)
    expected_initial = spec.initial_state()
    if kind == "native":
        from battery_dfn_composite_traces import snapshot

        if not same(trace["native_spec"], spec.to_dict()) or not same(
            trace["initial_state"], snapshot(spec, expected_initial)
        ):
            raise ValueError("canonical native initial-state binding")
        layout = None
        from battery_dfn_composite_traces import dispatch

        from chemworld.physchem.reactor_shared import ScalarControlProgram

        event_ids = [
            {
                e.event_id
                for e in dispatch(limit)[1](
                    spec, ScalarControlProgram(((0.0, target),), unit)
                ).all_events
            }
            for _, target, unit in data["waveform"]
        ]
    else:
        metadata = source_schema(
            json.dumps(data, sort_keys=True), json.dumps(grid, sort_keys=True), limit
        )
        if not same(trace["source_build"], metadata) or trace["source_version"] != "25.4.2":
            raise ValueError("actual source layout/scales binding")
        layout = {v["name"]: v for v in metadata["state_layout"]}
        event_ids = [set(metadata["termination_event_names"]) for _ in data["waveform"]]
    if trace["tolerances"]["rtol"] != RTOL or trace["tolerances"]["atol"] != ATOL:
        raise ValueError("solver tolerance identity")
    if not 96485 < trace["faraday_C_mol"] < 96486:
        raise ValueError("Faraday constant identity")
    start = 0.0
    for si, (s, (duration, _, _)) in enumerate(zip(segments, data["waveform"], strict=True)):
        t = np.asarray(s["times_s"], float)
        if t.ndim != 1 or len(t) < 2 or np.any(np.diff(t) <= 0) or not np.all(np.isfinite(t)):
            raise ValueError("invalid raw time axis")
        if abs(t[0] - start) > 1e-9 or s["end_time_s"] != t[-1]:
            raise ValueError("wrong original stage time or missing endpoint")
        if si == 0 and t[0] != 0.0:
            raise ValueError("original t0 missing")
        ending = event_kind(s["termination"])
        if data["expected"] == "complete":
            if ending is not None or abs(t[-1] - start - duration) > 1e-8:
                raise ValueError("short/incorrect complete history")
        elif ending != data["expected"] or not start < t[-1] <= start + duration:
            raise ValueError("actual cutoff event mismatch")
        if kind == "native" and s["outcome"] != ("success" if ending is None else "stopped"):
            raise ValueError("native outcome and event inconsistent")
        if not isinstance(s["solver"], dict) or not s["solver"] or not finite(s["solver"]):
            raise ValueError("missing/nonfinite actual solver diagnostics")
        events = s[("native" if kind == "native" else "source") + "_event_values_at_endpoint"]
        if not isinstance(events, dict) or set(events) != event_ids[si] or not finite(events):
            raise ValueError("missing actual endpoint event residuals")
        requested = start + np.linspace(0.0, duration, data["samples_per_segment"])
        expected_times = requested[requested <= t[-1]]
        if not len(expected_times) or abs(expected_times[-1] - t[-1]) > 1e-9:
            expected_times = np.r_[expected_times, t[-1]]
        if len(t) != len(expected_times) or np.max(abs(t - expected_times)) > 1e-8:
            raise ValueError("missing/replaced requested time samples")
        nt = len(t)
        values = arr(s["values"], (nt, 4))
        if si == 0 and not identity(values[0, 2:], np.zeros(2)):
            raise ValueError("nonzero initial Q/W")
        if set(s["raw_profiles"]) != {p["phase_id"] for p in phases}:
            raise ValueError("phase identity/count mismatch")
        if set(s["regions"]) != ({"negative", "separator", "positive"} if spatial else set()):
            raise ValueError("salt region completeness")
        if kind == "native":
            raw = s["raw_amounts_mol"]
            if len(raw) != nt or any(set(v) != set(spec.inventory_species) for v in raw):
                raise ValueError("native full inventory keys")
            coordinates = s["raw_coordinates"]
            if len(coordinates) != nt or any(
                set(v) != {"discharged_charge_C", "delivered_work_J"} for v in coordinates
            ):
                raise ValueError("native raw Q/W missing")
            if not identity(
                [[c["discharged_charge_C"], c["delivered_work_J"]] for c in coordinates],
                values[:, 2:],
            ):
                raise ValueError("native Q/W projection identity")
            if si == 0 and not identity(
                [raw[0][k] for k in expected_initial.amounts_mol],
                list(expected_initial.amounts_mol.values()),
            ):
                raise ValueError("native authored initial inventory")
            if (
                len(s["raw_algebraic_values"]) != nt
                or any(not isinstance(v, dict) or v != {} for v in s["raw_algebraic_values"])
                or not isinstance(s["roundoff_corrections_mol"], dict)
                or not isinstance(s["conservation_residuals"], dict)
            ):
                raise ValueError("native diagnostics missing")
        else:
            if not same(s["source_state_layout"], metadata["state_layout"]):
                raise ValueError("source segment state layout")
            raw_source = arr(s["source_state_vectors"], (nt, metadata["state_count"]))
            arr(s["source_initial_rhs"], (metadata["rhs_size"],))
            arr(s["source_initial_algebraic_residual"], (metadata["algebraic_size"],))
            for column, name in ((2, "Discharge capacity [A.h]"), (3, "Discharge energy [W.h]")):
                v = layout[name]
                if not identity(
                    values[:, column],
                    3600 * (raw_source[:, v["indices"]].reshape(nt) * v["scale"] + v["reference"]),
                ):
                    raise ValueError("source actual signed Q/W state projection")
            for name in ("Throughput capacity [A.h]", "Throughput energy [W.h]"):
                if si == 0 and not identity(raw_source[0, layout[name]["indices"]], [0.0]):
                    raise ValueError("source actual throughput initial value")
        offset = 0
        for pi, phase in enumerate(phases):
            name, domain = phase["phase_id"], phase["electrode"]
            p = s["raw_profiles"][name]
            region = 0 if domain == "negative" else 2
            nx, nr = (grid["x"][region] if spatial else 1), grid["r"]
            if not same(p["phase"], phase) or p["electrode"] != domain:
                raise ValueError("phase material declaration binding")
            xnodes = (np.arange(nx) + 0.5) * data["lengths_m"][region] / nx + sum(
                data["lengths_m"][:region]
            )
            edges = np.linspace(0.0, phase["radius_m"], nr + 1)
            if (
                not identity(p["x_nodes_m"], xnodes)
                or not identity(p["r_edges_m"], edges)
                or not identity(p["r_nodes_m"], (edges[1:] + edges[:-1]) / 2)
            ):
                raise ValueError("true x/r mesh identity")
            theta = arr(p["theta"], (nt, nx, nr))
            amounts = arr(p["shell_amounts_mol"], (nt, nx, nr))
            capacity_shell = (
                np.diff((edges / phase["radius_m"]) ** 3)
                * phase["active_volume_m3"]
                / nx
                * phase["maximum_concentration_mol_m3"]
            )
            if not identity(amounts, theta * capacity_shell):
                raise ValueError("phase shell amount/theta mapping")
            for key in ("surface", "average", "q", "j", "j0", "eta"):
                arr(p[key], (nt, nx))
            if not identity(
                p["average"],
                np.sum(amounts, axis=2)
                / (phase["active_volume_m3"] / nx * phase["maximum_concentration_mol_m3"]),
            ):
                raise ValueError("phase mean from true inventory")
            if not identity(
                p["q"], np.asarray(p["j"]) * 3 * phase["active_volume_m3"] / phase["radius_m"] / nx
            ):
                raise ValueError("j versus integrated q identity")
            if si == 0 and not identity(theta[0], np.full((nx, nr), phase["initial_fraction"])):
                raise ValueError("authored uniform original particle t0")
            if kind == "native":
                keys = [list(b[3]) for b in spec.particle_bins[offset : offset + nx]]
                if p["inventory_keys"] != keys or not identity(
                    [[[row[k] for k in ks] for ks in keys] for row in raw], amounts
                ):
                    raise ValueError("native each-shell raw inventory identity")
            else:
                label = (
                    ""
                    if limit == "merged-dfn" or domain == "positive"
                    else ("primary " if pi == 0 else "secondary ")
                )
                key = (
                    f"X-averaged {domain} {label}particle concentration [mol.m-3]"
                    if not spatial
                    else f"{domain.capitalize()} {label}particle concentration [mol.m-3]"
                )
                v = layout[key]
                concentration = raw_source[:, v["indices"]] * v["scale"] + v["reference"]
                if not identity(
                    concentration.reshape(nt, nx, nr), theta * phase["maximum_concentration_mol_m3"]
                ):
                    raise ValueError("source original full particle state projection")
            offset += nx
        if spatial:
            for ri, region in enumerate(("negative", "separator", "positive")):
                p = s["regions"][region]
                nx = grid["x"][ri]
                edges = np.linspace(
                    sum(data["lengths_m"][:ri]), sum(data["lengths_m"][: ri + 1]), nx + 1
                )
                if not identity(p["edges_m"], edges) or not identity(
                    p["nodes_m"], (edges[:-1] + edges[1:]) / 2
                ):
                    raise ValueError("salt mesh identity")
                ce = arr(p["salt_mol_m3"], (nt, nx))
                salt = arr(p["salt_amounts_mol"], (nt, nx))
                if not identity(
                    salt, ce * np.diff(edges) * data["area_m2"] * data["porosities"][ri]
                ):
                    raise ValueError("salt amount versus true pore volume")
                arr(p["electrolyte_potential_V"], (nt, nx))
                arr(p["electrolyte_face_current_A"], (nt, nx + 1))
                if region != "separator":
                    arr(p["solid_potential_V"], (nt, nx))
                    arr(p["solid_face_current_A"], (nt, nx + 1))
                if si == 0 and not identity(
                    ce[0], np.full(nx, data["initial_concentration_mol_m3"])
                ):
                    raise ValueError("authored salt initial concentration")
                if kind == "native":
                    keys = list(spec.electrolyte.keys[spec.electrolyte.region_slices[region]])
                    if p["inventory_keys"] != keys or not identity(
                        [[row[k] for k in keys] for row in raw], salt
                    ):
                        raise ValueError("native true salt state projection")
                else:
                    title = (
                        "Separator" if region == "separator" else region.capitalize() + " electrode"
                    )
                    v = layout[title + " porosity times concentration [mol.m-3]"]
                    if not identity(
                        raw_source[:, v["indices"]] * v["scale"] + v["reference"],
                        ce * data["porosities"][ri],
                    ):
                        raise ValueError("source true salt state projection")
                    if region != "separator":
                        v = layout[
                            region.capitalize() + " electrode surface potential difference [V]"
                        ]
                        if not identity(
                            raw_source[:, v["indices"]] * v["scale"] + v["reference"],
                            np.asarray(p["solid_potential_V"])
                            - np.asarray(p["electrolyte_potential_V"]),
                        ):
                            raise ValueError(
                                "source actual original potential-difference state projection"
                            )
        start = t[-1]
    return True


def metrics(trace, data):
    f, checks = trace["faraday_C_mol"], {}
    initial_phase = {
        p["phase_id"]: p["active_volume_m3"]
        * p["maximum_concentration_mol_m3"]
        * p["initial_fraction"]
        for p in declared_phases(data, trace["limit"])
    }
    phase_domain = {p["phase_id"]: p["electrode"] for p in declared_phases(data, trace["limit"])}
    n0 = sum(v for k, v in initial_phase.items() if phase_domain[k] == "negative")
    p0 = sum(v for k, v in initial_phase.items() if phase_domain[k] == "positive")
    salt0 = (
        data["initial_concentration_mol_m3"]
        * data["area_m2"]
        * np.dot(data["lengths_m"], data["porosities"])
    )

    def maximum(key, value):
        checks[key] = max(checks.get(key, 0.0), float(value))

    for si, s in enumerate(trace["segments"]):
        t, rows = np.asarray(s["times_s"]), np.asarray(s["values"])
        current, charge = rows[:, 1], rows[:, 2]
        totals, qtotals = (
            {"negative": np.zeros(len(t)), "positive": np.zeros(len(t))},
            {"negative": np.zeros(len(t)), "positive": np.zeros(len(t))},
        )
        for name, p in s["raw_profiles"].items():
            amount = np.asarray(p["shell_amounts_mol"]).sum(axis=(1, 2))
            q = np.asarray(p["q"]).sum(axis=1)
            totals[p["electrode"]] += amount
            qtotals[p["electrode"]] += q
            maximum(
                "phase_Faraday_quadrature_mol", abs(amount[-1] - amount[0] + trapezoid(q, x=t) / f)
            )
            if si:
                previous = trace["segments"][si - 1]["raw_profiles"][name]
                maximum(
                    "stage_particle_jump_mol",
                    np.max(
                        abs(
                            np.asarray(p["shell_amounts_mol"])[0]
                            - np.asarray(previous["shell_amounts_mol"])[-1]
                        )
                    ),
                )
            # Independent symmetric BV and declared ce-normalized j0.
            phase = p["phase"]
            theta = np.asarray(p["surface"])
            ce = (
                np.asarray(s["regions"][p["electrode"]]["salt_mol_m3"])
                if s["regions"]
                else data["initial_concentration_mol_m3"]
            )
            j0 = phase["exchange_current_scale_A_m2"] * np.sqrt(
                ce / data["initial_concentration_mol_m3"] * theta * (1 - theta)
            )
            eta = (
                2
                * 8.31446261815324
                * data["temperature_K"]
                / f
                * np.arcsinh(np.asarray(p["j"]) / (2 * j0))
            )
            maximum("kinetic_eta_identity_V", np.max(abs(np.asarray(p["eta"]) - eta)))
            maximum("kinetic_j0_identity_A_m2", np.max(abs(np.asarray(p["j0"]) - j0)))
            if s["regions"]:
                region = s["regions"][p["electrode"]]
                ocp = sum(
                    coefficient * theta**power
                    for power, coefficient in enumerate(phase["polynomial"])
                )
                residual = (
                    np.asarray(region["solid_potential_V"])
                    - np.asarray(region["electrolyte_potential_V"])
                    - ocp
                    - np.asarray(p["eta"])
                )
                maximum("electrochemical_closure_V", np.max(abs(residual)))
        maximum("total_Li_mol", np.max(abs(totals["negative"] + totals["positive"] - n0 - p0)))
        maximum("negative_Faraday_mol", np.max(abs(totals["negative"] - n0 + charge / f)))
        maximum("positive_Faraday_mol", np.max(abs(totals["positive"] - p0 - charge / f)))
        maximum(
            "electrode_current_A",
            max(
                np.max(abs(qtotals["negative"] - current)),
                np.max(abs(qtotals["positive"] + current)),
            ),
        )
        if s["regions"]:
            salt = sum(np.asarray(p["salt_amounts_mol"]).sum(axis=1) for p in s["regions"].values())
            maximum("total_salt_mol", np.max(abs(salt - salt0)))
            last_face = None
            for region, p in s["regions"].items():
                ionic = np.asarray(p["electrolyte_face_current_A"])
                qcell = sum(
                    (
                        np.asarray(v["q"])
                        for v in s["raw_profiles"].values()
                        if v["electrode"] == region
                    ),
                    start=np.zeros((len(t), ionic.shape[1] - 1)),
                )
                maximum("local_current_A", np.max(abs(np.diff(ionic, axis=1) - qcell)))
                if last_face is not None:
                    maximum("local_current_A", np.max(abs(ionic[:, 0] - last_face)))
                last_face = ionic[:, -1]
                if region != "separator":
                    solid = np.asarray(p["solid_face_current_A"])
                    maximum("local_current_A", np.max(abs(ionic + solid - current[:, None])))
                if si:
                    maximum(
                        "stage_salt_jump_mol",
                        np.max(
                            abs(
                                np.asarray(p["salt_amounts_mol"])[0]
                                - np.asarray(
                                    trace["segments"][si - 1]["regions"][region]["salt_amounts_mol"]
                                )[-1]
                            )
                        ),
                    )
            maximum(
                "local_current_A",
                max(
                    np.max(
                        abs(
                            np.asarray(s["regions"]["negative"]["electrolyte_face_current_A"])[:, 0]
                        )
                    ),
                    np.max(abs(last_face)),
                ),
            )
        integral = trapezoid(rows[:, 0] * current, x=t)
        maximum(
            "work_quadrature_ratio",
            abs(rows[-1, 3] - rows[0, 3] - integral)
            / (0.2 + 0.001 * abs(rows[-1, 3] - rows[0, 3])),
        )
        _duration, target, unit = data["waveform"][si]
        if unit == "A":
            maximum("CC_current_A", np.max(abs(current - target)))
            maximum("CC_charge_C", np.max(abs(charge - charge[0] - target * (t - t[0]))))
        else:
            if data["control_expectation"] == "interior":
                maximum("CV_voltage_V", np.max(abs(rows[:, 0] - target)))
                maximum(
                    "CV_current_excess_A",
                    np.max(np.maximum(0.0, abs(current) - data["maximum_current_A"])),
                )
            else:
                maximum("CV_limit_A", np.max(abs(current - data["maximum_current_A"])))
                maximum("CV_wrong_voltage_sign_V", np.max(np.maximum(0.0, target - rows[:, 0])))
        if si:
            previous = np.asarray(trace["segments"][si - 1]["values"])[-1]
            maximum("stage_Q_C", abs(rows[0, 2] - previous[2]))
            maximum("stage_W_J", abs(rows[0, 3] - previous[3]))
        if data["expected"] == "minimum-voltage":
            maximum("event_voltage_V", abs(rows[-1, 0] - data["minimum_voltage_V"]))
        if data["expected"].startswith("phase:"):
            phase = data["expected"].split(":")[1]
            maximum(
                "event_surface_fraction",
                abs(
                    np.min(np.asarray(s["raw_profiles"][phase]["surface"])[-1])
                    - data["surface_margin"]
                ),
            )
    if data.get("internal_exchange"):
        s = trace["segments"][0]
        a, b = (float(np.asarray(s["raw_profiles"][k]["q"])[0].sum()) for k in ("host_a", "host_b"))
        checks["internal_exchange_missing"] = (
            0.0 if a * b < 0 and min(abs(a), abs(b)) > 1e-8 else 1.0
        )
    return checks


LIMITS = {
    "total_Li_mol": 1e-9,
    "negative_Faraday_mol": 1e-9,
    "positive_Faraday_mol": 1e-9,
    "phase_Faraday_quadrature_mol": 2e-6,
    "total_salt_mol": 1e-10,
    "electrode_current_A": 1e-7,
    "local_current_A": 1e-7,
    "stage_particle_jump_mol": 1e-9,
    "stage_salt_jump_mol": 1e-9,
    "stage_Q_C": 1e-5,
    "stage_W_J": 1e-4,
    "work_quadrature_ratio": 1.0,
    "kinetic_eta_identity_V": 1e-6,
    "kinetic_j0_identity_A_m2": 1e-7,
    "electrochemical_closure_V": 1e-6,
    "CC_current_A": 1e-7,
    "CC_charge_C": 1e-5,
    "CV_voltage_V": 1e-4,
    "CV_current_excess_A": 1e-5,
    "CV_limit_A": 1e-5,
    "CV_wrong_voltage_sign_V": 1e-4,
    "event_voltage_V": 0.002,
    "event_surface_fraction": 0.002,
    "internal_exchange_missing": 0.0,
}


def sampled(segment, data, *, merge=False):
    """Five fixed x observation nodes; integrated q uses five fixed x intervals."""
    obs = {}
    rows = np.asarray(segment["values"])
    for i, k in enumerate(("V", "I", "Q", "W")):
        obs[k] = rows[:, i]
    for name, p in segment["raw_profiles"].items():
        phase = p["phase"]
        nx = len(p["x_nodes_m"])
        region = 0 if p["electrode"] == "negative" else 2
        xn = (np.asarray(p["x_nodes_m"]) - sum(data["lengths_m"][:region])) / data["lengths_m"][
            region
        ]
        rn = np.asarray(p["r_nodes_m"]) / phase["radius_m"]

        def xinterp(v, xn=xn):
            return np.array([np.interp(OBSERVATION_COORDINATES, xn, row) for row in np.asarray(v)])

        for key in ("surface", "average", "j", "j0", "eta"):
            obs[f"{name}:{key}"] = xinterp(p[key])
        prof = np.asarray(p["theta"])
        rsmall = np.array(
            [[np.interp(OBSERVATION_COORDINATES, rn, row) for row in at] for at in prof]
        )
        obs[name + ":theta"] = np.stack([xinterp(rsmall[:, :, i]) for i in range(5)], axis=2)
        q = np.asarray(p["q"])
        edges = np.linspace(0.0, 1.0, nx + 1)
        dest = np.linspace(0.0, 1.0, 6)
        weights = (
            np.maximum(
                0.0,
                np.minimum(dest[1:, None], edges[None, 1:])
                - np.maximum(dest[:-1, None], edges[None, :-1]),
            )
            * nx
        )
        obs[name + ":q"] = q @ weights.T
        obs[name + ":qtotal"] = q.sum(axis=1)
        obs[name + ":amount"] = np.asarray(p["shell_amounts_mol"]).sum(axis=(1, 2))
    if merge and "host_b:amount" in obs:
        a, b = data["phases"][:2]
        av, bv = a["active_volume_m3"], b["active_volume_m3"]
        for k in ("surface", "average", "theta", "j", "j0", "eta"):
            obs["host_a:" + k] = (av * obs["host_a:" + k] + bv * obs["host_b:" + k]) / (av + bv)
        for k in ("q", "qtotal", "amount"):
            obs["host_a:" + k] += obs["host_b:" + k]
        obs = {k: v for k, v in obs.items() if not k.startswith("host_b:")}
    for ri, (name, p) in enumerate(segment["regions"].items()):
        start = sum(data["lengths_m"][:ri])
        length = data["lengths_m"][ri]
        for key in (
            "salt_mol_m3",
            "electrolyte_potential_V",
            "solid_potential_V",
            "electrolyte_face_current_A",
            "solid_face_current_A",
        ):
            if key not in p:
                continue
            node = "edges_m" if "face_current" in key else "nodes_m"
            coords = (np.asarray(p[node]) - start) / length
            obs[name + ":" + key] = np.array(
                [np.interp(OBSERVATION_COORDINATES, coords, row) for row in p[key]]
            )
    return obs


def bound(key):
    suffix = key.split(":")[-1]
    if key == "V" or "potential_V" in key:
        return 0.005, 1e-5
    if key == "I" or suffix == "qtotal" or "face_current_A" in key:
        return 0.01, 0.01
    if key == "Q":
        return 0.05, 0.001
    if key == "W":
        return 0.2, 0.001
    if suffix == "eta":
        return 0.003, 1e-5
    if suffix in ("j", "j0"):
        return 0.05, 0.01
    if suffix == "q":
        return 0.002, 0.01
    if suffix == "amount":
        return 2e-6, 1e-5
    if suffix == "salt_mol_m3":
        return 5.0, 0.01
    return 0.005, 1e-5


def compare(left, right, data):
    result = []
    merge = left["limit"] == "merged-dfn" or right["limit"] == "merged-dfn"
    if len(left["segments"]) != len(right["segments"]):
        return [{"passed": False, "reason": "stage count mismatch"}]
    for i, (a, b) in enumerate(zip(left["segments"], right["segments"], strict=True)):
        t1, t2 = np.asarray(a["times_s"]), np.asarray(b["times_s"])
        # Structural checks independently enforce each authored endpoint.
        points = np.unique(np.r_[t1[t1 <= min(t1[-1], t2[-1])], t2[t2 <= min(t1[-1], t2[-1])]])
        oa, ob = sampled(a, data, merge=merge), sampled(b, data, merge=merge)
        if same(left["grid"], right["grid"]) and left["limit"] == right["limit"]:
            for original, observations in ((a, oa), (b, ob)):
                for phase, p in original["raw_profiles"].items():
                    for field in (
                        "theta",
                        "surface",
                        "average",
                        "j",
                        "j0",
                        "q",
                        "eta",
                        "shell_amounts_mol",
                    ):
                        observations[
                            phase + ":raw:" + ("amount" if field == "shell_amounts_mol" else field)
                        ] = np.asarray(p[field])
                for region, p in original["regions"].items():
                    for field in (
                        "salt_mol_m3",
                        "electrolyte_potential_V",
                        "solid_potential_V",
                        "electrolyte_face_current_A",
                        "solid_face_current_A",
                    ):
                        if field in p:
                            observations[region + ":raw:" + field] = np.asarray(p[field])
        common = set(oa) & set(ob)
        if not common:
            return [{"passed": False, "reason": "empty observations"}]
        if (
            left["limit"] != "composite-spm"
            and right["limit"] != "composite-spm"
            and set(oa) != set(ob)
        ):
            return [{"passed": False, "reason": "missing observations"}]
        for key in sorted(common):
            aa, bb = oa[key], ob[key]
            ashape = aa.shape[1:]
            a2 = aa.reshape(len(t1), -1)
            b2 = bb.reshape(len(t2), -1)
            va = np.stack([np.interp(points, t1, a2[:, j]) for j in range(a2.shape[1])], axis=1)
            vb = np.stack([np.interp(points, t2, b2[:, j]) for j in range(b2.shape[1])], axis=1)
            # Both actual original event endpoints remain a separately paired observation.
            va = np.vstack((va, a2[-1]))
            vb = np.vstack((vb, b2[-1]))
            atol, rtol = bound(key)
            error = abs(va - vb)
            ratio = error / (atol + rtol * abs(vb))
            where = np.unravel_index(np.argmax(ratio), ratio.shape)
            result.append(
                {
                    "passed": bool(np.all(np.isfinite(ratio)) and np.max(ratio) <= 1),
                    "stage": i,
                    "output": key,
                    "max_ratio": float(ratio[where]),
                    "absolute_error": float(error[where]),
                    "reference_value": float(vb[where]),
                    "time_s": float(points[where[0]]) if where[0] < len(points) else None,
                    "paired_actual_endpoint": bool(where[0] == len(points)),
                    "component": int(where[1]),
                    "shape": ashape,
                    "atol": atol,
                    "rtol": rtol,
                }
            )
        if data["expected"] != "complete":
            difference = abs(t1[-1] - t2[-1])
            tol = 1.0 + 0.005 * t2[-1]
            result.append(
                {
                    "passed": bool(difference <= tol),
                    "stage": i,
                    "output": "event_time_s",
                    "max_ratio": float(difference / tol),
                    "absolute_error": float(difference),
                }
            )
    return result


REFUSAL_ERRORS = {
    "duplicate-phase-id": ("construction", "negative material phase identities must be unique"),
    "wrong-topology": (
        "construction",
        "composite DFN requires exactly two negative and one positive phase",
    ),
    "nonmonotone-ocv": (
        "construction",
        "phase OCV monotonicity is uncertified for unique current sharing",
    ),
    "wrong-ocv-family": (
        "construction",
        "phase OCV/initial state must cover the declared DFN surfaces",
    ),
    "zero-particle-diffusivity": ("construction", "diffusivity_m2_s must be positive and finite"),
    "zero-electrolyte-conductivity": (
        "construction",
        "conductivity_S_m must be positive and finite",
    ),
    "missing-particle-Li": (
        "consumer",
        "composite DFN state requires every phase/salt inventory and original Q/W",
    ),
    "missing-salt": (
        "consumer",
        "composite DFN state requires every phase/salt inventory and original Q/W",
    ),
    "EIS": ("consumer", "composite DFN impedance closure is unsupported"),
    "SOH": ("consumer", "composite DFN automatic equilibrium-SOH/SOC mapping is unsupported"),
    "chemical-energy": (
        "consumer",
        "composite DFN chemical-energy and equilibrium-SOH closure is unsupported",
    ),
    "thermal": ("consumer", "composite DFN is isothermal; thermal coupling is unsupported"),
}


def declared_rejection(route, name):
    stage, message = REFUSAL_ERRORS[name]
    return route["stage"] == stage and route["error"] == {"type": "ValueError", "message": message}


def comparison_pairs(data):
    pairs = [("native_" + g, "reference_" + g) for g in GRIDS] + [
        (k + "_coarse", k + "_fine") for k in ("native", "reference")
    ]
    limit = data.get("limit")
    if limit:
        pairs += [
            (k + "_" + g, k + "_" + g + "_" + limit) for k in ("native", "reference") for g in GRIDS
        ]
        pairs += [("native_" + g + "_" + limit, "reference_" + g + "_" + limit) for g in GRIDS]
        pairs += [(k + "_coarse_" + limit, k + "_fine_" + limit) for k in ("native", "reference")]
    return pairs


def assess(item):
    data = item["inputs"]
    index = int(data["id"][2:])
    checks = {}
    diagnostics = {}
    plan = route_plan(index)
    checks["canonical_fixed_inputs"] = item["id"] == data["id"] and same(data, case_data(index))
    checks["exact_route_denominator"] = set(item["routes"]) == {r[0] for r in plan}
    for name, kind, level, limit in plan:
        route = item["routes"].get(name, {})
        try:
            if kind == "refusal":
                from battery_dfn_composite_traces import snapshot

                spec = native_spec(data, GRIDS[level])
                expected = snapshot(spec, spec.initial_state())
                checks[name + ":refusal"] = (
                    route["rejected"] is True
                    and route["kind"] == name
                    and route["unit"] == data["id"]
                    and same(route["inputs"], data)
                    and same(route["grid"], GRIDS[level])
                    and route["implementation"] == "refusal"
                    and declared_rejection(route, name)
                    and finite(route)
                    and same(route["state_before"], expected)
                    and same(route["state_after"], expected)
                )
                continue
            structural(route, data, GRIDS[level], kind, limit)
            checks[name + ":complete_raw_identity"] = True
            measures = metrics(route, data)
            diagnostics[name] = measures
            for key, value in measures.items():
                checks[name + ":" + key] = bool(np.isfinite(value) and value <= LIMITS[key])
        except Exception as exc:
            checks[name + ":complete_raw_identity"] = False
            diagnostics[name] = {"assessment_error": type(exc).__name__ + ": " + str(exc)}
    comparisons = {}
    if index != 14:
        pairs = comparison_pairs(data)
        for a, b in pairs:
            key = a + " vs " + b
            if not all(checks.get(n + ":complete_raw_identity", False) for n in (a, b)):
                checks[key] = False
                comparisons[key] = [{"passed": False, "reason": "incomplete raw identity gate"}]
                continue
            try:
                rows = compare(item["routes"][a], item["routes"][b], data)
                comparisons[key] = rows
                checks[key] = bool(rows) and all(v["passed"] for v in rows)
            except Exception as exc:
                comparisons[key] = [
                    {"passed": False, "reason": type(exc).__name__ + ": " + str(exc)}
                ]
                checks[key] = False
    item.update(
        checks=checks,
        diagnostics=diagnostics,
        comparisons=comparisons,
        failures=[k for k, v in checks.items() if not v],
        passed=bool(checks) and all(checks.values()),
    )
