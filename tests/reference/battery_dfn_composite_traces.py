"""Direct native histories and explicit domain refusals; no independent solver."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
from battery_dfn_composite_cases import ATOL, RTOL, UNITS, F, declared_phases, native_spec


def dispatch(limit):
    if limit == "composite-spm":
        from chemworld.physchem.battery_composite import composite_fields, composite_system

        return composite_fields, composite_system
    if limit == "merged-dfn":
        from chemworld.physchem.battery_dfn import dfn_fields, dfn_system

        return dfn_fields, dfn_system
    from chemworld.physchem.battery_dfn_composite import composite_dfn_fields, composite_dfn_system

    return composite_dfn_fields, composite_dfn_system


def native_time(data, grid, *, limit=None):
    from chemworld.physchem.reactor_shared import ScalarControlProgram

    spec = native_spec(data, grid, limit)
    fields_function, system_function = dispatch(limit)
    phases = declared_phases(data, limit)
    state, segments = spec.initial_state(), []
    spatial = limit != "composite-spm"
    initial = snapshot(spec, state)
    for index, (duration, target, unit) in enumerate(data["waveform"]):
        progress = {
            "completed_segments": segments,
            "stage": "integrate",
            "active_segment": {
                "index": index,
                "start_time_s": state.time_s,
                "duration_s": duration,
                "target": target,
                "unit": unit,
                "requested_end_time_s": state.time_s + duration,
            },
            "current_time_s": state.time_s,
        }
        try:
            program = ScalarControlProgram(((0.0, target),), unit)
            system = system_function(spec, program)
            system = replace(system, policy=replace(system.policy, rtol=RTOL, atol=ATOL))
            result = system.advance(
                state,
                duration,
                evaluation_times_s=np.linspace(0.0, duration, data["samples_per_segment"]),
            )
            states = list(result.trajectory)
            if not states or states[-1].time_s != result.final_state.time_s:
                states.append(result.final_state)
            progress.update(stage="extract", current_time_s=result.final_state.time_s)
            fields = [fields_function(spec, s, program=program) for s in states]
            record = {
                "times_s": [s.time_s for s in states],
                "end_time_s": result.final_state.time_s,
                "termination": result.stopping_event,
                "outcome": result.outcome,
                "values": [
                    [
                        f["voltage_V"],
                        f["current_A"],
                        s.coordinates["discharged_charge_C"],
                        s.coordinates["delivered_work_J"],
                    ]
                    for s, f in zip(states, fields, strict=True)
                ],
                "raw_profiles": {},
                "regions": {},
                "raw_amounts_mol": [dict(s.amounts_mol) for s in states],
                "raw_coordinates": [dict(s.coordinates) for s in states],
                "raw_algebraic_values": [dict(s.algebraic_values) for s in states],
                "solver": dict(result.solver_diagnostic),
                "conservation_residuals": dict(result.conservation_residuals),
                "roundoff_corrections_mol": dict(result.roundoff_corrections_mol),
            }
            progress["active_segment"]["record"] = record
            offset = 0
            for phase in phases:
                name, domain = phase["phase_id"], phase["electrode"]
                region = 0 if domain == "negative" else 2
                nx = grid["x"][region] if spatial else 1
                bins = spec.particle_bins[offset : offset + nx]
                particle = bins[0][2]
                amounts = np.array(
                    [[[s.amounts_mol[k] for k in b[3]] for b in bins] for s in states]
                )
                theta = amounts / (
                    particle.mesh.cell_volumes_m3 * particle.maximum_concentration_mol_m3
                )
                arrays = {
                    key: np.array([f[field][offset : offset + nx] for f in fields])
                    for key, field in (
                        ("surface", "surface_fraction"),
                        ("average", "average_fraction"),
                        ("q", "reaction_current_A"),
                        ("eta", "kinetic_overpotential_V"),
                    )
                }
                arrays["j"] = arrays["q"] / particle.area_m2
                if spatial:
                    e = spec.electrolyte
                    selected = e.region_slices[domain]
                    local_ce = np.array(
                        [f["electrolyte_concentration_mol_m3"][selected] for f in fields]
                    )
                    xnodes = e.mesh.centers_m[selected]
                else:
                    local_ce = np.full_like(arrays["q"], data["initial_concentration_mol_m3"])
                    xnodes = [sum(data["lengths_m"][:region]) + data["lengths_m"][region] / 2]
                arrays["j0"] = phase["exchange_current_scale_A_m2"] * np.sqrt(
                    local_ce
                    / data["initial_concentration_mol_m3"]
                    * arrays["surface"]
                    * (1 - arrays["surface"])
                )
                record["raw_profiles"][name] = {
                    "electrode": domain,
                    "phase": phase,
                    "inventory_keys": [list(b[3]) for b in bins],
                    "x_nodes_m": np.asarray(xnodes).tolist(),
                    "r_nodes_m": particle.mesh.centers_m.tolist(),
                    "r_edges_m": list(particle.mesh.faces_m),
                    "shell_amounts_mol": amounts.tolist(),
                    "theta": theta.tolist(),
                    **{k: v.tolist() for k, v in arrays.items()},
                }
                offset += nx
            if spatial:
                e = spec.electrolyte
                nodes, edges = e.mesh.centers_m, np.asarray(e.mesh.faces_m)
                for name in ("negative", "separator", "positive"):
                    selected = e.region_slices[name]
                    start, end = selected.start, selected.stop
                    face = slice(start, end + 1)
                    view = {
                        "nodes_m": nodes[selected].tolist(),
                        "edges_m": edges[face].tolist(),
                        "inventory_keys": list(e.keys[selected]),
                        "salt_mol_m3": [
                            f["electrolyte_concentration_mol_m3"][selected].tolist() for f in fields
                        ],
                        "salt_amounts_mol": [
                            [s.amounts_mol[k] for k in e.keys[selected]] for s in states
                        ],
                        "electrolyte_potential_V": [
                            f["electrolyte_potential_V"][selected].tolist() for f in fields
                        ],
                        "electrolyte_face_current_A": [
                            (f["electrolyte_face_current_A_m2"][face] * e.area_m2).tolist()
                            for f in fields
                        ],
                    }
                    if name != "separator":
                        solid_slice = (
                            slice(0, e.cells[0]) if name == "negative" else slice(e.cells[0], None)
                        )
                        view["solid_potential_V"] = [
                            f["solid_potential_V"][solid_slice].tolist() for f in fields
                        ]
                        view["solid_face_current_A"] = [
                            (f[name + "_solid_face_current_A_m2"] * e.area_m2).tolist()
                            for f in fields
                        ]
                    record["regions"][name] = view
            record["native_event_values_at_endpoint"] = {
                event.event_id: float(event.value(result.final_state))
                for event in system.all_events
            }
            segments.append(record)
            state = result.final_state
            if result.outcome != "success":
                break
        except Exception as exc:
            exc.partial_history = progress
            raise
    return {
        "implementation": "native",
        "family": "SPM" if not spatial else "DFN",
        "unit": data["id"],
        "inputs": data,
        "grid": grid,
        "limit": limit,
        "native_spec": spec.to_dict(),
        "initial_state": initial,
        "response_units": UNITS,
        "faraday_C_mol": F,
        "segments": segments,
        "tolerances": {"rtol": RTOL, "atol": ATOL},
    }


def snapshot(spec, state):
    return {
        "amounts_mol": dict(state.amounts_mol),
        "coordinates": dict(state.coordinates),
        "algebraic_values": dict(state.algebraic_values),
        "time_s": state.time_s,
        "temperature_K": state.temperature_K,
        "volume_l": state.volume_L,
        "spec": spec.to_dict(),
    }


def refusal(data, name, grid):
    from chemworld.physchem.battery_impedance import battery_impedance
    from chemworld.physchem.battery_models import battery_chemical_energy_j
    from chemworld.physchem.battery_soh import soh_from_battery
    from chemworld.physchem.reactor_shared import ScalarControlProgram

    spec = native_spec(data, grid)
    state = spec.initial_state()
    before = snapshot(spec, state)
    d = deepcopy(data)
    rejected, error, stage = False, None, "construction"
    try:
        if name == "duplicate-phase-id":
            d["phases"][1]["phase_id"] = d["phases"][0]["phase_id"]
            native_spec(d, grid)
        elif name == "wrong-topology":
            replace(spec, negative_phases=(spec.negative_phases[0],))
        elif name == "nonmonotone-ocv":
            d["phases"][0]["polynomial"] = (0.1, 0.2)
            native_spec(d, grid)
        elif name == "wrong-ocv-family":
            phase = spec.negative_phases[0]
            altered = replace(phase, ocv=replace(phase.ocv, supported_models=("SPM",)))
            replace(spec, negative_phases=(altered, spec.negative_phases[1]))
        elif name == "zero-particle-diffusivity":
            d["phases"][0]["diffusivity_m2_s"] = 0.0
            native_spec(d, grid)
        elif name == "zero-electrolyte-conductivity":
            d["conductivity_S_m"] = 0.0
            native_spec(d, grid)
        else:
            stage = "consumer"
            if name in ("missing-particle-Li", "missing-salt"):
                amounts = dict(state.amounts_mol)
                del amounts[
                    spec.particle_bins[0][3][0]
                    if name == "missing-particle-Li"
                    else spec.electrolyte.keys[0]
                ]
                dispatch(None)[0](spec, replace(state, amounts_mol=amounts), current_A=0.0)
            elif name == "EIS":
                battery_impedance(spec, [1.0], state=state)
            elif name == "SOH":
                soh_from_battery(spec, state)
            elif name == "chemical-energy":
                battery_chemical_energy_j(spec, state)
            elif name == "thermal":
                dispatch(None)[1](
                    spec, ScalarControlProgram(((0.0, 0.0),), "A"), thermal_device=object()
                )
            else:
                raise RuntimeError("Unknown declared refusal")
    except (ValueError, TypeError, NotImplementedError) as exc:
        rejected, error = True, {"type": type(exc).__name__, "message": str(exc)}
    return {
        "implementation": "refusal",
        "unit": data["id"],
        "inputs": data,
        "grid": grid,
        "kind": name,
        "rejected": rejected,
        "error": error,
        "stage": stage,
        "state_before": before,
        "state_after": snapshot(spec, state),
    }
