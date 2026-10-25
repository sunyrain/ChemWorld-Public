"""Fixed 54-source + 48-common-G closure blocks; create-only external output."""

import argparse
import itertools
import json
import sys
import time
import traceback
from dataclasses import replace
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chemworld.physchem.distributed_ro import (
    advance_distributed_ro,
    distributed_ro_inventory,
    distributed_ro_profile,
)
from chemworld.physchem.pressure_recovery import IsothermalLiquidPressureRecoverySpec
from chemworld.physchem.seawater_ro import seawater_ro_profile
from reference.ro47_fixtures import spec as binary_spec
from reference.ro47close_fixtures import (
    mcas_feed,
    mcas_spec,
    multisolute_feed,
    multisolute_spec,
    seawater_spec,
)
from reference.ro47close_source_canary import build_source
from reference.ro47close_source_solve import solve_source

NAMES = {
    "H2O": "H2O",
    "Na+": "Na_+",
    "Cl-": "Cl_-",
    "Ca2+": "Ca_2+",
    "SO4--": "SO4_2-",
    "urea": "urea",
}


def source_cases():
    rows = []
    for ti, si, pi, li in itertools.product(range(3), range(2), range(2), range(2)):
        rows.append(
            {
                "mode": "seawater",
                "temperature": (290.0, 310.0, 330.0)[ti],
                "salinity": (0.015, 0.045)[si],
                "pressure": (5e6, 7e6)[pi],
                "length": (3.0, 6.0)[li],
                "area": (5.0, 15.0)[(ti + si + pi + li) % 2],
                "nfe": 3,
            }
        )
    for ti, si, pi, ni in itertools.product(range(3), range(2), range(2), range(2)):
        parity = (ti + si + pi + ni) % 2
        rows.append(
            {
                "mode": "mcas",
                "temperature": (290.0, 310.0, 330.0)[ti],
                "feed_scale": (0.5, 1.5)[si],
                "pressure": (4e6, 6e6)[pi],
                "neutral": bool(ni),
                "length": (3.0, 6.0)[parity],
                "area": (15.0, 5.0)[parity],
                "nfe": 3,
            }
        )
    for mode, nfe in itertools.product(("seawater", "mcas"), (3, 6, 12)):
        rows.append(
            dict(
                mode=mode,
                temperature=298.15,
                pressure=5e6,
                length=5.0,
                area=10.0,
                nfe=nfe,
                **(
                    {"salinity": 0.035}
                    if mode == "seawater"
                    else {"neutral": True, "feed_scale": 1.0}
                ),
            )
        )
    return rows


def compare_source(case, output, *, source_domain="default"):
    from pyomo.environ import value

    args = dict(case)
    mode = args.pop("mode")
    t, p = case["temperature"], case["pressure"]
    if mode == "seawater":
        native = replace(
            seawater_spec(finite_elements=case["nfe"]),
            temperature_K=t,
            inlet_pressure_Pa=p,
            membrane_length_m=case["length"],
            membrane_area_m2=case["area"],
        )
        feed = {"H2O": 1 - case["salinity"], "TDS": case["salinity"]}
        profile = seawater_ro_profile(native, feed, native.properties.volume(feed, t, 101325.0))
    else:
        native = mcas_spec(neutral=case["neutral"], finite_elements=case["nfe"])
        native = replace(
            native,
            temperature_K=t,
            inlet_pressure_Pa=p,
            membrane_area_m2=case["area"],
            channel=replace(native.channel, length_m=case["length"]),
        )
        feed = mcas_feed(native)
        feed = {k: v if k == "H2O" else v * case["feed_scale"] for k, v in feed.items()}
        args["molecular_weights"] = {
            NAMES[k]: v for k, v in native.properties.component_mass_kg_mol.items()
        }
        profile = distributed_ro_profile(native, feed, native.properties.volume(feed, t, 101325.0))
    model = build_source(mode, **args, source_domain=source_domain)
    unit = model.fs.unit
    # Construct only physically used state properties before solving; the
    # unconnected x=0 interface/permeate source placeholders are not consumers.
    points = [*unit.feed_side.properties.values(), unit.mixed_permeate[0]]
    points.extend(unit.feed_side.properties_interface[0, x] for x in unit.difference_elements)
    points.extend(unit.permeate_side[0, x] for x in unit.difference_elements)
    for state in points:
        _ = state.dens_mass_phase
        _ = state.pressure_osm_phase
        _ = state.visc_d_phase
        _ = state.enth_mass_phase
        _ = state.diffus_phase_comp
    solved = solve_source(
        model,
        ipopt=True,
        initialize=mode == "mcas",
        source_continuation=mode == "seawater" and source_domain != "default",
    )
    (output / "source.json").write_text(json.dumps(solved, indent=2), encoding="utf8")
    (output / "native.json").write_text(json.dumps(profile, indent=2), encoding="utf8")
    comparisons = []

    def pair(key, actual, reference):
        reference = float(value(reference))
        actual = float(actual)
        error = abs(actual - reference)
        comparisons.append(
            {
                "quantity": key,
                "native": actual,
                "source": reference,
                "absolute_error": error,
                "relative_error": error / max(abs(reference), 1e-30),
                "passed": error <= 1e-10 + 5e-7 * abs(reference),
            }
        )

    stations = profile["stations"] if mode == "seawater" else profile["stations"][1:]
    charge = []
    for x, row in zip(unit.difference_elements, stations, strict=True):
        bulk = unit.feed_side.properties[0, x]
        interface = unit.feed_side.properties_interface[0, x]
        permeate = unit.permeate_side[0, x]
        prefix = f"x={x}/"
        pair(prefix + "pressure", row["pressure_Pa"], bulk.pressure)
        pair(
            prefix + "pressure_gradient", row["pressure_gradient_Pa_m"], -unit.feed_side.dP_dx[0, x]
        )
        if mode == "seawater":
            for nk, j in (("water", "H2O"), ("tds", "TDS")):
                pair(
                    prefix + j + "/flow",
                    row[nk + "_mass_flow_kg_s"],
                    bulk.flow_mass_phase_comp["Liq", j],
                )
                pair(
                    prefix + j + "/flux",
                    row[nk + "_flux_kg_m2_s"],
                    unit.flux_mass_phase_comp[0, x, "Liq", j],
                )
            pair(prefix + "K", row["film_coefficient_m_s"], unit.feed_side.K[0, x, "TDS"])
            pair(
                prefix + "CP",
                row["concentration_polarization_modulus"],
                interface.conc_mass_phase_comp["Liq", "TDS"]
                / bulk.conc_mass_phase_comp["Liq", "TDS"],
            )
            for label, state in (("bulk", bulk), ("interface", interface), ("permeate", permeate)):
                props = row[label]
                for nk, reference in (
                    ("density_kg_m3", state.dens_mass_phase["Liq"]),
                    ("viscosity_Pa_s", state.visc_d_phase["Liq"]),
                    ("osmotic_pressure_Pa", state.pressure_osm_phase["Liq"]),
                    ("specific_enthalpy_J_kg", state.enth_mass_phase["Liq"]),
                    ("osmotic_coefficient", state.osm_coeff),
                    ("diffusivity_m2_s", state.diffus_phase_comp["Liq", "TDS"]),
                ):
                    pair(prefix + label + "/" + nk, props[nk], reference)
        else:
            props = native.properties
            weights = props.component_mass_kg_mol
            for key in props.component_ids:
                source_key = NAMES[key]
                pair(
                    prefix + key + "/flow",
                    row["component_mol_s"][key] * weights[key],
                    bulk.flow_mass_phase_comp["Liq", source_key],
                )
                pair(
                    prefix + key + "/flux",
                    row["component_flux_mol_m2_s"][key] * weights[key],
                    unit.flux_mass_phase_comp[0, x, "Liq", source_key],
                )
            for channel in props.channels:
                cid = channel.channel_id
                for key, nu in channel.stoichiometry:
                    source_key = NAMES[key]
                    pair(
                        prefix + key + "/interface_concentration",
                        row["interface_concentrations_mol_m3"][cid] * nu,
                        interface.conc_mol_phase_comp["Liq", source_key],
                    )
                    pair(
                        prefix + key + "/permeate_concentration",
                        row["permeate_concentrations_mol_m3"][cid] * nu,
                        permeate.conc_mol_phase_comp["Liq", source_key],
                    )
                    pair(
                        prefix + key + "/K",
                        row["film_coefficients_m_s"][cid],
                        unit.feed_side.K[0, x, source_key],
                    )
                    pair(
                        prefix + key + "/CP",
                        row["interface_concentrations_mol_m3"][cid]
                        / row["bulk_concentrations_mol_m3"][cid],
                        interface.conc_mol_phase_comp["Liq", source_key]
                        / bulk.conc_mol_phase_comp["Liq", source_key],
                    )
            for label, state in (("bulk", bulk), ("interface", interface), ("permeate", permeate)):
                if label == "bulk":
                    n = row["component_mol_s"]
                    volume = props.volume(n, t, row["pressure_Pa"])
                    pi = 8.31446261815324 * t * sum(n[k] for k in props.solute_ids) / volume
                else:
                    concentrations = row[label + "_concentrations_mol_m3"]
                    n = props.amounts(0.0, concentrations)
                    solute_mass = sum(n[k] * weights[k] for k in props.solute_ids)
                    n["H2O"] = (1000 - solute_mass) / weights["H2O"]
                    pi = row[label + "_osmotic_pressure_Pa"]
                mass = sum(n[k] * weights[k] for k in n)
                pressure = 101325.0 if label == "permeate" else row["pressure_Pa"]
                pair(prefix + label + "/density", 1000.0, state.dens_mass_phase["Liq"])
                pair(prefix + label + "/viscosity", 0.001, state.visc_d_phase["Liq"])
                pair(prefix + label + "/osmotic", pi, state.pressure_osm_phase["Liq"])
                pair(
                    prefix + label + "/enthalpy",
                    props.enthalpy(n, t, pressure) / mass,
                    state.enth_mass_phase["Liq"],
                )
            local_source_charge = sum(
                value(unit.flux_mass_phase_comp[0, x, "Liq", NAMES[k]])
                / weights[k]
                * props.charges[k]
                for k in props.solute_ids
            )
            charge.append(
                {
                    "x": float(x),
                    "native": row["charge_flux_mol_m2_s"],
                    "source": local_source_charge,
                }
            )
    pair("pressure_drop", profile["pressure_drop_Pa"], -unit.deltaP[0])
    pair("recovery", profile["water_recovery"], unit.recovery_mass_phase_comp[0, "Liq", "H2O"])
    for native_key, source_key in (("permeate", "mixed_permeate"), ("retentate", "retentate")):
        state = (
            unit.mixed_permeate[0]
            if source_key == "mixed_permeate"
            else unit.feed_side.properties[0, 1]
        )
        if mode == "seawater":
            amounts = profile[native_key + "_kg_s"]
            mass = sum(amounts.values())
            h = native.properties.enthalpy(amounts, t, float(value(state.pressure)))
            for key in amounts:
                pair(native_key + "/" + key, amounts[key], state.flow_mass_phase_comp["Liq", key])
        else:
            amounts = profile[native_key + "_mol_s"]
            mass = sum(amounts[k] * native.properties.component_mass_kg_mol[k] for k in amounts)
            h = native.properties.enthalpy(amounts, t, float(value(state.pressure)))
            for key in amounts:
                pair(
                    native_key + "/" + key,
                    amounts[key] * native.properties.component_mass_kg_mol[key],
                    state.flow_mass_phase_comp["Liq", NAMES[key]],
                )
        pair(
            native_key + "/enthalpy_flow",
            h,
            sum(state.flow_mass_phase_comp["Liq", j] for j in state.params.component_list)
            * state.enth_mass_phase["Liq"],
        )
    (output / "comparisons.json").write_text(json.dumps(comparisons, indent=2), encoding="utf8")
    passed = (
        all(r["passed"] for r in comparisons)
        and solved["all_source_equalities_passed"]
        and solved["all_source_bounds_passed"]
        and solved["degrees_of_freedom"] == 0
        and all(abs(c["native"]) < 1e-12 and abs(c["source"]) < 1e-10 for c in charge)
    )
    return {
        "passed": passed,
        "comparisons": len(comparisons),
        "failed_comparisons": [r for r in comparisons if not r["passed"]],
        "maximum_relative_error": max(r["relative_error"] for r in comparisons),
        "source_maximum_scaled_residual": solved["max_scaled_residual"],
        "source_equalities": solved["original_equalities"],
        "source_dof": solved["degrees_of_freedom"],
        "source_bounds_passed": solved["all_source_bounds_passed"],
        "charge": charge,
        "water_recovery": profile["water_recovery"],
        "pressure_drop_Pa": profile["pressure_drop_Pa"],
    }


def common_cases():
    return [
        {
            "family": f,
            "temperature": t,
            "multiplier": c,
            "pressure": p,
            "recovery": r,
            "length": (0.5, 2.0)[(fi + ti + ci + pi + ri) % 2],
        }
        for fi, f in enumerate(("ideal", "Davies"))
        for ti, t in enumerate((290.0, 310.0, 330.0))
        for ci, c in enumerate((0.5, 1.5))
        for pi, p in enumerate((4e5, 8e5))
        for ri, r in enumerate((False, True))
    ]


def compare_common(case, output):
    model = multisolute_spec()
    original = binary_spec(case["family"], temperature=case["temperature"])
    activity = replace(
        original.properties.database.activity_model,
        charges=tuple(model.properties.charges.items()),
        source_id=model.properties.viscosity.source_id,
        parameters={
            k: replace(v, source_id=model.properties.viscosity.source_id)
            for k, v in original.properties.database.activity_model.parameters.items()
        },
    )
    properties = replace(
        model.properties,
        database=replace(model.properties.database, activity_model=activity),
        viscosity=replace(
            original.properties.viscosity, source_id=model.properties.viscosity.source_id
        ),
    )
    model = replace(
        model,
        properties=properties,
        temperature_K=case["temperature"],
        inlet_pressure_Pa=case["pressure"],
        channel=replace(model.channel, length_m=case["length"]),
        energy_recovery=IsothermalLiquidPressureRecoverySpec(0.7, 1.0, 2e6, 1e-4)
        if case["recovery"]
        else None,
    )
    feed = {
        k: v if k == "H2O" else v * case["multiplier"] for k, v in multisolute_feed(model).items()
    }
    q = 10e-6 / 60
    profile = distributed_ro_profile(model, feed, q)
    checks = []

    def check(name, error, tolerance):
        checks.append(
            {
                "quantity": name,
                "error": float(abs(error)),
                "tolerance": float(tolerance),
                "passed": abs(error) <= tolerance,
            }
        )

    for key in feed:
        check(
            key + "/balance",
            profile["inlet_mol_s"][key]
            - profile["permeate_mol_s"][key]
            - profile["retentate_mol_s"][key],
            1e-10 + 1e-8 * abs(profile["inlet_mol_s"][key]),
        )
    for row in profile["stations"]:
        vw = properties.water_volume_m3_mol
        nf = properties.amounts(
            1.0, {k: c * vw for k, c in row["interface_concentrations_mol_m3"].items()}
        )
        np_ = properties.amounts(
            1.0, {k: c * vw for k, c in row["permeate_concentrations_mol_m3"].items()}
        )
        mf = properties.database.chemical_potentials(
            nf, temperature_K=model.temperature_K, pressure_Pa=row["pressure_Pa"]
        )
        mp = properties.database.chemical_potentials(
            np_, temperature_K=model.temperature_K, pressure_Pa=model.reservoir_pressure_Pa
        )
        delta = mf.chemical_potentials_J_mol["H2O"] - mp.chemical_potentials_J_mol["H2O"]
        a = model.water_permeability.evaluate(
            model.temperature_K, row["pressure_Pa"], sum(row["bulk_concentrations_mol_m3"].values())
        )
        check("water/chemical-potential", row["water_flux_m_s"] - a * delta / vw, 1e-12)
        check("charge", row["charge_flux_mol_m2_s"], 1e-10)
        for c in properties.channels:
            key = c.channel_id
            b = c.permeability.evaluate(
                model.temperature_K,
                row["pressure_Pa"],
                sum(row["bulk_concentrations_mol_m3"].values()),
            )
            check(
                key + "/SD",
                row["channel_flux_mol_m2_s"][key]
                - b
                * (
                    row["interface_concentrations_mol_m3"][key]
                    - row["permeate_concentrations_mol_m3"][key]
                ),
                1e-12,
            )
    result, diagnostic = advance_distributed_ro(
        model, distributed_ro_inventory(model, feed), q, 2.0
    )
    check("finite/enthalpy", result.enthalpy_residual_J, 1e-6)
    for key, error in result.component_residuals_mol.items():
        check("finite/" + key, error, 1e-10 + 1e-8 * feed[key])
    if case["recovery"]:
        recovery = profile["pressure_recovery"]
        expected = recovery["inlet"]["gibbs_energy_J"] - recovery["outlet"]["gibbs_energy_J"]
        check("recovery/G", recovery["pressure_availability_W"] - expected, 1e-6)
        check(
            "recovery/H",
            recovery["device_heat_into_liquid_W"]
            - recovery["enthalpy_rate_change_W"]
            - recovery["recovered_shaft_power_W"],
            1e-6,
        )
        check(
            "recovery/S",
            model.temperature_K * recovery["entropy_production_W_K"]
            - recovery["irreversibility_W"],
            1e-6,
        )
    (output / "native.json").write_text(
        json.dumps({"profile": profile, "finite": diagnostic, "checks": checks}, indent=2),
        encoding="utf8",
    )
    return {
        "passed": all(c["passed"] for c in checks) and profile["pressure_drop_Pa"] > 0,
        "comparisons": len(checks),
        "failed_comparisons": [c for c in checks if not c["passed"]],
        "water_recovery": profile["water_recovery"],
        "pressure_drop_Pa": profile["pressure_drop_Pa"],
    }


def mesh_checks(rows):
    checks = []
    for offset in (48, 51):
        family = rows[offset : offset + 3]
        for quantity in ("water_recovery", "pressure_drop_Pa"):
            coarse = abs(family[1][quantity] - family[0][quantity])
            fine = abs(family[2][quantity] - family[1][quantity])
            checks.append(
                {
                    "mode": family[0]["case"]["mode"],
                    "quantity": quantity,
                    "coarse_change": float(coarse),
                    "fine_change": float(fine),
                    "passed": bool(fine < coarse),
                }
            )
    return checks


def run(output, block):
    output.mkdir(exist_ok=False, parents=True)
    cases = source_cases() if block.startswith("source") else common_cases()
    function = compare_source if block == "source" else compare_common
    if block == "source-explicit":
        function = partial(compare_source, source_domain="forward-pressure-and-nonnegative-solute")
    rows = []
    started = time.monotonic()
    for index, case in enumerate(cases):
        target = output / f"unit-{index + 1:03d}"
        target.mkdir()
        elapsed = time.monotonic() - started
        print(
            json.dumps(
                {
                    "stage": block,
                    "completed": index,
                    "total": len(cases),
                    "elapsed_s": elapsed,
                    "eta_s": elapsed / index * (len(cases) - index) if index else None,
                    "case": case,
                }
            ),
            flush=True,
        )
        row = {"unit": index + 1, "case": case}
        unit_start = time.monotonic()
        with (target / "execution.log").open("w", encoding="utf8") as stream:
            try:
                row.update(function(case, target))
            except Exception as exc:
                row.update(passed=False, error=repr(exc), traceback=traceback.format_exc())
                stream.write(row["traceback"])
        row["elapsed_s"] = time.monotonic() - unit_start
        rows.append(row)
        (target / "summary.json").write_text(json.dumps(row, indent=2), encoding="utf8")
        summary = {
            "block": block,
            "label": "development-qualification",
            "total": len(cases),
            "completed": len(rows),
            "passed": sum(r["passed"] for r in rows),
            "failed": sum(not r["passed"] for r in rows),
            "rows": rows,
            "elapsed_s": time.monotonic() - started,
        }
        (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
        print(
            json.dumps(
                {
                    k: v
                    for k, v in row.items()
                    if k not in ("traceback", "failed_comparisons", "charge")
                }
            ),
            flush=True,
        )
    if block.startswith("source") and all(row["passed"] for row in rows):
        mesh = mesh_checks(rows)
        summary["mesh_checks"] = mesh
        summary["mesh_passed"] = all(row["passed"] for row in mesh)
        (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block", choices=("source", "source-explicit", "common"), required=True)
    args = parser.parse_args()
    run(args.output, args.block)
