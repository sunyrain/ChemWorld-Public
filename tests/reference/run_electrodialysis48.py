"""Fixed 40-unit periodic interior electrodialysis development block."""

import argparse
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

from chemworld.physchem.electrodialysis import (
    advance_electrodialysis,
    electrodialysis_inventory,
    electrodialysis_point,
)
from chemworld.physchem.ion_exchange_pore import ion_exchange_partition, ion_exchange_pore_point
from chemworld.physchem.reactor_shared import ScalarControlProgram
from chemworld.physchem.residual_system import RootPolicy
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.electrodialysis import unpack_electrodialysis
from chemworld.world.electrodialysis_world import electrodialysis_world_spec


def run(output):
    from reference.electrodialysis48_fixtures import amounts, inventories, spec
    from reference.electrodialysis48_reference import F, R, finite, point, volume

    output.mkdir(parents=True, exist_ok=False)
    units, records, outputs = [], [], {}
    started = time.monotonic()

    def compare(name, actual, expected, *, relative=2e-7, absolute=1e-10):
        passed = (
            actual == expected
            if isinstance(expected, (bool, str)) or expected is None
            else math.isfinite(actual)
            and math.isclose(actual, expected, rel_tol=relative, abs_tol=absolute)
        )
        records.append(
            {
                "unit": units[-1]["name"],
                "quantity": name,
                "actual": actual,
                "expected": expected,
                "relative_tolerance": relative,
                "absolute_tolerance": absolute,
                "passed": bool(passed),
            }
        )

    def unit(name, function):
        before = len(records)
        row = {"name": name, "error": None}
        units.append(row)
        try:
            outputs[name] = function()
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        selected = records[before:]
        row.update(
            comparisons_total=len(selected),
            comparisons_passed=sum(r["passed"] for r in selected),
            failures=[r for r in selected if not r["passed"]],
        )
        row["passed"] = row["error"] is None and bool(selected) and not row["failures"]
        elapsed = time.monotonic() - started
        print(
            f"units={len(units)}/40 elapsed={elapsed:.2f}s "
            f"throughput={len(units) / max(elapsed, 0.001):.2f}/s "
            f"ETA={elapsed / len(units) * (40 - len(units)):.2f}s {name} passed={row['passed']}",
            flush=True,
        )

    def analytical(mode):
        model = spec(2)
        n = amounts(model)
        c = model.properties.concentrations(n)
        if mode == "equilibrium":
            actual = electrodialysis_point(model, (n, n), 0, "A")
            for key, value in actual["diluate_rates_mol_s"].items():
                compare(key, value, 0)
            compare("voltage", actual["voltage_V"], 0)
            return actual
        if mode == "donnan":
            actual = ion_exchange_partition(model.cem, model.properties.charges, c)
            phi = math.asinh(model.cem.fixed_charge_mol_m3 / (2 * c["Na+"]))
            compare("potential", actual["dimensionless_potential"], phi)
            for k in c:
                compare(
                    k,
                    actual["pore_concentrations_mol_m3"][k],
                    c[k] * math.exp(-model.properties.charges[k] * phi),
                )
            return actual
        if mode == "uncharged_conductance":
            pore = replace(model.cem, fixed_charge_mol_m3=0)
            actual = ion_exchange_pore_point(
                pore,
                model.properties,
                c,
                c,
                temperature_K=310,
                pressure_Pa=1e5,
                current_density_A_m2=10,
            )
            conductivity = (
                F * F / (R * 310) * sum(d * c[k] for k, d in pore.effective_diffusivities_m2_s)
            )
            compare("voltage", actual["voltage_drop_V"], 10 * pore.thickness_m / conductivity)
            compare("water", actual["water_flux_m_s"], 0)
            return actual
        if mode == "symmetric_pair_conductance":
            actual = electrodialysis_point(model, (n, n), 0.001, "A")
            resistance = model.solution_resistance_Ohm_m2
            for pore in (model.aem, model.cem):
                x = pore.fixed_charge_mol_m3
                ci = {
                    "Na+": (math.sqrt(x * x + 4 * c["Na+"] ** 2) - x) / 2,
                    "Cl-": (math.sqrt(x * x + 4 * c["Na+"] ** 2) + x) / 2,
                }
                conductivity = (
                    F * F / (R * 310) * sum(d * ci[k] for k, d in pore.effective_diffusivities_m2_s)
                )
                resistance += 1 / (
                    conductivity / pore.thickness_m
                    + pore.hydraulic_permeability_m_Pa_s * F * F * x * x
                )
            compare("voltage", actual["voltage_V"], 0.001 / model.membrane_area_m2 * resistance)
            return actual
        if mode in {"open_circuit_diffusion", "osmosis"}:
            ns = inventories(model)
            actual = electrodialysis_point(model, ns, 0, "A")
            expected = point(model, ns, 0, "A")
            compare("current", actual["current_A"], 0)
            compare(
                "voltage", actual["voltage_V"], expected["voltage_V"], relative=3e-3, absolute=2e-6
            )
            compare("water_to_more_concentrated", actual["diluate_rates_mol_s"]["H2O"] > 0, True)
            compare("salt_diffuses_down_gradient", actual["diluate_rates_mol_s"]["Na+"] < 0, True)
            compare("dissipation_nonnegative", actual["dissipation_W"] >= -2e-7, True)
            if mode == "osmosis":
                reverse = electrodialysis_point(model, tuple(reversed(ns)), 0, "A")
                compare(
                    "reversed_water",
                    reverse["diluate_rates_mol_s"]["H2O"],
                    -actual["diluate_rates_mol_s"]["H2O"],
                )
            return actual
        if mode == "electroosmotic_sign":
            actual = electrodialysis_point(model, (n, n), 0.001, "A")
            compare("AEM_water_against_current", actual["aem"]["water_flux_m_s"] < 0, True)
            compare("CEM_water_along_current", actual["cem"]["water_flux_m_s"] > 0, True)
            compare(
                "equal_current",
                actual["aem"]["current_density_A_m2"],
                actual["cem"]["current_density_A_m2"],
            )
            return actual
        pore = model.cem
        driven = ion_exchange_pore_point(
            pore, model.properties, c, c, temperature_K=310, pressure_Pa=1e5, current_density_A_m2=1
        )
        pressure = ion_exchange_pore_point(
            pore,
            model.properties,
            c,
            c,
            temperature_K=310,
            pressure_Pa=100100,
            pressure_right_Pa=100000,
            current_density_A_m2=0,
        )
        compare(
            "reciprocal_water_current_vs_voltage_pressure",
            driven["water_flux_m_s"],
            -pressure["voltage_drop_V"] / 100,
            relative=2e-4,
            absolute=1e-14,
        )
        compare("driven_dissipation", driven["dissipation_W_m2"] >= -2e-7, True)
        compare("pressure_dissipation", pressure["dissipation_W_m2"] >= -2e-7, True)
        return {"electrical": driven, "pressure": pressure}

    for mode in (
        "equilibrium",
        "donnan",
        "uncharged_conductance",
        "symmetric_pair_conductance",
        "open_circuit_diffusion",
        "osmosis",
        "electroosmotic_sign",
        "reciprocity_dissipation",
    ):
        unit("analytical:" + mode, lambda m=mode: analytical(m))

    def operating(count, direction, current):
        model = spec(count)
        ns = inventories(model, direction)
        actual = electrodialysis_point(model, ns, current, "A")
        expected = point(model, ns, current, "A")
        compare("status", actual["status"], "converged")
        for key in ("current_A", "voltage_V", "electrical_power_W"):
            compare(key, actual[key], expected[key], relative=3e-3, absolute=2e-6)
        for membrane in ("aem", "cem"):
            a, b = actual[membrane], expected[membrane]
            for key in ("water_flux_m_s", "voltage_drop_V"):
                compare(membrane + ":" + key, a[key], b[key], relative=3e-3, absolute=2e-6)
            for key in model.properties.ion_ids:
                compare(
                    membrane + ":flux:" + key,
                    a["ion_fluxes_mol_m2_s"][key],
                    b["ion_fluxes_mol_m2_s"][key],
                    relative=3e-3,
                    absolute=2e-6,
                )
            for i, row in enumerate(a["pore_concentrations_mol_m3"]):
                for k, key in enumerate(model.properties.ion_ids):
                    compare(
                        f"{membrane}:pore:{i}:{key}",
                        row[key],
                        b["pore_concentrations_mol_m3"][i][k],
                        relative=3e-3,
                        absolute=2e-6,
                    )
                compare(
                    f"{membrane}:potential:{i}",
                    a["pore_dimensionless_potential"][i],
                    b["pore_dimensionless_potential"][i],
                    relative=3e-3,
                    absolute=2e-6,
                )
            compare(membrane + ":charge", a["charge_residual_mol_m3"] / 100 <= 2e-7, True)
        for field in ("diluate_rates_mol_s", "concentrate_rates_mol_s"):
            charge = sum(
                model.properties.charges[k] * actual[field][k] for k in model.properties.ion_ids
            )
            scale = sum(
                abs(model.properties.charges[k] * actual[field][k])
                for k in model.properties.ion_ids
            )
            compare(field + ":charge", abs(charge) / max(scale, 1e-20) <= 2e-7, True)
        compare("dissipation_nonnegative", actual["dissipation_W"] >= -2e-7, True)
        return {"native": actual, "reference": expected}

    for count in (3, 4):
        for direction in (-1, 1):
            for current in (0, 0.001, 0.003):
                unit(
                    f"operating:{count}:{direction}:{current}",
                    lambda n=count, d=direction, i=current: operating(n, d, i),
                )

    def refinement(count, direction):
        model = spec(count)
        ns = inventories(model, direction)
        expected = point(model, ns, 0.001, "A")
        errors = []
        for segments in (4, 8, 16):
            refined = replace(
                model,
                aem=replace(model.aem, segments=segments),
                cem=replace(model.cem, segments=segments),
            )
            actual = electrodialysis_point(refined, ns, 0.001, "A")
            candidates = [abs(actual["voltage_V"] / expected["voltage_V"] - 1)]
            for membrane in ("aem", "cem"):
                for key in model.properties.ion_ids:
                    candidates.append(
                        abs(
                            actual[membrane]["ion_fluxes_mol_m2_s"][key]
                            / expected[membrane]["ion_fluxes_mol_m2_s"][key]
                            - 1
                        )
                    )
            error = max(candidates)
            if errors:
                compare(f"grid:{segments}", error < errors[-1] or error < 2e-5, True)
            errors.append(error)
        return {"segments": [4, 8, 16], "maximum_relative_errors": errors}

    for count in (3, 4):
        for direction in (-1, 1):
            unit(f"refinement:{count}:{direction}", lambda n=count, d=direction: refinement(n, d))

    def check_finite(model, initial, actual, receipt, expected, elapsed):
        for i, row in enumerate(actual.amounts_mol):
            for key, n in row.items():
                compare(
                    f"tank:{i}:{key}", n, expected["amounts_mol"][i][key], relative=0, absolute=2e-8
                )
            compare(
                f"H:{i}",
                actual.enthalpies_J[i],
                expected["enthalpies_J"][i],
                relative=0,
                absolute=2e-5,
            )
            charge = sum(model.properties.charges[k] * row[k] for k in model.properties.ion_ids)
            scale = sum(abs(model.properties.charges[k] * row[k]) for k in model.properties.ion_ids)
            compare(f"charge:{i}", abs(charge) / scale <= 2e-7, True)
        for key in ("electrical_work_J", "bath_heat_into_liquid_J"):
            compare(key, receipt[key], expected[key], relative=0, absolute=2e-5)
        compare("time", elapsed, expected["actual_duration_s"], relative=3e-3, absolute=1e-6)
        compare(
            "total_H",
            sum(actual.enthalpies_J),
            sum(initial.enthalpies_J),
            relative=0,
            absolute=2e-5,
        )

    def trajectory(count, unit_name, direction):
        model = spec(count)
        ns = inventories(model, direction)
        initial = electrodialysis_inventory(model, *ns)
        target = 0.001 if unit_name == "A" else 0.2
        result, receipt = advance_electrodialysis(
            model, initial, ScalarControlProgram(((0, target),), unit_name), 30
        )
        expected = finite(model, ns, target, unit_name, 30)
        check_finite(
            model, initial, result.final_state, receipt, expected, result.actual_duration_s
        )
        return {
            "native_final": {
                "amounts_mol": result.final_state.amounts_mol,
                "enthalpies_J": result.final_state.enthalpies_J,
            },
            "reference": expected,
            "receipt": receipt,
            "solver": result.solver_diagnostic,
        }

    for count in (3, 4):
        for unit_name in ("A", "V"):
            for direction in (-1, 1):
                unit(
                    f"trajectory:{count}:{unit_name}:{direction}",
                    lambda n=count, u=unit_name, d=direction: trajectory(n, u, d),
                )

    def world_case(mode, stop_mode=None):
        model = spec(4)
        ns = inventories(model)
        if stop_mode == "capacity":
            model = replace(model, capacities_L=(0.04, volume(ns[1]) * 1000 + 1e-8))
        if stop_mode == "water":
            model = replace(model, minimum_water_mol=0.9999998)
        runtime = ChemWorldRuntime(composition=electrodialysis_world_spec(model, *ns))
        restored = None
        try:
            runtime.reset(seed=48)
            initial = unpack_electrodialysis(runtime._state)[2]
            if mode == "ramp":
                actions = (
                    {"operation": "configure_electrical_control", "potential_V": 0.15},
                    {
                        "operation": "configure_electrical_control",
                        "potential_V": 0.3,
                        "ramp_duration_s": 2,
                    },
                )
                expected = finite(model, ns, 0.3, "V", 5, ramp=(0.15, 2))
            else:
                actions = ({"operation": "configure_electrical_control", "current_A": 0.001},)
                expected = finite(model, ns, 0.001, "A", 5)
            for action in actions:
                configured = runtime.execute(action)
                compare("configured", configured.info["transaction_status"], "committed")
            if mode == "snapshot":
                restored = ChemWorldRuntime.from_snapshot(
                    json.loads(json.dumps(runtime.snapshot().to_dict()))
                )
            executed = runtime.execute({"operation": "wait", "duration_s": 5})
            compare("committed", executed.info["transaction_status"], "committed")
            record, _, final = unpack_electrodialysis(runtime._state)
            receipt = record.settings["last_electrodialysis"]
            check_finite(model, initial, final, receipt, expected, runtime._state.ledger.time_s)
            compare(
                "electrical_ledger",
                runtime._state.ledger.flow_electrical_work_J,
                receipt["electrical_work_J"],
                relative=0,
                absolute=1e-10,
            )
            compare("no_duplicate_heater", runtime._state.ledger.energy_jacket_J, 0)
            compare(
                "thermal_ledger",
                runtime._state.ledger.heat_loss_J,
                -receipt["bath_heat_into_liquid_J"],
                relative=0,
                absolute=1e-10,
            )
            if stop_mode:
                compare("actual_stop", runtime._state.ledger.time_s < 5, True)
            for name in ("reactor", "concentrate"):
                tank = (
                    runtime._state if name == "reactor" else runtime._state.inactive_vessels[name]
                )
                compare("temperature:" + name, tank.temperature_K, 310, relative=0, absolute=1e-6)
                compare(
                    "local_bath:" + name,
                    tank.thermal.vessels[name].heat_loss_J,
                    -receipt["bath_heat_into_liquid_J"] / 2,
                    relative=0,
                    absolute=1e-10,
                )
            if restored:
                restored.execute({"operation": "wait", "duration_s": 5})
                compare("exact_replay", runtime._state.to_dict() == restored._state.to_dict(), True)
            (
                output / (mode + ("-" + stop_mode if stop_mode else "") + "-snapshot.json")
            ).write_text(json.dumps(runtime.snapshot().to_dict(), indent=2))
            return {
                "reference": expected,
                "receipt": receipt,
                "actual_time_s": runtime._state.ledger.time_s,
            }
        finally:
            runtime.close()
            if restored:
                restored.close()

    for mode in ("step", "ramp", "stops", "snapshot"):
        unit(
            "world:" + mode,
            (lambda m=mode: world_case(m))
            if mode != "stops"
            else lambda: {m: world_case("stops", m) for m in ("capacity", "water")},
        )

    def failure(mode):
        model = spec()
        ns = inventories(model)
        if mode == "compliance":
            model = replace(model, maximum_voltage_V=0.15)
            result, receipt = advance_electrodialysis(
                model,
                electrodialysis_inventory(model, *ns),
                ScalarControlProgram(((0, 0.003),), "A"),
                2,
            )
            compare("stop", result.stopping_event, "voltage_compliance")
            compare("time", result.actual_duration_s, 0)
            compare("work", receipt["electrical_work_J"], 0)
            compare("not_clipped_current", receipt["final"]["current_A"], 0.003)
            return receipt
        if mode == "iteration_limit":
            actual = electrodialysis_point(
                model, ns, 0.003, "A", policy=RootPolicy(maximum_iterations=1)
            )
            compare("status", actual["status"], "not_converged")
            compare("failure_reason", actual["solver"]["failure_reason"], "iteration_limit")
            return actual
        try:
            if mode == "unbalanced":
                electrodialysis_point(model, ({**ns[0], "Na+": 0.001}, ns[1]), 0.001, "A")
            else:
                replace(model, aem=replace(model.aem, fixed_charge_mol_m3=-100))
        except ValueError as exc:
            compare("explicit_failure", True, True)
            return {"error": str(exc)}
        compare("explicit_failure", False, True)
        return {}

    for mode in ("unbalanced", "orientation", "compliance", "iteration_limit"):
        unit("failure:" + mode, lambda m=mode: failure(m))

    summary = {
        "schema": "chemworld-electrodialysis48-development-v1",
        "units_total": len(units),
        "units_passed": sum(r["passed"] for r in units),
        "comparisons_total": len(records),
        "comparisons_passed": sum(r["passed"] for r in records),
        "units": units,
        "failures": [r for r in records if not r["passed"]],
        "elapsed_s": time.monotonic() - started,
    }
    for name, data in (("summary", summary), ("records", records), ("outputs", outputs)):
        (output / (name + ".json")).write_text(json.dumps(data, indent=2))
    print(
        json.dumps({k: v for k, v in summary.items() if k not in {"units", "failures"}}), flush=True
    )
    return summary["units_passed"] == 40


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    raise SystemExit(0 if run(parser.parse_args().output) else 1)
