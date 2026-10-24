"""Prospective fixed 36-unit dilute multi-ion nanofiltration experiment."""

import argparse
import json
import math
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from nanofiltration48_fixtures import amounts, concentrations, spec
from nanofiltration48_reference import EPS0, KB, E, R, factors, finite, point
from scipy.optimize import brentq

from chemworld.physchem.nanofiltration import (
    donnan_partition,
    nanofiltration_point,
    nanofiltration_profile,
    pore_hindrance,
    pore_partition_factors,
)
from chemworld.physchem.residual_system import RootPolicy
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.nanofiltration import unpack_nanofiltration
from chemworld.world.nanofiltration_world import nanofiltration_world_spec


def run(output):
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
            f"units={len(units)}/36 elapsed={elapsed:.2f}s "
            f"throughput={len(units) / max(elapsed, 0.001):.2f}/s "
            f"ETA={elapsed / len(units) * (36 - len(units)):.2f}s {name} passed={row['passed']}",
            flush=True,
        )

    def hindrance():
        compare("point_diffusion", pore_hindrance(0)[0], 1.0)
        compare("point_convection", pore_hindrance(0)[1], 1.0)
        compare("near_exclusion_diffusion", pore_hindrance(0.99)[0], 0.984 * (0.01 / 0.99) ** 2.5)
        return pore_hindrance(0.99)

    unit("analytical:hindrance_limits", hindrance)

    def steric():
        model = spec(2, 0)
        for radius in (0.2e-9, 0.5e-9, 0.97e-9):
            current = replace(
                model,
                pore_relative_permittivity=model.bulk_relative_permittivity,
                ions=tuple(replace(i, stokes_radius_m=radius) for i in model.ions),
            )
            for key, value in pore_partition_factors(current).items():
                compare(
                    key + str(radius) + ":steric",
                    value["steric"],
                    (1 - radius / model.pore_radius_m) ** 2,
                )
                compare(key + str(radius) + ":born", value["born"], 1.0)
        return True

    unit("analytical:steric_size_limits", steric)

    def born():
        model = spec()
        for ion in model.ions:
            z = model.properties.charges[ion.species_id]
            expected = math.exp(
                -z
                * z
                * E
                * E
                / (8 * math.pi * EPS0 * ion.born_radius_m * KB * model.temperature_K)
                * (1 / model.pore_relative_permittivity - 1 / model.bulk_relative_permittivity)
            )
            compare(
                ion.species_id + ":born",
                pore_partition_factors(model)[ion.species_id]["born"],
                expected,
            )
        return pore_partition_factors(model)

    unit("analytical:born_valence", born)

    def donnan(charge):
        model = spec(2, charge)
        actual = donnan_partition(model, concentrations(model))
        k = factors(model)[0][0]
        expected_phi = math.asinh(charge / (2 * k * 10))
        compare("potential", actual["dimensionless_potential"], expected_phi)
        for key in model.properties.ion_ids:
            compare(
                key,
                actual["pore_concentrations_mol_m3"][key],
                k * 10 * math.exp(-model.properties.charges[key] * expected_phi),
            )
        return actual

    for charge in (0, -20, 20):
        unit(f"analytical:donnan:{charge}", lambda c=charge: donnan(c))

    def water(mode):
        model = spec(2, 0)
        if mode == "radius":
            model = replace(model, pore_radius_m=2 * model.pore_radius_m)
        if mode == "thickness":
            model = replace(model, effective_thickness_m=2 * model.effective_thickness_m)
        if mode == "pressure":
            model = replace(
                model,
                inlet_pressure_Pa=model.reservoir_pressure_Pa
                + 2 * (model.inlet_pressure_Pa - model.reservoir_pressure_Pa),
            )
        actual = nanofiltration_point(model, dict.fromkeys(model.properties.ion_ids, 0.0))
        expected = (
            model.pore_radius_m**2
            * (model.inlet_pressure_Pa - model.reservoir_pressure_Pa)
            / (8 * 0.001 * model.effective_thickness_m)
        )
        compare("water_flux", actual["water_flux_m_s"], expected)
        return actual

    for mode in ("base", "radius", "thickness", "pressure"):
        unit("water:" + mode, lambda m=mode: water(m))

    def symmetric(dielectric):
        # Low-Peclet limit, where the midpoint spatial error is below the
        # analytical tolerance; all twelve operating units use the fixed MPa grid.
        model = spec(2, 0, 1.01e5)
        if not dielectric:
            model = replace(model, pore_relative_permittivity=model.bulk_relative_permittivity)
        k, diffusion, convection = [r[0] for r in factors(model)]
        hydraulic = model.pore_radius_m**2 / (8 * 0.001 * model.effective_thickness_m)
        dp = model.inlet_pressure_Pa - model.reservoir_pressure_Pa

        def cp(j):
            return (
                10
                * k
                * convection
                / (
                    1
                    + (k * convection - 1)
                    * math.exp(-convection * j * model.effective_thickness_m / diffusion)
                )
            )

        j = brentq(
            lambda x: x - hydraulic * (dp - 2 * R * model.temperature_K * (10 - cp(x))),
            0,
            hydraulic * dp,
            xtol=1e-18,
        )
        actual = nanofiltration_point(model, concentrations(model))
        compare("water_flux", actual["water_flux_m_s"], j)
        for key in model.properties.ion_ids:
            compare(key, actual["permeate_concentrations_mol_m3"][key], cp(j))
        return actual

    for dielectric in (False, True):
        unit(f"symmetric:{dielectric}", lambda d=dielectric: symmetric(d))

    def ions(count, charge, pressure):
        model = spec(count, charge, pressure)
        ns = amounts(model)
        profile = nanofiltration_profile(model, ns, 10e-6 / 60)
        actual = profile["pore"]
        qwater = profile["inlet_mol_s"]["H2O"] * 18e-6
        expected = point(model, concentrations(model), qwater)
        compare("status", actual["status"], "converged")
        compare(
            "water_flux",
            actual["water_flux_m_s"],
            expected["water_flux_m_s"],
            relative=2e-3,
            absolute=2e-6,
        )
        for key in model.properties.ion_ids:
            compare(
                "product:" + key,
                actual["permeate_concentrations_mol_m3"][key],
                expected["permeate_concentrations_mol_m3"][key],
                relative=2e-3,
                absolute=2e-6,
            )
            compare(
                "flux:" + key,
                actual["ion_fluxes_mol_m2_s"][key],
                expected["water_flux_m_s"] * expected["permeate_concentrations_mol_m3"][key],
                relative=2e-3,
                absolute=2e-6,
            )
        for i, row in enumerate(actual["pore_concentrations_mol_m3"]):
            for k, key in enumerate(model.properties.ion_ids):
                compare(
                    f"pore:{i}:{key}",
                    row[key],
                    expected["pore_concentrations_mol_m3"][i][k],
                    relative=2e-3,
                    absolute=2e-6,
                )
            compare(
                f"potential:{i}",
                actual["pore_dimensionless_potential"][i],
                expected["pore_dimensionless_potential"][i],
                relative=2e-3,
                absolute=2e-6,
            )
        charge_scale = max(
            sum(abs(model.properties.charges[k]) * c for k, c in concentrations(model).items()),
            abs(charge),
            1,
        )
        compare(
            "normalized_charge_within_threshold",
            actual["charge_residual_mol_m3"] / charge_scale <= 2e-7,
            True,
        )
        current_scale = sum(
            abs(model.properties.charges[k] * f) for k, f in actual["ion_fluxes_mol_m2_s"].items()
        )
        compare(
            "normalized_current_within_threshold",
            abs(actual["current_molar_charge_flux_mol_m2_s"]) / current_scale <= 2e-7,
            True,
        )
        return {"native": profile, "reference": expected}

    for count in (3, 4):
        for charge in (-20, 20):
            for pressure in (3e5, 5e5, 8e5):
                unit(
                    f"ions:{count}:{charge}:{pressure}",
                    lambda n=count, c=charge, p=pressure: ions(n, c, p),
                )

    def refinement(count, charge):
        model = spec(count, charge)
        expected = point(model, concentrations(model))
        reference = np.array(
            [expected["permeate_concentrations_mol_m3"][k] for k in model.properties.ion_ids]
        )
        errors = []
        for segments in (4, 8, 16):
            actual = nanofiltration_point(
                replace(model, pore_segments=segments), concentrations(model)
            )
            error = max(
                abs(
                    np.array(
                        [
                            actual["permeate_concentrations_mol_m3"][k]
                            for k in model.properties.ion_ids
                        ]
                    )
                    / reference
                    - 1
                )
            )
            if errors:
                compare(
                    "refinement:" + str(segments), bool(error < errors[-1] or error < 2e-5), True
                )
            errors.append(float(error))
        return {"segments": [4, 8, 16], "maximum_relative_errors": errors}

    for count in (3, 4):
        for charge in (-20, 20):
            unit(f"refinement:{count}:{charge}", lambda n=count, c=charge: refinement(n, c))

    def world(mode):
        model = spec(4)
        if mode == "capacity":
            model = replace(model, retentate_capacity_L=1e-4)
        kwargs = (
            {"permeate": amounts(model, 0.05, 0.5), "retentate": amounts(model, 0.05, 1.5)}
            if mode == "preloaded"
            else {}
        )
        runtime = ChemWorldRuntime(
            composition=nanofiltration_world_spec(model, amounts(model), **kwargs)
        )
        restored = None
        try:
            runtime.reset(seed=48)
            initial = unpack_nanofiltration(runtime._state)[2]
            expected = finite(model, initial.amounts_mol, 10e-6 / 60, 5)
            configured = runtime.execute({"operation": "set_flow_rate", "flow_rate_mL_min": 10})
            compare("configured", configured.info["transaction_status"], "committed")
            if mode == "snapshot":
                restored = ChemWorldRuntime.from_snapshot(
                    json.loads(json.dumps(runtime.snapshot().to_dict()))
                )
            result = runtime.execute({"operation": "wait", "duration_s": 5})
            compare("committed", result.info["transaction_status"], "committed")
            actual = unpack_nanofiltration(runtime._state)[2]
            for i, row in enumerate(actual.amounts_mol):
                for key, n in row.items():
                    compare(
                        f"tank:{i}:{key}",
                        n,
                        expected["amounts_mol"][i][key],
                        relative=0,
                        absolute=2e-8,
                    )
                compare(
                    f"H:{i}",
                    actual.enthalpies_J[i],
                    expected["enthalpies_J"][i],
                    relative=0,
                    absolute=2e-5,
                )
                compare(
                    f"charge:{i}",
                    sum(model.properties.charges[k] * row[k] for k in model.properties.ion_ids),
                    0,
                    absolute=2e-10,
                )
            compare(
                "actual_time",
                runtime._state.ledger.time_s,
                expected["actual_duration_s"],
                relative=2e-3,
                absolute=1e-7,
            )
            compare(
                "shaft_work",
                runtime._state.ledger.flow_pump_work_J,
                expected["shaft_work_J"],
                relative=0,
                absolute=2e-5,
            )
            receipt = next(
                r
                for r in runtime._state.equipment.equipment.values()
                if r.equipment_type == "declared_nanofiltration"
            ).settings["last_nanofiltration"]
            for key in ("hydraulic_work_J", "motor_loss_J", "bath_heat_into_liquid_J"):
                compare(key, receipt[key], expected[key], relative=0, absolute=2e-5)
            for name in ("reactor", "permeate", "retentate"):
                tank = (
                    runtime._state if name == "reactor" else runtime._state.inactive_vessels[name]
                )
                compare("temperature:" + name, tank.temperature_K, 310, relative=0, absolute=1e-6)
            if restored:
                restored.execute({"operation": "wait", "duration_s": 5})
                compare(
                    "exact_snapshot_replay",
                    runtime._state.to_dict() == restored._state.to_dict(),
                    True,
                )
            snapshot = runtime.snapshot().to_dict()
            (output / (mode + "-snapshot.json")).write_text(json.dumps(snapshot, indent=2))
            return {"reference": expected, "receipt": receipt, "info": result.info}
        finally:
            runtime.close()
            if restored:
                restored.close()

    for mode in ("empty", "preloaded", "capacity", "snapshot"):
        unit("world:" + mode, lambda m=mode: world(m))

    def failure(mode):
        model = spec()
        if mode == "iteration_limit":
            actual = nanofiltration_point(
                model, concentrations(model), policy=RootPolicy(maximum_iterations=1)
            )
            compare("status", actual["status"], "not_converged")
            compare("failure_reason", actual["solver"]["failure_reason"], "iteration_limit")
            return actual
        try:
            if mode == "absent_solvent":
                model.properties.validate_amounts({**amounts(model), "H2O": 0.0})
            elif mode == "excluded_radius":
                replace(model, pore_radius_m=1e-10)
            else:
                nanofiltration_point(model, {**concentrations(model), "Na+": 11.0})
        except ValueError as exc:
            compare("explicit_failure", True, True)
            return {"error": str(exc)}
        compare("explicit_failure", False, True)
        return {}

    for mode in ("absent_solvent", "excluded_radius", "unbalanced_ions", "iteration_limit"):
        unit("failure:" + mode, lambda m=mode: failure(m))

    summary = {
        "schema": "chemworld-nanofiltration48-development-v1",
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
    return summary["units_passed"] == 36


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    raise SystemExit(0 if run(parser.parse_args().output) else 1)
