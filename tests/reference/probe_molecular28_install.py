"""Actual installed parent28 consumers; standard library plus installed chemworld only.

JSON fixtures are authored separately. This file never imports tests/reference or
thermo/phasepy/teqp, including when replaying real-world snapshot continuations.
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import permutations
from math import fsum, log
from pathlib import Path
from threading import Event, Thread
from time import monotonic
from traceback import format_exc

import chemworld
from chemworld.foundation.state_ledgers import PhaseLedger, PhaseRecord
from chemworld.foundation.state_serialization import _phases
from chemworld.physchem.calphad_equilibrium import calphad_tp_equilibrium
from chemworld.physchem.heat_transfer_units import ThermalDeviceSpec
from chemworld.physchem.molecular_eos import molecular_eos_state
from chemworld.physchem.molecular_multiphase import MolecularMultiphaseSeed, MolecularMultiphaseSpec
from chemworld.physchem.molecular_phase_boundary import (
    MolecularBoundaryResult,
    MolecularBoundarySeed,
    MolecularPhaseBoundarySpec,
    solve_molecular_phase_boundary,
)
from chemworld.physchem.molecular_solid_boundary import (
    MolecularSolidBoundaryResult,
    MolecularSolidBoundarySpec,
    evaluate_molecular_solid_boundary,
)
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.molecular_multiphase_inventory import (
    MolecularMultiphaseInventorySpec,
    equilibrate_molecular_multiphase_inventory,
    molecular_multiphase_inventory_state,
    molecular_multiphase_stream,
)
from chemworld.runtime.molecular_multiphase_thermal import MolecularMultiphaseThermalSpec
from chemworld.runtime.molecular_phase_inventory import MolecularPhaseTarget
from chemworld.world.molecular_multiphase_world import build_molecular_multiphase_world


def encoded(value):
    return json.loads(json.dumps(value, allow_nan=False))


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def phase_check(phases, source, ids, *, amount_tolerance=2e-6):
    require(len(phases) == len(source["amounts"]), "unexpected phase count")
    order = min(
        permutations(range(len(phases))),
        key=lambda order: max(
            abs(phases[j].component_amounts_mol[k] - source["amounts"][i][n])
            for i, j in enumerate(order)
            for n, k in enumerate(ids)
        ),
    )
    amount_error = max(
        abs(phases[j].component_amounts_mol[k] - source["amounts"][i][n])
        for i, j in enumerate(order)
        for n, k in enumerate(ids)
    )
    density_error = max(
        abs(phases[j].state.molar_density_mol_m3 / source["densities"][i] - 1)
        for i, j in enumerate(order)
    )
    require(amount_error <= amount_tolerance, f"phase amount error {amount_error}")
    require(density_error <= 3e-5, f"phase density error {density_error}")
    return {"amount_error_mol": amount_error, "density_relative_error": density_error}


def vlle_unit(data, record):
    core = MolecularMultiphaseSpec.from_dict(data["spec"])
    spec = MolecularMultiphaseInventorySpec(
        core,
        (
            MolecularPhaseTarget("gas", "gas", "gas", True),
            MolecularPhaseTarget("hydrocarbon", "hydrocarbon_liquid", "liquid"),
            MolecularPhaseTarget("water", "water_liquid", "liquid"),
        ),
    )
    stock = PhaseLedger({"feed": PhaseRecord("feed", "reactor", "liquid", 1.0, data["feed"])})
    result = equilibrate_molecular_multiphase_inventory(
        stock,
        spec,
        vessel_id="reactor",
        source_phase_ids=("feed",),
        inputs={"T": 300.0, "P": 3e6},
        seed=MolecularMultiphaseSeed(302.0, 3.03e6),
    )
    record["closure"] = result.to_dict()
    require(result.success, result.failure_reason)
    source = data["source"]
    current = molecular_multiphase_inventory_state(
        result.ledger,
        spec,
        vessel_id="reactor",
        source_phase_ids=result.installed_phase_ids,
        temperature_K=300.0,
        pressure_Pa=3e6,
    )
    record["source_errors"] = phase_check(current.seed.phases, source, core.model.component_ids)
    for key, actual, tolerance in (
        ("H", current.enthalpy_J, 0.03),
        ("S", current.entropy_J_K, 3e-4),
        ("U", current.internal_energy_J, 0.03),
    ):
        require(abs(actual - source[key]) <= tolerance, f"VLLE {key} mismatch")
    require(
        _phases(encoded(result.ledger.to_dict())).to_dict() == result.ledger.to_dict(),
        "VLLE ledger JSON mismatch",
    )
    record["streams"] = []
    for phase_id in result.installed_phase_ids:
        stream = molecular_multiphase_stream(
            result.ledger,
            spec,
            vessel_id="reactor",
            temperature_K=300.0,
            molar_rate_mol_s=0.001,
            phase_id=phase_id,
        )
        live = result.ledger.phases[phase_id]
        total = fsum(live.species_amounts_mol.values())
        require(
            all(
                abs(stream.state.mole_fractions[i] - live.species_amounts_mol[k] / total) < 1e-12
                for i, k in enumerate(core.model.component_ids)
            ),
            "stream did not read live phase",
        )
        record["streams"].append(
            {
                "phase_id": phase_id,
                "state": stream.state.to_dict(),
                "molar_rate_mol_s": stream.molar_rate_mol_s,
                "enthalpy_rate_W": stream.enthalpy_rate_W,
            }
        )
    anchor = source["common_mu_RT_without_logP"]
    require(source["source_common_mu_error_RT"] <= 3e-7, "source common tangent unresolved")
    potentials = [
        [
            log(x) + phi
            for x, phi in zip(
                p.state.mole_fractions, p.state.log_fugacity_coefficients, strict=True
            )
        ]
        for p in current.seed.phases
    ]
    error = max(abs(a - b) for row in potentials for a, b in zip(row, anchor, strict=True))
    require(error <= 2e-6, f"independent common tangent mismatch {error}")
    record["common_tangent_error_RT"] = error
    record["finite_source_tpd"] = []
    for point in source["finite_tpd_points"]:
        state = molecular_eos_state(
            core.model.residual_model, 300.0, point["rho"], point["composition"]
        )
        value = fsum(
            x * (log(x) + phi - mu)
            for x, phi, mu in zip(
                point["composition"], state.log_fugacity_coefficients, anchor, strict=True
            )
        )
        row = point | {"native_tpd_RT": value, "error": abs(value - point["tpd_RT"])}
        record["finite_source_tpd"].append(row)
        require(point["tpd_RT"] >= -1e-7 and row["error"] <= 2e-6, "finite TPD mismatch")
    record["stability_scope"] = "ten fixed source composition/root probes; no global certificate"


def thermal_unit(data, record, boundary):
    core = MolecularMultiphaseSpec.from_dict(data["spec"])
    inventory = MolecularMultiphaseInventorySpec(
        core,
        (
            MolecularPhaseTarget("vapor", "gas", "gas", True),
            MolecularPhaseTarget("liquid", "liquid", "liquid"),
        ),
    )
    cold, hot = data["cold"], data["hot_" + boundary]
    coordinate = "H" if boundary == "P" else "U"
    power = (hot[coordinate] - cold[coordinate]) / 40
    spec = MolecularMultiphaseThermalSpec(
        inventory, boundary, 6e5 if boundary == "P" else cold["V"], 255.0, 390.0
    )
    require(
        MolecularMultiphaseThermalSpec.from_dict(encoded(spec.to_dict())) == spec,
        "thermal specification codec mismatch",
    )
    composition = build_molecular_multiphase_world(
        spec,
        data["feed"],
        energy_J=cold[coordinate],
        seed=MolecularMultiphaseSeed(302.0, 6.06e5),
        formulas={"propane": "C3H8", "butane": "C4H10"},
        device=ThermalDeviceSpec(
            conductance_W_K=10.0,
            maximum_power_W=power,
            maximum_cooling_power_W=power,
            setpoint_K=340.0,
            minimum_setpoint_K=250.0,
            maximum_setpoint_K=400.0,
        ),
        max_volume_L=0.2,
        max_pressure_Pa=2e6,
    )
    session = ChemWorldRuntime(composition=composition)
    clone = None
    try:
        session.reset(seed=28)
        out = session.execute(
            {"operation": "heat", "duration_s": 40.0, "target_temperature_K": 340.0}
        )
        record["action_info"] = out.info
        require(out.info["transaction_status"] == "committed", "installed thermal action rejected")
        state = session._state
        current = molecular_multiphase_inventory_state(
            state.phases,
            inventory,
            vessel_id="reactor",
            source_phase_ids=tuple(state.phases.phases),
            temperature_K=state.temperature_K,
            pressure_Pa=state.pressure_Pa,
        )
        record["hot_state"] = state.to_dict()
        record["source_errors"] = phase_check(
            current.seed.phases, hot, core.model.component_ids, amount_tolerance=2e-8
        )
        require(abs(state.temperature_K - hot["T"]) <= 0.002, "hot temperature mismatch")
        require(abs(state.pressure_Pa / hot["P"] - 1) <= 2e-5, "hot pressure mismatch")
        require(
            abs(
                (current.enthalpy_J if boundary == "P" else current.internal_energy_J)
                - hot[coordinate]
            )
            <= 3e-4,
            "hot energy mismatch",
        )
        require(abs(state.ledger.energy_jacket_J - 40 * power) <= 3e-5, "finite heat mismatch")
        require(state.ledger.time_s == 40.0, "shared clock mismatch")
        snapshot = encoded(session.snapshot().to_dict())
        clone = ChemWorldRuntime.from_snapshot(snapshot)
        for target in (session, clone):
            out = target.execute({"operation": "wait", "duration_s": 1.0})
            require(out.info["transaction_status"] == "committed", "JSON suffix action rejected")
        require(session._state.to_dict() == clone._state.to_dict(), "exact JSON suffix mismatch")
        record["suffix_state"] = session._state.to_dict()
        record["exact_json_suffix"] = True
    finally:
        session.close()
        if clone is not None:
            clone.close()


def bubble_unit(data, record):
    spec = MolecularPhaseBoundarySpec.from_dict(data["spec"])
    result = solve_molecular_phase_boundary(
        spec, MolecularBoundarySeed.from_dict(data["seed"]), temperature_K=data["source"]["T"]
    )
    record["native"] = result.to_dict()
    require(result.success, result.failure_reason)
    require(
        abs(result.liquid.pressure_Pa / data["source"]["P"] - 1) <= 2e-5,
        "installed bubble pressure mismatch",
    )
    require(
        MolecularBoundaryResult.from_dict(encoded(result.to_dict())).to_dict() == result.to_dict(),
        "bubble result codec mismatch",
    )


def solid_unit(data, record):
    spec = MolecularSolidBoundarySpec.from_dict(data["spec"])
    case, source = data["case"], data["source"]
    result = evaluate_molecular_solid_boundary(
        spec, case["T"], case["P"] / (8.3145 * case["T"]), case["amounts"]
    )
    record["boundary"] = result.to_dict()
    require(result.success, result.failure_reason)
    require(
        abs(result.minimum_distance_J_mol_atoms - source["minimum"]) <= 2e-5,
        "installed solid tangent mismatch",
    )
    require(
        MolecularSolidBoundaryResult.from_dict(encoded(result.to_dict())).to_dict()
        == result.to_dict(),
        "solid result codec mismatch",
    )
    solid = calphad_tp_equilibrium(
        spec.solids,
        case["T"],
        case["P"],
        result.element_amounts_mol,
        initial_phases=tuple(c.calphad_seed() for c in result.candidates),
    )
    record["solid_consumer"] = solid.to_dict()
    require(solid.success, solid.failure_reason)
    for element, amount in result.element_amounts_mol.items():
        require(
            abs(sum(p.element_amounts_mol.get(element, 0) for p in solid.phases) - amount) <= 1e-10,
            "original solid consumer lost material",
        )


def run(payload_path, output_path, *, package_root):
    imported = Path(chemworld.__file__).resolve()
    require(imported.is_relative_to(Path(package_root).resolve()), f"wrong package: {imported}")
    payload = json.loads(Path(payload_path).read_text(encoding="utf-8"))
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=False)
    records, started, stop, status = [], monotonic(), Event(), {"unit": "starting"}

    def progress():
        elapsed = monotonic() - started
        rate = len(records) / elapsed if elapsed else 0
        print(
            json.dumps(
                {
                    "stage": "installed-function-probe",
                    "completed": len(records),
                    "total": 5,
                    "unit": status["unit"],
                    "elapsed_s": elapsed,
                    "units_per_min": rate * 60,
                    "eta_s": (5 - len(records)) / rate if rate else None,
                }
            ),
            flush=True,
        )

    def heartbeat():
        while not stop.wait(30):
            progress()

    thread = Thread(target=heartbeat, daemon=True)
    thread.start()
    units = (
        ("I01-vlle-inventory-streams-tpd", vlle_unit, "vlle", ()),
        ("I02-P-world", thermal_unit, "thermal", ("P",)),
        ("I03-V-world", thermal_unit, "thermal", ("V",)),
        ("I04-bubble-codec", bubble_unit, "bubble", ()),
        ("I05-solid-codec-consumer", solid_unit, "solid", ()),
    )
    try:
        with (output / "records.jsonl").open("x", encoding="utf-8") as stream:
            for name, function, key, arguments in units:
                status["unit"] = name
                begin = monotonic()
                row = {"id": name, "success": False}
                progress()
                try:
                    function(payload[key], row, *arguments)
                    row["success"] = True
                except Exception:
                    row["failure"] = format_exc()
                row["elapsed_s"] = monotonic() - begin
                records.append(row)
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                progress()
    finally:
        stop.set()
        thread.join()
    forbidden = [
        name
        for name in sys.modules
        if name.split(".")[0] in {"tests", "reference", "thermo", "phasepy", "teqp"}
    ]
    summary = {
        "mode": "installed_packaging_function_check",
        "package": str(imported),
        "units": 5,
        "attempted": len(records),
        "passed": sum(r["success"] for r in records),
        "failures": [{"id": r["id"], "failure": r["failure"]} for r in records if not r["success"]],
        "forbidden_imports": forbidden,
        "success": all(r["success"] for r in records) and not forbidden,
        "elapsed_s": monotonic() - started,
    }
    (output / "SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--package-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.payload, args.output, package_root=args.package_root), indent=2))
