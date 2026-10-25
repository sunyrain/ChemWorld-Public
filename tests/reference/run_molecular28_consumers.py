"""Fixed six-unit parent28 original-inventory and thermal development block."""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import replace
from math import fsum
from pathlib import Path

import numpy as np
from molecular28_general_cases import native_spec, regular_spec
from molecular28_general_reference import (
    regular_state,
    thermo_fixed_volume,
    thermo_oracle,
    thermo_state,
)

from chemworld.foundation.state_ledgers import PhaseLedger, PhaseRecord
from chemworld.foundation.state_serialization import _phases
from chemworld.physchem.heat_transfer_units import ThermalDeviceSpec
from chemworld.physchem.molecular_multiphase import MolecularMultiphaseSeed
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

FEED = {"propane": 0.006, "butane": 0.004}
PLANNED = {"C01": 14, "C02": 36, "C03": 36, "C04": 40, "C05": 6, "C06": 4}


def encode(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"unsupported result type: {type(value).__name__}")


def dumps(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, default=encode)


def inventory_spec(core=None, count=3):
    core = core or native_spec("vle")
    targets = (
        tuple(MolecularPhaseTarget(f"liquid-{i}", "fluid", "liquid", i == 0) for i in range(count))
        if core.model.component_ids[0] == "a"
        else (
            MolecularPhaseTarget("vapor", "gas", "gas", True),
            MolecularPhaseTarget("liquid", "liquid", "liquid"),
        )
    )
    return MolecularMultiphaseInventorySpec(core, targets)


def read(ledger, spec, phase_ids, temperature, pressure):
    stock = molecular_multiphase_inventory_state(
        ledger,
        spec,
        vessel_id="reactor",
        source_phase_ids=phase_ids,
        temperature_K=temperature,
        pressure_Pa=pressure,
    )
    return {
        "T": temperature,
        "P": pressure,
        "H": stock.enthalpy_J,
        "U": stock.internal_energy_J,
        "S": stock.entropy_J_K,
        "V": stock.volume_m3,
        "amounts": [
            [p.component_amounts_mol[k] for k in spec.equilibrium.model.component_ids]
            for p in stock.seed.phases
        ],
        "densities": [p.state.molar_density_mol_m3 for p in stock.seed.phases],
    }, stock


class Checks:
    def __init__(self):
        self.rows = []

    def test(self, name, passed, **details):
        self.rows.append({"name": name, "passed": bool(passed), **details})

    def close(self, name, actual, expected, tolerance, relative=False):
        error = abs(actual - expected) / abs(expected) if relative else abs(actual - expected)
        self.test(
            name,
            error <= tolerance,
            actual=actual,
            expected=expected,
            error=error,
            tolerance=tolerance,
            relative=relative,
        )

    def state(self, label, actual, source, feed, regular=False):
        for key, tol, relative in (
            ("T", 0.002, False),
            ("P", 2e-5, True),
            ("H", 0.03 if regular else 3e-4, False),
            ("U", 0.03 if regular else 3e-4, False),
            ("S", 3e-4 if regular else 3e-6, False),
            ("V", 3e-5, True),
        ):
            self.close(f"{label}:{key}", actual[key], source[key], tol, relative)
        amounts = np.asarray(actual["amounts"])
        expected = np.asarray(source["amounts"])
        material = float(np.max(np.abs(amounts.sum(axis=0) - np.array(tuple(feed.values())))))
        self.close(f"{label}:component_balance", material, 0.0, 2e-10 if regular else 2e-12)
        count_ok = len(amounts) == len(expected)
        self.test(f"{label}:phase_count", count_ok, actual=len(amounts), expected=len(expected))
        if count_ok:
            permutation = min(
                itertools.permutations(range(len(amounts))),
                key=lambda order: float(np.max(np.abs(amounts[list(order)] - expected))),
            )
            error = float(np.max(np.abs(amounts[list(permutation)] - expected)))
            density_error = max(
                abs(actual["densities"][i] / source["densities"][j] - 1)
                for j, i in enumerate(permutation)
            )
        else:
            error = density_error = None
        self.test(
            f"{label}:phase_allocation",
            error is not None and error <= (2e-6 if regular else 2e-8),
            error=error,
            vector_shape=list(expected.shape),
            tolerance=2e-6 if regular else 2e-8,
        )
        self.test(
            f"{label}:phase_density",
            density_error is not None and density_error <= 3e-5,
            error=density_error,
            tolerance=3e-5,
            vector_length=len(expected),
        )


def sources():
    cold = thermo_state((0.6, 0.4), total=0.01, T=300.0, P=6e5)
    return (
        cold,
        thermo_state((0.6, 0.4), total=0.01, T=320.0, P=6e5),
        thermo_fixed_volume((0.6, 0.4), 320.0, cold["V"], total=0.01),
    )


def world(boundary):
    cold, hot_p, hot_v = sources()
    hot = hot_p if boundary == "P" else hot_v
    coordinate = "H" if boundary == "P" else "U"
    power = (hot[coordinate] - cold[coordinate]) / 40.0
    spec = MolecularMultiphaseThermalSpec(
        inventory_spec(), boundary, 6e5 if boundary == "P" else cold["V"], 255.0, 390.0
    )
    composition = build_molecular_multiphase_world(
        spec,
        FEED,
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
    return spec, composition, cold, hot, power


def execute(session, action):
    result = session.execute(action)
    if result.info["transaction_status"] != "committed":
        raise RuntimeError(dumps({"action": action, "info": result.info}))
    return result


def c01(check, record):
    spec = inventory_spec()
    original = PhaseLedger({"feed": PhaseRecord("feed", "reactor", "liquid", 0.01, FEED)})
    result = equilibrate_molecular_multiphase_inventory(
        original,
        spec,
        vessel_id="reactor",
        source_phase_ids=("feed",),
        inputs={"T": 300.0, "P": 6e5},
    )
    check.test("closure_success", result.success, failure_reason=result.failure_reason)
    record["closure"] = result.to_dict()
    if not result.success:
        raise RuntimeError(result.failure_reason)
    actual, _ = read(result.ledger, spec, result.installed_phase_ids, 300.0, 6e5)
    source = thermo_state((0.6, 0.4), total=0.01, T=300.0, P=6e5)
    record.update(actual=actual, source=source)
    check.state("inventory", actual, source, FEED)
    stream = molecular_multiphase_stream(
        result.ledger, spec, vessel_id="reactor", temperature_K=300.0, molar_rate_mol_s=0.001
    )
    gas = thermo_oracle().flash(zs=[0.6, 0.4], T=300.0, P=6e5).gas
    check.close(
        "stream_composition",
        max(abs(a - b) for a, b in zip(stream.state.mole_fractions, gas.zs, strict=True)),
        0.0,
        2e-6,
    )
    check.close("stream_enthalpy", stream.state.enthalpy_J_mol, gas.H(), 0.03)
    check.test(
        "ledger_json",
        _phases(json.loads(dumps(result.ledger.to_dict()))).to_dict() == result.ledger.to_dict(),
    )


def thermal(check, record, boundary):
    spec, composition, cold, hot, power = world(boundary)
    record.update(specification=spec.to_dict(), source_cold=cold, source_hot=hot, power_W=power)
    session = ChemWorldRuntime(composition=composition)
    session.reset(seed=28)
    try:
        state = session._state
        initial_volume = state.volume_L * 1e-3
        actual, _ = read(
            state.phases,
            spec.inventory,
            tuple(state.phases.phases),
            state.temperature_K,
            state.pressure_Pa,
        )
        check.state("initial", actual, cold, FEED)
        history = [{"state": state.to_dict(), "properties": actual}]
        record["history"] = history
        for index, (target, source) in enumerate(((340.0, hot), (280.0, cold)), start=1):
            action = {"operation": "heat", "duration_s": 40.0, "target_temperature_K": target}
            outcome = execute(session, action)
            state = session._state
            actual, _ = read(
                state.phases,
                spec.inventory,
                tuple(state.phases.phases),
                state.temperature_K,
                state.pressure_Pa,
            )
            history.append(
                {
                    "state": state.to_dict(),
                    "properties": actual,
                    "action": action,
                    "info": outcome.info,
                }
            )
            check.state(f"stage{index}", actual, source, FEED)
            check.close(
                f"stage{index}:jacket",
                state.ledger.energy_jacket_J,
                40.0 * power if index == 1 else 0.0,
                3e-5,
            )
            work = 6e5 * (actual["V"] - initial_volume) if boundary == "P" else 0.0
            check.close(f"stage{index}:work", state.ledger.boundary_work_J, work, 3e-5)
            check.close(f"stage{index}:clock", state.ledger.time_s, 40.0 * index, 0.0)
    finally:
        session.close()


def regular_stock():
    return PhaseLedger(
        {
            "feed": PhaseRecord("feed", "reactor", "liquid", 1.0, {"a": 1.0, "b": 1.0, "c": 1.0}),
            "other": PhaseRecord("other", "other-vessel", "liquid", 3.0, {"a": 2.0}),
            "solid": PhaseRecord("solid", "reactor", "solid", 0.01, {"inert": 1.0}),
        }
    )


def c04(check, record):
    omega = 11224.524534506875
    spec = inventory_spec(regular_spec(omega=omega))
    original = current = regular_stock()
    phase_ids, seed = ("feed",), None
    record["history"] = []
    for index, temperature in enumerate((300.0, 650.0, 300.0)):
        result = equilibrate_molecular_multiphase_inventory(
            current,
            spec,
            vessel_id="reactor",
            source_phase_ids=phase_ids,
            inputs={"T": temperature, "P": 1e5},
            seed=seed,
        )
        record["history"].append({"closure": result.to_dict()})
        if not result.success:
            raise RuntimeError(result.failure_reason)
        current, phase_ids = result.ledger, result.installed_phase_ids
        actual, stock = read(current, spec, phase_ids, temperature, 1e5)
        seed = stock.seed
        source = regular_state((1 / 3, 1 / 3, 1 / 3), temperature, omega=omega, total=3.0)
        record["history"][-1].update(actual=actual, source=source)
        check.state(f"state{index}", actual, source, {"a": 1.0, "b": 1.0, "c": 1.0}, regular=True)
        for name in ("other", "solid"):
            check.test(
                f"state{index}:untouched:{name}", current.phases[name] == original.phases[name]
            )
        check.test(
            f"state{index}:no_ghosts",
            set(current.phases) == set(phase_ids) | {"other", "solid"}
            and all(fsum(current.phases[p].species_amounts_mol.values()) > 0 for p in phase_ids),
        )
    check.test(
        "ledger_json", _phases(json.loads(dumps(current.to_dict()))).to_dict() == current.to_dict()
    )


def c05(check, record):
    spec = inventory_spec(regular_spec(omega=11224.524534506875))
    original = regular_stock()
    before = dumps(original.to_dict())
    record["attempts"] = []
    variants = (
        replace(spec, phase_targets=spec.phase_targets[:2]),
        replace(spec, equilibrium=replace(spec.equilibrium, max_phases=2)),
        replace(spec, equilibrium=replace(spec.equilibrium, maximum_phase_updates=1)),
    )
    for index, bad in enumerate(variants):
        result = equilibrate_molecular_multiphase_inventory(
            original,
            bad,
            vessel_id="reactor",
            source_phase_ids=("feed",),
            inputs={"T": 300.0, "P": 1e5},
        )
        record["attempts"].append(result.to_dict())
        check.test(
            f"case{index}:rejected",
            not result.success and bool(result.failure_reason),
            failure_reason=result.failure_reason,
        )
        check.test(f"case{index}:unchanged", dumps(result.ledger.to_dict()) == before)


def c06(check, record):
    child_code = (
        "import json,sys; from chemworld.runtime import ChemWorldRuntime; "
        "r=ChemWorldRuntime.from_snapshot(json.load(sys.stdin)); "
        "out=r.execute({'operation':'wait','duration_s':1}); "
        "assert out.info['transaction_status']=='committed',out.info; "
        "print(json.dumps(r._state.to_dict(),allow_nan=False)); r.close()"
    )
    record["processes"] = []
    for boundary in ("P", "V"):
        spec, composition, *_ = world(boundary)
        encoded = json.loads(dumps(spec.to_dict()))
        check.test(f"{boundary}:codec", MolecularMultiphaseThermalSpec.from_dict(encoded) == spec)
        session = ChemWorldRuntime(composition=composition)
        session.reset(seed=28)
        try:
            snapshot = session.snapshot().to_dict()
            execute(session, {"operation": "wait", "duration_s": 1})
            child = subprocess.run(
                [sys.executable, "-c", child_code],
                input=dumps(snapshot),
                text=True,
                encoding="utf-8",
                capture_output=True,
                timeout=300,
                cwd=Path(__file__).resolve().parents[2],
                check=False,
            )
            row = {
                "boundary": boundary,
                "exit_code": child.returncode,
                "stderr": child.stderr,
                "snapshot": snapshot,
                "expected": session._state.to_dict(),
            }
            record["processes"].append(row)
            if child.returncode:
                row["stdout"] = child.stdout
                raise RuntimeError(f"fresh-process continuation failed: {child.stderr}")
            row["actual"] = json.loads(child.stdout)
            check.test(f"{boundary}:exact_suffix", row["actual"] == row["expected"])
        finally:
            session.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    start, completed, current = time.monotonic(), [], [""]
    stop = threading.Event()

    def progress():
        elapsed = time.monotonic() - start
        rate = len(completed) / elapsed if elapsed else 0.0
        print(
            dumps(
                {
                    "stage": "original_inventory_consumers",
                    "completed": len(completed),
                    "total": 6,
                    "current": current[0],
                    "elapsed_s": elapsed,
                    "units_per_minute": rate * 60,
                    "eta_s": (6 - len(completed)) / rate if rate else None,
                }
            ),
            flush=True,
        )

    def heartbeat():
        while not stop.wait(30):
            progress()

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    units = {
        "C01": c01,
        "C02": lambda c, r: thermal(c, r, "P"),
        "C03": lambda c, r: thermal(c, r, "V"),
        "C04": c04,
        "C05": c05,
        "C06": c06,
    }
    try:
        for key, function in units.items():
            current[0] = key
            progress()
            check, record, unit_start = Checks(), {"unit": key}, time.monotonic()
            try:
                function(check, record)
            except Exception:
                record["exception"] = traceback.format_exc()
            record.update(
                checks=check.rows,
                planned_checks=PLANNED[key],
                elapsed_s=time.monotonic() - unit_start,
            )
            record["passed"] = (
                "exception" not in record
                and len(check.rows) == PLANNED[key]
                and all(c["passed"] for c in check.rows)
            )
            with (args.output / "records.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(dumps(record) + "\n")
            completed.append(
                {
                    "unit": key,
                    "passed": record["passed"],
                    "planned_checks": PLANNED[key],
                    "executed_checks": len(check.rows),
                    "passed_checks": sum(c["passed"] for c in check.rows),
                    "failures": [c for c in check.rows if not c["passed"]],
                    "exception": record.get("exception"),
                    "elapsed_s": record["elapsed_s"],
                }
            )
            progress()
    finally:
        stop.set()
        thread.join()
    summary = {
        "mode": "development",
        "task": "NATIVE-CONV-28/C",
        "planned_units": 6,
        "executed_units": len(completed),
        "passed_units": sum(r["passed"] for r in completed),
        "planned_checks": 136,
        "executed_checks": sum(r["executed_checks"] for r in completed),
        "passed_checks": sum(r["passed_checks"] for r in completed),
        "elapsed_s": time.monotonic() - start,
        "raw_directory": str(args.output.resolve()),
        "units": completed,
        "passed": all(r["passed"] for r in completed),
    }
    (args.output / "SUMMARY.json").write_text(dumps(summary) + "\n", encoding="utf-8")
    print(dumps(summary), flush=True)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
