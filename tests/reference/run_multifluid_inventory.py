"""Fixed parent27 source EOS -> codec -> finite inventory development block."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import threading
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import teqp
from run_multifluid_core import R, source_input
from scipy.optimize import brentq

from chemworld.data.multifluid import load_multifluid_model
from chemworld.physchem.coupled_system import (
    CoupledProcess,
    CoupledState,
    CoupledSystem,
    HeatFlux,
    MaterialPort,
)
from chemworld.physchem.helmholtz_terms import HelmholtzTerm
from chemworld.physchem.molecular_mixture import (
    MolecularCaloricModel,
    MolecularIsochoricThermodynamics,
    MolecularStream,
)
from chemworld.physchem.solver_backend import ODESolverPolicy

NAMES = (
    "Methane",
    "Ethane",
    "n-Propane",
    "n-Butane",
    "IsoButane",
    "Nitrogen",
    "CarbonDioxide",
    "Hydrogen",
)


def reference_state(ref, temperature, rho, xs):
    """External residual derivatives and explicitly analytical ideal reference."""
    xs = np.array(xs)
    ar10 = ref.get_Ar10(temperature, rho, xs)
    ar01 = ref.get_Ar01(temperature, rho, xs)
    ideal_u = sum(x * R * ((1.5 + i) * temperature + (3 + i) * 300) for i, x in enumerate(xs))
    u = ideal_u + R * temperature * ar10
    pressure = rho * R * temperature * (1 + ar01)
    return float(u), float(u + pressure / rho), float(pressure)


def execute(count, form, mode):
    names = NAMES[:count]
    native = load_multifluid_model(source_input(names), source="teqp 0.23.2 explicit source")
    ref = teqp.build_multifluid_model(list(names), teqp.get_datapath(), flags={"Rmodel": "CODATA"})
    ideal = tuple(
        replace(
            f,
            reducing_temperature_K=300.0,
            reducing_molar_density_mol_m3=1000.0,
            ideal_terms=(
                HelmholtzTerm("IdealGasHelmholtzLead", {"a1": 2.0 + i, "a2": 3.0 + i}),
                HelmholtzTerm("IdealGasHelmholtzLogTau", {"a": 1.5 + i}),
            ),
            source="explicit analytical ideal reference; not measured material heat capacity",
        )
        for i, f in enumerate(native.caloric_model().ideal_references)
    )
    thermal = MolecularIsochoricThermodynamics(MolecularCaloricModel(native, ideal), 4.0, 5.0)
    rebuilt = MolecularIsochoricThermodynamics.from_dict(json.loads(json.dumps(thermal.to_dict())))
    codec = rebuilt == thermal
    ids = native.component_ids
    amounts = np.array([2.0, 0.0] if count == 2 else [2 / count] * count)
    start = CoupledState(dict(zip(ids, amounts, strict=True)), 450.0, 4.0)
    feed_x = np.arange(count, 0, -1, dtype=float)
    feed_x /= feed_x.sum()
    options = {
        "numerical_form": form,
        "policy": ODESolverPolicy("multifluid-inventory", method="BDF", rtol=1e-10, atol=1e-12),
    }
    if mode == "heat":
        options["heat_flux"] = lambda _: HeatFlux(5.0)
        expected_amounts = amounts.copy()
        added_energy = 1.25
    else:
        stream = MolecularStream(rebuilt.model.state(500.0, 400.0, feed_x), 0.01)
        ports = {key: f"feed-{i}" for i, key in enumerate(ids)}
        contribution = stream.contribution(ports)
        options["processes"] = (CoupledProcess("feed", lambda _: contribution),)
        options["ports"] = tuple(
            MaterialPort(port, key, "source", "vessel", "in") for key, port in ports.items()
        )
        expected_amounts = amounts + 0.0025 * feed_x
        added_energy = 0.0025 * reference_state(ref, 500.0, 400.0, feed_x)[1]

    def reference_energy(t, ns):
        n = float(sum(ns))
        return n * reference_state(ref, t, n / 0.004, ns / n)[0] + 5 * (t - 298.15)

    expected_energy = reference_energy(450.0, amounts) + added_energy
    expected_t = brentq(
        lambda t: reference_energy(t, expected_amounts) - expected_energy, 440.0, 470.0, xtol=1e-11
    )
    expected_p = reference_state(
        ref, expected_t, sum(expected_amounts) / 0.004, expected_amounts / sum(expected_amounts)
    )[2]
    result = CoupledSystem(rebuilt, **options).advance(start, 0.25)
    final = result.final_state
    actual_amounts = np.array([final.amounts_mol[k] for k in ids])
    final_reference_energy = reference_energy(final.temperature_K, actual_amounts)
    expected = {
        **{f"n:{k}": float(v) for k, v in zip(ids, expected_amounts, strict=True)},
        "temperature_K": expected_t,
        "pressure_Pa": expected_p,
        "independent_energy_residual_J": 0.0,
        "ledger_energy_residual_J": 0.0,
        "internal_energy_J": final_reference_energy,
        "time_s": 0.25,
    }
    actual = {
        **{f"n:{k}": float(v) for k, v in final.amounts_mol.items()},
        "temperature_K": final.temperature_K,
        "pressure_Pa": rebuilt.pressure_pa(final),
        "independent_energy_residual_J": final_reference_energy - expected_energy,
        "ledger_energy_residual_J": result.energy_balance_residual_J,
        "internal_energy_J": rebuilt.internal_energy_j(final),
        "time_s": final.time_s,
    }
    rows = []
    for key, value in expected.items():
        atol = (
            2e-10
            if key.startswith("n:") or key == "time_s"
            else 2e-7
            if key == "temperature_K"
            else 2e-4
            if key == "pressure_Pa"
            else 2e-5
        )
        rows.append(
            {
                "quantity": key,
                "native": actual[key],
                "reference": value,
                "atol": atol,
                "rtol": 2e-7,
                "passed": math.isclose(actual[key], value, rel_tol=2e-7, abs_tol=atol),
            }
        )
    return {
        "components": names,
        "codec_passed": codec,
        "comparisons": rows,
        "model": rebuilt.to_dict(),
    }


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    progress = {"completed": 0, "total": 16, "active": "startup"}
    stopped = threading.Event()

    def heartbeat():
        while not stopped.wait(20):
            elapsed = time.monotonic() - started
            rate = progress["completed"] / elapsed
            print(
                json.dumps(
                    {
                        **progress,
                        "elapsed_s": elapsed,
                        "units_per_minute": rate * 60,
                        "eta_s": (16 - progress["completed"]) / rate if rate else None,
                    }
                ),
                flush=True,
            )

    threading.Thread(target=heartbeat, daemon=True).start()
    units, failures = [], []
    try:
        for count, form, mode in itertools.product(
            (2, 3, 4, 8), ("ode", "residual"), ("heat", "feed")
        ):
            key = f"n{count}-{form}-{mode}"
            progress["active"] = key
            try:
                unit = {"unit": key, **execute(count, form, mode)}
                unit["passed"] = unit["codec_passed"] and all(
                    r["passed"] for r in unit["comparisons"]
                )
            except Exception as exc:
                unit = {
                    "unit": key,
                    "passed": False,
                    "codec_passed": False,
                    "comparisons": [],
                    "unexecuted_comparisons": count + 6,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            (output / f"{key}.json").write_text(
                json.dumps(unit, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
            compact = {k: v for k, v in unit.items() if k != "model"}
            units.append(compact)
            if not unit["passed"]:
                failures.append(compact)
            progress["completed"] += 1
            print(
                json.dumps(
                    {**progress, "passed": unit["passed"], "elapsed_s": time.monotonic() - started}
                ),
                flush=True,
            )
    finally:
        stopped.set()
    rows = [r for unit in units for r in unit["comparisons"]]
    summary = {
        "mode": "development",
        "reference": "teqp 0.23.2 plus analytical ideal U/H",
        "units_planned": 16,
        "units_executed": len(units),
        "units_passed": sum(u["passed"] for u in units),
        "comparisons_planned": 164,
        "comparisons_executed": len(rows),
        "comparisons_passed": sum(r["passed"] for r in rows),
        "codec_planned": 16,
        "codec_passed": sum(u["codec_passed"] for u in units),
        "elapsed_s": time.monotonic() - started,
        "failures": failures,
        "units": units,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps({k: v for k, v in summary.items() if k not in {"units", "failures"}}, indent=2)
    )
    return bool(failures)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    raise SystemExit(run(parser.parse_args().output))
