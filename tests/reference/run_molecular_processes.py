"""Fixed analytic process block plus actual native PC-SAFT receiver closure."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from math import fsum, log
from pathlib import Path
from threading import Event, Thread
from time import monotonic

from chemworld.foundation import WorldState
from chemworld.foundation.state_ledgers import PhaseLedger, PhaseRecord, VesselLedger, VesselRecord
from chemworld.physchem.helmholtz import HelmholtzFluid
from chemworld.physchem.helmholtz_terms import HelmholtzTerm
from chemworld.physchem.molecular_flash import MolecularDensityBasin
from chemworld.physchem.molecular_mixture import MolecularCaloricModel, MolecularStream
from chemworld.physchem.molecular_processes import (
    MolecularHeatDuty,
    MolecularPressureStage,
    MolecularProcessDomain,
    molecular_heat_duty,
    molecular_mix_streams,
    molecular_process_state,
    molecular_split_stream,
)
from chemworld.physchem.pcsaft import PCSAFTModel, SAFTSegment
from chemworld.runtime.molecular_process_path import (
    MolecularProcessPath,
    advance_molecular_process_receiver,
)

R = 8.31446261815324


@dataclass(frozen=True)
class IdealResidual:
    component_ids = ("A", "B")
    gas_constant_J_mol_K = R

    def alphar(self, temperature_K, molar_density_mol_m3, mole_fractions):
        return 0 * temperature_K + 0 * molar_density_mol_m3 + 0 * sum(mole_fractions)


def domain(real=False):
    residual = (
        PCSAFTModel(
            (
                SAFTSegment("A", 1.0, 3.4e-10, 120.0, "declared test"),
                SAFTSegment("B", 2.0, 3.8e-10, 180.0, "declared test"),
            ),
            ((0.0, 0.02), (0.02, 0.0)),
            (100.0, 1000.0),
            "declared test",
        )
        if real
        else IdealResidual()
    )
    ideal = tuple(
        HelmholtzFluid(
            k,
            "",
            0.04 + 0.02 * i,
            R,
            300.0,
            1000.0,
            100.0,
            1000.0,
            1e9,
            (
                HelmholtzTerm("IdealGasHelmholtzLead", {"a1": 2.0 + i, "a2": -3.5 - i}),
                HelmholtzTerm("IdealGasHelmholtzLogTau", {"a": 2.5 + i}),
            ),
            (),
            "explicit constant-Cv analytic reference",
        )
        for i, k in enumerate(("A", "B"))
    )
    return MolecularProcessDomain(
        MolecularCaloricModel(residual, ideal),
        (250.0, 900.0),
        MolecularDensityBasin("gas", (1.0, 1200.0), 100.0),
    )


def stream(d, temperature=400.0, pressure=2e5, fractions=(0.25, 0.75), rate=0.05):
    return MolecularStream(
        d.model.state_tp(
            temperature,
            pressure,
            fractions,
            initial_density_mol_m3=pressure / (R * temperature),
            density_bounds_mol_m3=d.density_basin.density_bounds_mol_m3,
        ),
        rate,
    )


def ideal(temperature, density, fractions):
    cv = R * fsum(x * (2.5 + i) for i, x in enumerate(fractions))
    h0 = R * 300 * fsum(x * (-3.5 - i) for i, x in enumerate(fractions))
    entropy = R * fsum(
        x * (-log(density / 1000) + (2.5 + i) * log(temperature / 300) + 0.5)
        for i, x in enumerate(fractions)
    )
    entropy -= R * fsum(x * log(x) for x in fractions if x > 0)
    return {
        "P": R * temperature * density,
        "H": h0 + (cv + R) * temperature,
        "U": h0 + cv * temperature,
        "S": entropy,
        "Cv": cv,
        "Cp": cv + R,
    }


def path_and_world(real=False, *, pressure_limit=1e6):
    d = domain(real)
    inlet = stream(d, rate=0.02)
    path = MolecularProcessPath(
        d,
        "external-feed",
        inlet,
        (
            MolecularPressureStage(d, "compressor", 4e5, 0.8),
            MolecularHeatDuty(d, 10.0, boundary_temperature_K=800.0),
            MolecularPressureStage(d, "throttle", 3e5),
        ),
    )
    n, volume, temperature, x = 1.0, 20.0, 300.0, (0.25, 0.75)
    s = d.model.state(temperature, n / (volume * 1e-3), x)
    amounts = dict(zip(d.model.component_ids, x, strict=True))
    phase = PhaseRecord(
        "bulk",
        "receiver",
        "gas",
        volume,
        amounts,
        metadata={"carrier_representation": "species_inventory"},
    )
    vessel = VesselRecord(
        "receiver", "tank", volume, 800.0, pressure_limit, ("bulk",), temperature, s.pressure_Pa
    )
    world = WorldState(
        amounts,
        volume,
        temperature,
        s.pressure_Pa,
        "gas",
        "receiver",
        phases=PhaseLedger({"bulk": phase}),
        vessels=VesselLedger({"receiver": vessel}),
    )
    return path, world


def run_unit(name, unit):
    d, rows, checks = domain(), unit["runs"], unit["comparisons"]

    def compare(label, actual, expected, *, relative=1e-8, absolute=1e-7):
        checks.append(
            {
                "name": label,
                "actual": actual,
                "expected": expected,
                "pass": abs(actual - expected) <= absolute + relative * abs(expected),
            }
        )

    def success(label, result):
        rows.append(result.to_dict())
        checks.append({"name": label + "_success", "pass": result.success})
        if not result.success:
            raise ValueError(f"{label}: {result.failure_reason}")

    if name == "local_closure":
        for temperature, pressure in ((350.0, 1e5), (450.0, 3e5), (600.0, 5e5)):
            density, x = pressure / (R * temperature), (0.25, 0.75)
            expected = ideal(temperature, density, x)
            for coordinate in ("T", "H", "S"):
                r = molecular_process_state(
                    d,
                    pressure_Pa=pressure,
                    coordinate=coordinate,
                    value=temperature if coordinate == "T" else expected[coordinate],
                    mole_fractions=x,
                    initial_temperature_K=400.0,
                )
                success(f"{temperature}_{coordinate}", r)
                compare("T", r.state.temperature_K, temperature)
                compare("rho", r.state.molar_density_mol_m3, density)
                compare("P", r.state.pressure_Pa, pressure)
    elif name in {"compression", "expansion", "throttle"}:
        mode = {"compression": "compressor", "expansion": "expander", "throttle": "throttle"}[name]
        t, p2 = (
            (500.0, 1e5) if mode == "expander" else (400.0, 4e5 if mode == "compressor" else 1e5)
        )
        feed = stream(d, temperature=t)
        cp = ideal(t, feed.state.molar_density_mol_m3, feed.state.mole_fractions)["Cp"]
        for eta in (1.0,) if mode == "throttle" else (1.0, 0.8):
            r = MolecularPressureStage(d, mode, p2, eta).evaluate(feed)
            success(str(eta), r)
            iso = t * (p2 / 2e5) ** (R / cp)
            target = (
                t
                if mode == "throttle"
                else t + (iso - t) / (eta if mode == "compressor" else 1 / eta)
            )
            compare("T", r.outlet.state.temperature_K, target)
            compare("work", r.shaft_work_W, 0.05 * cp * (target - t))
            compare("energy", r.diagnostics["energy_residual_W"], 0, absolute=1e-6)
    elif name == "heat":
        feed = stream(d)
        cp = ideal(400, feed.state.molar_density_mol_m3, feed.state.mole_fractions)["Cp"]
        for heat, bath in ((20.0, 800.0), (-20.0, 250.0)):
            r = molecular_heat_duty(d, feed, heat_duty_W=heat, boundary_temperature_K=bath)
            success(str(heat), r)
            compare("T", r.outlet.state.temperature_K, 400 + heat / (0.05 * cp))
            compare("energy", r.diagnostics["energy_residual_W"], 0.0, absolute=1e-6)
    elif name == "mix_split":
        feeds = (
            stream(d, 350, fractions=(0.8, 0.2), rate=0.02),
            stream(d, 500, fractions=(0.2, 0.8), rate=0.03),
        )
        r = molecular_mix_streams(d, feeds, outlet_pressure_Pa=2e5, initial_temperature_K=400.0)
        success("mix", r)
        x = (0.44, 0.56)
        coefficients = ideal(300, 100, x)
        h = (
            fsum(
                s.molar_rate_mol_s
                * ideal(
                    s.state.temperature_K, s.state.molar_density_mol_m3, s.state.mole_fractions
                )["H"]
                for s in feeds
            )
            / 0.05
        )
        t = 300 + h / coefficients["Cp"]
        compare("mixed_T", r.outlet.state.temperature_K, t)
        for actual, expected in zip(r.outlet.state.mole_fractions, x, strict=True):
            compare("composition", actual, expected, absolute=1e-12)
        split = molecular_split_stream(d, r.outlet, (0.25, 0.75, 0.0))
        success("split", split)
        compare(
            "split_amount", fsum(s.molar_rate_mol_s for s in split.outlets), 0.05, absolute=1e-12
        )
        compare("split_H", fsum(s.enthalpy_rate_W for s in split.outlets), 0.05 * h, absolute=1e-6)
    else:
        real = name == "pcsaft_receiver"
        path, world = path_and_world(real)
        evaluated = path.evaluate()
        success("path", evaluated)
        for form in ("ode", "residual"):
            r = advance_molecular_process_receiver(
                world, path, 10.0, phase_id="bulk", maximum_amount_mol=1.1, numerical_form=form
            )
            rows.append(
                {
                    "form": form,
                    "success": r.success,
                    "diagnostics": r.diagnostics,
                    "failure_reason": r.failure_reason,
                    "state": None if r.state is None else r.state.to_dict(include_hidden=True),
                }
            )
            checks.append({"name": form + "_success", "pass": r.success})
            if not r.success:
                raise ValueError(str(r.diagnostics))
            final = r.state
            compare("stop_time", r.actual_duration_s, 5.0, relative=0.0, absolute=1e-7)
            for k, x in zip(("A", "B"), (0.25, 0.75), strict=True):
                compare(
                    "amount_" + k, final.species_amounts[k], 1.1 * x, relative=0.0, absolute=1e-8
                )
            for key in ("receiver_energy_residual_J", "path_energy_residual_J"):
                compare(key, r.diagnostics[key], 0.0, relative=0.0, absolute=2e-5)
            compare("heat_receipt", final.ledger.flow_heat_J, 50.0, relative=0.0, absolute=2e-5)
            compare(
                "work_receipt",
                final.ledger.flow_shaft_work_J,
                5 * evaluated.shaft_work_W,
                relative=0.0,
                absolute=2e-5,
            )
            compare(
                "no_receiver_heat_duplicate",
                final.thermal.vessels["receiver"].energy_jacket_J,
                0.0,
                relative=0.0,
                absolute=1e-12,
            )
            if not real:
                v = ideal(300, 50, (0.25, 0.75))
                end_u = v["U"] + r.diagnostics["received_enthalpy_J"]
                h0 = -v["Cp"] * 300
                temperature = (end_u / 1.1 - h0) / v["Cv"]
                compare("finite_T", final.temperature_K, temperature, relative=0.0, absolute=2e-7)
    return unit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    names = (
        "local_closure",
        "compression",
        "expansion",
        "throttle",
        "heat",
        "mix_split",
        "ideal_receiver",
        "pcsaft_receiver",
    )
    units, started = [], monotonic()
    finished = Event()

    def heartbeat():
        while not finished.wait(30):
            elapsed = monotonic() - started
            print(
                f"Process block alive; completed={len(units)}/8; elapsed={elapsed:.1f}s; "
                f"rate={len(units) / max(elapsed, 1e-6):.3f}/s",
                flush=True,
            )

    Thread(target=heartbeat, daemon=True).start()
    for name in names:
        unit = {"name": name, "runs": [], "comparisons": []}
        try:
            run_unit(name, unit)
        except Exception as exc:
            unit["exception"] = f"{type(exc).__name__}: {exc}"
            unit["comparisons"].append({"name": "unit_execution", "pass": False})
        units.append(unit)
        checks = [c for u in units for c in u["comparisons"]]
        elapsed = monotonic() - started
        report = {
            "scope": "development_eight_fixed_units",
            "expected_units": 8,
            "expected_comparisons_when_fully_executed": 109,
            "completed_units": len(units),
            "comparisons": len(checks),
            "passed": sum(c["pass"] for c in checks),
            "units": units,
            "numerical_elapsed_s": elapsed,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf8"
        )
        print(
            f"Process units {len(units)}/8; comparisons={report['passed']}/{len(checks)}; "
            f"elapsed={elapsed:.1f}s; rate={len(units) / max(elapsed, 1e-6):.3f}/s; "
            f"ETA={(8 - len(units)) * elapsed / len(units):.1f}s",
            flush=True,
        )
    finished.set()
    return int(any(not c["pass"] for c in checks))


if __name__ == "__main__":
    raise SystemExit(main())
