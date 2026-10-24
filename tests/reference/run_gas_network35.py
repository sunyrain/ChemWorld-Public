"""Independent Cantera network comparison; retain all attempts, including failures."""

import argparse
import json
import time
from pathlib import Path

import cantera as ct
import numpy as np
import yaml
from gas_network35_cases import AMOUNTS, document, network_spec
from scipy.integrate import simpson
from scipy.optimize import brentq

from chemworld.physchem.elements import molecular_weight
from chemworld.physchem.gas_reactor_network import (
    advance_gas_reactor_network,
    pack_gas_reactor_network_state,
)
from chemworld.physchem.mechanism_import import convert_mechanism


def source_network(spec):
    payload = document()
    payload["elements"] = [{"symbol": "H", "atomic-weight": molecular_weight({"H": 1})}]
    reactors = {}
    for node in spec.nodes:
        phase = ct.Solution(yaml=yaml.safe_dump(payload), name="synthetic")
        n = AMOUNTS[node.vessel_id]
        pressure = sum(n.values()) * ct.gas_constant / 1000 * 700 / (node.volume_L / 1000)
        phase.TPX = 700, pressure, n
        reactors[node.vessel_id] = ct.IdealGasReactor(
            phase, energy="off", volume=node.volume_L / 1000, clone=True
        )
    edge = spec.connections[0]
    src, dst = reactors["source"], reactors["receiver"]
    coefficient = edge.coefficient if edge.enabled else 0
    if edge.mode == "pressure-valve":
        flow = ct.Valve(src, dst, K=coefficient)
    elif edge.mode == "mass-flow":
        flow = ct.MassFlowController(src, dst, mdot=coefficient)
    else:
        flow = ct.MassFlowController(
            src, dst, mdot=lambda _: coefficient * src.phase.mean_molecular_weight / 1000
        )
    reactor_net = ct.ReactorNet(list(reactors.values()))
    reactor_net.rtol, reactor_net.atol = 1e-12, 1e-20
    reactor_net.initialize()
    return reactors, flow, reactor_net


def source_state(reactors):
    return {
        key: {
            "amounts_mol": dict(
                zip(r.phase.species_names, r.phase.concentrations * r.volume * 1000, strict=True)
            ),
            "pressure_Pa": r.phase.P,
            "internal_energy_J": r.mass * r.phase.int_energy_mass,
        }
        for key, r in reactors.items()
    }


def source_stop(spec, case):
    node = next(
        n
        for n in spec.nodes
        if n.vessel_id == ("source" if case == "source-cutoff" else "receiver")
    )

    def residual(t):
        reactors, _, solver = source_network(spec)
        solver.advance(t)
        state = source_state(reactors)[node.vessel_id]
        return (
            sum(state["amounts_mol"].values()) - node.minimum_amount_mol
            if case == "source-cutoff"
            else state["pressure_Pa"] - node.maximum_pressure_Pa
        )

    return brentq(residual, 0, 9, xtol=1e-11)


def reference(spec, duration):
    reactors, flow, solver = source_network(spec)
    initial = source_state(reactors)
    times = np.linspace(0, duration, 1001)
    rates, enthalpies = [], []
    for t in times:
        solver.advance(float(t))
        source = reactors["source"].phase
        molar_rate = flow.mass_flow_rate / (source.mean_molecular_weight / 1000)
        rates.append(molar_rate * source.X)
        enthalpies.append(flow.mass_flow_rate * source.enthalpy_mass)
    transfer = simpson(np.asarray(rates), x=times, axis=0)
    enthalpy = float(simpson(np.asarray(enthalpies), x=times))
    final = source_state(reactors)
    for key in final:
        delta_u = final[key]["internal_energy_J"] - initial[key]["internal_energy_J"]
        final[key]["bath_heat_J"] = delta_u + (enthalpy if key == "source" else -enthalpy)
    return {
        "nodes": final,
        "transferred_mol": dict(zip(("A", "B"), map(float, transfer), strict=True)),
        "enthalpy_J": enthalpy,
        "actual_duration_s": duration,
    }


def run(output):
    rows, failures = [], []
    cases = [(case, t) for case in ("molar", "mass", "valve", "closed") for t in (0.1, 0.5, 1)]
    cases += [(case, 10) for case in ("source-cutoff", "pressure-limit")]
    start = time.monotonic()
    for index, (case, horizon) in enumerate(cases, 1):
        checks = []
        record = {"case": case, "requested_duration_s": horizon, "checks": checks}
        try:
            spec, network = network_spec(case), convert_mechanism(document())
            duration = source_stop(spec, case) if horizon == 10 else horizon
            source = reference(spec, duration)
            result, native = advance_gas_reactor_network(
                spec, network, pack_gas_reactor_network_state(spec, network, AMOUNTS), horizon
            )
            record.update(native=native, reference=source)

            def check(label, actual, expected, atol, rtol=2e-6, checks=checks):
                actual, expected = float(actual), float(expected)
                tolerance = atol + rtol * abs(expected)
                checks.append(
                    {
                        "quantity": label,
                        "native": actual,
                        "reference": expected,
                        "absolute_error": abs(actual - expected),
                        "tolerance": tolerance,
                        "passed": abs(actual - expected) <= tolerance,
                    }
                )

            check("time_s", result.actual_duration_s, duration, 1e-7)
            for node, detail in native["nodes"].items():
                target = source["nodes"][node]
                for species, amount in detail["amounts_mol"].items():
                    check(f"{node}:{species}:mol", amount, target["amounts_mol"][species], 2e-10)
                check(f"{node}:pressure_Pa", detail["pressure_Pa"], target["pressure_Pa"], 1e-3)
                check(f"{node}:bath_J", detail["bath_heat_J"], target["bath_heat_J"], 1e-7)
                check(
                    f"{node}:energy_residual_J",
                    detail["energy_residual_J"],
                    0,
                    1e-7
                    + 2e-6
                    * (
                        abs(detail["bath_heat_J"])
                        + abs(detail["enthalpy_in_J"])
                        + abs(detail["enthalpy_out_J"])
                    ),
                    0,
                )
            edge = native["connections"]["feed"]
            for species, amount in edge["amounts_mol"].items():
                check(f"feed:{species}:mol", amount, source["transferred_mol"][species], 2e-10)
            check("feed:enthalpy_J", edge["enthalpy_J"], source["enthalpy_J"], 1e-7)
            check(
                "total:energy_residual_J",
                native["energy_residual_J"],
                0,
                1e-7 + 2e-6 * abs(native["bath_heat_J"]),
                0,
            )
            for item in checks:
                if not item["passed"]:
                    failures.append({"case": case, "horizon": horizon, **item})
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            failures.append({"case": case, "horizon": horizon, "error": record["error"]})
        rows.append(record)
        elapsed = time.monotonic() - start
        print(
            f"35A Cantera cases {index}/{len(cases)}; "
            f"{index / max(elapsed, 1e-9):.2f} cases/s; "
            f"ETA {elapsed / index * (len(cases) - index):.1f}s",
            flush=True,
        )
    total = sum(len(row["checks"]) for row in rows)
    summary = {
        "evidence_status": "development",
        "source": f"Cantera {ct.__version__}",
        "expected_cases": 14,
        "completed_cases": sum("error" not in r for r in rows),
        "expected_checks": 210,
        "completed_checks": total,
        "passed_checks": sum(c["passed"] for row in rows for c in row["checks"]),
        "failures": failures,
        "rows": rows,
        "elapsed_s": time.monotonic() - start,
    }
    text = json.dumps(summary, indent=2, allow_nan=False)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(text)
    print(
        json.dumps({k: v for k, v in summary.items() if k not in {"rows", "failures"}}), flush=True
    )
    return int(bool(failures) or total != 210)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    raise SystemExit(run(parser.parse_args().output))
