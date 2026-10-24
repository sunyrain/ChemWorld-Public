"""Fixed 24-unit / 234-comparison finite aqueous throughflow qualification."""

import argparse
import json
import os
import subprocess
import time
import traceback
from dataclasses import replace
from pathlib import Path

import numpy as np
from aqueous_flow40_reference import TIMES, VW, R, T, front_errors, history

from chemworld.examples.aqueous_flow_column import example_inputs
from chemworld.physchem.aqueous_reactive_transport import (
    AqueousReactiveTransportSpec,
    advance_aqueous_reactive_transport,
    aqueous_transport_faces,
    prepare_aqueous_transport_state,
)
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.aqueous_reactive_transport_world import build_aqueous_reactive_transport_world


def clean(value):
    return json.loads(json.dumps(value))


def native_front_convergence(kind, condition, details):
    native_fronts = {}
    keys = ("A-",) if kind == "salt" else ("A-", "HA")
    if kind == "slow":
        keys += ("S",)
    for count in (4, 8, 16):
        previous = next(
            d["native"]["trajectory"][-1]
            for d in details
            if d["unit"] == f"{kind}-{condition}-{count}" and "native" in d
        )
        native_fronts[count] = [
            sum(previous["amounts_by_vessel"][f"cell-{i:02d}"][k] for k in keys)
            / previous["cells"][f"cell-{i:02d}"]["volume_m3"]
            for i in range(count)
        ]
    return front_errors(kind, condition, native_fronts)


class Block:
    def __init__(self, output, baseline_src=None):
        output.mkdir(parents=True, exist_ok=False)
        self.output, self.rows, self.units, self.details, self.case = output, [], [], [], {}
        self.baseline_src = baseline_src

    def check(self, name, actual, expected, *, rtol=0.0, atol=0.0):
        if isinstance(expected, (bool, str, dict)):
            passed, error = clean(actual) == clean(expected), None
        else:
            a, b = np.asarray(actual), np.asarray(expected)
            passed = bool(
                a.shape == b.shape
                and np.all(np.isfinite(a))
                and np.allclose(a, b, rtol=rtol, atol=atol)
            )
            error = float(np.max(abs(a - b))) if a.size and a.shape == b.shape else None
        self.rows.append(
            {
                "unit": self.case["name"],
                "quantity": name,
                "passed": passed,
                "actual": actual,
                "expected": expected,
                "rtol": rtol,
                "atol": atol,
                "maximum_absolute_error": error,
            }
        )

    def profile(self):
        c = self.case
        spec, initial = example_inputs(c["kind"], c["condition"], c["cells"])
        reference = history(c["kind"], c["condition"], c["cells"])
        if c["kind"] == "slow":
            refined = history(c["kind"], c["condition"], c["cells"], max_step=0.125)
            discrepancy = max(
                abs(p["concentrations"][k] - q["concentrations"][k]).max()
                for p, q in zip(reference, refined, strict=True)
                for k in p["concentrations"]
            )
            if discrepancy > 1e-8:
                raise ValueError("independent precursor refinement failed: " + str(discrepancy))
        native = advance_aqueous_reactive_transport(spec, initial, 50.0, evaluation_times_s=TIMES)
        self.details.append({"unit": c["name"], "native": native.to_dict()})
        species = [s.species_id for s in spec.equilibrium.database.species]
        names = spec.member_vessel_ids

        def flatten(stocks):
            return [stocks[v][k] for v in names for k in species]

        self.check(
            "all_species_inventory",
            [flatten(p["amounts_by_vessel"]) for p in native.trajectory],
            [flatten(p["amounts"]) for p in reference],
            rtol=3e-6,
            atol=2e-11,
        )
        self.check(
            "water_and_volume",
            [
                [p["amounts_by_vessel"][v]["H2O"] for v in names]
                + [p["cells"][v]["volume_m3"] for v in names]
                for p in native.trajectory
            ],
            [[p["amounts"][v]["H2O"] for v in names] + list(p["volumes"]) for p in reference],
            rtol=2e-10,
            atol=1e-15,
        )
        elements = sorted({k for s in spec.equilibrium.database.species for k in s.composition})

        def conserved(stocks):
            return [
                sum(
                    s.composition.get(e, 0) * stocks[v][s.species_id]
                    for v in names
                    for s in spec.equilibrium.database.species
                )
                for e in elements
            ] + [
                sum(s.charge * stocks[v][s.species_id] for s in spec.equilibrium.database.species)
                for v in names
            ]

        self.check(
            "elements_and_charge",
            [conserved(p["amounts_by_vessel"]) for p in native.trajectory],
            [conserved(initial)] * len(TIMES),
            atol=2e-10,
        )
        native_flows, reference_flows, native_phi, reference_phi, currents = [], [], [], [], []
        for p, ref in zip(native.trajectory, reference, strict=True):
            native_flows.append(
                [
                    row[field][k]
                    for row in p["faces"].values()
                    for field in (
                        "advection_amount_rates_mol_s",
                        "dispersion_amount_rates_mol_s",
                        "electrodiffusion_amount_rates_mol_s",
                        "amount_rates_mol_s",
                    )
                    for k in species
                ]
            )
            reference_flows.append(
                [
                    row[field][k]
                    for row in ref["faces"]
                    for field in ("advection", "dispersion", "molecular", "total")
                    for k in species
                ]
            )
            native_phi.append(
                [row.get("potential_difference_V", 0.0) for row in p["faces"].values()]
            )
            reference_phi.append([row["potential_V"] for row in ref["faces"]])
            currents.append([row["current_molar_charge_rate_mol_s"] for row in p["faces"].values()])
        self.check("all_flow_components", native_flows, reference_flows, rtol=2e-5, atol=2e-13)
        self.check("diffusion_potential", native_phi, reference_phi, atol=2e-7)
        self.check("zero_current", currents, np.zeros_like(currents).tolist(), atol=1e-17)
        affinity = []
        for point in native.trajectory:
            for values in point["amounts_by_vessel"].values():
                if c["kind"] != "salt" and values["H2O"]:
                    volume = values["H2O"] * VW
                    affinity.append(
                        float(
                            R
                            * T
                            * np.log(
                                values["H+"]
                                * values["A-"]
                                / values["HA"]
                                / volume
                                / (1.0, 10.0)[c["condition"]]
                            )
                        )
                    )
        self.check("fast_affinity", affinity, [0.0] * len(affinity), atol=2e-5)
        actual_energy, expected_energy = [], []
        for point, ref in zip(native.trajectory, reference, strict=True):
            for i, v in enumerate(names):
                local = point["cells"][v]
                actual_energy.append(
                    [
                        local["enthalpy_change_J"],
                        local["transported_enthalpy_J"],
                        local["bath_heat_J"],
                        local["boundary_work_J"],
                        local["internal_energy_change_J"]
                        - local["bath_heat_J"]
                        - local["transported_enthalpy_J"]
                        + local["boundary_work_J"],
                    ]
                )
                expected_energy.append(
                    [
                        ref["enthalpy"][v] - reference[0]["enthalpy"][v],
                        float(ref["transported_h"][i]),
                        ref["heat"][v],
                        float(1e5 * (ref["volumes"][i] - reference[0]["volumes"][i])),
                        0.0,
                    ]
                )
        self.check("local_first_law", actual_energy, expected_energy, atol=3e-6)
        self.check(
            "global_bath",
            [p["bath_heat_J"] for p in native.trajectory],
            [sum(p["heat"].values()) for p in reference],
            atol=3e-6,
        )
        self.check(
            "time_complete_codec",
            {
                "times": [p["time_s"] for p in native.trajectory],
                "duration": native.actual_duration_s,
                "success": native.diagnostics["solver"]["success"],
                "codec": AqueousReactiveTransportSpec.from_dict(clean(spec.to_dict())).to_dict(),
            },
            {"times": TIMES.tolist(), "duration": 50.0, "success": True, "codec": spec.to_dict()},
        )
        if c["cells"] == 16:
            front = native_front_convergence(c["kind"], c["condition"], self.details)
            self.details.append({"unit": c["name"], "front_convergence": front})
            self.check("front_convergence", all(r > 1.2 for r in front["ratios"]), True)

    def event(self):
        c = self.case
        spec, n = example_inputs("salt", 1, 4)
        if c["boundary"] == "feed":
            spec = replace(spec, flow=replace(spec.flow, minimum_feed_volume_m3=1e-7))
            n["feed"] = {k: v * 3e-7 / 5e-6 for k, v in n["feed"].items()}
            expected_t, event, feed_volume = 40.0, "aqueous_feed_reserve", 3e-7
        else:
            spec = replace(spec, flow=replace(spec.flow, collector_capacity_m3=1e-7))
            expected_t, event, feed_volume = 20.0, "aqueous_collector_capacity", 5e-6
        result = advance_aqueous_reactive_transport(spec, n, 60.0)
        final = result.final
        ref = history("salt", 1, 4, times=np.array([0.0, expected_t]), feed_volume=feed_volume)[-1]
        self.check("earliest_time", result.actual_duration_s, expected_t, atol=1e-10)
        self.check(
            "event_identity", {"events": result.diagnostics["stopping_events"]}, {"events": [event]}
        )
        self.check(
            "finite_reservoir_water",
            [final["amounts_by_vessel"][v]["H2O"] for v in ("feed", "collector")],
            [ref["amounts"][v]["H2O"] for v in ("feed", "collector")],
            rtol=2e-10,
            atol=1e-13,
        )
        self.check(
            "cell_water_unchanged",
            {v: final["amounts_by_vessel"][v]["H2O"] for v in spec.cell_vessel_ids},
            {v: n[v]["H2O"] for v in spec.cell_vessel_ids},
        )
        keys = list(n["feed"])
        self.check(
            "global_material_and_heat",
            [sum(ns[k] for ns in final["amounts_by_vessel"].values()) for k in keys]
            + [final["bath_heat_J"]],
            [sum(ns[k] for ns in n.values()) for k in keys] + [0.0],
            atol=3e-6,
        )
        self.check(
            "independent_endpoint",
            [final["amounts_by_vessel"][v][k] for v in spec.member_vessel_ids for k in keys],
            [ref["amounts"][v][k] for v in spec.member_vessel_ids for k in keys],
            rtol=3e-6,
            atol=2e-11,
        )
        runtime = ChemWorldRuntime(composition=build_aqueous_reactive_transport_world(spec, n))
        try:
            runtime.reset(seed=40)
            reply = runtime.execute({"operation": "wait", "duration_s": 60.0})
            self.check(
                "world_event_commit",
                {"status": reply.info["transaction_status"], "time": runtime._state.ledger.time_s},
                {"status": "committed", "time": result.actual_duration_s},
            )
            before = stocks(runtime)
            runtime.execute({"operation": "wait", "duration_s": 1.0})
            self.check(
                "boundary_does_not_overrun",
                {"stocks": stocks(runtime), "time": runtime._state.ledger.time_s},
                {"stocks": before, "time": result.actual_duration_s},
            )
            self.details.append(
                {
                    "unit": c["name"],
                    "physical": result.to_dict(),
                    "world_status": reply.info["transaction_status"],
                }
            )
        finally:
            runtime.close()

    def world(self):
        c = self.case
        kind, condition = c["kind"], c["condition"]
        spec, n = example_inputs(kind, condition, 4)
        runtime = ChemWorldRuntime(composition=build_aqueous_reactive_transport_world(spec, n))
        restored = None
        reference = history(kind, condition, 4, times=np.array([0.0, 10.0, 20.0]))[-1]
        try:
            runtime.reset(seed=40)
            runtime.execute({"operation": "select_vessel", "vessel_id": "cell-02"})
            statuses = [
                runtime.execute({"operation": "wait", "duration_s": 10.0}).info[
                    "transaction_status"
                ]
                for _ in range(2)
            ]
            values, locals_ = stocks(runtime), local_states(runtime)
            keys = list(n["feed"])
            self.check(
                "world_inventory",
                [values[v][k] for v in spec.member_vessel_ids for k in keys],
                [reference["amounts"][v][k] for v in spec.member_vessel_ids for k in keys],
                rtol=3e-6,
                atol=2e-11,
            )
            self.check(
                "unique_phases_and_volumes",
                all(
                    s.phases.phases["aqueous"].species_amounts_mol == s.species_amounts
                    and abs(s.volume_L - s.species_amounts["H2O"] * VW * 1000) <= 1e-12
                    for s in locals_.values()
                ),
                True,
            )
            self.check(
                "shared_time_and_commits",
                {"statuses": statuses, "time": runtime._state.ledger.time_s},
                {"statuses": ["committed"] * 2, "time": 20.0},
            )
            faces = aqueous_transport_faces(spec, values)["faces"]
            self.check(
                "real_water_salt_enthalpy_flow",
                all(
                    row["amount_rates_mol_s"]["H2O"] > 0 and row["enthalpy_W"] != 0
                    for row in faces.values()
                )
                and values["collector"]["A-"] > 0,
                True,
            )
            self.check(
                "world_energy",
                [runtime._state.ledger.energy_jacket_J, runtime._state.ledger.boundary_work_J],
                [sum(reference["heat"].values()), 0.0],
                atol=3e-6,
            )
            snapshot = clean(runtime.snapshot().to_dict())
            path = self.output / (c["name"] + "-snapshot.json")
            path.write_text(json.dumps(snapshot), encoding="utf-8")
            restored = ChemWorldRuntime.from_snapshot(snapshot)
            for world in (runtime, restored):
                world.execute({"operation": "wait", "duration_s": 5.0})
            self.check(
                "exact_json_continuation",
                restored.snapshot().to_dict(),
                runtime.snapshot().to_dict(),
            )
            target = self.output / (c["name"] + "-fresh.json")
            process = subprocess.run(
                [
                    "uv",
                    "run",
                    "--no-sync",
                    "python",
                    str(Path(__file__).resolve()),
                    "--resume",
                    str(path),
                    "--output",
                    str(target),
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            (self.output / (c["name"] + "-fresh.log")).write_text(
                process.stdout + process.stderr, encoding="utf-8"
            )
            fresh = json.loads(target.read_text()) if target.exists() else {}
            self.check(
                "exact_fresh_process",
                {"exit": process.returncode, "snapshot": fresh},
                {"exit": 0, "snapshot": runtime.snapshot().to_dict()},
            )
            from chemworld.runtime.thermal_material_routing import exchange_packet

            before = clean(runtime.snapshot().to_dict())
            rejected = False
            try:
                current = runtime._state
                exchange_packet(
                    current,
                    current.replace(vessel_id="outside"),
                    runtime.compiled_composition.spec.physics,
                    {"A-": 1e-8},
                )
            except ValueError:
                rejected = True
            self.check(
                "external_packet_refusal",
                {"rejected": rejected, "snapshot": runtime.snapshot().to_dict()},
                {"rejected": True, "snapshot": before},
            )
        finally:
            runtime.close()
            if restored is not None:
                restored.close()

    def compatibility(self):
        from chemworld.examples.native_aqueous_reactive_transport import (
            example_inputs as closed_inputs,
        )

        if self.baseline_src is None:
            raise ValueError("full block requires --baseline-src with the read-only pre-40E source")
        environment = dict(os.environ, PYTHONPATH=str(self.baseline_src))
        code = (
            "import json; "
            "from chemworld.examples.native_aqueous_reactive_transport import example_inputs; "
            "from chemworld.physchem.aqueous_reactive_transport "
            "import advance_aqueous_reactive_transport; "
            "s,n=example_inputs('salt',0,4); print(json.dumps({'codec':s.to_dict(),"
            "'result':advance_aqueous_reactive_transport(s,n,2.).to_dict()}))"
        )
        process = subprocess.run(
            ["uv", "run", "--no-sync", "python", "-c", code],
            capture_output=True,
            text=True,
            timeout=60,
            env=environment,
        )
        (self.output / "old-source.log").write_text(
            process.stdout + process.stderr, encoding="utf-8"
        )
        if process.returncode:
            raise ValueError("old source reference process failed: " + process.stderr)
        baseline = json.loads(process.stdout)
        closed, n = closed_inputs("salt", 0, 4)
        self.check("old_v1_full_codec", closed.to_dict(), baseline["codec"])
        before = baseline["result"]
        after = advance_aqueous_reactive_transport(closed, n, 2.0).to_dict()
        self.check("old_closed_full_numerics_and_diagnostics", after, before)
        opened, stocks_ = example_inputs("slow", 0, 4)
        opened = replace(opened, flow=replace(opened.flow, flow_m3_s=0.0, dispersion_m2_s=0.0))
        shut = replace(opened, flow=None)
        shut_stocks = {v: stocks_[v] for v in shut.cell_vessel_ids}
        first = advance_aqueous_reactive_transport(opened, stocks_, 2.0)
        second = advance_aqueous_reactive_transport(shut, shut_stocks, 2.0)
        keys = list(stocks_["feed"])
        self.check(
            "zero_flow_original_chemistry",
            [first.final["amounts_by_vessel"][v][k] for v in shut.cell_vessel_ids for k in keys],
            [second.final["amounts_by_vessel"][v][k] for v in shut.cell_vessel_ids for k in keys],
            rtol=3e-6,
            atol=2e-11,
        )
        self.check(
            "zero_flow_reservoirs_unchanged",
            [first.final["amounts_by_vessel"][v][k] for v in ("feed", "collector") for k in keys],
            [stocks_[v][k] for v in ("feed", "collector") for k in keys],
            atol=1e-13,
        )
        self.check(
            "v2_codec",
            AqueousReactiveTransportSpec.from_dict(clean(opened.to_dict())).to_dict(),
            opened.to_dict(),
        )
        self.check(
            "actual_members_and_faces",
            {"members": opened.member_vessel_ids, "edges": opened.topology},
            {
                "members": ("feed", *shut.cell_vessel_ids, "collector"),
                "edges": (
                    ("aqueous-inlet", "feed", "cell-00"),
                    *shut.topology,
                    ("aqueous-outlet", "cell-03", "collector"),
                ),
            },
        )
        runtime = ChemWorldRuntime(
            composition=build_aqueous_reactive_transport_world(opened, stocks_)
        )
        try:
            runtime.reset(seed=40)
            runtime.execute({"operation": "wait", "duration_s": 2.0})
            self.check("no_invented_pump_work", runtime._state.ledger.flow_shaft_work_J, 0.0)
        finally:
            runtime.close()
        self.check("same_source_reverse_rates", opened.reverse_rates(), shut.reverse_rates())
        self.details.append(
            {
                "unit": self.case["name"],
                "old_source": str(self.baseline_src),
                "old_result": before,
                "new_result": after,
                "zero_flow": first.to_dict(),
            }
        )

    def refusals(self):
        import chemworld.physchem.aqueous_reactive_transport as transport

        spec, original = example_inputs("slow", 0, 4)
        failures = []
        for kind in ("charge", "feed", "capacity", "half_empty"):
            n = clean(original)
            if kind == "charge":
                n["feed"]["H+"] *= 2
            elif kind == "feed":
                n["feed"] = {k: value * 0.01 for k, value in n["feed"].items()}
            elif kind == "capacity":
                n["collector"] = {k: value * 2 for k, value in n["feed"].items()}
            else:
                n["collector"]["HA"] = 1e-6
            prior, reason = clean(n), ""
            try:
                prepare_aqueous_transport_state(spec, n)
            except ValueError as error:
                reason = str(error)
            failures.append({"kind": kind, "reason": reason})
            self.check(
                "reject_" + kind,
                {"reason": bool(reason), "stocks": n},
                {"reason": True, "stocks": prior},
            )
        runtime = ChemWorldRuntime(
            composition=build_aqueous_reactive_transport_world(spec, original)
        )
        saved = transport.solve_aqueous_reaction_equilibrium
        try:
            runtime.reset(seed=40)
            before, heat, clock = (
                stocks(runtime),
                runtime._state.ledger.energy_jacket_J,
                runtime._state.ledger.time_s,
            )

            def fail(*args, **kwargs):
                raise ValueError("injected fast closure failure")

            transport.solve_aqueous_reaction_equilibrium = fail
            reply = runtime.execute({"operation": "wait", "duration_s": 1.0})
            self.check(
                "fast_failure_physical_rollback",
                {
                    "rejected": reply.info["transaction_status"] != "committed",
                    "stocks": stocks(runtime),
                    "heat": runtime._state.ledger.energy_jacket_J,
                    "time": runtime._state.ledger.time_s,
                },
                {"rejected": True, "stocks": before, "heat": heat, "time": clock},
            )
        finally:
            transport.solve_aqueous_reaction_equilibrium = saved
            runtime.close()
        for kind in ("source", "negative_flow", "invalid_capacity"):
            reason = ""
            try:
                values = (
                    {"source_id": "missing"}
                    if kind == "source"
                    else {"flow_m3_s": -1.0}
                    if kind == "negative_flow"
                    else {"collector_capacity_m3": 0.0}
                )
                replace(spec, flow=replace(spec.flow, **values))
            except ValueError as error:
                reason = str(error)
            failures.append({"kind": kind, "reason": reason})
            self.check("reject_" + kind, bool(reason), True)
        self.details.append({"unit": self.case["name"], "refusals": failures})


def local_states(runtime):
    return {runtime._state.vessel_id: runtime._state, **runtime._state.inactive_vessels}


def stocks(runtime):
    return {v: s.species_amounts for v, s in local_states(runtime).items()}


def cases():
    rows = [
        {
            "name": f"{kind}-{condition}-{cells}",
            "group": "profile",
            "kind": kind,
            "condition": condition,
            "cells": cells,
            "comparisons": 11 if cells == 16 else 10,
        }
        for kind in ("salt", "fast", "slow")
        for condition in (0, 1)
        for cells in (4, 8, 16)
    ]
    rows += [
        {"name": "event-" + b, "group": "event", "boundary": b, "comparisons": 8}
        for b in ("feed", "collector")
    ]
    rows += [
        {"name": "world-" + k, "group": "world", "kind": k, "condition": c, "comparisons": 8}
        for k, c in (("salt", 1), ("slow", 0))
    ]
    rows += [{"name": k, "group": k, "comparisons": 8} for k in ("compatibility", "refusals")]
    return rows


def run(output, canary=False, baseline_src=None):
    fixed = cases()
    if canary:
        fixed = [fixed[i] for i in (0, 18, 21)]
    block, started = Block(output, baseline_src), time.monotonic()
    planned = sum(c["comparisons"] for c in fixed)
    for index, case in enumerate(fixed, 1):
        block.case = case
        before, error = len(block.rows), None
        try:
            getattr(block, case["group"])()
        except Exception:
            error = traceback.format_exc()
        selected = block.rows[before:]
        passed = (
            error is None
            and len(selected) == case["comparisons"]
            and all(r["passed"] for r in selected)
        )
        record = {
            **case,
            "passed": passed,
            "error": error,
            "executed": len(selected),
            "passed_comparisons": sum(r["passed"] for r in selected),
        }
        block.units.append(record)
        for file, rows in (("unit_results.jsonl", [record]), ("comparisons.jsonl", selected)):
            with (output / file).open("a", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(json.dumps(row) + "\n")
        elapsed = time.monotonic() - started
        print(
            f"40E {index}/{len(fixed)} {case['name']} {'PASS' if passed else 'FAIL'} "
            f"comparisons={len(block.rows)}/{planned} "
            f"rate={index / max(elapsed, 0.001):.2f} units/s "
            f"ETA={(len(fixed) - index) * elapsed / index:.1f}s {error or ''}",
            flush=True,
        )
    summary = {
        "schema": "chemworld-aqueous40e-development-v1",
        "planned_units": len(fixed),
        "units_passed": sum(r["passed"] for r in block.units),
        "planned_comparisons": planned,
        "comparisons_executed": len(block.rows),
        "comparisons_passed": sum(r["passed"] for r in block.rows),
        "unexecuted_comparisons": planned - len(block.rows),
        "failures": [r for r in block.units if not r["passed"]],
        "failed_comparisons": [r for r in block.rows if not r["passed"]],
        "elapsed_s": time.monotonic() - started,
        "passed": all(r["passed"] for r in block.units) and len(block.rows) == planned,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "details.json").write_text(json.dumps(block.details, indent=2), encoding="utf-8")
    return summary


def recheck_fronts(source, output):
    """Re-evaluate the same six criteria from retained complete native raw states."""
    details = json.loads((source / "details.json").read_text(encoding="utf-8"))
    block, started = Block(output), time.monotonic()
    for case in (c for c in cases() if c.get("cells") == 16):
        block.case = case
        front = native_front_convergence(case["kind"], case["condition"], details)
        block.details.append({"unit": case["name"], "front_convergence": front})
        block.check("front_convergence", all(r > 1.2 for r in front["ratios"]), True)
    summary = {
        "schema": "chemworld-aqueous40e-front-reevaluation-v1",
        "source_raw_directory": str(source),
        "physical_reruns": 0,
        "planned_comparisons": 6,
        "comparisons_executed": len(block.rows),
        "comparisons_passed": sum(r["passed"] for r in block.rows),
        "unexecuted_comparisons": 6 - len(block.rows),
        "failed_comparisons": [r for r in block.rows if not r["passed"]],
        "elapsed_s": time.monotonic() - started,
        "passed": len(block.rows) == 6 and all(r["passed"] for r in block.rows),
    }
    (output / "comparisons.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in block.rows), encoding="utf-8"
    )
    (output / "details.json").write_text(json.dumps(block.details, indent=2), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--canary", action="store_true")
    parser.add_argument("--baseline-src", type=Path)
    parser.add_argument("--recheck-fronts", type=Path)
    args = parser.parse_args()
    if args.resume:
        runtime = ChemWorldRuntime.from_snapshot(json.loads(args.resume.read_text()))
        try:
            reply = runtime.execute({"operation": "wait", "duration_s": 5.0})
            if reply.info["transaction_status"] != "committed":
                raise ValueError(str(reply.info.get("outcome_diagnostics")))
            with args.output.open("x", encoding="utf-8") as stream:
                json.dump(runtime.snapshot().to_dict(), stream)
        finally:
            runtime.close()
    else:
        result = (
            recheck_fronts(args.recheck_fronts, args.output)
            if args.recheck_fronts
            else run(args.output, args.canary, args.baseline_src)
        )
        print(
            json.dumps(
                {k: v for k, v in result.items() if k not in {"failures", "failed_comparisons"}}
            ),
            flush=True,
        )
        raise SystemExit(0 if result["passed"] else 1)
