"""Fixed 22-unit / 156-comparison joint gas-aqueous-mineral development block."""

import argparse
import json
import os
import shutil
import subprocess
import time
import traceback
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
from aqueous_mineral_gas40_reference import final_root, history
from run_aqueous_gas40_full import Block as CheckBlock

from chemworld.examples.aqueous_headspace_minerals import example_inputs
from chemworld.physchem.aqueous_headspace_minerals import (
    AqueousHeadspaceMineralSpec,
    advance_aqueous_headspace_minerals,
    headspace_mineral_properties,
    prepare_aqueous_headspace_mineral_state,
)
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.aqueous_gas_transfer import AqueousGasTransferRuntime
from chemworld.world.aqueous_gas_transfer_world import build_aqueous_headspace_world

TIMES = (0.0, 0.5, 2.0, 5.0, 30.0, 120.0, 600.0)


def clean(value):
    return json.loads(json.dumps(value))


def physical(runtime):
    s = runtime._state
    return {
        "amounts": s.species_amounts,
        "phases": s.phases.to_dict(),
        "slots": s.state_slots.to_dict(),
        "P": s.pressure_Pa,
        "T": s.temperature_K,
        "V": s.volume_L,
        "ledger": s.ledger.to_dict(),
    }


def physical_without_attempt_cost(value):
    return {
        **value,
        "ledger": {k: v for k, v in value["ledger"].items() if k not in {"cost", "risk"}},
    }


class Block(CheckBlock):
    def profile(self):
        c = self.case
        spec, initial, geometry = example_inputs(c["kind"], c["condition"])
        ref, events = history(c["kind"], c["condition"], TIMES)
        refined, _ = history(c["kind"], c["condition"], TIMES, max_step=1.0)
        if (
            max(
                abs(p["amounts"][k] - q["amounts"][k])
                for p, q in zip(ref, refined, strict=True)
                for k in initial
            )
            > 1e-10
        ):
            raise ValueError("independent joint reference refinement failed")
        native = advance_aqueous_headspace_minerals(
            spec, initial, geometry, TIMES[-1], evaluation_times_s=TIMES
        )
        self.details.append(
            {
                "unit": c["name"],
                "spec": spec.to_dict(),
                "initial": initial,
                "geometry": {k: g.to_dict() for k, g in geometry.items()},
                "native": native.to_dict(),
                "reference": ref,
                "reference_events": events,
            }
        )
        points, ids = native.trajectory, list(initial)
        self.check(
            "all_species",
            [[p["amounts_mol"][k] for k in ids] for p in points],
            [[p["amounts"][k] for k in ids] for p in ref],
            rtol=4e-6,
            atol=3e-10,
        )
        self.check(
            "actual_pressure_and_gas_volume",
            [[p["properties"]["pressure_Pa"], p["properties"]["gas"]["volume_m3"]] for p in points],
            [[p["properties"]["P"], p["properties"]["Vg"]] for p in ref],
            rtol=4e-7,
            atol=1e-9,
        )
        energies = [("enthalpy_J", "H"), ("internal_energy_J", "U")]
        self.check(
            "all_phase_energy",
            [
                [p["properties"][key] for key, _ in energies]
                + [
                    p["properties"][phase][key]
                    for phase in ("liquid", "gas")
                    for key, _ in energies
                ]
                + [
                    p["properties"]["minerals"][m][key]
                    for m in spec.mineral_ids
                    for key, _ in energies
                ]
                for p in points
            ],
            [
                [p["properties"][key] for _, key in energies]
                + [p["properties"][key + phase] for phase in ("l", "g") for _, key in energies]
                + [p["properties"][key + "m"][m] for m in spec.mineral_ids for _, key in energies]
                for p in ref
            ],
            atol=4e-6,
            rtol=1e-9,
        )
        self.check(
            "rigid_heat_work",
            [[p["bath_heat_J"], p["boundary_work_J"]] for p in points],
            [[p["heat"], 0.0] for p in ref],
            atol=4e-6,
        )
        self.check(
            "forward_reverse_transfer",
            [
                [
                    p["transfers"][t.reaction_id][k]
                    for t in spec.transfers
                    for k in ("forward_mol_s", "reverse_mol_s")
                ]
                for p in points
            ],
            [[v for t in spec.transfers for v in p["gas"][t.reaction_id]] for p in ref],
            atol=1e-11,
            rtol=4e-6,
        )
        self.check(
            "mineral_rates_and_area",
            [
                [
                    p["minerals"][m][k]
                    for m in spec.mineral_ids
                    for k in ("surface_area_m2", "dissolution_mol_s")
                ]
                for p in points
            ],
            [[p["mineral"][m][k] for m in spec.mineral_ids for k in ("area", "rate")] for p in ref],
            atol=1e-9,
            rtol=4e-6,
        )
        masks = [
            [p["minerals"][m]["saturation_index"] is None for m in spec.mineral_ids] for p in points
        ]
        expected_masks = [[p["mineral"][m]["si"] is None for m in spec.mineral_ids] for p in ref]
        si = [
            [p["minerals"][m]["saturation_index"] or 0.0 for m in spec.mineral_ids] for p in points
        ]
        expected_si = [[p["mineral"][m]["si"] or 0.0 for m in spec.mineral_ids] for p in ref]
        self.check(
            "mineral_affinity",
            masks == expected_masks and np.allclose(si, expected_si, rtol=4e-6, atol=2e-7),
            True,
        )
        db = spec.equilibrium.database
        elements = sorted({e for s in db.species for e in s.composition})
        invariant_error = max(
            abs(
                sum(
                    s.composition.get(e, 0)
                    * (p["amounts_mol"][s.species_id] - initial[s.species_id])
                    for s in db.species
                )
            )
            for p in points
            for e in elements
        )
        charge = max(
            abs(sum(s.charge * p["amounts_mol"][s.species_id] for s in db.species)) for p in points
        )
        self.check(
            "finite_original_invariants",
            invariant_error < 3e-10
            and charge < 1e-11
            and all(
                p["amounts_mol"]["H2O"] == initial["H2O"]
                and p["amounts_mol"]["I(g)"] == initial["I(g)"]
                and min(p["amounts_mol"].values()) >= 0
                for p in points
            ),
            True,
        )
        actual_events = native.diagnostics["events"]
        self.check(
            "exact_exhaustion_no_nucleation",
            [e["mineral_id"] for e in actual_events] == [e["mineral_id"] for e in events]
            and np.allclose(
                [e["time_s"] for e in actual_events],
                [e["time_s"] for e in events],
                atol=2e-6,
                rtol=2e-6,
            )
            and all(native.final["amounts_mol"][e["mineral_id"]] == 0 for e in events)
            and (
                c["kind"] != "unseeded"
                or all(p["amounts_mol"][m] == 0 for p in points for m in spec.mineral_ids)
            ),
            True,
        )

    def world(self):
        c = self.case
        spec, n, g = example_inputs(c["kind"], c["condition"])
        runtime = ChemWorldRuntime(
            composition=build_aqueous_headspace_world(spec, n, mineral_geometries=g)
        )
        restored = None
        try:
            runtime.reset(seed=40)
            slots = runtime._state.state_slots
            a = runtime.execute({"operation": "wait", "duration_s": 30.0})
            restored = ChemWorldRuntime.from_snapshot(clean(runtime.snapshot().to_dict()))
            replies = [
                r.execute({"operation": "wait", "duration_s": 20.0}) for r in (runtime, restored)
            ]
            self.check(
                "complete_json_continuation",
                runtime.snapshot().to_dict(),
                restored.snapshot().to_dict(),
            )
            source, dest = (
                self.output / (c["name"] + "-snapshot.json"),
                self.output / (c["name"] + "-fresh.json"),
            )
            source.write_text(json.dumps(runtime.snapshot().to_dict()), encoding="utf-8")
            final = runtime.execute({"operation": "wait", "duration_s": 5.0})
            proc = subprocess.run(
                [
                    "uv",
                    "run",
                    "--no-sync",
                    "python",
                    str(Path(__file__).resolve()),
                    "--resume",
                    str(source),
                    "--output",
                    str(dest),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            (self.output / (c["name"] + "-fresh.log")).write_text(
                proc.stdout + proc.stderr, encoding="utf-8"
            )
            self.check(
                "fresh_process_continuation",
                {
                    "code": proc.returncode,
                    "snapshot": json.loads(dest.read_text()) if dest.exists() else None,
                },
                {"code": 0, "snapshot": runtime.snapshot().to_dict()},
            )
            self.check(
                "commits_and_clock",
                {
                    "statuses": [r.info["transaction_status"] for r in (a, *replies, final)],
                    "time": runtime._state.ledger.time_s,
                },
                {"statuses": ["committed"] * 4, "time": 55.0},
            )
            ref = history(c["kind"], c["condition"], (0.0, 55.0))[0][-1]
            s = runtime._state
            self.check(
                "world_inventory",
                list(s.species_amounts.values()),
                [ref["amounts"][k] for k in s.species_amounts],
                atol=3e-10,
                rtol=4e-6,
            )
            self.check(
                "world_pressure", s.pressure_Pa, ref["properties"]["P"], atol=1e-4, rtol=4e-7
            )
            self.check("world_heat", s.ledger.energy_jacket_J, ref["heat"], atol=4e-6)
            self.check(
                "original_geometry_phase_owner",
                s.state_slots == slots
                and s.phases.total_amounts_mol() == s.species_amounts
                and abs(sum(p.volume_L for p in s.phases.phases.values()) - s.volume_L) < 1e-12
                and s.vessels.vessels["reactor"].pressure_Pa == s.pressure_Pa,
                True,
            )
            before = clean(physical(runtime))
            denied = runtime.execute(
                {"operation": "add_solvent", "solvent": "water", "volume_L": 0.001}
            )
            self.check(
                "external_material_refused",
                denied.info["transaction_status"] != "committed"
                and physical_without_attempt_cost(clean(physical(runtime)))
                == physical_without_attempt_cost(before),
                True,
            )
            self.details.append(
                {
                    "unit": c["name"],
                    "final": before,
                    "reference": ref,
                    "first_info": a.info,
                    "last_info": final.info,
                    "denied_info": denied.info,
                    "after_denial": clean(physical(runtime)),
                }
            )
        finally:
            runtime.close()
            if restored is not None:
                restored.close()

    def root(self):
        c = self.case
        spec, n, g = example_inputs(c["kind"], c["condition"])
        root_n, residual = final_root(c["kind"], c["condition"])
        result = advance_aqueous_headspace_minerals(spec, n, g, 1800.0)
        final = result.final
        self.details.append(
            {"unit": c["name"], "native": result.to_dict(), "root": root_n, "residual": residual}
        )
        self.check(
            "coexistence_root",
            list(final["amounts_mol"].values()),
            [root_n[k] for k in final["amounts_mol"]],
            atol=5e-9,
            rtol=5e-6,
        )
        self.check(
            "zero_joint_rates",
            [p["net_mol_s"] for p in final["transfers"].values()]
            + [p["dissolution_mol_s"] for p in final["minerals"].values()],
            [0.0] * (len(spec.transfers) + len(spec.mineral_ids)),
            atol=5e-11,
        )
        root_state = prepare_aqueous_headspace_mineral_state(spec, root_n)
        stationary = advance_aqueous_headspace_minerals(spec, root_state, g, 5.0)
        self.check(
            "root_stationary",
            list(stationary.final["amounts_mol"].values()),
            list(root_state.values()),
            atol=3e-10,
        )
        self.check(
            "v2_codec",
            AqueousHeadspaceMineralSpec.from_dict(clean(spec.to_dict())).to_dict(),
            spec.to_dict(),
        )

    def limits(self):
        spec, n, g = example_inputs()
        kind = self.case["kind"]
        if kind == "thermodynamics":
            before = headspace_mineral_properties(spec, n)
            after = headspace_mineral_properties(spec, {**n, "I(g)": 2 * n["I(g)"]})
            dp = after["pressure_Pa"] - before["pressure_Pa"]
            for phase in ("liquid", "mineral"):
                a, b = (
                    (before["liquid"], after["liquid"])
                    if phase == "liquid"
                    else (before["minerals"]["MA(s)"], after["minerals"]["MA(s)"])
                )
                self.check(
                    phase + "_H_G_pressure",
                    [b[k] - a[k] for k in ("enthalpy_J", "gibbs_energy_J")],
                    [dp * a["volume_m3"]] * 2,
                    atol=2e-9,
                )
                self.check(
                    phase + "_U_pressure", b["internal_energy_J"], a["internal_energy_J"], atol=2e-9
                )
        elif kind == "refusals":
            for bad in ("capacity", "water", "charge", "geometry"):
                error = None
                try:
                    if bad == "capacity":
                        prepare_aqueous_headspace_mineral_state(spec, {**n, "MA(s)": 0.2})
                    elif bad == "water":
                        prepare_aqueous_headspace_mineral_state(spec, {**n, "H2O": n["H2O"] * 0.9})
                    elif bad == "charge":
                        prepare_aqueous_headspace_mineral_state(spec, {**n, "M+": n["M+"] * 2})
                    else:
                        composition = build_aqueous_headspace_world(spec, n, mineral_geometries=g)
                        state = composition.physics.initial_states["reactor"]
                        slot = state.state_slots.slots[0]
                        state = state.replace(
                            state_slots=replace(
                                state.state_slots,
                                slots=(
                                    replace(
                                        slot, values={**slot.values, "reference_area_m2": 0.02}
                                    ),
                                ),
                            )
                        )
                        AqueousGasTransferRuntime(composition).validate_state(state)
                except ValueError as exc:
                    error = str(exc)
                self.check("reject_" + bad, error is not None, True)
                self.details.append({"unit": self.case["name"], "refusal": bad, "reason": error})
        else:
            runtime = ChemWorldRuntime(
                composition=build_aqueous_headspace_world(spec, n, mineral_geometries=g)
            )
            try:
                runtime.reset(seed=40)
                before = clean(physical(runtime))
                with patch(
                    "chemworld.physchem.aqueous_headspace_minerals.solve_aqueous_reaction_equilibrium",
                    side_effect=ValueError("injected joint failure"),
                ):
                    reply = runtime.execute({"operation": "wait", "duration_s": 5.0})
                after = clean(physical(runtime))
                self.check("rejected", reply.info["transaction_status"] != "committed", True)
                self.check(
                    "material_pressure_geometry_rollback",
                    {k: v for k, v in after.items() if k != "ledger"},
                    {k: v for k, v in before.items() if k != "ledger"},
                )
                self.check(
                    "clock_and_heat_rollback",
                    physical_without_attempt_cost(after)["ledger"],
                    physical_without_attempt_cost(before)["ledger"],
                )
                self.check(
                    "explicit_failure", "injected joint failure" in json.dumps(reply.info), True
                )
                self.details.append(
                    {
                        "unit": self.case["name"],
                        "info": reply.info,
                        "before": before,
                        "after": after,
                    }
                )
            finally:
                runtime.close()

    def compatibility(self):
        c = self.case
        outputs = []
        for source in (self.baseline, None):
            name = "old" if source else "current"
            dest = self.output / (c["name"] + "-" + name + ".json")
            if source and self.retained is not None:
                shutil.copyfile(self.retained / dest.name, dest)
                outputs.append(json.loads(dest.read_text()))
                continue
            env = dict(os.environ)
            env["PYTHONPATH"] = str(source or Path("src").resolve())
            proc = subprocess.run(
                [
                    "uv",
                    "run",
                    "--no-sync",
                    "python",
                    str(Path(__file__).with_name("aqueous_gas40_compat.py")),
                    "--kind",
                    c["kind"],
                    "--condition",
                    str(c["condition"]),
                    "--output",
                    str(dest),
                    *(
                        []
                        if source
                        else ["--legacy-source", str(self.output / (c["name"] + "-old.json"))]
                    ),
                ],
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            (self.output / (c["name"] + "-" + name + ".log")).write_text(
                proc.stdout + proc.stderr, encoding="utf-8"
            )
            if proc.returncode:
                raise ValueError(proc.stdout + proc.stderr)
            outputs.append(json.loads(dest.read_text()))
        self.check("v1_complete_spec_and_numerics", outputs[1]["direct"], outputs[0]["direct"])
        expected = clean(
            {k: outputs[0]["world"][k] for k in ("snapshot", "results", "final", "restored")}
        )
        migration_paths = []
        for endpoint in ("final", "restored"):
            for index, module in enumerate(expected[endpoint]["execution"]["modules"]):
                if "scope_notes" in module:
                    module.pop("scope_notes")
                    migration_paths.append(f"{endpoint}.execution.modules[{index}].scope_notes")
        self.details.append(
            {
                "unit": c["name"],
                "explicit_07A_encoding_migration": migration_paths,
                "scope": (
                    "Only obsolete module display notes are removed from expected legacy "
                    "endpoints; complete original input, outputs and all physical fields retained"
                ),
            }
        )
        self.check(
            "v1_complete_world_and_json",
            outputs[1]["world"],
            expected,
        )


def run(output, baseline, retained=None):
    block = Block(output)
    block.baseline = baseline
    block.retained = retained
    cases = [
        {"name": f"{k}-{c}", "kind": k, "condition": c, "group": "profile", "comparisons": 9}
        for k in ("uptake", "degassing", "exhaustion", "unseeded", "double")
        for c in (0, 1)
    ]
    cases += [
        {"name": k, "kind": k, "condition": 0, "group": "profile", "comparisons": 9}
        for k in ("zero_gas", "zero_mineral")
    ]
    cases += [
        {"name": "world-" + k, "kind": k, "condition": c, "group": "world", "comparisons": 8}
        for k, c in (("uptake", 0), ("double", 1), ("exhaustion", 0))
    ]
    cases += [
        {"name": "root-" + k, "kind": k, "condition": c, "group": "root", "comparisons": 4}
        for k, c in (("uptake", 0), ("double", 1))
    ]
    cases += [
        {"name": "limit-" + k, "kind": k, "group": "limits", "comparisons": 4}
        for k in ("thermodynamics", "refusals", "rollback")
    ]
    cases += [
        {
            "name": "compat-" + k,
            "kind": k,
            "condition": c,
            "group": "compatibility",
            "comparisons": 2,
        }
        for k, c in (("acid", 0), ("double", 1))
    ]
    retained_summary = None
    if retained is not None:
        previous = json.loads((retained / "raw.json").read_text())
        retained_summary = json.loads((retained / "summary.json").read_text())
        block.units = [u for u in previous["units"] if u["group"] != "compatibility"]
        names = {u["name"] for u in block.units}
        if len(names) != 20 or not all(u["passed"] for u in block.units):
            raise ValueError("reevaluation requires all twenty unchanged scientific units")
        block.rows = [r for r in previous["comparisons"] if r["unit"] in names]
        block.details = [r for r in previous["details"] if r["unit"] in names]
    active_cases = [c for c in cases if retained is None or c["group"] == "compatibility"]
    started = time.monotonic()
    for index, case in enumerate(active_cases, 1):
        block.case = case
        before, error = len(block.rows), None
        try:
            getattr(block, case["group"])()
        except Exception:
            error = traceback.format_exc()
        selected = block.rows[before:]
        unit = {
            **case,
            "executed": len(selected),
            "passed_comparisons": sum(r["passed"] for r in selected),
            "error": error,
            "passed": error is None
            and len(selected) == case["comparisons"]
            and all(r["passed"] for r in selected),
        }
        block.units.append(unit)
        elapsed = time.monotonic() - started
        print(
            f"40G {index}/{len(active_cases)} {case['name']} "
            f"{unit['passed_comparisons']}/{case['comparisons']} passed={unit['passed']} "
            f"{index / elapsed:.3f} units/s "
            f"ETA={elapsed / index * (len(active_cases) - index):.1f}s",
            flush=True,
        )
        (output / "raw.json").write_text(
            json.dumps(
                {"units": block.units, "comparisons": block.rows, "details": block.details},
                indent=2,
            ),
            encoding="utf-8",
        )
    failures = [u for u in block.units if not u["passed"]]
    summary = {
        "schema": "chemworld-native40g-development-v1",
        "planned_units": len(cases),
        "units_passed": len(cases) - len(failures),
        "planned_comparisons": sum(c["comparisons"] for c in cases),
        "comparisons_executed": len(block.rows),
        "comparisons_passed": sum(r["passed"] for r in block.rows),
        "unexecuted_comparisons": sum(c["comparisons"] for c in cases) - len(block.rows),
        "failures": failures,
        "failed_comparisons": [r for r in block.rows if not r["passed"]],
        "elapsed_s": time.monotonic() - started,
        "artifact_directory": str(output.resolve()),
        "passed": not failures,
        "retained_scientific_evidence": None
        if retained is None
        else {
            "artifact_directory": str(retained.resolve()),
            "units": 20,
            "comparisons": 152,
            "source_elapsed_s": retained_summary["elapsed_s"],
            "reason": (
                "Only old snapshot consumption is reevaluated after shared codec fix; "
                "all scientific raw data retained unchanged"
            ),
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k not in ("failures", "failed_comparisons")}
        ),
        flush=True,
    )
    return not failures


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-src", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--reevaluate-from", type=Path)
    args = parser.parse_args()
    if args.resume:
        runtime = ChemWorldRuntime.from_snapshot(json.loads(args.resume.read_text()))
        try:
            runtime.execute({"operation": "wait", "duration_s": 5.0})
            args.output.write_text(json.dumps(runtime.snapshot().to_dict()), encoding="utf-8")
        finally:
            runtime.close()
    else:
        if args.baseline_src is None:
            parser.error("--baseline-src required for original v1 compatibility")
        raise SystemExit(0 if run(args.output, args.baseline_src, args.reevaluate_from) else 1)
