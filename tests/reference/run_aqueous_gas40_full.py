"""Fixed sixteen-unit / 120-comparison finite gas/liquid development block."""

import argparse
import json
import subprocess
import time
import traceback
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
from aqueous_gas40_reference import VOLUME, WATER_KG, R, history, parameters, rates

from chemworld.examples.aqueous_headspace import example_inputs
from chemworld.physchem.aqueous_gas_transfer import (
    AqueousHeadspaceSpec,
    advance_aqueous_headspace,
    headspace_pressure,
    headspace_transfer_rates,
    prepare_aqueous_headspace_state,
)
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.aqueous_gas_transfer_world import build_aqueous_headspace_world

TIMES = (0.0, 1.0, 5.0, 30.0, 120.0, 600.0)


def clean(value):
    return json.loads(json.dumps(value))


def physical(runtime):
    state = runtime._state
    return {
        "amounts": state.species_amounts,
        "phases": state.phases.to_dict(),
        "P": state.pressure_Pa,
        "T": state.temperature_K,
        "V": state.volume_L,
        "time": state.ledger.time_s,
        "heat": state.ledger.energy_jacket_J,
    }


class Block:
    def __init__(self, output):
        output.mkdir(parents=True, exist_ok=False)
        self.output, self.rows, self.units, self.details, self.case = output, [], [], [], {}

    def check(self, name, actual, expected, *, atol=0.0, rtol=0.0):
        if isinstance(expected, (bool, dict, str)):
            passed, error = clean(actual) == clean(expected), None
        else:
            a, b = np.asarray(actual), np.asarray(expected)
            passed = bool(
                a.shape == b.shape
                and np.all(np.isfinite(a))
                and np.allclose(a, b, atol=atol, rtol=rtol)
            )
            error = float(np.max(abs(a - b))) if a.size and a.shape == b.shape else None
        self.rows.append(
            {
                "unit": self.case["name"],
                "quantity": name,
                "passed": passed,
                "actual": actual,
                "expected": expected,
                "atol": atol,
                "rtol": rtol,
                "maximum_absolute_error": error,
            }
        )

    def profile(self):
        c = self.case
        spec, initial = example_inputs(c["kind"], c["condition"])
        reference = history(c["kind"], c["condition"], TIMES)
        if c["kind"] in ("acid", "double"):
            refined = history(c["kind"], c["condition"], TIMES, max_step=0.5)
            discrepancy = max(
                abs(p["amounts"][k] - q["amounts"][k])
                for p, q in zip(reference, refined, strict=True)
                for k in initial
            )
            if discrepancy > 1e-10:
                raise ValueError("independent acid reference refinement failed")
        native = advance_aqueous_headspace(spec, initial, 600.0, evaluation_times_s=TIMES)
        self.details.append({"unit": c["name"], "native": native.to_dict(), "reference": reference})
        ids = list(initial)
        self.check(
            "all_species",
            [[p["amounts_mol"][k] for k in ids] for p in native.trajectory],
            [[p["amounts"][k] for k in ids] for p in reference],
            rtol=3e-6,
            atol=2e-10,
        )
        self.check(
            "partial_and_total_pressure",
            [
                [p["properties"]["pressure_Pa"], *p["properties"]["partial_pressures_Pa"].values()]
                for p in native.trajectory
            ],
            [
                [
                    p["properties"]["P"],
                    *[
                        p["amounts"][k] * R * spec.temperature_K / VOLUME
                        for k in spec.gas_species_ids
                    ],
                ]
                for p in reference
            ],
            rtol=3e-7,
            atol=1e-4,
        )
        params = parameters(c["kind"], c["condition"])
        self.check(
            "forward_reverse_transfer",
            [
                [
                    v[key]
                    for v in p["transfers"].values()
                    for key in ("forward_mol_s", "reverse_mol_s")
                ]
                for p in native.trajectory
            ],
            [
                [value for row in rates(params, p["amounts"]).values() for value in row]
                for p in reference
            ],
            rtol=3e-6,
            atol=2e-12,
        )
        self.check(
            "phase_enthalpy_and_internal_energy",
            [
                [
                    p["properties"][phase][key]
                    for phase in ("liquid", "gas")
                    for key in ("enthalpy_J", "internal_energy_J")
                ]
                for p in native.trajectory
            ],
            [
                [p["properties"][key] for key in ("H_liquid", "U_liquid", "H_gas", "U_gas")]
                for p in reference
            ],
            atol=3e-6,
        )
        self.check(
            "rigid_first_law",
            [
                [
                    p["bath_heat_J"],
                    p["properties"]["internal_energy_J"]
                    - p["properties"]["enthalpy_J"]
                    + p["properties"]["pressure_Pa"] * 2 * VOLUME,
                ]
                for p in native.trajectory
            ],
            [[p["heat"], 0.0] for p in reference],
            atol=3e-6,
        )
        elements = sorted({e for s in spec.equilibrium.database.species for e in s.composition})

        def conserved(n):
            return [
                sum(
                    s.composition.get(e, 0) * n[s.species_id]
                    for s in spec.equilibrium.database.species
                )
                for e in elements
            ] + [sum(s.charge * n[s.species_id] for s in spec.equilibrium.database.species)]

        self.check(
            "elements_charge",
            [conserved(p["amounts_mol"]) for p in native.trajectory],
            [conserved(initial)] * len(TIMES),
            atol=2e-10,
        )
        affinities = []
        dissipation = []
        for p in native.trajectory:
            n = p["amounts_mol"]
            if params["acid"]:
                affinities.append(
                    R
                    * spec.temperature_K
                    * np.log(n["H+"] * n["Cl-"] / (WATER_KG * n["A(aq)"] * params["Ka"]))
                )
            dissipation += [
                -v["net_mol_s"] * v["reaction_gibbs_J_mol"]
                for v in p["transfers"].values()
                if v["reaction_gibbs_J_mol"] is not None
            ]
        self.check(
            "fast_affinity_and_transfer_dissipation",
            all(abs(a) <= 2e-5 for a in affinities) and all(d >= -1e-12 for d in dissipation),
            True,
        )
        self.details[-1]["affinities_J_mol"] = affinities
        self.details[-1]["transfer_dissipation_W"] = dissipation
        self.check(
            "fixed_water_geometry_inert",
            [
                [
                    p["amounts_mol"]["H2O"],
                    p["amounts_mol"]["I(g)"],
                    p["properties"]["liquid"]["volume_m3"],
                    p["properties"]["gas"]["volume_m3"],
                ]
                for p in native.trajectory
            ],
            [[initial["H2O"], initial["I(g)"], VOLUME, VOLUME]] * len(TIMES),
        )
        self.check(
            "times_and_codec",
            {
                "times": [p["time_s"] for p in native.trajectory],
                "codec": AqueousHeadspaceSpec.from_dict(clean(spec.to_dict())).to_dict(),
            },
            {"times": list(TIMES), "codec": spec.to_dict()},
        )

    def world(self):
        c = self.case
        spec, initial = example_inputs(c["kind"], c["condition"])
        runtime = ChemWorldRuntime(composition=build_aqueous_headspace_world(spec, initial))
        restored = None
        try:
            runtime.reset(seed=40)
            first = runtime.execute({"operation": "wait", "duration_s": 30.0})
            restored = ChemWorldRuntime.from_snapshot(clean(runtime.snapshot().to_dict()))
            replies = [
                r.execute({"operation": "wait", "duration_s": 20.0}) for r in (runtime, restored)
            ]
            self.check(
                "full_json_continuation",
                runtime.snapshot().to_dict(),
                restored.snapshot().to_dict(),
            )
            snapshot = self.output / (c["name"] + "-snapshot.json")
            snapshot.write_text(json.dumps(runtime.snapshot().to_dict()), encoding="utf-8")
            final = runtime.execute({"operation": "wait", "duration_s": 5.0})
            dest = self.output / (c["name"] + "-fresh.json")
            proc = subprocess.run(
                [
                    "uv",
                    "run",
                    "--no-sync",
                    "python",
                    str(Path(__file__).resolve()),
                    "--output",
                    str(dest),
                    "--resume",
                    str(snapshot),
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
                    "returncode": proc.returncode,
                    "snapshot": json.loads(dest.read_text()) if dest.exists() else None,
                },
                {"returncode": 0, "snapshot": runtime.snapshot().to_dict()},
            )
            self.check(
                "commits_and_clock",
                {
                    "commits": [r.info["transaction_status"] for r in (first, *replies, final)],
                    "time": runtime._state.ledger.time_s,
                },
                {"commits": ["committed"] * 4, "time": 55.0},
            )
            ref = history(c["kind"], c["condition"], (0.0, 55.0))[-1]
            self.check(
                "world_inventory",
                list(runtime._state.species_amounts.values()),
                [ref["amounts"][k] for k in runtime._state.species_amounts],
                rtol=3e-6,
                atol=2e-10,
            )
            self.check(
                "world_pressure",
                runtime._state.pressure_Pa,
                ref["properties"]["P"],
                rtol=3e-7,
                atol=1e-4,
            )
            state = runtime._state
            self.check(
                "original_phase_owner",
                {
                    "total": state.phases.total_amounts_mol(),
                    "volumes": {k: p.volume_L for k, p in state.phases.phases.items()},
                },
                {
                    "total": state.species_amounts,
                    "volumes": {"aqueous": VOLUME * 1000, "headspace": VOLUME * 1000},
                },
            )
            self.check("world_heat", state.ledger.energy_jacket_J, ref["heat"], atol=3e-6)
            before = clean(physical(runtime))
            denied = runtime.execute(
                {"operation": "add_solvent", "solvent": "water", "volume_L": 0.001}
            )
            self.check(
                "external_material_rejected",
                denied.info["transaction_status"] != "committed" and physical(runtime) == before,
                True,
            )
            self.details.append(
                {
                    "unit": c["name"],
                    "final": before,
                    "reference": ref,
                    "first_info": first.info,
                    "final_info": final.info,
                }
            )
        finally:
            runtime.close()
            if restored is not None:
                restored.close()

    def limits(self):
        name = self.case["kind"]
        if name in ("zero", "equilibrium"):
            spec, n = example_inputs("neutral")
            if name == "zero":
                spec = replace(
                    spec, transfers=tuple(replace(t, coefficient_mol_s=0.0) for t in spec.transfers)
                )
            else:
                p = parameters("neutral", 0)
                total = n["A(g)"] + n["A(aq)"]
                n["A(g)"] = total / (1 + WATER_KG * p["K"]["A"] * R * p["T"] / VOLUME / 1e5)
                n["A(aq)"] = total - n["A(g)"]
            result = advance_aqueous_headspace(spec, n, 600.0)
            self.details.append({"unit": self.case["name"], "native": result.to_dict()})
            self.check(
                "stationary_inventory",
                list(result.final["amounts_mol"].values()),
                list(n.values()),
                atol=0.0 if name == "zero" else 2e-10,
            )
            if name == "zero":
                self.check(
                    "stationary_pressure",
                    result.final["properties"]["pressure_Pa"],
                    headspace_pressure(spec, n),
                )
            else:
                self.check(
                    "zero_net_transfer",
                    [r["net_mol_s"] for r in headspace_transfer_rates(spec, n).values()],
                    [0.0],
                    atol=1e-13,
                )
            self.check(
                "stationary_heat",
                result.final["bath_heat_J"],
                0.0,
                atol=0.0 if name == "zero" else 3e-6,
            )
            if name == "zero":
                self.check(
                    "codec",
                    AqueousHeadspaceSpec.from_dict(clean(spec.to_dict())).to_dict(),
                    spec.to_dict(),
                )
            else:
                runtime = ChemWorldRuntime(composition=build_aqueous_headspace_world(spec, n))
                try:
                    runtime.reset(seed=40)
                    reply = runtime.execute({"operation": "wait", "duration_s": 5.0})
                    self.check(
                        "equilibrium_world",
                        reply.info["transaction_status"] == "committed"
                        and np.allclose(
                            list(runtime._state.species_amounts.values()),
                            list(n.values()),
                            rtol=3e-6,
                            atol=2e-10,
                        ),
                        True,
                    )
                finally:
                    runtime.close()
        elif name == "refusals":
            reasons = []
            for kind in ("zero_gas", "charge", "water", "transfer"):
                spec, n = example_inputs("acid")
                error = None
                try:
                    if kind == "zero_gas":
                        for k in spec.gas_species_ids:
                            n[k] = 0.0
                    elif kind == "charge":
                        n["H+"] *= 2
                    elif kind == "water":
                        n["H2O"] *= 0.5
                    else:
                        spec = replace(
                            spec, transfers=(replace(spec.transfers[0], reaction_id="acid"),)
                        )
                    prepare_aqueous_headspace_state(spec, n)
                except ValueError as exc:
                    error = str(exc)
                reasons.append({"kind": kind, "reason": error})
                self.check("reject_" + kind, error is not None, True)
            self.details.append({"unit": self.case["name"], "refusals": reasons})
        else:
            spec, n = example_inputs("acid")
            runtime = ChemWorldRuntime(composition=build_aqueous_headspace_world(spec, n))
            try:
                runtime.reset(seed=40)
                before = clean(physical(runtime))
                with patch(
                    "chemworld.physchem.aqueous_gas_transfer.solve_aqueous_reaction_equilibrium",
                    side_effect=ValueError("injected fast failure"),
                ):
                    reply = runtime.execute({"operation": "wait", "duration_s": 5.0})
                after = physical(runtime)
                self.check("rollback_status", reply.info["transaction_status"] != "committed", True)
                self.check(
                    "rollback_material",
                    {k: after[k] for k in ("amounts", "phases", "P", "T", "V")},
                    {k: before[k] for k in ("amounts", "phases", "P", "T", "V")},
                )
                self.check(
                    "rollback_clock_heat",
                    [after["time"], after["heat"]],
                    [before["time"], before["heat"]],
                )
                self.check(
                    "failure_reason", "injected fast failure" in json.dumps(reply.info), True
                )
                self.details.append({"unit": self.case["name"], "info": reply.info})
            finally:
                runtime.close()


def run(output):
    block = Block(output)
    cases = [
        {"name": f"{k}-{c}", "group": "profile", "kind": k, "condition": c, "comparisons": 9}
        for k in ("neutral", "acid", "desorption", "double")
        for c in (0, 1)
    ]
    cases += [
        {"name": "world-" + k, "group": "world", "kind": k, "condition": c, "comparisons": 8}
        for k, c in (("neutral", 1), ("acid", 0), ("desorption", 0), ("double", 1))
    ]
    cases += [
        {"name": "limit-" + k, "group": "limits", "kind": k, "comparisons": 4}
        for k in ("zero", "equilibrium", "refusals", "rollback")
    ]
    started = time.monotonic()
    for index, case in enumerate(cases, 1):
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
        for name, data in (
            ("comparisons.json", block.rows),
            ("units.json", block.units),
            ("details.json", block.details),
        ):
            (output / name).write_text(json.dumps(data, indent=2), encoding="utf-8")
        elapsed = time.monotonic() - started
        print(
            f"40F {index}/16 {case['name']} {'PASS' if unit['passed'] else 'FAIL'} "
            f"comparisons={len(block.rows)}/120 rate={index / max(elapsed, 0.001):.2f} units/s "
            f"ETA={(16 - index) * elapsed / index:.1f}s {error or ''}",
            flush=True,
        )
    summary = {
        "schema": "chemworld-aqueous40f-development-v1",
        "planned_units": 16,
        "units_passed": sum(u["passed"] for u in block.units),
        "planned_comparisons": 120,
        "comparisons_executed": len(block.rows),
        "comparisons_passed": sum(r["passed"] for r in block.rows),
        "unexecuted_comparisons": 120 - len(block.rows),
        "failures": [u for u in block.units if not u["passed"]],
        "failed_comparisons": [r for r in block.rows if not r["passed"]],
        "elapsed_s": time.monotonic() - started,
        "passed": all(u["passed"] for u in block.units),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if args.resume:
        runtime = ChemWorldRuntime.from_snapshot(json.loads(args.resume.read_text()))
        try:
            reply = runtime.execute({"operation": "wait", "duration_s": 5.0})
            if reply.info["transaction_status"] != "committed":
                raise ValueError(str(reply.info))
            with args.output.open("x", encoding="utf-8") as stream:
                json.dump(runtime.snapshot().to_dict(), stream)
        finally:
            runtime.close()
    else:
        result = run(args.output)
        print(
            json.dumps(
                {k: v for k, v in result.items() if k not in ("failures", "failed_comparisons")}
            )
        )
        raise SystemExit(0 if result["passed"] else 1)
