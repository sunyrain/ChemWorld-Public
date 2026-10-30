"""Fixed 20 units / 240 comparisons for harmonic RRKM development qualification."""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from unittest.mock import patch

import numpy as np
import rrkm38_cases as cases
import rrkm38_reference as oracle

from chemworld.physchem.energy_grain_master_equation import energy_grain_generator
from chemworld.physchem.gas_reactor_network import (
    GasFlowConnectionSpec,
    GasReactorNetworkSpec,
    GasReactorNodeSpec,
    advance_gas_reactor_network,
    pack_gas_reactor_network_state,
)
from chemworld.physchem.rrkm_grain_binding import (
    RRKMGrainSpec,
    compile_rrkm_grains,
    rrkm_canonical_report,
    rrkm_grain_network,
    rrkm_microcanonical_report,
)
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.scheduling import local_states
from chemworld.world.energy_grain import rrkm_grain_world


class Record:
    def __init__(self, name):
        self.name, self.checks, self.evidence = name, [], {}

    def check(self, label, actual, expected, atol=None, rtol=3e-12):
        passed = (
            actual == expected
            if atol is None
            else np.allclose(actual, expected, atol=atol, rtol=rtol)
        )
        self.checks.append(
            {
                "label": label,
                "actual": actual,
                "expected": expected,
                "atol": atol,
                "rtol": rtol if atol is not None else None,
                "passed": bool(passed),
            }
        )


def numbers(values):
    return [v for row in values for v in row if v is not None]


def counting(record, name):
    card = cases.count_card(name)
    spec = RRKMGrainSpec.from_dict(card)
    native = rrkm_microcanonical_report(spec)
    reference = oracle.micro(card)
    canonical = rrkm_canonical_report(spec, 600.0)
    expected = oracle.canonical(card, 600.0)
    rows = list(native["channels"].values())
    refs = list(reference["channels"].values())
    record.check("integer-density-counts", list(native["well"]["counts"]), reference["counts"])
    record.check(
        "integer-cumulative-counts", list(native["well"]["sum_of_states"]), reference["sums"]
    )
    record.check(
        "dimensioned-density",
        list(native["well"]["density_states_per_J_mol"]),
        [g / card["grid"]["grain_width_J_mol"] for g in reference["counts"]],
        1e-12,
    )
    actual_energy = [
        native["well"]["ground_energy_J_mol"],
        native["well"]["zero_point_added_J_mol"],
    ]
    expected_energy = [
        oracle.ground(card["well"]),
        oracle.ground(card["well"]) - card["well"]["reference_energy_J_mol"],
    ]
    for actual, source in [
        (native["well"], card["well"]),
        *[
            (r["transition_state"], c["transition_state"])
            for r, c in zip(rows, card["channels"], strict=True)
        ],
    ]:
        actual_energy.extend(q["error_J_mol"] for q in actual["quantization"])
        expected_energy.extend(
            oracle.index(q, card["grid"]) * card["grid"]["grain_width_J_mol"] - q
            for q in source["vibrational_quanta_J_mol"]
        )
    record.check("ZPE-once-and-explicit-quantization", actual_energy, expected_energy, 1e-10)
    record.check(
        "transition-state-counts",
        [list(r["transition_state"]["counts"]) for r in rows],
        [r["counts"] for r in refs],
    )
    record.check(
        "barrier-indices", [r["barrier"]["index"] for r in rows], [r["barrier"] for r in refs]
    )
    record.check(
        "microcanonical-rate-with-h-NA",
        numbers([r["rate_s"] for r in rows]),
        numbers([r["rates"] for r in refs]),
        1e-8,
    )
    record.check(
        "finite-canonical-average",
        [r["finite_grid_average_s"] for r in canonical["channels"].values()],
        [r["finite"] for r in expected["channels"].values()],
        1e-8,
    )
    record.check(
        "finite-partition-fraction",
        canonical["well_partition"]["retained_partition_fraction"],
        expected["retention"],
        1e-13,
    )
    record.check(
        "dense-formula-applicability",
        canonical["infinite_grid_formula_applicable"],
        expected["dense"],
    )
    actual = [r["infinite_grid_average_s"] for r in canonical["channels"].values()]
    wanted = [r["infinite"] for r in expected["channels"].values()]
    record.check(
        "conditional-infinite-grid-reference", actual, wanted, 1e-8 if expected["dense"] else None
    )
    record.check(
        "source-declaration-codec",
        RRKMGrainSpec.from_dict(json.loads(json.dumps(spec.to_dict()))).to_dict(),
        spec.to_dict(),
    )
    record.evidence = {
        "card": card,
        "native": native,
        "independent": reference,
        "canonical": canonical,
        "reference_canonical": expected,
    }


def inputs(card, temperature, amounts, flow=None):
    model = RRKMGrainSpec.from_dict(card)
    network = rrkm_grain_network(model)
    nodes = (GasReactorNodeSpec("cell", 1.0),)
    edges = ()
    if flow is not None:
        nodes += (GasReactorNodeSpec("feed", 1.0, chemistry_enabled=False),)
        edges = (GasFlowConnectionSpec("bath-feed", "feed", "cell", coefficient=flow),)
    reactor = GasReactorNetworkSpec(nodes, edges, temperature, "cell")
    full = {v: {s: row.get(s, 0.0) for s in network.species_ids} for v, row in amounts.items()}
    return model, network, reactor, full


def physical(record, name):
    card, temperature, initial, duration = cases.physical_card(name)
    flow = None
    amounts = {"cell": initial}
    if name == "finite-flow":
        card, temperature, amounts, flow = cases.flow_card()
    model, network, reactor, full = inputs(card, temperature, amounts, flow)
    state = pack_gas_reactor_network_state(reactor, network, full)
    result, diag = advance_gas_reactor_network(reactor, network, state, duration)
    if flow is None:
        ref = oracle.closed(card, temperature, initial, duration)
        expected = {"cell": {**full["cell"], **ref["amounts"]}}
        heat = {"cell": ref["heat_J"]}
    else:
        ref = oracle.finite_flow(card, temperature, full, flow, duration)
        expected = ref["amounts"]
        heat = ref["heat_J"]
    actual = diag["nodes"]
    names = network.species_ids
    record.check(
        "original-material-endpoint",
        [[actual[v]["amounts_mol"][s] for s in names] for v in full],
        [[expected[v][s] for s in names] for v in full],
        3e-10,
        5e-6,
    )
    record.check(
        "original-node-heat",
        [actual[v]["bath_heat_J"] for v in full],
        [heat[v] for v in full],
        5e-6,
        5e-6,
    )
    record.check("aggregate-heat-once", diag["bath_heat_J"], sum(heat.values()), 5e-6, 5e-6)
    record.check(
        "actual-inventory-pressure",
        [actual[v]["pressure_Pa"] for v in full],
        [sum(expected[v].values()) * 1000 * cases.R * temperature for v in full],
        0.02,
        5e-6,
    )
    molecular = [s for s in names if s not in {b["species_id"] for b in card["bath_gases"]}]
    record.check(
        "total-molecular-inventory",
        sum(actual[v]["amounts_mol"][s] for v in full for s in molecular),
        sum(full[v][s] for v in full for s in molecular),
        2e-11,
        0.0,
    )
    record.check(
        "original-elements",
        list(diag["element_residuals_mol"].values()),
        [0.0] * len(diag["element_residuals_mol"]),
        2e-10,
        0.0,
    )
    record.check("original-first-law", diag["energy_residual_J"], 0.0, 1e-5, 0.0)
    grains = compile_rrkm_grains(model)
    native_matrix = energy_grain_generator(
        grains,
        temperature_K=temperature,
        bath_concentrations_mol_L={
            b["species_id"]: full["cell"][b["species_id"]] for b in card["bath_gases"]
        },
    )
    reference_names, matrix, _ = oracle.generator(card, temperature, full["cell"])
    record.check(
        "independent-full-generator", native_matrix["generator_s_inverse"], matrix.tolist(), 1e-10
    )
    record.check(
        "compiled-state-density",
        [g.statistical_weight for g in grains.wells[0].grains],
        [g for g in oracle.micro(card)["counts"] if g],
        0.0,
        0.0,
    )
    record.check(
        "one-original-species-per-state", list(native_matrix["species_ids"]), reference_names
    )
    record.check("full-requested-time", result.actual_duration_s, duration, 1e-12, 0.0)
    record.check(
        "input-not-replaced",
        state.amounts_mol == pack_gas_reactor_network_state(reactor, network, full).amounts_mol,
        True,
    )
    record.evidence = {
        "card": card,
        "temperature": temperature,
        "initial": full,
        "native": diag,
        "reference": ref,
        "resource": dict(result.solver_diagnostic),
        "native_generator": native_matrix,
    }


def refinement(record, name):
    evidence = []
    for card in cases.refinement_cards(name):
        spec = RRKMGrainSpec.from_dict(card)
        micro = rrkm_microcanonical_report(spec)
        ref = oracle.micro(card)
        canonical = rrkm_canonical_report(spec, 600.0)
        expected = oracle.canonical(card, 600.0)
        width = card["grid"]["grain_width_J_mol"]
        record.check(f"{width}:counts", list(micro["well"]["counts"]), ref["counts"])
        record.check(
            f"{width}:rates",
            numbers([c["rate_s"] for c in micro["channels"].values()]),
            numbers([c["rates"] for c in ref["channels"].values()]),
            1e-8,
        )
        record.check(
            f"{width}:canonical",
            [c["finite_grid_average_s"] for c in canonical["channels"].values()],
            [c["finite"] for c in expected["channels"].values()],
            1e-8,
        )
        record.check(
            f"{width}:dense-eligibility",
            canonical["infinite_grid_formula_applicable"],
            expected["dense"],
        )
        evidence.append(
            {"card": card, "native": micro, "canonical": canonical, "reference": expected}
        )
    record.evidence = {
        "refinement": evidence,
        "claim": "model-specific finite-grid values; no universal continuum convergence",
    }


def execute(runtime, action):
    result = runtime.execute(action)
    if result.info["transaction_status"] != "committed":
        raise AssertionError(
            {k: result.info.get(k) for k in ("error_message", "outcome_diagnostics")}
        )
    return result


def world(record, kind, directory):
    card, temperature, amounts, flow = cases.flow_card()
    model, network, reactor, full = inputs(card, temperature, amounts, flow)
    runtime = ChemWorldRuntime(composition=rrkm_grain_world(model, reactor, full))
    restored = None
    try:
        runtime.reset(seed=38)
        execute(runtime, {"operation": "select_vessel", "vessel": "feed"})
        execute(runtime, {"operation": "wait", "duration_s": 0.05})
        snapshot = json.loads(json.dumps(runtime.snapshot().to_dict()))
        replay = True
        record.evidence["snapshot"] = snapshot
        if kind == "rollback":
            before = json.loads(json.dumps(runtime._state.to_dict()))
            with patch(
                "chemworld.runtime.gas_reactor_network.advance_gas_reactor_network",
                side_effect=RuntimeError("38C injected solver fault"),
            ):
                failed = runtime.execute({"operation": "wait", "duration_s": 0.1})
            after = json.loads(json.dumps(runtime._state.to_dict()))
            record.evidence.update(
                rollback_before=before, rollback_after=after, rollback_info=dict(failed.info)
            )
            for key in ("ledger", "process"):
                for field in ("cost", "risk"):
                    before[key].pop(field)
                    after[key].pop(field)
            replay = (
                before == after
                and failed.info["actual_duration_s"] == 0
                and failed.info["operation_outcome"] == "numerical_failure"
            )
        if kind == "json":
            restored = ChemWorldRuntime.from_snapshot(snapshot)
        action = {"operation": "wait", "duration_s": 0.1}
        execute(runtime, action)
        if restored:
            execute(restored, action)
            replay = restored._state.to_dict() == runtime._state.to_dict()
        if kind == "process":
            source, target = directory / "snapshot.json", directory / "continued.json"
            source.write_text(json.dumps(snapshot), encoding="utf-8")
            code = (
                "import json;from pathlib import Path;"
                "from chemworld.runtime import ChemWorldRuntime;"
                f"r=ChemWorldRuntime.from_snapshot(json.loads(Path({str(source)!r}).read_text(encoding='utf-8')));"
                f"o=r.execute({action!r});assert o.info['transaction_status']=='committed';"
                f"Path({str(target)!r}).write_text(json.dumps(r._state.to_dict()),encoding='utf-8');r.close()"
            )
            subprocess.run([sys.executable, "-c", code], check=True)
            replay = json.loads(target.read_text(encoding="utf-8")) == json.loads(
                json.dumps(runtime._state.to_dict())
            )
        states = local_states(runtime._state)
        ref = oracle.finite_flow(card, temperature, full, flow, 0.15)
        names = network.species_ids
        record.check(
            "world-original-material",
            [[states[v].species_amounts[s] for s in names] for v in full],
            [[ref["amounts"][v][s] for s in names] for v in full],
            3e-10,
            5e-6,
        )
        record.check(
            "world-local-heat",
            [states[v].thermal.vessels[v].energy_jacket_J for v in full],
            [ref["heat_J"][v] for v in full],
            5e-6,
            5e-6,
        )
        record.check(
            "world-aggregate-heat",
            runtime._state.ledger.energy_jacket_J,
            sum(ref["heat_J"].values()),
            5e-6,
            5e-6,
        )
        record.check(
            "shared-clock", list(runtime._state.vessel_elapsed_s.values()), [0.15] * 2, 1e-12, 0.0
        )
        record.check(
            "original-phase-material",
            [s.phases.phases["gas"].species_amounts_mol for s in states.values()],
            [s.species_amounts for s in states.values()],
        )
        record.check(
            "competing-products",
            [states["cell"].species_amounts[s] > 0 for s in ("P", "Q")],
            [True, True],
        )
        record.check(
            "finite-source-depleted",
            states["feed"].species_amounts["He"],
            0.02 - flow * 0.15,
            1e-12,
            0.0,
        )
        record.check("no-second-well-inventory", model.well.state_id in names, False)
        record.check(
            "one-original-accounting-owner",
            [
                s.spec.owner_id
                for v in states.values()
                for s in v.state_slots.slots
                if s.spec.state_type == "gas_network_accounting_v1"
            ],
            ["cell:gas-network"],
        )
        record.check("source-card-retained", network.metadata["rrkm_declaration"], model.to_dict())
        record.check("selected-member", runtime._state.vessel_id, "feed")
        record.check("exact-JSON-process-or-rollback", replay, True)
        record.evidence.update(final_state=runtime._state.to_dict(), reference=ref, card=card)
    finally:
        runtime.close()
        if restored:
            restored.close()


def refusals(record):
    modifications = (
        ("missing-source-version", lambda c: c["well"]["source"].update(version="")),
        (
            "negative-stable-mode",
            lambda c: c["well"].update(vibrational_quanta_J_mol=[-1000.0] * 30),
        ),
        ("zero-grid-width", lambda c: c["grid"].update(grain_width_J_mol=0.0)),
        (
            "zero-rounded-mode",
            lambda c: c["grid"].update(
                grain_width_J_mol=10000.0,
                quantization="nearest",
                maximum_quantization_error_J_mol=5000.0,
            ),
        ),
        ("exact-off-grid", lambda c: c["well"]["vibrational_quanta_J_mol"].__setitem__(0, 1100.0)),
        (
            "nearest-error-bound",
            lambda c: c["grid"].update(
                grain_width_J_mol=900.0,
                quantization="nearest",
                maximum_quantization_error_J_mol=10.0,
            ),
        ),
        (
            "wrong-baseline",
            lambda c: c["baseline"].update(excluded_internal_modes=["full NASA curve"]),
        ),
        (
            "positive-imaginary",
            lambda c: c["channels"][0].update(imaginary_frequency_cm_inverse=100.0),
        ),
        ("zero-degeneracy", lambda c: c["channels"][0].update(path_degeneracy=0)),
        (
            "negative-barrier",
            lambda c: c["channels"][0]["transition_state"].update(reference_energy_J_mol=-1e6),
        ),
        (
            "wrong-TS-mode-count",
            lambda c: c["channels"][0]["transition_state"]["vibrational_quanta_J_mol"].append(
                1000.0
            ),
        ),
        (
            "product-higher-than-source",
            lambda c: c["channels"][0]["product"].update(energy_J_mol=1e6),
        ),
    )
    for name, modify in modifications:
        card = cases.base()
        modify(card)
        try:
            compile_rrkm_grains(RRKMGrainSpec.from_dict(card))
            error = None
        except (ValueError, TypeError) as exc:
            error = str(exc)
        record.check(name, error is not None, True)
        record.evidence[name] = {"input": card, "error": error}


PROBE = r"""
import json,sys
from pathlib import Path
from dataclasses import fields
from collections.abc import Mapping
sys.path.insert(0,sys.argv[2])
import chemworld.world.energy_grain as binding
from chemworld.physchem.energy_grain_specs import EnergyGrainMasterEquationSpec
from chemworld.physchem.gas_reactor_network import (
    GasReactorNetworkSpec,GasReactorNodeSpec,pack_gas_reactor_network_state,
    advance_gas_reactor_network)
from energy_grain38_cases import declaration,initial_amounts
rows=[]
for topology in ('branch','two-well','six-grain'):
    card=declaration(topology=topology);model=EnergyGrainMasterEquationSpec.from_dict(card)
    network=binding.compile_energy_grain_network(model)
    reactor=GasReactorNetworkSpec((GasReactorNodeSpec('cell',1),),(),600,'cell')
    amounts=initial_amounts(card,temperature=600,pressure=1e4)
    state=pack_gas_reactor_network_state(reactor,network,{'cell':amounts})
    result,diag=advance_gas_reactor_network(reactor,network,state,.05)
    endpoint={f.name:(dict(v) if isinstance(v,Mapping) else v)
        for f in fields(result.final_state) for v in [getattr(result.final_state,f.name)]}
    rows.append({'topology':topology,'declaration':model.to_dict(),'network':network.to_dict(),'endpoint':endpoint,'diagnostic':diag})
Path(sys.argv[1]).write_text(json.dumps({'module_path':binding.__file__,'rows':rows},allow_nan=False),encoding='utf-8')
"""


def compatibility(record, directory, baseline):
    reports = []
    root = Path(__file__).resolve().parents[2]
    for name, source in (("old", baseline), ("current", root / "src")):
        target = directory / (name + ".json")
        subprocess.run(
            [sys.executable, "-c", PROBE, str(target), str(Path(__file__).parent.resolve())],
            cwd=directory,
            env={**os.environ, "PYTHONPATH": str(source)},
            check=True,
        )
        report = json.loads(target.read_text(encoding="utf-8"))
        if not Path(report["module_path"]).is_relative_to(source):
            raise ValueError("wrong source loaded")
        reports.append(report)
    for a, b in zip(reports[0]["rows"], reports[1]["rows"], strict=True):
        for field in ("declaration", "network", "endpoint", "diagnostic"):
            record.check(a["topology"] + ":" + field, a[field], b[field])
    record.evidence = {"baseline_commit": "68adce7b", "reports": reports}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--baseline-src", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.summary.exists():
        raise FileExistsError("preserve existing attempts")
    names = [
        *cases.COUNT_CASES,
        *cases.PHYSICAL_CASES,
        "sparse-refinement",
        "quantized-refinement",
        "world-ownership",
        "world-json",
        "world-process",
        "world-rollback",
        "refusals",
        "compatibility",
    ]
    start = time.perf_counter()
    records = []
    progress = {"active": "starting", "completed": 0}
    finished = threading.Event()

    def heartbeat():
        while not finished.wait(30):
            elapsed = time.perf_counter() - start
            completed = progress["completed"]
            rate = completed / elapsed
            eta = (20 - completed) / rate if rate else None
            print(
                json.dumps(
                    {
                        "stage": progress["active"],
                        "completed": completed,
                        "total": 20,
                        "elapsed_s": elapsed,
                        "units_per_s": rate,
                        "eta_s": eta,
                    }
                ),
                flush=True,
            )

    threading.Thread(target=heartbeat, daemon=True).start()
    for i, name in enumerate(names, 1):
        progress["active"] = name
        print(
            f"38C starting {i}/20 {name}; completed={i - 1}; "
            f"elapsed={time.perf_counter() - start:.1f}s",
            flush=True,
        )
        record = Record(name)
        directory = args.output.parent / (args.output.stem + "-" + name)
        try:
            if name in cases.COUNT_CASES:
                counting(record, name)
            elif name in cases.PHYSICAL_CASES:
                physical(record, name)
            elif name.endswith("refinement"):
                refinement(record, name)
            elif name == "refusals":
                refusals(record)
            else:
                directory.mkdir(exist_ok=False)
                if name == "compatibility":
                    compatibility(record, directory, args.baseline_src.resolve())
                else:
                    world(record, name[6:], directory)
        except Exception:
            record.evidence["exception"] = traceback.format_exc()
        passed = sum(c["passed"] for c in record.checks)
        records.append(
            {
                "unit": name,
                "passed": passed == 12
                and len(record.checks) == 12
                and "exception" not in record.evidence,
                "checks": record.checks,
                "evidence": record.evidence,
            }
        )
        elapsed = time.perf_counter() - start
        progress["completed"] = i
        print(
            f"38C {i}/20 {name}: {'PASS' if records[-1]['passed'] else 'FAIL'}; "
            f"{i / elapsed:.2f} units/s; ETA {(20 - i) * elapsed / i:.1f}s",
            flush=True,
        )
        args.output.write_text(
            json.dumps({"records": records}, indent=2, allow_nan=False), encoding="utf-8"
        )
    finished.set()
    summary = {
        "scope": "38C harmonic RRKM finite-grid development",
        "planned_units": 20,
        "planned_comparisons": 240,
        "passed_units": sum(r["passed"] for r in records),
        "completed_comparisons": sum(len(r["checks"]) for r in records),
        "passed_comparisons": sum(c["passed"] for r in records for c in r["checks"]),
        "elapsed_s": time.perf_counter() - start,
        "failures": [
            {
                "unit": r["unit"],
                "checks": [c for c in r["checks"] if not c["passed"]],
                "exception": r["evidence"].get("exception"),
            }
            for r in records
            if not r["passed"]
        ],
        "units": [
            {"unit": r["unit"], "passed": r["passed"], "checks": len(r["checks"])} for r in records
        ],
    }
    args.summary.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    args.output.write_text(
        json.dumps({"summary": summary, "records": records}, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(
        json.dumps({k: v for k, v in summary.items() if k not in {"failures", "units"}}), flush=True
    )
    return 0 if summary["passed_units"] == 20 else 1


if __name__ == "__main__":
    raise SystemExit(main())
