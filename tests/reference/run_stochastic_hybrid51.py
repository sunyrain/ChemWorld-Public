"""Fixed 24-unit analytic/distribution/world qualification; every failure retained."""

from __future__ import annotations

import argparse
import copy
import json
import math
import subprocess
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp
from stochastic_hybrid51_cases import CASES, author, shared_world
from stochastic_hybrid51_reference import simulate

from chemworld.foundation.balances import TransitionBalance
from chemworld.foundation.discrete_inventory import AVOGADRO
from chemworld.physchem.stochastic_hybrid import (
    HYBRID_ALGORITHM,
    advance_stochastic_hybrid,
    compile_hybrid,
    initial_hybrid_continuation,
)
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.composition import WorldCompositionSpec
from chemworld.world.stochastic_hybrid_world import stochastic_hybrid_world

REPLICATES = 2048
REFERENCE_REPLICATES = 65536
CDF_LIMIT = sum(math.sqrt(math.log(4 * 8 / .001) / (2 * n))
                for n in (REPLICATES, REFERENCE_REPLICATES))


class Record:
    def __init__(self, identity):
        self.identity, self.checks, self.evidence = identity, [], {}

    def equal(self, name, actual, expected=True):
        self.checks.append({"check": name, "passed": bool(actual == expected),
                            "actual": actual, "expected": expected})

    def bound(self, name, actual, limit=2e-7):
        self.checks.append({"check": name, "passed": bool(actual <= limit),
                            "actual": float(actual), "limit": float(limit)})

    def rejects(self, name, call):
        try:
            call()
        except (ValueError, ArithmeticError, RuntimeError) as exc:
            self.equal(name, True)
            self.evidence[name] = str(exc)
        else:
            self.equal(name, False)

    def payload(self):
        return {"unit": self.identity, "passed": bool(self.checks) and all(
            c["passed"] for c in self.checks), "checks": self.checks, "evidence": self.evidence}


def native(case, seed, duration=3., *, max_events=10000):
    network, spec, amounts = author(case, max_events=max_events)
    memory = initial_hybrid_continuation(network, spec, amounts, seed=seed)
    return advance_stochastic_hybrid(network, spec, amounts, memory, duration)


def analytic_unit(index):
    record = Record(f"{index + 1:02d}/analytic")
    case = dict(CASES[0])
    if index == 0:
        case.update(alpha=0., beta=0., k1=0.)
        result = native(case, 17)
        record.bound("exponential_continuous_flow", abs(result.amounts_mol["A"] /
            case["A0_mol"] - math.exp(-3 * case["k0"])))
        record.equal("no_jumps", len(result.events), 0)
    elif index == 1:
        case.update(k0=0., k1=0., beta=0.)
        result = native(case, 17)
        threshold = -math.log1p(-float(np.random.Generator(np.random.PCG64(17)).random()))
        record.equal("one_irreversible_jump", len(result.events), 1)
        record.bound("constant_hazard_inverse", abs(result.events[0]["time_s"]
                     - threshold / case["alpha"]))
        record.equal("continuous_frozen", result.amounts_mol["A"], case["A0_mol"])
    elif index == 2:
        case.update(beta=0.)
        result, reference = native(case, 17), simulate(case, 17)
        record.equal("one_switch", len(result.events), 1)
        record.bound("decaying_hazard_inverse", abs(result.events[0]["time_s"]
                     - reference["events"][0]["time_s"]))
        record.bound("changed_continuous_rate", abs(result.amounts_mol["A"] /
                     case["A0_mol"] - reference["fraction_A"]))
    elif index == 3:
        case.update(alpha=.1, beta=0., k0=1., k1=0.)
        result = native(case, 17, 30.)
        record.equal("finite_hazard_no_event", len(result.events), 0)
        record.bound("integrated_risk", abs(result.continuation["pending"]["hazards"][0]
                                           - .1 * (1 - math.exp(-30))))
        record.equal("threshold_exceeds_total_risk", result.continuation[
            "pending"]["targets"][0] > .1)
        record.evidence["analytic_no_event_probability_at_infinity"] = math.exp(-.1)
    elif index == 4:
        case.update(k0=0., k1=0., alpha=0., beta=0.)
        result = native(case, 17)
        record.equal("frozen_inventory", result.amounts_mol, author(case)[2])
        record.equal("zero_accumulated_risk", result.continuation["pending"]["hazards"], [0., 0.])
        record.equal("only_initial_poisson_targets", result.work["random_draws"], 2)
    else:
        case.update(k0=0., k1=0., alpha=2., beta=1.)
        reference = simulate(case, 17)
        stop = reference["events"][0]["time_s"] + .01
        first = native(case, 17, stop)
        network, spec, _ = author(case)
        second = advance_stochastic_hybrid(network, spec, first.amounts_mol,
                                          first.continuation, .01)
        record.equal("first_channel_dormant", first.amounts_mol["E0"], 0.)
        record.equal("retained_dormant_risk", second.continuation["pending"]["hazards"][0],
                     first.continuation["pending"]["hazards"][0])
        record.equal("retained_targets", second.continuation["pending"]["targets"],
                     first.continuation["pending"]["targets"])
        record.equal("no_redraw_while_dormant", second.work["random_draws"], 0)
        result = second
    record.evidence.update(case=case, amounts=result.amounts_mol, events=result.events,
                           pending=result.continuation["pending"], work=result.work)
    return record


def pathwise_unit(index):
    case = CASES[index]
    record = Record(f"{index + 7:02d}/feedback-pathwise/{index}")
    samples, failures = [], []
    for replicate in range(32):
        seed = 511000 + 1000 * index + replicate
        try:
            actual, reference = native(case, seed), simulate(case, seed)
        except Exception as exc:
            failures.append({"seed": seed, "replicate": replicate, "error": repr(exc),
                "failed_prefix": getattr(exc, "events", ()), "work": getattr(exc, "work", {})})
            record.equal(str(seed) + "/completed_without_exception", False)
            continue
        label = str(seed)
        actual_ids = [e["reaction_id"] for e in actual.events]
        reference_ids = [e["reaction_id"] for e in reference["events"]]
        record.equal(label + "/event_identities", actual_ids, reference_ids)
        if len(actual.events) == len(reference["events"]):
            record.bound(label + "/event_time_error", max((abs(a["time_s"] - b["time_s"])
                for a, b in zip(actual.events, reference["events"], strict=True)), default=0))
        record.bound(label + "/A_fraction_error", abs(actual.amounts_mol["A"] /
                     case["A0_mol"] - reference["fraction_A"]))
        record.equal(label + "/E1", round(actual.amounts_mol["E1"] * AVOGADRO), reference["E1"])
        record.equal(label + "/RNG", actual.continuation["rng_state"], reference["rng_state"])
        record.equal(label + "/targets", actual.continuation["pending"]["targets"],
                     reference["targets"])
        record.bound(label + "/hazards", max(abs(a - b) for a, b in zip(
            actual.continuation["pending"]["hazards"], reference["hazards"], strict=True)))
        record.bound(label + "/continuous_atoms", abs(sum(actual.amounts_mol[k]
            for k in ("A", "B")) / case["A0_mol"] - 1), 1e-9)
        record.equal(label + "/integer_atoms", sum(round(actual.amounts_mol[k] * AVOGADRO)
                                                   for k in ("E0", "E1")), 1)
        samples.append({"seed": seed, "native_events": actual.events,
                        "native_amounts": actual.amounts_mol, "reference": reference,
                        "work": actual.work})
    record.equal("all_fixed_paths_finished", len(samples), 32)
    record.equal("no_failed_paths", len(failures), 0)
    record.evidence.update(case=case, trajectories=samples, failures=failures, denominator=32)
    return record


def distribution_unit(index, started, progress):
    record = Record(f"{index + 11:02d}/distribution/{index}")
    case = CASES[index]
    reference = []
    last_print = time.monotonic()
    for replicate in range(REFERENCE_REPLICATES):
        result = simulate(case, 51300000 + 100000 * index + replicate)
        reference.append([result["fraction_A"], result["E1"]])
        now = time.monotonic()
        if now - last_print >= 10:
            print(f"analytic reference {index + 1}/4: {replicate + 1}/{REFERENCE_REPLICATES}",
                  flush=True)
            last_print = now
    samples, work, failures = [], [], []
    for replicate in range(REPLICATES):
        seed = 51200000 + 100000 * index + replicate
        try:
            result = native(case, seed)
            samples.append([result.amounts_mol["A"] / case["A0_mol"],
                            round(result.amounts_mol["E1"] * AVOGADRO)])
            work.append(result.work)
        except Exception as exc:
            failures.append({"seed": seed, "replicate": replicate, "error": repr(exc),
                "failed_prefix": getattr(exc, "events", ()), "work": getattr(exc, "work", {})})
        progress[0] += 1
        now = time.monotonic()
        if now - last_print >= 5 or replicate + 1 == REPLICATES:
            rate = progress[0] / (now - started)
            print(f"native distribution {index + 1}/4: {replicate + 1}/{REPLICATES}; "
                f"total={progress[0]}/8192 rate={rate:.2f}/s ETA={(8192-progress[0])/rate:.1f}s",
                flush=True)
            last_print = now
    record.evidence.update(case=case, native_first_seed=51200000 + 100000 * index,
        reference_first_seed=51300000 + 100000 * index,
        native_samples=samples, reference_samples=reference, work=work, failures=failures,
        native_denominator=REPLICATES, reference_denominator=REFERENCE_REPLICATES)
    record.equal("all_native_trajectories", len(samples), REPLICATES)
    record.equal("no_failed_trajectories", len(failures), 0)
    if len(samples) != REPLICATES:
        return record
    data, ref = np.asarray(samples), np.asarray(reference)
    record.equal("bounded_continuous_and_binary_integer_support", bool(
        np.all((data[:, 0] >= 0) & (data[:, 0] <= 1))
        and np.all(np.isin(data[:, 1], [0, 1]))))
    for column, label in ((0, "A_fraction"), (1, "E1")):
        distance = ks_2samp(np.round(data[:, column], 7), np.round(ref[:, column], 7)).statistic
        record.bound(label + "/full_cdf", distance, CDF_LIMIT)
    for label, transform in (("A", lambda x: x[:, 0]), ("A_squared", lambda x: x[:, 0]**2),
                             ("E1", lambda x: x[:, 1]),
                             ("A_times_E1", lambda x: x[:, 0] * x[:, 1])):
        a, b = transform(data), transform(ref)
        limit = 7 * math.sqrt(float(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))) + 2e-7
        record.bound(label + "/moment", abs(float(a.mean() - b.mean())), limit)
    return record


def world(*, max_events=10000):
    network, spec, amounts = author(max_events=max_events)
    runtime = ChemWorldRuntime(composition=stochastic_hybrid_world(network, spec, amounts, seed=17))
    runtime.reset(seed=7)
    return runtime


def execute(record, runtime, duration, label):
    result = runtime.execute({"operation": "wait", "duration_s": duration})
    record.equal(label + "/committed", result.info["transaction_status"], "committed")
    record.evidence[label] = {k: result.info.get(k) for k in (
        "transaction_status", "operation_outcome", "outcome_diagnostics", "actual_duration_s")}
    return result


def consumer_unit(index):
    record = Record(f"{index + 15:02d}/consumer")
    if index == 5:
        network, spec, amounts = author()
        memory = initial_hybrid_continuation(network, spec, amounts, seed=17)
        whole = advance_stochastic_hybrid(network, spec, amounts, memory, 3.)
        events = []
        for stop in (.123, .51, 1.25, 2.2, 3.):
            result = advance_stochastic_hybrid(network, spec, amounts, memory,
                                              stop - memory["time_s"])
            amounts, memory = result.amounts_mol, json.loads(json.dumps(result.continuation))
            events.extend(result.events)
        record.equal("event_ids", [e["reaction_id"] for e in events],
                     [e["reaction_id"] for e in whole.events])
        record.bound("event_times", max(abs(a["time_s"] - b["time_s"])
            for a, b in zip(events, whole.events, strict=True)))
        record.bound("continuous_fraction", abs(amounts["A"] - whole.amounts_mol["A"])
                     / CASES[0]["A0_mol"])
        record.equal("RNG", memory["rng_state"], whole.continuation["rng_state"])
        return record
    if index == 4:
        runtime = world(max_events=1)
        before = runtime._state.to_dict()
        result = runtime.execute({"operation": "wait", "duration_s": 3.})
        after = runtime._state.to_dict()
        record.equal("numerical_failure", result.info["operation_outcome"], "numerical_failure")
        record.equal("inventory_rollback", after["species_amounts"], before["species_amounts"])
        record.equal("RNG_hazards_rollback", after["process"]["stochastic_continuation"],
                     before["process"]["stochastic_continuation"])
        record.equal("clock_rollback", after["ledger"]["time_s"], 0.)
        record.evidence["failed_candidate"] = result.info["outcome_diagnostics"]
        return record
    runtime = (ChemWorldRuntime(composition=shared_world()) if index == 3 else world())
    if index == 3:
        runtime.reset(seed=7)
    result = execute(record, runtime, 1., "first_wait")
    if index == 0:
        record.equal("original_strategy", result.info["outcome_diagnostics"]["advance_strategy"],
                     "gillespie-direct")
        record.equal("declared_algorithm", result.info["outcome_diagnostics"]["solver"],
                     HYBRID_ALGORITHM)
        before = runtime._state.to_dict()
        runtime.observe()
        record.equal("observation_is_read_only", runtime._state.to_dict(), before)
        record.equal("single_original_clock", runtime._state.ledger.time_s, 1.)
        return record
    payload = json.loads(json.dumps(runtime.snapshot().to_dict()))
    if index in (1, 3):
        restored = ChemWorldRuntime.from_snapshot(payload)
        execute(record, runtime, 2., "original_continue")
        execute(record, restored, 2., "restored_continue")
        record.equal("exact_full_state", runtime._state.to_dict(), restored._state.to_dict())
        if index == 3:
            record.equal("all_vessel_clocks", set(runtime._state.vessel_elapsed_s.values()) == {3.})
            for local in (runtime._state, *runtime._state.inactive_vessels.values()):
                record.equal(local.vessel_id + "/process_clock",
                             local.process.stochastic_continuation["time_s"], 3.)
    else:
        execute(record, runtime, 2., "original_continue")
        with tempfile.TemporaryDirectory(prefix="chemworld-hybrid51-fresh-") as directory:
            source, output = Path(directory) / "snapshot.json", Path(directory) / "result.json"
            source.write_text(json.dumps(payload), encoding="utf-8")
            code = ("import json; from pathlib import Path; "
                "from chemworld.runtime import ChemWorldRuntime; "
                f"r=ChemWorldRuntime.from_snapshot(json.loads(Path({str(source)!r}).read_text())); "
                "x=r.execute({'operation':'wait','duration_s':2.}); "
                "assert x.info['transaction_status']=='committed'; "
                f"Path({str(output)!r}).write_text(json.dumps(r._state.to_dict()))")
            subprocess.run([sys.executable, "-c", code], check=True)
            record.equal("exact_fresh_process_state", json.loads(output.read_text()),
                         json.loads(json.dumps(runtime._state.to_dict())))
    return record


def rejection_unit(index):
    record = Record(f"{index + 21:02d}/rejection")
    network, spec, amounts = author()
    if index == 0:
        for label, broken in (("jump_changes_continuous", replace(spec,
                               stochastic_reactions=("basal",))),
                              ("ODE_changes_discrete", replace(spec, discrete_species=("E0",)))):
            record.rejects(label, lambda broken=broken: compile_hybrid(network, broken))
    elif index == 1:
        record.rejects("fractional_discrete_not_rounded", lambda: initial_hybrid_continuation(
            network, spec, {**amounts, "E0": .5 / AVOGADRO}, seed=17))
    elif index == 2:
        runtime = world()
        execute(record, runtime, 1., "valid_prefix")
        payload = runtime.snapshot().to_dict()
        for field, value in (("algorithm", "unknown"), ("discrete_species", ["E0"]),
                             ("pending", None)):
            broken = copy.deepcopy(payload)
            broken["world_state"]["process"]["stochastic_continuation"][field] = value
            record.rejects("snapshot/" + field, lambda broken=broken:
                           ChemWorldRuntime.from_snapshot(broken))
        broken = copy.deepcopy(payload)
        broken["world_state"]["process"]["stochastic_continuation"]["pending"]["targets"][0] = 0.
        record.rejects("consumed_target", lambda: ChemWorldRuntime.from_snapshot(broken))
        before = runtime._state
        for key in ("A", "E0"):
            changed = before.replace(species_amounts={**before.species_amounts, key: 0.})
            record.equal("forged_candidate/" + key,
                runtime.constitution.check_transition(before, changed, TransitionBalance()).passed,
                False)
    else:
        for boundary in ("dose", "sample"):
            payload = stochastic_hybrid_world(network, spec, amounts).to_dict()
            if boundary == "dose":
                payload["components"].append({"kind": "material_input", "parameters": {}})
            else:
                payload["components"].append({"kind": "observation",
                    "parameters": {"instruments": ["molecule_capture"]}})
                payload["task"].update(operations=["wait", "measure", "terminate"],
                                       instruments=["molecule_capture"])
            record.rejects(boundary, lambda payload=payload: ChemWorldRuntime(
                composition=WorldCompositionSpec.from_dict(payload)))
        runtime = world()
        execute(record, runtime, 1., "valid_prefix")
        before = runtime._state.to_dict()
        record.rejects("partial_reset", lambda: runtime.reset(options={"scope": "material"}))
        record.equal("failed_reset_preserves_everything", runtime._state.to_dict(), before)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.summary.exists():
        raise FileExistsError("qualification output is immutable; choose a new attempt")
    started, progress, records = time.monotonic(), [0], []
    jobs = [("analytic", lambda i=i: analytic_unit(i)) for i in range(6)]
    jobs += [("pathwise", lambda i=i: pathwise_unit(i)) for i in range(4)]
    jobs += [("distribution", lambda i=i: distribution_unit(i, started, progress))
             for i in range(4)]
    jobs += [("consumer", lambda i=i: consumer_unit(i)) for i in range(6)]
    jobs += [("rejection", lambda i=i: rejection_unit(i)) for i in range(4)]
    for index, (stage, job) in enumerate(jobs):
        try:
            record = job()
        except Exception as exc:
            record = Record(f"{index + 1:02d}/{stage}/uncaught")
            record.equal("completed_without_exception", False)
            record.evidence["error"] = repr(exc)
        payload = record.payload()
        records.append(payload)
        print(f"unit {index + 1}/24 {stage}: {'PASS' if payload['passed'] else 'FAIL'}; "
              f"elapsed={time.monotonic()-started:.1f}s", flush=True)
    elapsed = time.monotonic() - started
    checks = [c for r in records for c in r["checks"]]
    summary = {"task": "NATIVE-CONV-51B", "mode": "development", "passed": all(
        r["passed"] for r in records), "units_passed": sum(r["passed"] for r in records),
        "units_total": 24, "comparisons_passed": sum(c["passed"] for c in checks),
        "comparisons_total": len(checks), "native_distribution_attempts": progress[0],
        "native_distribution_denominator": 8192, "pathwise_denominator": 128,
        "independent_distribution_denominator": 262144, "cdf_limit": CDF_LIMIT,
        "wall_time_s": elapsed, "raw_output": str(args.output.resolve()),
        "failures": [{"unit": r["unit"], "checks": [c for c in r["checks"] if not c["passed"]],
                      "errors": r["evidence"].get("failures", r["evidence"].get("error"))}
                     for r in records if not r["passed"]],
        "units": [{"unit": r["unit"], "passed": r["passed"], "comparisons": len(r["checks"])}
                  for r in records]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "units": records}, indent=2,
                                     allow_nan=False), encoding="utf-8")
    args.summary.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(summary), flush=True)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
