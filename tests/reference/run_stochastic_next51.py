"""Prospectively fixed 29-unit distribution/replay/resource qualification."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import tempfile
import time
from math import log, sqrt
from pathlib import Path

import numpy as np
from stochastic_next51_cases import (
    DIRECT,
    NEXT,
    fixed_cases,
    independent_law,
    make_network,
    shared_world,
)

from chemworld.physchem.stochastic_kinetics import (
    EventBudgetError,
    advance_stochastic,
    initial_continuation,
)
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.stochastic_world import stochastic_world

REPLICATES = 2048
CDF_LIMIT = sqrt(log(2 * 24 / .001) / (2 * REPLICATES))


class Record:
    def __init__(self, identity):
        self.identity, self.checks, self.evidence = identity, [], {}

    def equal(self, name, actual, expected=True):
        passed = bool(actual == expected)
        self.checks.append({"check": name, "passed": passed,
                            "actual": actual, "expected": expected})

    def bound(self, name, actual, limit):
        self.checks.append({"check": name, "passed": bool(actual <= limit),
                            "actual": float(actual), "limit": float(limit)})

    def payload(self):
        return {"unit": self.identity, "passed": all(c["passed"] for c in self.checks),
                "checks": self.checks, "evidence": self.evidence}


def distribution_unit(index, algorithm, case, started, progress):
    record = Record(f"{index + 1:02d}/{algorithm}/{case['kind']}/{case['copies']}")
    network = make_network(case["kind"], algorithm=algorithm)
    support, probability = independent_law(**case)
    species = network.species_ids
    samples, failures, work = [], [], []
    first_seed = 51000000 + 100000 * index
    last_print = time.monotonic()
    for replicate in range(REPLICATES):
        try:
            result = advance_stochastic(
                network, {"A": case["copies"]},
                initial_continuation(first_seed + replicate, algorithm=algorithm), case["duration"],
            )
            samples.append([result.counts[key] for key in species])
            work.append({**result.work, "events_committed": len(result.events)})
        except Exception as exc:
            failures.append({"replicate": replicate, "seed": first_seed + replicate,
                             "error": repr(exc)})
        progress[0] += 1
        now = time.monotonic()
        if now - last_print >= 5 or replicate + 1 == REPLICATES:
            rate = progress[0] / (now - started)
            print(f"distribution {index + 1}/24 replicate {replicate + 1}/{REPLICATES}; "
                  f"total={progress[0]}/49152 rate={rate:.1f}/s "
                  f"ETA={(49152 - progress[0]) / rate:.1f}s", flush=True)
            last_print = now
    record.evidence.update(case=case, algorithm=algorithm, first_seed=first_seed,
                           samples=samples, work=work, failures=failures,
                           support=support.tolist(), probability=probability.tolist())
    record.equal("all_trajectories_finished", len(samples), REPLICATES)
    record.equal("no_solver_failures", len(failures), 0)
    if len(samples) != REPLICATES:
        return record
    data = np.asarray(samples, dtype=int)
    locations = {tuple(row): i for i, row in enumerate(support)}
    valid = all(tuple(row) in locations for row in data)
    record.equal("exact_integer_nonnegative_conserved_support", valid)
    if not valid:
        return record
    histogram = np.bincount([locations[tuple(row)] for row in data], minlength=len(support))
    distance = float(np.max(np.abs(np.cumsum(histogram) / REPLICATES - np.cumsum(probability))))
    record.bound("full_distribution_cdf", distance, CDF_LIMIT)
    means = probability @ support
    centered_support = support - means
    centered_data = data - means
    for i, key in enumerate(species):
        variance = float(probability @ centered_support[:, i] ** 2)
        record.bound(f"mean/{key}", abs(float(data[:, i].mean() - means[i])),
                     7 * sqrt(variance / REPLICATES) + 1e-11)
        for j in range(i, len(species)):
            reference_values = centered_support[:, i] * centered_support[:, j]
            expected = float(probability @ reference_values)
            moment_variance = float(probability @ (reference_values - expected) ** 2)
            actual = float(np.mean(centered_data[:, i] * centered_data[:, j]))
            record.bound(f"centered_second/{key}/{species[j]}", abs(actual - expected),
                         7 * sqrt(moment_variance / REPLICATES) + 1e-11)
    return record


def replay_unit():
    record = Record("25/exact-segmented-json-replay")
    network = make_network()
    for seed in range(510100, 510112):
        original = initial_continuation(seed, algorithm=NEXT)
        whole = advance_stochastic(network, {"A": 2}, original, 10)
        counts, continuation, events = {"A": 2}, original, []
        for end in range(1, 11):
            part = advance_stochastic(network, counts, continuation, end - continuation["time_s"])
            counts, continuation = part.counts, json.loads(json.dumps(part.continuation))
            events.extend(part.events)
        record.equal(f"{seed}/events", tuple(events), whole.events)
        record.equal(f"{seed}/counts", counts, whole.counts)
        record.equal(f"{seed}/continuation", continuation, whole.continuation)
        record.equal(f"{seed}/input_unchanged", original,
                     initial_continuation(seed, algorithm=NEXT))
    return record


def execute(record, runtime, label, action):
    result = runtime.execute(action)
    record.equal(label, result.info["transaction_status"], "committed")
    if result.info["transaction_status"] != "committed":
        record.evidence[label] = result.info
    return result


def world_unit():
    record = Record("26/original-world-and-shared-clock")
    for seed in range(510200, 510204):
        runtime = ChemWorldRuntime(composition=stochastic_world(
            make_network(), {"A": 100}, seed=seed,
        ))
        runtime.reset(seed=7)
        result = execute(record, runtime, f"{seed}/wait", {"operation": "wait", "duration_s": 1})
        record.equal(f"{seed}/actual_algorithm", result.info["outcome_diagnostics"]["solver"], NEXT)
        before = runtime._state.to_dict()
        runtime.observe()
        record.equal(f"{seed}/observation_read_only", runtime._state.to_dict(), before)
        payload = json.loads(json.dumps(runtime.snapshot().to_dict()))
        restored = ChemWorldRuntime.from_snapshot(payload)
        action = {"operation": "wait", "duration_s": 2}
        execute(record, runtime, f"{seed}/continue", action)
        execute(record, restored, f"{seed}/restored_continue", action)
        record.equal(f"{seed}/json_resume", restored._state.to_dict(), runtime._state.to_dict())
        with tempfile.TemporaryDirectory(prefix="chemworld-next51-process-") as directory:
            source, output = Path(directory) / "snapshot.json", Path(directory) / "result.json"
            source.write_text(json.dumps(payload), encoding="utf-8")
            code = (
                "import json; from pathlib import Path; "
                "from chemworld.runtime import ChemWorldRuntime; "
                f"r=ChemWorldRuntime.from_snapshot(json.loads(Path({str(source)!r}).read_text())); "
                f"r.execute({action!r}); "
                f"Path({str(output)!r}).write_text(json.dumps(r._state.to_dict()))"
            )
            subprocess.run([sys.executable, "-c", code], check=True)
            record.equal(f"{seed}/independent_process", json.loads(output.read_text()),
                         json.loads(json.dumps(runtime._state.to_dict())))
        lab = ChemWorldRuntime(composition=shared_world(seed))
        lab.reset(seed=7)
        execute(record, lab, f"{seed}/shared_wait", {"operation": "wait", "duration_s": 1})
        record.equal(f"{seed}/shared_time", lab._state.ledger.time_s, 1)
        record.equal(f"{seed}/all_local_times", set(lab._state.vessel_elapsed_s.values()) == {1})
        record.equal(f"{seed}/process_time",
                     lab._state.process.stochastic_continuation["time_s"], 1)
        lab_copy = ChemWorldRuntime.from_snapshot(json.loads(json.dumps(lab.snapshot().to_dict())))
        for label, target in (("shared_continue", lab), ("shared_resume", lab_copy)):
            execute(record, target, f"{seed}/{label}", action)
        record.equal(f"{seed}/shared_replay", lab._state.to_dict(), lab_copy._state.to_dict())
    return record


def intervention_unit():
    record = Record("27/intervention-zero-and-dormant-clocks")
    network = make_network()
    initial = initial_continuation(510300, algorithm=NEXT)
    for counts, duration in (({"A": 2}, 0), ({}, 10)):
        result = advance_stochastic(network, counts, initial, duration)
        record.equal(f"zero/{duration}/rng", result.continuation["rng_state"], initial["rng_state"])
        record.equal(f"zero/{duration}/draws", result.work["random_draws"], 0)
    before = advance_stochastic(network, {"A": 20}, initial, 1)
    counts = {**before.counts, "A": before.counts["A"] + 10}
    reset = copy.deepcopy(before.continuation)
    reset.update(pending=None, signature=None)
    left = advance_stochastic(network, counts, before.continuation, 1)
    right = advance_stochastic(network, counts, reset, 1)
    record.equal("inventory_change_invalidates_only_clocks", left == right)
    # Start with the two competing B exits. The loser stores remaining hazard
    # while B is empty, then resumes it on the next A->B or C->B event.
    network = make_network("multiscale")
    prepared = advance_stochastic(network, {"B": 1}, initial, 1e-12)
    times = prepared.continuation["pending"]["times_s"]
    when, fired = min((t, i) for i, t in enumerate(times) if t is not None)
    other = 2 if fired == 1 else 1
    old_propensity = prepared.continuation["pending"]["propensities"][other]
    disabled = advance_stochastic(network, prepared.counts, prepared.continuation, when - 1e-12)
    pending = disabled.continuation["pending"]
    hazard = old_propensity * (times[other] - when)
    record.equal("competing_channel_disabled", pending["times_s"][other], None)
    record.equal("unspent_internal_hazard", pending["dormant_hazards"][other], hazard)
    revival = min(t for t in pending["times_s"] if t is not None)
    revived = advance_stochastic(network, disabled.counts, disabled.continuation, revival - when)
    record.equal("revived_clock", revived.continuation["pending"]["times_s"][other],
                 revival + hazard / old_propensity)
    return record


def rollback_unit():
    record = Record("28/budget-rollback-and-corruption")
    network = make_network(max_events=1)
    initial = initial_continuation(510400, algorithm=NEXT)
    counts = {"A": 100}
    failed = False
    try:
        advance_stochastic(network, counts, initial, 10)
    except EventBudgetError:
        failed = True
    record.equal("pure_budget_failure", failed)
    record.equal("pure_input_counts", counts, {"A": 100})
    record.equal("pure_input_rng", initial, initial_continuation(510400, algorithm=NEXT))
    runtime = ChemWorldRuntime(composition=stochastic_world(network, counts, seed=510400))
    runtime.reset(seed=7)
    before = runtime._state.to_dict()
    result = runtime.execute({"operation": "wait", "duration_s": 10})
    record.equal("world_numerical_failure", result.info["operation_outcome"], "numerical_failure")
    after = runtime._state.to_dict()
    record.equal("rollback_counts", after["species_amounts"], before["species_amounts"])
    record.equal("rollback_process", after["process"]["stochastic_continuation"],
                 before["process"]["stochastic_continuation"])
    record.equal("rollback_time", after["ledger"]["time_s"], 0)
    network = make_network()
    prepared = advance_stochastic(network, counts, initial, 1e-12)
    prepared.continuation["pending"]["times_s"][0] = -1
    refused = False
    try:
        advance_stochastic(network, prepared.counts, prepared.continuation, 0)
    except ValueError:
        refused = True
    record.equal("corrupt_pending_rejected", refused)
    return record


def resource_unit():
    record = Record("29/sparse-propensity-work")
    evaluations, events = {}, {}
    for algorithm in (DIRECT, NEXT):
        network = make_network("sparse", algorithm=algorithm)
        rows = []
        started = time.monotonic()
        for seed in range(510500, 510532):
            result = advance_stochastic(
                network, {f"A{i}": 20 for i in range(16)},
                initial_continuation(seed, algorithm=algorithm), 1,
            )
            rows.append({"seed": seed, **result.work, "events": len(result.events)})
        evaluations[algorithm] = sum(row["propensity_evaluations"] for row in rows)
        events[algorithm] = sum(row["events"] for row in rows)
        record.evidence[algorithm] = {"replicates": rows, "elapsed_s": time.monotonic() - started,
                                     "propensity_evaluations": evaluations[algorithm],
                                     "events": events[algorithm]}
    relative = (evaluations[NEXT] / events[NEXT]) / (evaluations[DIRECT] / events[DIRECT])
    record.bound("evaluations_per_event_relative_to_direct", relative, .25)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()
    output, summary = Path(args.output), Path(args.summary)
    if output.exists() or summary.exists():
        raise FileExistsError("retain previous attempts; choose new evidence paths")
    started, progress, records = time.monotonic(), [0], []
    for algorithm in (DIRECT, NEXT):
        for case in fixed_cases():
            records.append(distribution_unit(len(records), algorithm, case, started, progress))
    for callback in (replay_unit, world_unit, intervention_unit, rollback_unit, resource_unit):
        record = Record(f"{len(records) + 1:02d}/{callback.__name__}")
        try:
            record = callback()
        except Exception as exc:
            record.equal("consumer_completed", False)
            record.evidence["exception"] = repr(exc)
        records.append(record)
        print(f"consumer {len(records)}/29: {record.identity} "
              f"passed={all(c['passed'] for c in record.checks)}", flush=True)
    payload = {"schema": "native-next-reaction51-development-v1", "expected_units": 29,
               "expected_statistical_trajectories": 49152,
               "statistical_trajectories_attempted": progress[0],
               "elapsed_s": time.monotonic() - started,
               "units": [record.payload() for record in records]}
    checks = [check for record in records for check in record.checks]
    payload.update(units_passed=sum(unit["passed"] for unit in payload["units"]),
                   comparisons=len(checks), comparisons_passed=sum(c["passed"] for c in checks))
    payload["failures"] = [{"unit": record.identity, **check} for record in records
                           for check in record.checks if not check["passed"]]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    compact = copy.deepcopy(payload)
    for unit in compact["units"]:
        evidence = unit["evidence"]
        for key in ("samples", "work", "support", "probability"):
            evidence.pop(key, None)
        for check in unit["checks"]:
            if check["passed"] and isinstance(check.get("actual"), dict | list | tuple):
                check["actual"] = "exact match; full values in external evidence"
                check.pop("expected", None)
    compact["full_evidence"] = str(output.resolve())
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(compact, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({key: payload[key] for key in (
        "units_passed", "expected_units", "comparisons_passed", "comparisons", "elapsed_s",
    )}), flush=True)
    return 0 if payload["units_passed"] == 29 else 1


if __name__ == "__main__":
    raise SystemExit(main())
