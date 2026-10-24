"""Fixed 30-unit/192-comparison explicit tau-leap development block."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, replace
from math import log, sqrt
from pathlib import Path

import numpy as np
from stochastic_tau51_cases import (
    CASES,
    DURATION,
    POLICIES,
    REPLICATES,
    convert_world,
    make_network,
)
from stochastic_tau51_reference import Law

from chemworld.examples.native_domains import mixed_clock_world
from chemworld.foundation.balances import TransitionBalance
from chemworld.physchem.stochastic_kinetics import (
    AVOGADRO,
    TAU_ALGORITHM,
    EventBudgetError,
    StochasticSpec,
    _signature,
    advance_stochastic,
    initial_continuation,
    mol_to_counts,
)
from chemworld.physchem.stochastic_tau_leap import TauLeapPolicy, advance_tau_leap
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.biochemical_instruments import capture_outlet, capture_plan
from chemworld.world.biochemical_worlds import biochemical_world
from chemworld.world.stochastic_world import stochastic_world

DIRECT = "gillespie-direct-v1"
CDF_SAMPLING = sqrt(log(2 * 24 / 0.001) / (2 * REPLICATES))
CDF_ERROR = (0.40, 0.18, 0.08)
MEAN_ERROR = (0.08, 0.025, 0.009)
COV_ERROR = (0.40, 0.15, 0.06)
SLOTS = (
    ("completed", "zero_exceptions", "exact_support", "distribution_cdf", "means", "covariance"),
    ("counts", "continuation", "events", "batches", "inputs", "time", "batching", "hold"),
    (
        "committed",
        "algorithm",
        "firings",
        "observations",
        "json",
        "process",
        "shared_time",
        "shared_resume",
    ),
    (
        "dose",
        "process_stream",
        "capture",
        "sampling_stream",
        "reset_pending",
        "reset_stream",
        "time",
        "atoms",
    ),
    (
        "zero_duration",
        "zero_propensity",
        "exhausted",
        "critical",
        "rejected",
        "no_clip",
        "reject_budget",
        "third_order",
    ),
    (
        "steps",
        "firings",
        "world_inventory",
        "world_process",
        "world_time",
        "pending",
        "policy",
        "time",
    ),
    (
        "completed",
        "batching",
        "firings",
        "relative_work",
        "step_bound",
        "work_accounting",
        "stiff_failure",
        "stiff_immutable",
    ),
)
WORK_KEYS = (
    "propensity_evaluations",
    "random_draws",
    "poisson_draws",
    "uniform_draws",
    "steps_attempted",
    "rejected_leaps",
    "accepted_leaps",
    "ssa_steps",
    "firings_planned",
    "firings_committed",
    "leap_firings",
    "pending_firings",
)


def initial(seed):
    return initial_continuation(seed, algorithm=TAU_ALGORITHM)


def refused(callback, expected=(ValueError, ArithmeticError, EventBudgetError)):
    try:
        callback()
    except expected as exc:
        return str(exc)
    return None


def run_action(runtime, action, history):
    result = runtime.execute(action)
    history.append({"action": action, "info": result.info})
    if result.info["transaction_status"] != "committed":
        raise RuntimeError(f"native action failed: {action}: {result.info}")
    return result


def distribution(unit, case, policy_index, progress):
    law, network = Law(case, DURATION), make_network(case, POLICIES[policy_index])
    samples, failures, work = [], [], []
    for replicate in range(REPLICATES):
        seed = 513000000 + 10000 * unit + replicate
        try:
            result = advance_stochastic(network, case["initial"], initial(seed), DURATION)
            samples.append([result.counts[k] for k in case["species"]])
            work.append(
                {
                    "seed": seed,
                    **{k: result.work[k] for k in WORK_KEYS},
                    "output_hold_s": result.work["output_hold_s"],
                }
            )
        except Exception as exc:
            failures.append({"seed": seed, "exception": repr(exc)})
        progress()
    evidence = {
        "case": case,
        "policy": POLICIES[policy_index].to_dict(),
        "samples": samples,
        "work": work,
        "trajectory_failures": failures,
        "work_totals": {k: sum(row[k] for row in work) for k in WORK_KEYS},
        "expected_mean": law.mean.tolist(),
        "expected_covariance": law.covariance.tolist(),
    }
    checks = {"completed": len(samples) == REPLICATES, "zero_exceptions": not failures}
    if len(samples) != REPLICATES:
        return checks, evidence
    rows = np.asarray(samples)
    checks["exact_support"] = law.valid(rows)
    if not checks["exact_support"]:
        return checks, evidence
    mean = rows.mean(axis=0)
    covariance = np.atleast_2d(np.cov(rows, rowvar=False, ddof=1))
    cdf = law.cdf_distance(rows)
    mean_bounds = 7 * np.sqrt(np.diag(law.covariance) / REPLICATES)
    mean_bounds += MEAN_ERROR[policy_index] * max(1, law.total)
    covariance_bounds = 7 * np.sqrt(np.maximum(law.product_variance, 0.0) / REPLICATES)
    covariance_bounds += COV_ERROR[policy_index] * np.maximum(
        1.0, np.sqrt(np.outer(np.diag(law.covariance), np.diag(law.covariance)))
    )
    checks.update(
        distribution_cdf=cdf <= CDF_SAMPLING + CDF_ERROR[policy_index],
        means=bool(np.all(abs(mean - law.mean) <= mean_bounds)),
        covariance=bool(np.all(abs(covariance - law.covariance) <= covariance_bounds)),
    )
    evidence.update(
        observed_mean=mean.tolist(),
        observed_covariance=covariance.tolist(),
        cdf_error=cdf,
        cdf_limit=CDF_SAMPLING + CDF_ERROR[policy_index],
        mean_errors=abs(mean - law.mean).tolist(),
        mean_limits=mean_bounds.tolist(),
        covariance_errors=abs(covariance - law.covariance).tolist(),
        covariance_limits=covariance_bounds.tolist(),
    )
    return checks, evidence


def replay():
    checks = dict.fromkeys(SLOTS[1], True)
    evidence = {"paths": []}
    network = make_network()
    for seed in range(513900000, 513900004):
        source, counts = initial(seed), {"A": 1000}
        whole = advance_stochastic(network, counts, source, 3.0)
        state, events, batches, holds = source, [], [], []
        for endpoint in (0.017, 0.311, 0.973, 1.007, 2.319, 3.0):
            part = advance_stochastic(network, counts, state, endpoint - state["time_s"])
            counts, state = part.counts, json.loads(json.dumps(part.continuation))
            events.extend(part.events)
            batches.extend(part.batches)
            holds.append(part.work["output_hold_s"])
        checks["counts"] &= counts == whole.counts
        checks["continuation"] &= state == whole.continuation
        checks["events"] &= tuple(events) == whole.events
        checks["batches"] &= tuple(batches) == whole.batches
        checks["inputs"] &= source == initial(seed)
        checks["time"] &= state["time_s"] == 3.0
        checks["batching"] &= bool(batches) and whole.work["leap_firings"] > len(batches)
        checks["hold"] &= all(0 <= h <= POLICIES[1].max_tau_s + 1e-14 for h in holds)
        evidence["paths"].append(
            {
                "seed": seed,
                "counts": counts,
                "hold_s": holds,
                "batches": [asdict(b) for b in batches],
            }
        )
    return checks, evidence


def world():
    history = []
    runtime = ChemWorldRuntime(
        composition=stochastic_world(make_network(), {"A": 1000}, seed=513900010)
    )
    runtime.reset(seed=7)
    first = run_action(runtime, {"operation": "wait", "duration_s": 1}, history)
    before = runtime._state.to_dict()
    runtime.observe()
    checks = {
        "observations": runtime._state.to_dict() == before,
        "algorithm": first.info["outcome_diagnostics"]["solver"] == TAU_ALGORITHM,
        "firings": first.info["outcome_diagnostics"]["events_executed"]
        == mol_to_counts(runtime._state.species_amounts)["B"],
    }
    payload = json.loads(json.dumps(runtime.snapshot().to_dict()))
    restored = ChemWorldRuntime.from_snapshot(payload)
    action = {"operation": "wait", "duration_s": 2}
    run_action(runtime, action, history)
    run_action(restored, action, history)
    checks["json"] = restored._state.to_dict() == runtime._state.to_dict()
    with tempfile.TemporaryDirectory(prefix="chemworld-tau51-") as directory:
        source, output = Path(directory) / "snapshot.json", Path(directory) / "state.json"
        source.write_text(json.dumps(payload), encoding="utf-8")
        code = (
            "import json; from pathlib import Path; "
            "from chemworld.runtime import ChemWorldRuntime; "
            f"r=ChemWorldRuntime.from_snapshot(json.loads(Path({str(source)!r}).read_text())); "
            f"r.execute({action!r}); "
            f"Path({str(output)!r}).write_text(json.dumps(r._state.to_dict()))"
        )
        subprocess.run([sys.executable, "-c", code], check=True)
        checks["process"] = json.loads(output.read_text()) == json.loads(
            json.dumps(runtime._state.to_dict())
        )
    lab = ChemWorldRuntime(composition=convert_world(mixed_clock_world(), 513900010))
    lab.reset(seed=7)
    run_action(lab, {"operation": "wait", "duration_s": 1}, history)
    checks["shared_time"] = lab._state.ledger.time_s == 1 and set(
        lab._state.vessel_elapsed_s.values()
    ) == {1}
    lab_copy = ChemWorldRuntime.from_snapshot(json.loads(json.dumps(lab.snapshot().to_dict())))
    run_action(lab, action, history)
    run_action(lab_copy, action, history)
    checks["shared_resume"] = lab_copy._state.to_dict() == lab._state.to_dict()
    checks["committed"] = all(row["info"]["transaction_status"] == "committed" for row in history)
    return checks, {"history": history, "final_ledger": runtime._state.ledger.to_dict()}


def interventions():
    history = []
    lab = ChemWorldRuntime(composition=convert_world(biochemical_world("reversible"), 513900020))
    lab.reset(seed=7)
    run_action(lab, {"operation": "wait", "duration_s": 1}, history)
    before = lab._state
    old = copy.deepcopy(before.process.stochastic_continuation)
    run_action(
        lab, {"operation": "add_component", "component": 0, "amount_mol": 10 / AVOGADRO}, history
    )
    after = lab._state.process.stochastic_continuation
    old_counts, new_counts = (
        mol_to_counts(before.species_amounts),
        mol_to_counts(lab._state.species_amounts),
    )
    checks = {
        "dose": sum(new_counts.values()) - sum(old_counts.values()) == 10
        and after["pending"] is None,
        "process_stream": after["rng_state"] == old["rng_state"]
        and after["events_committed"] == old["events_committed"],
    }
    left = ChemWorldRuntime.from_snapshot(json.loads(json.dumps(lab.snapshot().to_dict())))
    run_action(left, {"operation": "measure", "instrument": "molecule_counter"}, history)
    pre_capture = lab._state
    selected, _ = capture_plan(pre_capture)
    run_action(lab, {"operation": "measure", "instrument": "molecule_capture"}, history)
    run_action(left, {"operation": "measure", "instrument": "molecule_capture"}, history)
    captured = mol_to_counts(lab._state.species_amounts)
    checks["capture"] = all(type(n) is int and n >= 0 for n in captured.values()) and sum(
        captured.values()
    ) <= sum(new_counts.values())
    checks["sampling_stream"] = (
        lab._state.species_amounts == left._state.species_amounts
        and lab._state.process.stochastic_continuation
        == left._state.process.stochastic_continuation
    )
    balance = TransitionBalance(capture_outlet(pre_capture, {}, {}))
    checks["atoms"] = lab.constitution.check_transition(pre_capture, lab._state, balance).passed
    checks["atoms"] &= all(new_counts[k] == captured[k] + selected[k] for k in new_counts)
    # The original mixed-clock recipe intentionally offers queued analysis, not
    # single-vessel dosing. Exercise its real material lifecycle separately.
    lab = ChemWorldRuntime(composition=convert_world(mixed_clock_world(), 513900020))
    lab.reset(seed=7)
    run_action(lab, {"operation": "wait", "duration_s": 1}, history)
    old = copy.deepcopy(lab._state.process.stochastic_continuation)
    when = lab._state.ledger.time_s
    lab.reset(options={"scope": "material", "vessel_ids": [lab._state.vessel_id]})
    current = lab._state.process.stochastic_continuation
    checks["reset_pending"] = current["pending"] is None and current["signature"] is None
    checks["reset_stream"] = all(
        current[k] == old[k] for k in ("rng_state", "sampling_rng_state", "events_committed")
    )
    run_action(lab, {"operation": "wait", "duration_s": 1}, history)
    checks["time"] = (
        lab._state.ledger.time_s == when + 1 == lab._state.process.stochastic_continuation["time_s"]
    )
    return checks, {
        "history": history,
        "dose_before": old_counts,
        "dose_after": new_counts,
        "after_capture": captured,
        "final_time": lab._state.ledger.time_s,
    }


class ForcedPoisson:
    """Fixed numerical proposal fixture, never a statistical trajectory source."""

    def __init__(self, values):
        self.generator = np.random.default_rng(513900030)
        self.bit_generator = self.generator.bit_generator
        self.values = iter(values)

    def poisson(self, mean):
        return next(self.values)

    def random(self):
        return self.generator.random()


def forced_negative(*, reject_limit=32):
    policy = replace(POLICIES[1], max_rejections_per_leap=reject_limit)
    network = make_network(policy=policy)
    spec, counts, state = (
        StochasticSpec(**network.stochastic),
        {"A": 1000, "B": 0},
        initial(513900030),
    )
    return advance_tau_leap(
        network.stochastic_channels,
        counts,
        state,
        spec,
        ForcedPoisson([1001, 14]),
        0.001,
        _signature(counts, network.stochastic_channels, spec),
    )


def boundaries():
    source = initial(513900030)
    zero = advance_stochastic(make_network(), {"A": 1000}, source, 0)
    empty = advance_stochastic(make_network(), {"B": 1000}, source, 1)
    low = make_network(CASES[0])
    exact = advance_stochastic(low, {"A": 3}, source, 1)
    direct = advance_stochastic(
        make_network(CASES[0], algorithm=DIRECT), {"A": 3}, initial_continuation(513900030), 1
    )
    forced = forced_negative()
    third = advance_stochastic(make_network(CASES[7]), {"A": 150}, source, 1)
    checks = {
        "zero_duration": zero.continuation["rng_state"] == source["rng_state"]
        and zero.work["random_draws"] == 0,
        "zero_propensity": empty.continuation["rng_state"] == source["rng_state"]
        and empty.work["random_draws"] == 0,
        "exhausted": empty.counts == {"A": 0, "B": 1000},
        "critical": exact.events == direct.events
        and exact.counts == direct.counts
        and not exact.batches,
        "rejected": forced.work["rejected_leaps"] == 1,
        "no_clip": forced.counts == {"A": 1000, "B": 0}
        and forced.continuation["pending"]["firings"] == [14],
        "reject_budget": bool(refused(lambda: forced_negative(reject_limit=1))),
        "third_order": third.counts["A"] + 3 * third.counts["C"] == 150
        and min(third.counts.values()) >= 0,
    }
    return checks, {
        "forced_work": forced.work,
        "forced_pending": forced.continuation["pending"],
        "third_counts": third.counts,
    }


def rollback():
    source, counts = initial(513900040), {"A": 1000}
    before = copy.deepcopy(source)
    step_network = make_network(policy=replace(POLICIES[1], max_steps_per_advance=1))
    event_network = make_network(max_events=1)
    errors = {
        "steps": refused(lambda: advance_stochastic(step_network, counts, source, 10)),
        "firings": refused(lambda: advance_stochastic(event_network, counts, source, 10)),
    }
    checks = {
        name: bool(error) and source == before and counts == {"A": 1000}
        for name, error in errors.items()
    }
    runtime = ChemWorldRuntime(composition=stochastic_world(event_network, counts, seed=513900040))
    runtime.reset(seed=7)
    before_world = runtime._state.to_dict()
    result = runtime.execute({"operation": "wait", "duration_s": 10})
    after = runtime._state.to_dict()
    checks["world_inventory"] = (
        after["species_amounts"] == before_world["species_amounts"]
        and result.info["operation_outcome"] == "numerical_failure"
    )
    checks["world_process"] = (
        after["process"]["stochastic_continuation"]
        == before_world["process"]["stochastic_continuation"]
    )
    checks["world_time"] = after["ledger"]["time_s"] == before_world["ledger"]["time_s"]
    prepared = advance_stochastic(make_network(), counts, source, 0.001)
    mutations = (
        ("signature", "bad"),
        ("policy", {}),
        ("end_time_s", 0.0),
        ("firings", [-1]),
        ("firings", [1001]),
        ("algorithm", DIRECT),
        ("channel_ids", ["wrong"]),
        ("extra", 1),
    )
    pending_errors = []
    for key, value in mutations:
        altered = copy.deepcopy(prepared.continuation)
        altered["pending"][key] = value
        pending_errors.append(
            {
                "field": key,
                "value": value,
                "error": refused(
                    lambda altered=altered: advance_stochastic(
                        make_network(), prepared.counts, altered, 0
                    )
                ),
            }
        )
    checks["pending"] = all(row["error"] for row in pending_errors)
    policy_errors = [
        refused(lambda: TauLeapPolicy(epsilon=0)),
        refused(lambda: TauLeapPolicy.from_dict({"extra": 1})),
        refused(lambda: StochasticSpec(1.0, algorithm=TAU_ALGORITHM)),
    ]
    checks["policy"] = all(policy_errors)
    enormous = initial_continuation(513900040, 1e20, algorithm=TAU_ALGORITHM)
    errors["time"] = refused(lambda: advance_stochastic(make_network(), counts, enormous, 0.001))
    checks["time"] = bool(errors["time"])
    return checks, {
        "errors": errors,
        "pending_errors": pending_errors,
        "policy_errors": policy_errors,
        "world_failure": result.info,
    }


def resource():
    results = {}
    for algorithm in (TAU_ALGORITHM, DIRECT):
        started, rows = time.monotonic(), []
        network = make_network(algorithm=algorithm)
        for seed in range(513900050, 513900058):
            result = advance_stochastic(
                network, {"A": 10000}, initial_continuation(seed, algorithm=algorithm), 1
            )
            rows.append(
                {
                    "seed": seed,
                    "work": result.work,
                    "counts": result.counts,
                    "firings": result.continuation["events_committed"],
                    "batches": [asdict(b) for b in result.batches],
                }
            )
        results[algorithm] = {"rows": rows, "elapsed_s": time.monotonic() - started}
    tau, exact = results[TAU_ALGORITHM]["rows"], results[DIRECT]["rows"]
    relative = (
        sum(r["work"]["propensity_evaluations"] for r in tau) / sum(r["firings"] for r in tau)
    ) / (sum(r["work"]["propensity_evaluations"] for r in exact) / sum(r["firings"] for r in exact))
    stiff_case = {**CASES[2], "rows": (("ab", "A => B", 1e6, (1,)), ("ba", "B => A", 1e6, (1,)))}
    stiff = make_network(stiff_case, replace(POLICIES[1], max_steps_per_advance=1))
    source, counts = initial(513900050), {"A": 1000}
    saved = copy.deepcopy(source)
    error = refused(lambda: advance_stochastic(stiff, counts, source, 1))
    checks = {
        "completed": len(tau) == len(exact) == 8,
        "batching": all(
            r["batches"] and r["work"]["leap_firings"] > len(r["batches"]) for r in tau
        ),
        "firings": all(r["firings"] == r["counts"]["B"] for r in (*tau, *exact)),
        "relative_work": relative <= 0.35,
        "step_bound": all(
            b["end_time_s"] <= b["start_time_s"] + POLICIES[1].max_tau_s
            for r in tau
            for b in r["batches"]
        ),
        "work_accounting": all(
            r["work"]["firings_planned"]
            == r["work"]["firings_committed"] + r["work"]["pending_firings"]
            for r in tau
        ),
        "stiff_failure": bool(error),
        "stiff_immutable": source == saved and counts == {"A": 1000},
    }
    return checks, {"methods": results, "relative_propensity_work": relative, "stiff_error": error}


def make_unit(identity, slots, callback):
    try:
        checks, evidence = callback()
    except Exception as exc:
        checks, evidence = {}, {"exception": repr(exc)}
    if set(checks) - set(slots):
        raise ValueError("runner produced an undeclared comparison")
    comparisons = [
        {"name": name, "passed": bool(checks.get(name, False)), "completed": name in checks}
        for name in slots
    ]
    return {
        "id": identity,
        "checks": comparisons,
        "evidence": evidence,
        "passed": all(c["passed"] for c in comparisons),
    }


def compact_payload(payload, full_evidence):
    """Derive a readable summary from retained raw results, without execution."""
    compact = copy.deepcopy(payload)
    info_keys = (
        "transaction_status", "operation_outcome", "actual_duration_s", "cost_delta",
        "sample_delta", "measurement_status", "outcome_diagnostics", "constitution_checks",
        "material_balance", "rollback_reason", "error_message",
    )
    for unit in compact["units"]:
        evidence = unit["evidence"]
        for key in ("samples", "work"):
            evidence.pop(key, None)
        for row in evidence.get("history", ()):
            row["info"] = {k: v for k, v in row["info"].items() if k in info_keys}
        if "world_failure" in evidence:
            evidence["world_failure"] = {
                k: v for k, v in evidence["world_failure"].items() if k in info_keys}
    compact["full_evidence"] = str(Path(full_evidence).resolve())
    return compact


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()
    output, summary = Path(args.output), Path(args.summary)
    if output.exists() or summary.exists():
        raise FileExistsError("retain all attempts; choose new output paths")
    started, last, completed = time.monotonic(), time.monotonic(), 0

    def progress():
        nonlocal last, completed
        completed += 1
        now = time.monotonic()
        if now - last >= 30:
            speed = completed / (now - started)
            print(
                f"distribution {completed}/49152; {speed:.1f} trajectories/s; "
                f"ETA {(49152 - completed) / speed:.1f}s",
                flush=True,
            )
            last = now

    units = []
    for case_index, case in enumerate(CASES):
        for policy_index in range(3):
            unit = 3 * case_index + policy_index
            units.append(
                make_unit(
                    f"{unit + 1:02d}/{case['id']}/{policy_index}",
                    SLOTS[0],
                    lambda unit=unit, case=case, policy_index=policy_index: distribution(
                        unit, case, policy_index, progress
                    ),
                )
            )
            print(
                f"unit {len(units)}/30 completed; trajectories {completed}/49152; "
                f"passed={units[-1]['passed']}",
                flush=True,
            )
    for index, callback in enumerate(
        (replay, world, interventions, boundaries, rollback, resource), 1
    ):
        units.append(make_unit(f"{len(units) + 1:02d}/{callback.__name__}", SLOTS[index], callback))
        print(f"unit {len(units)}/30 {callback.__name__}; passed={units[-1]['passed']}", flush=True)
    comparisons = [c for u in units for c in u["checks"]]
    payload = {
        "schema": "native-explicit-tau51-development-v1",
        "expected_units": 30,
        "units_passed": sum(u["passed"] for u in units),
        "expected_comparisons": 192,
        "comparisons": len(comparisons),
        "comparisons_passed": sum(c["passed"] for c in comparisons),
        "expected_trajectories": 49152,
        "attempted_trajectories": completed,
        "elapsed_s": time.monotonic() - started,
        "units": units,
        "failures": [{"unit": u["id"], **c} for u in units for c in u["checks"] if not c["passed"]],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    compact = compact_payload(payload, output)
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(compact, indent=2, allow_nan=False), encoding="utf-8")
    print(
        json.dumps(
            {
                k: payload[k]
                for k in (
                    "units_passed",
                    "comparisons_passed",
                    "comparisons",
                    "attempted_trajectories",
                    "elapsed_s",
                    "failures",
                )
            }
        ),
        flush=True,
    )
    return 0 if payload["units_passed"] == 30 else 1


if __name__ == "__main__":
    raise SystemExit(main())
