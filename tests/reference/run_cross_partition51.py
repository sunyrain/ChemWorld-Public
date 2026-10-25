"""Fixed complete 36-unit cross-partition development block; retain every failure."""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
from scipy.stats import binom, ks_2samp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cross_partition51_cases import CASES, author
from cross_partition51_reference import simulate

import test_cross_partition51 as contracts
from chemworld.foundation.discrete_inventory import AVOGADRO, mol_to_counts
from chemworld.physchem.stochastic_cross_partition import validate_transfer
from chemworld.physchem.stochastic_hybrid import (
    advance_stochastic_hybrid,
    initial_hybrid_continuation,
)

NATIVE_SAMPLES = 512
REFERENCE_SAMPLES = 8192
DKW_NATIVE = math.sqrt(math.log(2 * 5 / .001) / (2 * NATIVE_SAMPLES))
DKW_TWO = DKW_NATIVE + math.sqrt(math.log(2 * 5 / .001) / (2 * REFERENCE_SAMPLES))


class Record:
    def __init__(self, identity):
        self.identity, self.checks, self.evidence = identity, [], {}

    def equal(self, name, actual, expected=True):
        self.checks.append({"check": name, "passed": bool(actual == expected),
                            "actual": actual, "expected": expected})

    def bound(self, name, actual, limit=2e-7):
        self.checks.append({"check": name, "passed": bool(actual <= limit),
                            "actual": float(actual), "limit": float(limit)})

    def contract(self, name, call):
        try:
            call()
        except Exception:
            self.equal(name, False)
            self.evidence[name] = traceback.format_exc()
        else:
            self.equal(name, True)

    def payload(self):
        return {"unit": self.identity, "passed": bool(self.checks) and all(
            c["passed"] for c in self.checks), "checks": self.checks, "evidence": self.evidence}


def native(case, seed):
    network, spec, amounts = author(case)
    memory = initial_hybrid_continuation(network, spec, amounts, seed=seed)
    result = advance_stochastic_hybrid(network, spec, amounts, memory, 3.)
    residuals = validate_transfer(network, spec, amounts, result.amounts_mol,
        memory["partition_transfer"], result.continuation["partition_transfer"])
    return result, max(map(abs, residuals.values()), default=0.)


def trajectory(index):
    case, seed = CASES[index // 4], (0, 17, 41, 103)[index % 4]
    record = Record(f"{index+1:02d}/analytic-case-{index//4}-seed-{seed}")
    actual, residual = native(case, seed)
    expected = simulate(case, seed)
    record.equal("reaction_sequence", [e["reaction_id"] for e in actual.events],
                 [e["reaction_id"] for e in expected["events"]])
    record.equal("event_count", len(actual.events), len(expected["events"]))
    record.bound("event_time_max_error", max((abs(a["time_s"] - b["time_s"])
        for a, b in zip(actual.events, expected["events"], strict=True)), default=0.))
    record.equal("integer_D", mol_to_counts({"D": actual.amounts_mol["D"]})["D"], expected["D"])
    for key in ("A", "B"):
        record.bound(f"continuous_{key}_molecule_error",
                     abs(actual.amounts_mol[key] * AVOGADRO - expected[key]))
    record.equal("rng", actual.continuation["rng_state"], expected["rng_state"])
    pending = actual.continuation["pending"]
    record.equal("pending_targets", pending["targets"], expected["targets"])
    record.bound("pending_hazard_max_error", max(abs(a - b) for a, b in zip(
        pending["hazards"], expected["hazards"], strict=True)))
    record.equal("receipt_firings", actual.continuation["partition_transfer"]["firings"],
        {name: sum(e["reaction_id"] == name for e in expected["events"])
         for name in ("release", "capture")})
    record.bound("atom_residual_molecules", residual, 1e-4)
    record.equal("nonnegative", all(v >= 0 for v in actual.amounts_mol.values()))
    record.equal("draw_count", actual.work["random_draws"], 2 + len(expected["events"]))
    record.evidence.update(case=case, seed=seed, work=actual.work,
        availability_crossings=actual.work["availability_crossings"],
        native_events=list(actual.events), reference_events=expected["events"])
    return record


def consumer(index):
    record = Record(f"{index:02d}/original-consumer")
    selected = {
        25: [("availability_roots", contracts.
              test_availability_roots_keep_original_poisson_targets_and_real_fractional_stock)],
        26: [("split_continuation", contracts.
              test_arbitrary_split_agrees_numerically_without_claiming_bitwise_ode_equivalence)],
        27: [("original_world_json", contracts.test_original_world_and_complete_json_canary)],
        28: [("selected_other_vessel_shared_wait", contracts.
              test_shared_clock_wait_on_other_vessel_canary)],
        30: [("atomic_budget_rollback", contracts.
              test_event_budget_rolls_back_inventory_rng_receipt_and_clock)],
        31: [("distinct_codec", contracts.test_new_codec_does_not_relax_old_partition),
             ("zero_purity_quantization", contracts.
              test_zero_duration_is_pure_and_unresolved_continuous_quantum_fails_at_authoring)],
    }
    if index == 29:
        with tempfile.TemporaryDirectory(prefix="chemworld51D-resume-") as directory:
            record.contract("fresh_process_json", lambda:
                contracts.test_fresh_process_json_continuation_canary(Path(directory)))
    elif index == 32:
        for kind in ("wrong_stoichiometry", "one_molecule_loss", "invented_events"):
            record.contract(kind, lambda kind=kind:
                contracts.test_original_runtime_checks_source_bound_receipt_history(kind))
    elif index in (33, 34):
        record.contract(f"quantum_{index-31}_falling_and_fractional", lambda:
            contracts.test_multiple_quantum_consumption_and_discrete_falling_factorials(index - 31))
    else:
        for name, call in selected[index]:
            record.contract(name, call)
    return record


def distribution(index, output):
    record = Record(f"{index:02d}/{'finite-CME' if index == 35 else 'dynamic-PDMP'}")
    case = ({**CASES[4], "D": 8, "A": 0., "B": 0.} if index == 35 else CASES[0])
    seed_start = 20000 if index == 35 else 30000
    samples, failures, work, worst = [], [], {}, 0.
    started = time.monotonic()
    for n in range(NATIVE_SAMPLES):
        seed = seed_start + n
        try:
            actual, residual = native(case, seed)
            worst = max(worst, residual)
            samples.append({"seed": seed,
                "D": mol_to_counts({"D": actual.amounts_mol["D"]})["D"],
                "A": actual.amounts_mol["A"] * AVOGADRO,
                "B": actual.amounts_mol["B"] * AVOGADRO, "events": len(actual.events)})
            for key, value in actual.work.items():
                work[key] = work.get(key, 0) + value
        except Exception:
            failures.append({"seed": seed, "exception": traceback.format_exc()})
        if (n + 1) % 32 == 0:
            elapsed = time.monotonic() - started
            print(f"unit {index}/36 native {n+1}/{NATIVE_SAMPLES}; "
                  f"rate={(n+1)/elapsed:.2f}/s; ETA={elapsed/(n+1)*(NATIVE_SAMPLES-n-1):.1f}s",
                  flush=True)
    (output / f"{index}-native-samples.json").write_text(json.dumps(
        {"samples": samples, "failures": failures}, indent=2), encoding="utf-8")
    record.equal("native_samples_completed", len(samples), NATIVE_SAMPLES)
    record.equal("native_failure_count", len(failures), 0)
    record.bound("maximum_atom_residual_molecules", worst, 1e-4)
    record.evidence.update(case=case, seed_start=seed_start, failures=failures, work=work)
    if index == 35:
        k = case["release"] + case["capture"]
        probability = case["capture"] / k + case["release"] / k * math.exp(-3 * k)
        actual_cdf = [sum(s["D"] <= d for s in samples) / NATIVE_SAMPLES for d in range(9)]
        expected_cdf = binom.cdf(np.arange(9), 8, probability)
        record.bound("finite_CME_CDF", max(abs(a - b) for a, b in zip(
            actual_cdf, expected_cdf, strict=True)), DKW_NATIVE)
        record.evidence.update(probability=probability, actual_cdf=actual_cdf,
                               expected_cdf=expected_cdf.tolist())
    else:
        reference, reference_failures = [], []
        started = time.monotonic()
        for n in range(REFERENCE_SAMPLES):
            seed = 100000 + n
            try:
                result = simulate(case, seed)
                reference.append({"seed": seed, **{k: result[k] for k in ("D", "A", "B")},
                                  "events": len(result["events"])})
            except Exception:
                reference_failures.append({"seed": seed, "exception": traceback.format_exc()})
            if (n + 1) % 512 == 0:
                elapsed = time.monotonic() - started
                print(f"unit 36/36 analytic {n+1}/{REFERENCE_SAMPLES}; "
                      f"rate={(n+1)/elapsed:.2f}/s; "
                      f"ETA={elapsed/(n+1)*(REFERENCE_SAMPLES-n-1):.1f}s", flush=True)
        (output / "36-reference-samples.json").write_text(json.dumps(
            {"samples": reference, "failures": reference_failures}, indent=2), encoding="utf-8")
        record.equal("reference_samples_completed", len(reference), REFERENCE_SAMPLES)
        record.equal("reference_failure_count", len(reference_failures), 0)
        record.evidence["reference_failures"] = reference_failures
        for key in ("D", "A", "B", "events"):
            record.bound(f"dynamic_{key}_CDF", ks_2samp([s[key] for s in samples],
                         [s[key] for s in reference]).statistic, DKW_TWO)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    records, started = [], time.monotonic()
    for index in range(1, 37):
        try:
            record = (trajectory(index - 1) if index <= 24 else consumer(index)
                      if index <= 34 else distribution(index, args.output))
        except Exception:
            record = Record(f"{index:02d}/unhandled")
            record.equal("completed_without_exception", False)
            record.evidence["exception"] = traceback.format_exc()
        payload = record.payload()
        records.append(payload)
        (args.output / f"{index:02d}.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8")
        print(f"unit {index}/36 {'PASS' if payload['passed'] else 'FAIL'}; "
              f"elapsed={time.monotonic()-started:.1f}s", flush=True)
    checks = [c for r in records for c in r["checks"]]
    summary = {"schema": "cross-partition51-development-v1", "units_total": 36,
        "units_passed": sum(r["passed"] for r in records), "checks_total": len(checks),
        "checks_passed": sum(c["passed"] for c in checks),
        "failures": [r for r in records if not r["passed"]], "units": records,
        "native_ensemble_samples": 2 * NATIVE_SAMPLES,
        "analytic_ensemble_samples": REFERENCE_SAMPLES,
        "dkw_native": DKW_NATIVE, "dkw_two": DKW_TWO,
        "elapsed_s": time.monotonic() - started}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in {"units", "failures"}}), flush=True)
    return 0 if summary["units_passed"] == 36 else 1


if __name__ == "__main__":
    raise SystemExit(main())
