"""Fixed 8288-path development block; keep all failures and previous attempts."""

import argparse
import json
import time
from dataclasses import replace
from math import exp, log, sqrt
from pathlib import Path

import numpy as np
from native_copasi_reference import CopasiReference
from scipy.stats import binom, ks_2samp

from chemworld.foundation.discrete_inventory import AVOGADRO, counts_to_mol
from chemworld.physchem.reaction_network import BatchSolverOptions
from chemworld.physchem.stochastic_kinetics import advance_stochastic, initial_continuation
from chemworld.world.biochemical_worlds import biochemical_network

ALPHA = 0.01 / 12


def enzyme_balance(counts, initial):
    return (
        counts["E"] + counts["ES"] == initial["E"]
        and counts["S"] + counts["ES"] + counts["P"] == initial["S"]
        and all(type(n) is int and n >= 0 for n in counts.values())
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve previous attempt, choose a new output")
    started = time.monotonic()
    rows, failures, checks = [], [], []

    def progress(stage):
        done = len(rows) + len(failures)
        elapsed = time.monotonic() - started
        print(
            f"{stage}: {done}/8288; {elapsed:.1f}s elapsed; "
            f"{done / max(elapsed, 0.001):.1f} runs/s; "
            f"rough ETA={(8288-done)*elapsed/max(done,1):.0f}s (copy scales differ)",
            flush=True,
        )

    def run(stage, network, counts, seed, duration, reference=None, **case):
        context = dict(stage=stage, seed=seed, **case)
        try:
            final = (
                reference.run(seed, duration)
                if reference
                else advance_stochastic(
                    network, counts, initial_continuation(seed), duration
                ).counts
            )
            ok = (
                sum(final.values()) == counts["A"]
                if stage == "AB"
                else enzyme_balance(final, counts)
            )
            if not ok:
                raise ValueError("integer conservation failure")
            rows.append({**context, "counts": final})
            return final
        except Exception as exc:
            failures.append({**context, "error": repr(exc)})
            return None

    network = biochemical_network("reversible")
    for total in (100, 1000):
        endpoints = []
        for seed in range(2000):
            final = run("AB", network, {"A": total}, seed, 100, total=total)
            if final is not None:
                endpoints.append(final["B"])
            if (seed + 1) % 100 == 0:
                progress("AB")
        p = (1 - exp(-30)) / 3
        variance = total * p * (1 - p)
        mu4 = 3 * variance**2 + variance * (1 - 6 * p * (1 - p))
        width = sqrt((mu4 - (1997 / 1999) * variance**2) / 2000 / ALPHA)
        bounds = (
            [float(binom.ppf(q, 2000 * total, p) / 2000) for q in (ALPHA / 2, 1 - ALPHA / 2)],
            [max(0, variance - width), variance + width],
        )
        for label, value, limits in zip(
            ("mean", "variance"),
            (float(np.mean(endpoints)), float(np.var(endpoints, ddof=1))),
            bounds,
            strict=True,
        ):
            checks.append(
                {
                    "stage": "AB",
                    "total": total,
                    "check": label,
                    "observed": value,
                    "bounds": limits,
                    "passed": len(endpoints) == 2000 and limits[0] <= value <= limits[1],
                }
            )

    reference_version = None
    network = biochemical_network()
    for enzyme in (10, 100):
        for substrate in (100, 1000):
            initial = {"E": enzyme, "S": substrate}
            reference = CopasiReference(network, initial)
            reference_version = reference.version
            ends = {"native": [], "copasi": []}
            for engine, offset, ref in (("native", 0, None), ("copasi", 10001, reference)):
                for i in range(512):
                    result = run(
                        engine, network, initial, i + offset, 20, ref, E=enzyme, S=substrate
                    )
                    if result is not None:
                        ends[engine].append(result)
                    if (i + 1) % 32 == 0:
                        progress(f"enzyme {engine} E={enzyme} S={substrate}")
            reference.close()
            bound = 2 * sqrt(log(4 / ALPHA) / (2 * 512))
            for key in ("P", "ES"):
                left = [r[key] for r in ends["native"]]
                right = [r[key] for r in ends["copasi"]]
                distance = float(ks_2samp(left, right).statistic) if left and right else None
                checks.append(
                    {
                        "stage": "CDF",
                        "E": enzyme,
                        "S": substrate,
                        "species": key,
                        "distance": distance,
                        "bound": bound,
                        "native_mean": float(np.mean(left)),
                        "reference_mean": float(np.mean(right)),
                        "native_variance": float(np.var(left, ddof=1)),
                        "reference_variance": float(np.var(right, ddof=1)),
                        "passed": len(left) == len(right) == 512 and distance <= bound,
                    }
                )

    convergence = []
    for enzyme in (10, 100):
        for substrate in (100, 1000):
            base = biochemical_network()
            ode = replace(base, stochastic=None).integrate_batch(
                counts_to_mol({"E": enzyme, "S": substrate}),
                volume_L=1e-15,
                temperature_K=298.15,
                duration_s=20,
                solver_options=BatchSolverOptions(rtol=1e-9, atol_mol=1e-30),
            )
            expected = {k: n / 1e-15 for k, n in ode.final_amounts_mol.items()}
            for scale in (1, 100, 1000):
                network = biochemical_network(scale=scale)
                initial = {"E": enzyme * scale, "S": substrate * scale}
                endpoints = []
                for seed in range(16):
                    result = run(
                        "high_copy", network, initial, seed, 20, E=enzyme, S=substrate, scale=scale
                    )
                    if result is not None:
                        endpoints.append(result)
                    progress(f"high-copy E={enzyme} S={substrate} scale={scale}")
                for key in network.species_ids:
                    values = [r[key] / AVOGADRO / (1e-15 * scale) for r in endpoints]
                    mean = float(np.mean(values))
                    se = float(np.std(values, ddof=1) / sqrt(len(values))) if len(values) > 1 else 0
                    pool = enzyme if key in ("E", "ES") else substrate
                    limit = 0.01 * pool / AVOGADRO / 1e-15 + 6 * se
                    error = abs(mean - expected[key])
                    convergence.append(
                        {
                            "E": enzyme,
                            "S": substrate,
                            "scale": scale,
                            "species": key,
                            "mean_mol_L": mean,
                            "ode_mol_L": expected[key],
                            "error_mol_L": error,
                            "limit_mol_L": limit,
                            "passed": len(values) == 16 and error <= limit,
                        }
                    )
    summary = {
        "status": "development",
        "planned_runs": 8288,
        "completed_runs": len(rows),
        "failures": failures,
        "statistical_checks": checks,
        "convergence": convergence,
        "passed_statistical_checks": sum(c["passed"] for c in checks),
        "passed_high_copy_checks": sum(c["passed"] for c in convergence if c["scale"] == 1000),
        "copasi_version": reference_version,
        "familywise_alpha": 0.01,
        "elapsed_wall_s": time.monotonic() - started,
        "runs": rows,
    }
    args.output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in summary.items()
                if k not in {"runs", "convergence", "statistical_checks"}
            },
            indent=2,
        ),
        flush=True,
    )
    if (
        failures
        or summary["passed_statistical_checks"] != 12
        or summary["passed_high_copy_checks"] != 16
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
