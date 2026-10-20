"""Fixed 4000-run transient binomial block; development evidence only."""

import argparse
import json
import time
from math import exp, sqrt
from pathlib import Path

import numpy as np
from scipy.stats import binom

from chemworld.physchem.reaction_network import (
    RateLawSpec,
    ReactionNetworkSpec,
    ReactionSpec,
    SpeciesSpec,
)
from chemworld.physchem.stochastic_kinetics import advance_stochastic, initial_continuation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve earlier attempt; choose a new output path")
    network = ReactionNetworkSpec(
        "synthetic-binomial-ab",
        (SpeciesSpec("A", "H2"), SpeciesSpec("B", "H2")),
        tuple(
            ReactionSpec.from_equation(
                reaction_id=name,
                equation=eq,
                rate_law=RateLawSpec(name, "mass_action", {"k": rate}),
            )
            for name, eq, rate in (("forward", "A => B", 0.1), ("reverse", "B => A", 0.2))
        ),
        stochastic={"volume_L": 0.01},
    )
    started = time.monotonic()
    rows, failures, checks = [], [], []
    for total in (100, 1000):
        final_counts = []
        for seed in range(2000):
            try:
                result = advance_stochastic(
                    network, {"A": total, "B": 0}, initial_continuation(seed), 20
                )
                conserved = sum(result.counts.values()) == total
                if not conserved:
                    raise ValueError("integer conservation failure")
                final_counts.append(result.counts["B"])
                rows.append(
                    {
                        "population": total,
                        "seed": seed,
                        "B": result.counts["B"],
                        "events": len(result.events),
                    }
                )
            except (ValueError, ArithmeticError, RuntimeError) as exc:
                failures.append({"population": total, "seed": seed, "error": str(exc)})
            if (seed + 1) % 100 == 0:
                done = len(rows) + len(failures)
                elapsed = time.monotonic() - started
                print(
                    f"SSA {done}/4000; {done / elapsed:.1f} runs/s; "
                    f"ETA {(4000 - done) * elapsed / done:.0f}s",
                    flush=True,
                )
        p = (1 - exp(-6)) / 3
        expected_mean, variance = total * p, total * p * (1 - p)
        alpha = 0.01 / 4
        mean_bounds = [
            float(binom.ppf(q, 2000 * total, p) / 2000) for q in (alpha / 2, 1 - alpha / 2)
        ]
        mu4 = 3 * variance**2 + variance * (1 - 6 * p * (1 - p))
        variance_of_s2 = (mu4 - (2000 - 3) / (2000 - 1) * variance**2) / 2000
        width = sqrt(variance_of_s2 / alpha)
        variance_bounds = [max(0, variance - width), variance + width]
        observed = [float(np.mean(final_counts)), float(np.var(final_counts, ddof=1))]
        for kind, value, expected, bounds in zip(
            ("mean", "variance"),
            observed,
            (expected_mean, variance),
            (mean_bounds, variance_bounds),
            strict=True,
        ):
            checks.append(
                {
                    "population": total,
                    "check": kind,
                    "observed": value,
                    "expected": expected,
                    "bounds": bounds,
                    "passed": len(final_counts) == 2000 and bounds[0] <= value <= bounds[1],
                }
            )
    payload = {
        "status": "development",
        "algorithm": "gillespie-direct-v1",
        "planned_runs": 4000,
        "completed_runs": len(rows),
        "failures": failures,
        "planned_statistical_checks": 4,
        "passed_statistical_checks": sum(c["passed"] for c in checks),
        "familywise_alpha": 0.01,
        "duration_s": 20,
        "checks": checks,
        "runs": rows,
        "elapsed_wall_s": time.monotonic() - started,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: v for k, v in payload.items() if k != "runs"}, indent=2))
    if failures or payload["passed_statistical_checks"] != 4:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
