"""Fixed 72-state finite-water comparison to independent scalar mass action."""

import argparse
import json
import math
import sys
import time
from itertools import product
from pathlib import Path

from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_aqueous39 import FAMILIES, PRESSURES, TEMPERATURES, binary_reference

from chemworld.physchem.aqueous_reaction_equilibrium import (
    solve_aqueous_reaction_equilibrium,
)
from reference.aqueous40_fixtures import KW, WATER_MASS, feed, model


def independent_molality(family):
    def equation(m):
        gamma, phi = binary_reference(family, m)
        return 2 * (math.log(m) + gamma) + 2 * WATER_MASS * m * phi - math.log(KW)

    return brentq(equation, 1e-10, 2.0, xtol=1e-14)


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    grid = list(product(FAMILIES, TEMPERATURES, PRESSURES, (0.01, 1.0)))
    (output / "units.json").write_text(json.dumps(grid, indent=2), encoding="utf-8")
    failures, records, passed = [], [], 0
    started = time.monotonic()
    print(f"start units=0/{len(grid)} comparisons={len(grid) * 10}", flush=True)

    def compare(unit, name, actual, expected, atol=2e-10):
        nonlocal passed
        valid = actual is not None and math.isfinite(actual)
        ok = valid and math.isclose(actual, expected, rel_tol=2e-7, abs_tol=atol)
        item = {
            "unit": unit,
            "quantity": name,
            "actual": actual,
            "expected": expected,
            "passed": bool(ok),
        }
        records.append(item)
        passed += bool(ok)
        if not ok:
            failures.append(item)

    with (output / "states.jsonl").open("w", encoding="utf-8") as states:
        for unit, (family, t, p, water_kg) in enumerate(grid, 1):
            spec = model(family, network="water")
            result = solve_aqueous_reaction_equilibrium(
                spec, feed(spec, water_kg=water_kg), temperature_K=t, pressure_Pa=p
            )
            states.write(
                json.dumps(
                    {"unit": unit, "definition": grid[unit - 1], "result": result.to_dict()},
                    allow_nan=False,
                )
                + "\n"
            )
            m = independent_molality(family)
            nw = water_kg / (WATER_MASS * (1 + WATER_MASS * m))
            extent = water_kg * m / (1 + WATER_MASS * m)
            amounts = result.amounts_mol or {}
            compare(unit, "success", int(result.success), 1)
            for key, expected in (("H2O", nw), ("H+", extent), ("OH-", extent)):
                compare(unit, key, amounts.get(key), expected)
            compare(
                unit, "solvent_mass_kg", result.diagnostics.get("solvent_mass_kg"), nw * WATER_MASS
            )
            compare(unit, "water_log_affinity", result.reaction_residuals.get("water"), 0, 2e-8)
            compare(unit, "charge_residual_mol", result.charge_residual_mol, 0)
            for label in ("H", "O"):
                compare(unit, "element_" + label, result.element_residuals_mol.get(label), 0)
            compare(
                unit,
                "maximum_scaled_conservation_residual",
                result.diagnostics.get("maximum_scaled_conservation_residual"),
                0,
                2e-9,
            )
            if unit % 12 == 0:
                elapsed = time.monotonic() - started
                print(
                    f"progress units={unit}/{len(grid)} rate={unit / elapsed:.2f}/s "
                    f"eta={(len(grid) - unit) * elapsed / unit:.1f}s",
                    flush=True,
                )
    (output / "records.jsonl").write_text(
        "".join(json.dumps(row, allow_nan=False) + "\n" for row in records), encoding="utf-8"
    )
    summary = {
        "mode": "development",
        "source": "synthetic constant-K finite water",
        "units": len(grid),
        "comparisons": len(records),
        "passed": passed,
        "failed": len(records) - passed,
        "failures": failures,
        "elapsed_seconds": time.monotonic() - started,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"complete units={len(grid)}/{len(grid)} passed={passed}/{len(records)}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
