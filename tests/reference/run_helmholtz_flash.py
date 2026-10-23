"""Fixed 432-unit independent pure-fluid two-phase and pressure flash block."""

import argparse
import json
import time
from itertools import combinations
from pathlib import Path

from chemworld.data.helmholtz_phase import load_helmholtz_phase_model
from chemworld.physchem.helmholtz_flash import flash_at_pressure, solve_two_phase_state

NAMES = ("Water", "CarbonDioxide", "R134a", "Nitrogen", "Methanol", "Ammonia")
PROPERTIES = {"T": "temperature_K", "P": "pressure_Pa", "rho": "molar_density_mol_m3",
              "H": "enthalpy_J_mol", "U": "internal_energy_J_mol", "S": "entropy_J_mol_K",
              "Q": "vapor_fraction"}
REFERENCE = {"T": "T", "P": "P", "rho": "Dmolar", "H": "Hmolar", "U": "Umolar",
             "S": "Smolar", "Q": "Q"}
ATOL = {"T": 2e-5, "P": 2e-3, "rho": 2e-5, "H": 2e-4, "U": 2e-4, "S": 2e-6, "Q": 2e-7}


def comparisons(state, targets):
    result = {}
    for key, reference in targets.items():
        native = float(getattr(state, PROPERTIES[key]))
        error = abs(native-reference)
        result[key] = {"native": native, "reference": reference, "error": error,
                       "passed": error <= ATOL[key]+3e-7*abs(reference)}
    return result


def main():
    import CoolProp
    import CoolProp.CoolProp as CP  # noqa: N817

    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if CoolProp.__version__ != "8.0.0":
        raise ValueError("reference version changed")
    args.output.mkdir(parents=True, exist_ok=False)
    sources = {}
    for path in args.sources.glob("*.json"):
        data = json.loads(path.read_text())
        sources[data["INFO"]["NAME"]] = data
    started = time.monotonic()
    rows = []
    for name in NAMES:
        model = load_helmholtz_phase_model(sources[name], source=f"CoolProp 8.0.0/{name}",
                                          eos_index=0)
        minimum = model.fluid.minimum_temperature_K
        gap = model.critical_temperature_K-minimum
        for fraction in (.2, .6, .95):
            temperature = minimum+fraction*gap
            targets = {key: CP.PropsSI(value, "T", temperature, "Q", .35, name)
                       for key, value in REFERENCE.items()}
            for pair in combinations(PROPERTIES, 2):
                if set(pair) == {"T", "P"}:
                    continue
                row = {"kind": "coexistence_pair", "fluid": name, "fraction": fraction,
                       "pair": pair, "expected_comparisons": 7, "metrics": {}}
                try:
                    report = solve_two_phase_state(model, {k: targets[k] for k in pair},
                        initial_temperature_K=temperature+.002*gap, initial_vapor_fraction=.36)
                    row["diagnostics"] = dict(report.diagnostics)
                    if not report.success:
                        raise ValueError(report.failure_reason)
                    row["metrics"] = comparisons(report.state, targets)
                    row["passed"] = all(m["passed"] for m in row["metrics"].values())
                except Exception as exc:
                    row.update(passed=False, error=f"{type(exc).__name__}: {exc}")
                rows.append(row)
                (args.output / "partial.json").write_text(json.dumps(rows, indent=2),
                                                          encoding="utf-8")
            elapsed = time.monotonic()-started
            print(f"Flash {name} T fraction {fraction}: {len(rows)}/432 units, "
                  f"{sum(r['passed'] for r in rows)} passed, {elapsed:.1f}s, "
                  f"ETA {elapsed/len(rows)*(432-len(rows)):.1f}s", flush=True)
        temperature = minimum+.6*gap
        pressure = CP.PropsSI("P", "T", temperature, "Q", 0, name)
        for branch, T in (("liquid", temperature-.03*gap), ("two_phase", temperature),
                          ("gas", temperature+.03*gap)):
            targets = {key: CP.PropsSI(value, "T", T,
                "Q" if branch == "two_phase" else "P", .35 if branch == "two_phase" else pressure,
                name) for key, value in REFERENCE.items() if key != "Q"}
            for coordinate in ("H", "U", "S", "rho"):
                row = {"kind": "pressure_flash", "fluid": name, "branch": branch,
                       "coordinate": coordinate, "expected_comparisons": 6, "metrics": {}}
                try:
                    report = flash_at_pressure(model, pressure_Pa=pressure,
                        coordinate=coordinate, value=targets[coordinate])
                    row["diagnostics"] = dict(report.diagnostics)
                    if not report.success:
                        raise ValueError(report.failure_reason)
                    row["metrics"] = comparisons(report.state, targets)
                    row["passed"] = (report.phase == branch
                                     and all(m["passed"] for m in row["metrics"].values()))
                    row["phase"] = report.phase
                except Exception as exc:
                    row.update(passed=False, error=f"{type(exc).__name__}: {exc}")
                rows.append(row)
                (args.output / "partial.json").write_text(json.dumps(rows, indent=2),
                                                          encoding="utf-8")
    summary = {"reference": "CoolProp 8.0.0", "unit_count": len(rows),
        "units_passed": sum(r["passed"] for r in rows), "expected_comparisons": 2952,
        "executed_comparisons": sum(len(r["metrics"]) for r in rows),
        "comparisons_passed": sum(m["passed"] for r in rows for m in r["metrics"].values()),
        "elapsed_s": time.monotonic()-started, "results": rows}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False),
                                             encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}))


if __name__ == "__main__":
    main()
