"""Fixed 144-unit pure-fluid saturation and homogeneous closure reference block."""

import argparse
import json
import time
from itertools import combinations
from pathlib import Path

from chemworld.data.helmholtz_phase import load_helmholtz_phase_model

NAMES = ("Water", "CarbonDioxide", "R134a", "Nitrogen", "Methanol", "Ammonia")
PROPERTIES = {"rho": "molar_density_mol_m3", "H": "enthalpy_J_mol", "U": "internal_energy_J_mol",
              "S": "entropy_J_mol_K", "Cp": "cp_J_mol_K", "Cv": "cv_J_mol_K",
              "w": "speed_of_sound_m_s", "T": "temperature_K", "P": "pressure_Pa"}
REFERENCE = {"rho": "Dmolar", "H": "Hmolar", "U": "Umolar", "S": "Smolar",
             "Cp": "Cpmolar", "Cv": "Cvmolar", "w": "A", "T": "T", "P": "P"}


def metric(native, reference, atol=1e-5):
    return {"native": native, "reference": reference, "error": abs(native-reference),
            "passed": abs(native-reference) <= atol+2e-7*abs(reference)}


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2, allow_nan=False),
                    encoding="utf-8")


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
    rows = []
    started = time.monotonic()
    for index, name in enumerate(NAMES):
        model = load_helmholtz_phase_model(sources[name], source=f"CoolProp 8.0.0/{name}",
                                          eos_index=0)
        minimum, critical = model.fluid.minimum_temperature_K, model.critical_temperature_K
        for fraction in (.2, .6, .95):
            temperature = minimum+fraction*(critical-minimum)
            pressure = CP.PropsSI("P", "T", temperature, "Q", 0, name)
            row = {"kind": "saturation", "fluid": name, "temperature_K": temperature,
                   "expected_comparisons": 7, "metrics": {}}
            try:
                report = model.saturation(temperature)
                row["diagnostics"] = dict(report.diagnostics)
                if not report.success:
                    raise ValueError(report.failure_reason)
                row["metrics"]["P"] = metric(report.pressure_Pa, pressure, 1e-3)
                for label, quality, state in (("liquid", 0, report.liquid), ("gas", 1, report.gas)):
                    for key, atol in (("rho", 1e-5), ("H", 1e-4), ("S", 1e-6)):
                        reference = CP.PropsSI(REFERENCE[key], "T", temperature, "Q", quality, name)
                        row["metrics"][label+"_"+key] = metric(getattr(state, PROPERTIES[key]),
                                                                reference, atol)
                row["passed"] = all(m["passed"] for m in row["metrics"].values())
            except Exception as exc:
                row.update(passed=False, error=f"{type(exc).__name__}: {exc}")
            rows.append(row)
            for multiplier in (.5, 1.5):
                target = multiplier*pressure
                row = {"kind": "TP", "fluid": name, "temperature_K": temperature,
                       "pressure_Pa": target, "expected_comparisons": 7, "metrics": {}}
                try:
                    report = model.solve_tp(temperature_K=temperature, pressure_Pa=target)
                    row["diagnostics"] = dict(report.diagnostics)
                    if not report.success:
                        raise ValueError(report.failure_reason)
                    for key in ("rho", "H", "U", "S", "Cp", "Cv", "w"):
                        reference = CP.PropsSI(REFERENCE[key], "T", temperature, "P", target, name)
                        row["metrics"][key] = metric(getattr(report.state, PROPERTIES[key]),
                                                      reference)
                    row["passed"] = all(m["passed"] for m in row["metrics"].values())
                except Exception as exc:
                    row.update(passed=False, error=f"{type(exc).__name__}: {exc}")
                rows.append(row)
        T = (critical+model.fluid.maximum_temperature_K)/2
        rho = .25*model.fluid.reducing_molar_density_mol_m3
        target = {key: CP.PropsSI(REFERENCE[key], "T", T, "Dmolar", rho, name)
                  for key in ("T", "P", "rho", "H", "U", "S")}
        initial = model.fluid.state(temperature_K=.99*T, molar_density_mol_m3=1.01*rho)
        for pair in combinations(target, 2):
            row = {"kind": "homogeneous_pair", "fluid": name, "pair": pair,
                   "expected_comparisons": 6, "metrics": {}}
            try:
                report = model.solve_homogeneous_state({k: target[k] for k in pair},
                                                        initial_state=initial)
                row["diagnostics"] = dict(report.diagnostics)
                if not report.success:
                    raise ValueError(report.failure_reason)
                row["metrics"] = {key: metric(getattr(report.state, PROPERTIES[key]), reference)
                                   for key, reference in target.items()}
                row["passed"] = all(m["passed"] for m in row["metrics"].values())
            except Exception as exc:
                row.update(passed=False, error=f"{type(exc).__name__}: {exc}")
            rows.append(row)
        elapsed = time.monotonic()-started
        print(f"Helmholtz solvers {index+1}/6 fluids, {len(rows)}/144 units, "
              f"{sum(r['passed'] for r in rows)} passed; {elapsed:.1f}s; "
              f"ETA {elapsed/(index+1)*(5-index):.1f}s", flush=True)
    summary = {"reference": "CoolProp 8.0.0", "unit_count": len(rows),
        "units_passed": sum(r["passed"] for r in rows), "expected_comparisons": 918,
        "executed_comparisons": sum(len(r["metrics"]) for r in rows),
        "comparisons_passed": sum(m["passed"] for r in rows for m in r["metrics"].values()),
        "elapsed_s": time.monotonic()-started, "results": rows}
    write(args.output / "summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}))


if __name__ == "__main__":
    main()
