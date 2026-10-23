"""Full136 source-fluid phase-domain block with fixed3092 units and no hidden retries."""

import argparse
import json
import time
from functools import partial
from itertools import combinations
from pathlib import Path

from chemworld.data.helmholtz_phase import load_helmholtz_phase_model
from chemworld.physchem.helmholtz_flash import solve_two_phase_state

PROPERTIES = {"rho": ("molar_density_mol_m3", "Dmolar", 2e-5),
    "H": ("enthalpy_J_mol", "Hmolar", 2e-4), "U": ("internal_energy_J_mol", "Umolar", 2e-4),
    "S": ("entropy_J_mol_K", "Smolar", 2e-6), "Cp": ("cp_J_mol_K", "Cpmolar", 2e-4),
    "Cv": ("cv_J_mol_K", "Cvmolar", 2e-4), "w": ("speed_of_sound_m_s", "A", 2e-4),
    "T": ("temperature_K", "T", 2e-5), "P": ("pressure_Pa", "P", 2e-3)}
COORDINATES = ("T", "P", "rho", "H", "U", "S")


def metric(native, reference, atol):
    return {"native": float(native), "reference": float(reference),
            "error": abs(float(native-reference)),
            "passed": bool(abs(native-reference) <= atol+3e-6*abs(reference))}


def required(report):
    if not report.success:
        raise ValueError(report.failure_reason+":"+json.dumps(dict(report.diagnostics)))
    return report


def rejection(call):
    try:
        call()
    except ValueError as exc:
        return {"passed": True, "reason": str(exc)}
    return {"passed": False, "reason": "unsupported request was accepted"}


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
    sources = sorted((json.loads(p.read_text()) for p in args.sources.glob("*.json")),
                     key=lambda d: d["INFO"]["NAME"])
    if len(sources) != 136:
        raise ValueError("fixed136 source directory changed")
    rows = []
    started = time.monotonic()
    for index, source in enumerate(sources):
        name = source["INFO"]["NAME"]
        pseudo = source["EOS"][0].get("pseudo_pure", False)
        specs = [("homogeneous_pair", pair, 6) for pair in combinations(COORDINATES, 2)]
        specs.append(("boundaries", None, 4))
        if pseudo:
            specs.append(("pseudo_pure_rejections", None, 3))
        else:
            specs.extend(("saturation", f, 7) for f in (.05, .5, .95, .9999))
            specs.extend(("TP", p, 7) for p in (.5, 1.5))
            specs.append(("supercritical_TP", None, 7))
        setup_error = None
        try:
            model = load_helmholtz_phase_model(source, source=f"CoolProp8/{name}", eos_index=0)
            minimum, critical = model.fluid.minimum_temperature_K, model.critical_temperature_K
            T = (critical+model.fluid.maximum_temperature_K)/2
            rho = .25*model.fluid.reducing_molar_density_mol_m3
            targets = {k: CP.PropsSI(PROPERTIES[k][1], "T", T, "Dmolar", rho, name)
                       for k in COORDINATES}
            initial = model.fluid.state(temperature_K=.99*T, molar_density_mol_m3=1.01*rho)
        except Exception as exc:
            setup_error = f"{type(exc).__name__}: {exc}"
        for kind, specification, expected in specs:
            row = {"fluid": name, "pseudo_pure": pseudo, "kind": kind,
                   "specification": specification, "expected_comparisons": expected,
                   "metrics": {}}
            try:
                if setup_error:
                    raise ValueError("fluid_setup:"+setup_error)
                if kind == "homogeneous_pair":
                    report = required(model.solve_homogeneous_state(
                        {k: targets[k] for k in specification}, initial_state=initial))
                    row["diagnostics"] = dict(report.diagnostics)
                    row["metrics"] = {k: metric(getattr(report.state, PROPERTIES[k][0]),
                        value, PROPERTIES[k][2]) for k, value in targets.items()}
                elif kind == "saturation":
                    temperature = minimum+specification*(critical-minimum)
                    report = required(model.saturation(temperature))
                    row["diagnostics"] = dict(report.diagnostics)
                    row["metrics"]["P"] = metric(report.pressure_Pa,
                        CP.PropsSI("P", "T", temperature, "Q", 0, name), 2e-3)
                    for phase, quality, endpoint in (("liquid", 0, report.liquid),
                                                      ("gas", 1, report.gas)):
                        for key in ("rho", "H", "S"):
                            native, ref, atol = PROPERTIES[key]
                            row["metrics"][phase+":"+key] = metric(getattr(endpoint, native),
                                CP.PropsSI(ref, "T", temperature, "Q", quality, name), atol)
                elif kind in {"TP", "supercritical_TP"}:
                    temperature = T if kind == "supercritical_TP" else (minimum+critical)/2
                    pressure = (targets["P"] if kind == "supercritical_TP" else
                        specification*CP.PropsSI("P", "T", temperature, "Q", 0, name))
                    report = required(model.solve_tp(temperature_K=temperature,
                                                      pressure_Pa=pressure))
                    row["diagnostics"] = dict(report.diagnostics)
                    for key in ("rho", "H", "U", "S", "Cp", "Cv", "w"):
                        native, ref, atol = PROPERTIES[key]
                        row["metrics"][key] = metric(getattr(report.state, native),
                            CP.PropsSI(ref, "T", temperature, "P", pressure, name), atol)
                elif kind == "boundaries":
                    row["metrics"] = {
                        "below_Tmin": rejection(partial(model.fluid.state,
                            temperature_K=.99*minimum, molar_density_mol_m3=rho)),
                        "above_Tmax": rejection(partial(model.fluid.state,
                            temperature_K=1.01*model.fluid.maximum_temperature_K,
                            molar_density_mol_m3=rho)),
                        "negative_rho": rejection(partial(model.fluid.state,
                            temperature_K=T, molar_density_mol_m3=-1)),
                        "critical_saturation": rejection(partial(model.saturation, critical))}
                else:
                    row["metrics"] = {"saturation": rejection(partial(model.saturation,
                        (minimum+critical)/2)), "automatic_TP": rejection(partial(model.solve_tp,
                            temperature_K=T, pressure_Pa=targets["P"])),
                        "two_phase_TQ": rejection(partial(solve_two_phase_state, model,
                            {"T": (minimum+critical)/2, "Q": .5}))}
                row["passed"] = all(m["passed"] for m in row["metrics"].values())
            except Exception as exc:
                row.update(passed=False, error=f"{type(exc).__name__}: {exc}")
            rows.append(row)
        (args.output/"partial.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        elapsed = time.monotonic()-started
        print(f"Phase catalog {index+1}/136 {name}, {len(rows)}/3092 units: "
              f"{sum(r['passed'] for r in rows)} passed; {elapsed:.1f}s; "
              f"ETA {elapsed/(index+1)*(135-index):.1f}s", flush=True)
    result = {"reference": "CoolProp8.0.0", "fluids": 136, "pure_fluids": 130,
        "pseudo_pure_fluids": 6, "units": len(rows), "units_passed": sum(r["passed"] for r in rows),
        "expected_comparisons": 19172, "executed_comparisons": sum(len(r["metrics"]) for r in rows),
        "comparisons_passed": sum(m["passed"] for r in rows for m in r["metrics"].values()),
        "elapsed_s": time.monotonic()-started, "results": rows}
    (args.output/"summary.json").write_text(json.dumps(result, indent=2, allow_nan=False),
                                           encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "results"}))


if __name__ == "__main__":
    main()
