"""Separate fixed25-unit source-domain and low-pressure reference supplement."""

import argparse
import json
import time
from functools import partial
from itertools import combinations
from pathlib import Path

from run_helmholtz_phase_catalog import COORDINATES, PROPERTIES, metric, rejection, required

from chemworld.data.helmholtz_phase import load_helmholtz_phase_model


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
    sources = {d["INFO"]["NAME"]: d for p in args.sources.glob("*.json")
               if (d := json.loads(p.read_text()))}
    rows = []
    started = time.monotonic()
    for name in ("MM", "R14", "R161", "R236EA", "PropyleneGlycol"):
        model = load_helmholtz_phase_model(sources[name], source=f"CoolProp8/{name}", eos_index=0)
        minimum, critical = model.fluid.minimum_temperature_K, model.critical_temperature_K
        maximum = model.fluid.maximum_temperature_K
        if name == "PropyleneGlycol":
            specs = [("low_pressure_potential", None, 11)]
        else:
            specs = [("original_domain", None, 3 if name == "R236EA" else 2)]
            specs.append(("saturation", maximum-.1 if name == "R236EA" else
                          minimum+.99*(critical-minimum), 7))
        if name == "R236EA":
            specs.extend(("homogeneous_pair", p, 6) for p in combinations(COORDINATES, 2))
            specs.append(("TP", None, 7))
        for kind, specification, count in specs:
            row = {"fluid": name, "kind": kind, "specification": specification,
                   "expected_comparisons": count, "metrics": {}}
            try:
                if kind == "original_domain":
                    original_T = minimum+.9999*(critical-minimum)
                    if name == "R236EA":
                        target_T = (critical+maximum)/2
                        row["metrics"] = {"maximum_below_critical": {"passed": maximum < critical},
                            "original_target_outside": {"passed": target_T > maximum},
                            "native_rejects": rejection(partial(model.fluid.state,
                                temperature_K=target_T, molar_density_mol_m3=100))}
                    else:
                        try:
                            native = model.saturation(original_T)
                            rejected = not native.success
                        except ValueError:
                            rejected = True
                        row["metrics"]["native_rejects_original"] = {"passed": rejected}
                        if name == "R161":
                            pressure = CP.PropsSI("P", "T", original_T, "Q", 0, name)
                            row["metrics"]["reference_over_source_Pmax"] = {
                                "passed": pressure > model.fluid.maximum_pressure_Pa,
                                "pressure_Pa": pressure,
                                "source_limit_Pa": model.fluid.maximum_pressure_Pa}
                        else:
                            row["metrics"]["reference_rejects_original"] = rejection(partial(
                                CP.PropsSI, "P", "T", original_T, "Q", 0, name))
                elif kind == "saturation":
                    report = required(model.saturation(specification))
                    row["metrics"]["P"] = metric(report.pressure_Pa,
                        CP.PropsSI("P", "T", specification, "Q", 0, name), 2e-3)
                    for phase, q, endpoint in (("liquid", 0, report.liquid),
                                               ("gas", 1, report.gas)):
                        for key in ("rho", "H", "S"):
                            native, ref, atol = PROPERTIES[key]
                            row["metrics"][phase+key] = metric(getattr(endpoint, native),
                                CP.PropsSI(ref, "T", specification, "Q", q, name), atol)
                elif kind == "low_pressure_potential":
                    T = minimum+.05*(critical-minimum)
                    report = required(model.saturation(T))
                    potentials, pressures = [], []
                    for phase, endpoint, phase_key in (("liquid", report.liquid, CP.iphase_liquid),
                                                       ("gas", report.gas, CP.iphase_gas)):
                        reference = CP.AbstractState("HEOS", name)
                        reference.specify_phase(phase_key)
                        reference.update(CP.DmolarT_INPUTS, endpoint.molar_density_mol_m3, T)
                        for key, ref in (("P", reference.p()), ("H", reference.hmolar()),
                                         ("U", reference.umolar()), ("S", reference.smolar())):
                            attr, _, atol = PROPERTIES[key]
                            row["metrics"][phase+key] = metric(getattr(endpoint, attr), ref, atol)
                        potentials.append(reference.hmolar()-T*reference.smolar())
                        pressures.append(reference.p())
                    g_error = abs(potentials[0]-potentials[1])
                    p_error = abs(pressures[0]-pressures[1])
                    row["metrics"]["independent_equal_g"] = {"passed": g_error <= 2e-5,
                                                              "error_J_mol": g_error}
                    row["metrics"]["independent_equal_p"] = {"passed": p_error <= 2e-3,
                                                              "error_Pa": p_error}
                    qt_g = [CP.PropsSI("Hmolar", "T", T, "Q", q, name)
                            -T*CP.PropsSI("Smolar", "T", T, "Q", q, name) for q in (0, 1)]
                    row["metrics"]["reference_QT_chemical_potential_disagreement"] = {
                        "passed": abs(qt_g[0]-qt_g[1]) > 1,
                        "reference_difference_J_mol": abs(qt_g[0]-qt_g[1])}
                else:
                    T, rho = (minimum+maximum)/2, .005*model.fluid.reducing_molar_density_mol_m3
                    target = {k: CP.PropsSI(PROPERTIES[k][1], "T", T, "Dmolar", rho, name)
                              for k in COORDINATES}
                    if kind == "TP":
                        report = required(model.solve_tp(temperature_K=T, pressure_Pa=target["P"]))
                        keys = ("rho", "H", "U", "S", "Cp", "Cv", "w")
                    else:
                        initial = model.fluid.state(temperature_K=.99*T,
                                                     molar_density_mol_m3=1.01*rho)
                        report = required(model.solve_homogeneous_state(
                            {k: target[k] for k in specification}, initial_state=initial))
                        keys = COORDINATES
                    row["metrics"] = {k: metric(getattr(report.state, PROPERTIES[k][0]),
                        CP.PropsSI(PROPERTIES[k][1], "T", T, "Dmolar", rho, name),
                        PROPERTIES[k][2]) for k in keys}
                row["passed"] = all(m["passed"] for m in row["metrics"].values())
            except Exception as exc:
                row.update(passed=False, error=f"{type(exc).__name__}: {exc}")
            rows.append(row)
        print(f"Source-domain supplement {len(rows)}/25 units, "
              f"{sum(r['passed'] for r in rows)} passed", flush=True)
    result = {"reference": "CoolProp8.0.0", "units": len(rows),
        "units_passed": sum(r["passed"] for r in rows), "expected_comparisons": 145,
        "executed_comparisons": sum(len(r["metrics"]) for r in rows),
        "comparisons_passed": sum(m["passed"] for r in rows for m in r["metrics"].values()),
        "elapsed_s": time.monotonic()-started, "results": rows}
    (args.output/"summary.json").write_text(json.dumps(result, indent=2, allow_nan=False),
                                           encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "results"}))


if __name__ == "__main__":
    main()
