"""Fixed full CoolProp 8.0.0 catalog qualification; reference imports only here."""

import argparse
import json
import time
from collections import Counter
from itertools import product
from math import isfinite
from pathlib import Path

from chemworld.data.helmholtz import load_helmholtz_fluid

ORDERS = tuple((d, total-d) for total in range(4) for d in range(total+1))
PROPERTIES = {
    "pressure_Pa": "p", "enthalpy_J_mol": "hmolar", "internal_energy_J_mol": "umolar",
    "entropy_J_mol_K": "smolar", "cp_J_mol_K": "cpmolar", "cv_J_mol_K": "cvmolar",
    "speed_of_sound_m_s": "speed_sound", "compressibility_factor": "compressibility_factor",
}


def derivative_name(kind, d, t):
    base = "alpha0" if kind == "ideal" else "alphar"
    if d+t == 0:
        return base
    suffix = []
    for name, order in (("Delta", d), ("Tau", t)):
        if order:
            suffix.append("d" + name + (str(order) if order > 1 else ""))
    return "d" + (str(d+t) if d+t > 1 else "") + base + "_" + "_".join(suffix)


def metric(native, reference, *, atol):
    error = abs(native-reference)
    return {"native": native, "reference": reference, "absolute_error": error,
            "passed": isfinite(native) and isfinite(reference)
            and error <= atol+2e-8*abs(reference)}


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=True, indent=2, allow_nan=False),
                    encoding="utf-8")


def main():
    import CoolProp
    import CoolProp.CoolProp as CP  # noqa: N817

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if CoolProp.__version__ != "8.0.0":
        raise ValueError("fixed qualification requires CoolProp 8.0.0")
    args.output.mkdir(parents=True, exist_ok=False)
    raw = args.output / "sources"
    raw.mkdir()
    started = time.monotonic()
    names = sorted(CP.FluidsList())
    results, imports, term_counts = [], [], Counter()
    # Persist the fixed denominator before evaluating native numerical values.
    write(args.output / "design.json", {"reference": "CoolProp 8.0.0", "fluids": names,
          "points_per_fluid": 8, "alpha_comparisons_per_point": 20,
          "property_comparisons_per_eligible_point": 10,
          "temperature_fractions": [.25, .6], "density_fractions": [.01, .25, 1.2, 2.2]})
    for index, name in enumerate(names):
        source = json.loads(CP.get_fluid_param_string(name, "JSON"))[0]
        write(raw / f"{index:03d}.json", source)
        model = None
        for eos_index, eos in enumerate(source["EOS"]):
            term_counts.update(t["type"] for key in ("alpha0", "alphar") for t in eos[key])
            row = {"fluid": name, "eos_index": eos_index}
            try:
                candidate = load_helmholtz_fluid(
                    source, eos_index=eos_index, source=f"CoolProp 8.0.0/{name}/{eos_index}")
                if eos_index == 0:
                    model = candidate
                row["passed"] = True
            except Exception as exc:
                row.update(passed=False, error=f"{type(exc).__name__}: {exc}")
            imports.append(row)
        reference = CP.AbstractState("HEOS", name)
        reference.specify_phase(CP.iphase_gas)
        eos = source["EOS"][0]
        minimum = eos["STATES"]["sat_min_liquid"]["T"]
        reducing = eos["STATES"]["reducing"]
        for f, density in product((.25, .6), (.01, .25, 1.2, 2.2)):
            temperature = minimum + f*(eos["T_max"]-minimum)
            rho = density*reducing["rhomolar"]
            row = {"fluid": name, "temperature_K": temperature,
                   "molar_density_mol_m3": rho, "alpha_metrics": {}, "property_metrics": {}}
            try:
                if model is None:
                    raise ValueError("native default EOS import failed")
                reference.update(CP.DmolarT_INPUTS, rho, temperature)
                delta, tau = density, reducing["T"]/temperature
                for kind, (d, t) in product(("ideal", "residual"), ORDERS):
                    key = derivative_name(kind, d, t)
                    try:
                        value = model.alpha_derivative(
                            kind, delta=delta, tau=tau, delta_order=d, tau_order=t)
                        comparison = metric(value, getattr(reference, key)(), atol=2e-8)
                    except Exception as exc:
                        comparison = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
                    row["alpha_metrics"][key] = comparison
                pressure, cv = reference.p(), reference.cvmolar()
                stiffness = reference.first_partial_deriv(CP.iP, CP.iDmolar, CP.iT)
                eligible = pressure > 0 and cv > 0 and stiffness > 0 and pressure <= eos["p_max"]
                row["property_eligibility"] = {"eligible": eligible, "pressure_Pa": pressure,
                    "cv_J_mol_K": cv, "dP_drho_T": stiffness, "pmax_Pa": eos["p_max"]}
                if eligible:
                    state = model.state(temperature_K=temperature, molar_density_mol_m3=rho)
                    refs = {key: getattr(reference, method)() for key, method in PROPERTIES.items()}
                    refs["dP_dT_rho_Pa_K"] = reference.first_partial_deriv(CP.iP, CP.iT, CP.iDmolar)
                    refs["dP_drho_T_Pa_m3_mol"] = stiffness
                    row["property_metrics"] = {
                        key: metric(getattr(state, key), value, atol=2e-7)
                        for key, value in refs.items()}
                row["passed"] = all(m["passed"] for group in ("alpha_metrics", "property_metrics")
                                    for m in row[group].values())
            except Exception as exc:
                row.update(passed=False, error=f"{type(exc).__name__}: {exc}")
            results.append(row)
        elapsed = time.monotonic()-started
        print(f"Helmholtz {index+1}/{len(names)} {name}: "
              f"{sum(r['passed'] for r in results)}/{len(results)} points passed; "
              f"{elapsed:.1f}s, ETA {elapsed/(index+1)*(len(names)-index-1):.1f}s", flush=True)
    summary = {"reference": "CoolProp 8.0.0", "development_only": True,
               "fluid_count": len(names), "import_count": len(imports),
               "imports_passed": sum(r["passed"] for r in imports), "imports": imports,
               "term_counts": dict(sorted(term_counts.items())),
               "point_count": len(results), "points_passed": sum(r["passed"] for r in results),
               "alpha_comparisons": sum(len(r["alpha_metrics"]) for r in results),
               "property_comparisons": sum(len(r["property_metrics"]) for r in results),
               "property_eligible_points": sum(
                   r.get("property_eligibility", {}).get("eligible", False) for r in results),
               "elapsed_s": time.monotonic()-started, "results": results}
    write(args.output / "summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in {"results", "imports"}}))


if __name__ == "__main__":
    main()
