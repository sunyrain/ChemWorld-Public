"""All 23 nondefault equations plus two synthetic GERG ideal-term families."""

import argparse
import copy
import json
import time
from itertools import product
from pathlib import Path

from chemworld.data.helmholtz import load_helmholtz_fluid

ORDERS = tuple((d, total-d) for total in range(4) for d in range(total+1))


def derivative_name(kind, d, t):
    base = "alpha0" if kind == "ideal" else "alphar"
    if d+t == 0:
        return base
    suffix = ["d"+name+(str(order) if order > 1 else "")
              for name, order in (("Delta", d), ("Tau", t)) if order]
    return "d"+(str(d+t) if d+t > 1 else "")+base+"_"+"_".join(suffix)


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
    sources = [json.loads(p.read_text()) for p in sorted(args.sources.glob("*.json"))]
    models = []
    for source in sources:
        for index in range(1, len(source["EOS"])):
            models.append((source["INFO"]["NAME"]+f"/{index}", source, index, False))
    nitrogen = next(s for s in sources if s["INFO"]["NAME"] == "Nitrogen")
    for family in ("GERG2004Sinh", "GERG2004Cosh"):
        source = copy.deepcopy(nitrogen)
        source["EOS"] = source["EOS"][:1]
        source["EOS"][0]["alphar"] = []
        source["EOS"][0]["alpha0"] = [
            {"type": "IdealGasHelmholtzLead", "a1": .2, "a2": -.3},
            {"type": "IdealGasHelmholtzLogTau", "a": 1.5},
            {"type": "IdealGasHelmholtz"+family, "n": [1.25, -.3], "theta": [.3, 1.7],
             "Tcrit": source["EOS"][0]["STATES"]["reducing"]["T"]}]
        models.append((family, source, 0, True))
    if len(models) != 25:
        raise ValueError("fixed block requires 23 alternate and 2 synthetic models")
    rows = []
    started = time.monotonic()
    for index, (label, source, eos_index, synthetic) in enumerate(models):
        data = copy.deepcopy(source)
        data["EOS"] = [source["EOS"][eos_index]]
        alias = f"CW24ReferenceVariant{index:03d}"
        data["INFO"].update(NAME=alias, CAS=alias, ALIASES=[])
        write(args.output/f"source-{index:02d}.json", data)
        setup_error = None
        try:
            CP.add_fluids_as_JSON("HEOS", json.dumps([data]))
            reference = CP.AbstractState("HEOS", alias)
            reference.specify_phase(CP.iphase_gas)
            native = load_helmholtz_fluid(source, source=f"CoolProp8/{label}", eos_index=eos_index)
        except Exception as exc:
            setup_error = f"{type(exc).__name__}: {exc}"
        eos = source["EOS"][eos_index]
        minimum = eos["STATES"]["sat_min_liquid"]["T"]
        reducing = eos["STATES"]["reducing"]
        for fraction, delta in product((.25, .6), (.01, .25, 1.2, 2.2)):
            temperature = minimum+fraction*(eos["T_max"]-minimum)
            tau, density = reducing["T"]/temperature, delta*reducing["rhomolar"]
            row = {"model": label, "synthetic": synthetic, "temperature_K": temperature,
                   "molar_density_mol_m3": density, "metrics": {}}
            try:
                if setup_error:
                    raise ValueError(setup_error)
                reference.update(CP.DmolarT_INPUTS, density, temperature)
                for kind, (d, t) in product(("ideal", "residual"), ORDERS):
                    name = derivative_name(kind, d, t)
                    a = native.alpha_derivative(kind, delta=delta, tau=tau,
                                                 delta_order=d, tau_order=t)
                    b = getattr(reference, name)()
                    row["metrics"][name] = {"native": a, "reference": b,
                        "error": abs(a-b), "passed": abs(a-b) <= 2e-8+2e-8*abs(b)}
                row["passed"] = all(m["passed"] for m in row["metrics"].values())
            except Exception as exc:
                row.update(passed=False, error=f"{type(exc).__name__}: {exc}")
            rows.append(row)
        print(f"Helmholtz variants {index+1}/25 {label}; "
              f"{sum(r['passed'] for r in rows)}/{len(rows)} points passed", flush=True)
    result = {"reference": "CoolProp 8.0.0", "models": 25, "real_alternate_EOS": 23,
              "synthetic_ideal_families": 2, "points": len(rows),
              "points_passed": sum(r["passed"] for r in rows), "expected_comparisons": 4000,
              "executed_comparisons": sum(len(r["metrics"]) for r in rows),
              "comparisons_passed": sum(m["passed"] for r in rows for m in r["metrics"].values()),
              "elapsed_s": time.monotonic()-started, "results": rows}
    write(args.output/"summary.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "results"}))


if __name__ == "__main__":
    main()
