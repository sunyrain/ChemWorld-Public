"""All 36 original inputs: independent CP primitive energy balance, not HA B fallback."""

import argparse
import json
import math
from pathlib import Path

from chemworld.physchem import humid_air_wet_bulb


def run(source):
    import CoolProp
    from CoolProp.CoolProp import HAProps_Aux, HAPropsSI, PropsSI
    from scipy.optimize import brentq

    if CoolProp.__version__ != "8.0.0":
        raise RuntimeError("fixed block requires CoolProp 8.0.0")
    data = json.loads(source.read_text())
    inputs = [r["inputs"] for r in data["rows"] if r["source"] == "native wet bulb"
              and r["check"] == "pressure_Pa"]
    if len(inputs) != 36:
        raise ValueError("expected all 36 original inlet records")
    rows = []

    def point(supplied):
        phase, T, p, W = (supplied[k] for k in ("phase", "T_K", "p_Pa", "W_kg_water_kg_dry_air"))
        checks = []
        metadata = {}
        try:
            h_in = HAPropsSI("H", "T", T, "P", p, "W", W)
            # The composition diagnostic is allowed above RH=1; the HA output
            # limiter may reject it, so compute the same mole-fraction ratio.
            epsilon = .621945
            Ws_in = HAPropsSI("W", "T", T, "P", p, "R", 1)
            metadata["reference_relative_humidity"] = (W/(epsilon+W))/(Ws_in/(epsilon+Ws_in))

            def properties(Tw):
                Ws = HAPropsSI("W", "T", Tw, "P", p, "R", 1)
                hs = HAPropsSI("H", "T", Tw, "P", p, "R", 1)
                hc = (HAProps_Aux("h_Ice", Tw, p, 0)[0] if phase == "ice"
                      else PropsSI("Hmass", "T|liquid", Tw, "P", p, "HEOS::Water"))
                return h_in+(Ws-W)*hc-hs, Ws-W

            lower, upper = ((193., min(T+1., 273.16)) if phase == "ice" else
                            (273.160001, min(T+1., PropsSI("T", "P", p, "Q", 0, "Water")-2.)))
            reference = brentq(lambda Tw: properties(Tw)[0], lower, upper,
                               xtol=1e-12, maxiter=100)
            balance, added = properties(reference)
            native = humid_air_wet_bulb(T, p, supplied["water_mass_fraction"],
                                       condensed_phase=phase)
            metadata.update(reference_temperature_K=reference,
                            reference_water_added_kg_kg_dry_air=added,
                            reference_positive_feed=added >= 0,
                            bracket_K=[lower, upper])
            checks = [("reference_energy_residual_J_kg_dry_air", balance, 0., 1e-5),
                      ("native_vs_reference_temperature_K",
                       native.temperature_K, reference, .02)]
        except Exception as exc:
            for name in ("reference_energy_residual_J_kg_dry_air",
                         "native_vs_reference_temperature_K"):
                rows.append({"inputs": supplied, "metadata": metadata, "check": name,
                             "passed": False, "error": f"{type(exc).__name__}: {exc}"})
        else:
            for name, actual, expected, absolute in checks:
                rows.append({"inputs": supplied, "metadata": metadata, "check": name,
                             "actual": actual, "expected": expected,
                             "absolute_error": abs(actual-expected), "relative_tolerance": 0,
                             "absolute_tolerance": absolute,
                             "passed": math.isclose(actual, expected, rel_tol=0, abs_tol=absolute)})
        print(f"CP primitive balance {len(rows)}/72 comparisons", flush=True)

    for supplied in inputs:
        point(supplied)
    return {"development_evidence_only": True, "source_file": source.name,
            "inputs": 36, "comparisons": len(rows), "passed": sum(r["passed"] for r in rows),
            "failed": sum(not r["passed"] for r in rows), "rows": rows,
            "reference": "CoolProp 8.0.0 HA H/W, HEOS liquid h or HAProps_Aux ice h; "
                         "independent SciPy brentq, not native roots or CP direct B solver",
            "limitation": "Reference supersaturation/negative feed is reported, not qualified "
                          "as humidification. Original 13 direct-B failures remain failures."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("retain earlier runs; select a fresh path")
    result = run(args.source)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}))
    raise SystemExit(bool(result["failed"]))
