"""Fixed 36-inlet, 360-comparison G8 wet-bulb/PH development block."""

import argparse
import json
import math
from pathlib import Path

from chemworld.physchem import (
    humid_air_from_relative_humidity,
    humid_air_ph,
    humid_air_wet_bulb,
)

INPUTS = tuple((phase, T, p, rh)
               for phase, temperatures in (("liquid", (290, 320, 370)), ("ice", (210, 240, 270)))
               for T in temperatures for p in (101325., 2e6) for rh in (0., .4, 1.))


def run():
    from CoolProp.CoolProp import HAPropsSI

    rows = []

    def evaluate(function):
        try:
            return function()
        except Exception as exc:
            return exc

    def obtain(value, function):
        if isinstance(value, Exception):
            raise value
        return function(value)

    def point(phase, T, p, rh):
        inlet = evaluate(lambda: humid_air_from_relative_humidity(T, p, rh, condensed_phase=phase))
        wet = evaluate(lambda: obtain(inlet, lambda s: humid_air_wet_bulb(
            T, p, s.water_mass_fraction, condensed_phase=phase)))
        recovered = evaluate(lambda: obtain(inlet, lambda s: humid_air_ph(
            s.water_mass_fraction, p, s.enthalpy_J_kg)))

        def compare(source, name, actual, expected, relative, absolute):
            row = {"source": source, "inputs": {"phase": phase, "T_K": T, "p_Pa": p, "RH": rh},
                   "check": name, "relative_tolerance": relative, "absolute_tolerance": absolute}
            if not isinstance(inlet, Exception):
                row["inputs"].update(water_mass_fraction=inlet.water_mass_fraction,
                                     W_kg_water_kg_dry_air=inlet.humidity_ratio_kg_water_kg_dry_air)
            try:
                a, b = float(actual()), float(expected())
                row.update(actual=a, expected=b, absolute_error=abs(a-b),
                           passed=math.isclose(a, b, rel_tol=relative, abs_tol=absolute))
            except Exception as exc:
                row.update(passed=False, error=f"{type(exc).__name__}: {exc}")
            rows.append(row)

        compare("native wet bulb", "pressure_Pa",
                lambda: obtain(wet, lambda r: r.outlet.gas.pressure_Pa), lambda: p, 5e-11, 0)
        compare("native wet bulb", "chemical_potential_residual_J_kg",
                lambda: obtain(wet, lambda r: r.outlet.chemical_potential_residual_J_kg),
                lambda: 0, 0, 1e-5)
        compare("native wet bulb", "water_residual_kg_kg_dry_air", lambda: obtain(wet, lambda r:
            r.outlet.gas.humidity_ratio_kg_water_kg_dry_air
            -r.inlet.humidity_ratio_kg_water_kg_dry_air-r.water_added_kg_per_kg_dry_air),
            lambda: 0, 0, 1e-12)
        compare("native wet bulb", "enthalpy_residual_J_kg_dry_air", lambda: obtain(wet, lambda r:
            r.outlet.gas.enthalpy_J_kg_dry_air-r.inlet.enthalpy_J_kg_dry_air
            -r.water_added_kg_per_kg_dry_air*r.outlet.condensate.enthalpy_J_kg), lambda: 0, 0, 1e-6)
        compare("native wet bulb", "positive_temperature_rise_K",
                lambda: obtain(wet, lambda r: max(0., r.temperature_K-T)), lambda: 0, 0, 2e-8)
        compare("native wet bulb", "negative_water_feed_kg_kg_dry_air",
                lambda: obtain(wet, lambda r: max(0., -r.water_added_kg_per_kg_dry_air)),
                lambda: 0, 0, 1e-12)
        for field, relative, absolute in (("temperature_K", 0, 2e-7),
                                          ("mass_density_kg_m3", 2e-9, 0),
                                          ("enthalpy_J_kg", 0, 1e-6)):
            compare("native PH", field,
                    lambda field=field: obtain(recovered, lambda r: getattr(r, field)),
                    lambda field=field: obtain(inlet, lambda r: getattr(r, field)),
                    relative, absolute)
        compare("CoolProp HA wet bulb model comparison", "temperature_K",
                lambda: obtain(wet, lambda r: r.temperature_K),
                lambda: obtain(inlet, lambda s: HAPropsSI(
                    "B", "T", T, "P", p, "W", s.humidity_ratio_kg_water_kg_dry_air)), 0, .02)
        print(f"wet bulb / PH {len(rows)}/360 comparisons", flush=True)

    for args in INPUTS:
        point(*args)
    assert len(rows) == 360
    return {"development_evidence_only": True, "inputs": 36, "comparisons": 360,
            "passed": sum(r["passed"] for r in rows),
            "failed": sum(not r["passed"] for r in rows),
            "rows": rows, "reference": "CoolProp 8.0.0 HA, same T/P/W input, not same RH",
            "scope": "Thermodynamic adiabatic saturation with liquid/ice supply at outlet T;"
                     " not a kinetic psychrometer model or actual process inventory validation."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    path = parser.parse_args().output
    if path.exists():
        raise FileExistsError("retain earlier runs; select a fresh path")
    result = run()
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}))
    raise SystemExit(bool(result["failed"]))
