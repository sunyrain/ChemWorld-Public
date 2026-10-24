"""Fixed 379-comparison development block; every error and model difference retained."""

import argparse
import json
import math
from pathlib import Path

from tests.reference.humid_air_official import G8_INPUTS, STATES

from chemworld.physchem import (
    condensed_water_state,
    humid_air_condensation_point,
    humid_air_from_relative_humidity,
    humid_air_relative_humidity,
    humid_air_saturation,
)

GRID = (tuple((T, p) for T in (193, 230, 260, 270) for p in (1e4, 101325, 5e6))
        + tuple((T, p) for T in (280, 300, 350, 473) for p in (2e6, 3e6, 5e6))
        + tuple((T, p) for T in (273, 273.16) for p in (101325, 5e6)))


def run():
    from CoolProp.CoolProp import HAPropsSI, PropsSI

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

    def compare(source, inputs, name, expected, actual, relative, absolute):
        row = {"source": source, "inputs": inputs, "check": name,
               "relative_tolerance": relative, "absolute_tolerance": absolute}
        try:
            a, b = float(actual()), float(expected())
            row.update(actual=a, expected=b, absolute_error=abs(a-b),
                       passed=math.isclose(a, b, rel_tol=relative, abs_tol=absolute))
        except Exception as exc:
            row.update(passed=False, error=f"{type(exc).__name__}: {exc}")
        rows.append(row)

    for i, ((_, T, rho), p) in enumerate(zip(G8_INPUTS, (1., 1e5, 1e6), strict=True)):
        saturated = evaluate(lambda T=T, p=p: humid_air_saturation(T, p))
        expectations = {"dry_air_mass_fraction": G8_INPUTS[i][0], "mass_density_kg_m3": rho,
                        "water_chemical_potential_J_kg": STATES[i][4],
                        "enthalpy_J_kg": STATES[i][1], "entropy_J_kg_K": STATES[i][3],
                        "cp_J_kg_K": STATES[i][5], "speed_of_sound_m_s": STATES[i][6]}
        for field, value in expectations.items():
            compare("G8 Table 13 saturation", [T, p], field, lambda value=value: value,
                    lambda field=field, saturated=saturated:
                        obtain(saturated, lambda result: getattr(result.gas, field)), 5e-7, 1e-8)
    def gas_point(T, p):
        saturated = evaluate(lambda: humid_air_saturation(T, p))
        dew = evaluate(lambda: obtain(saturated, lambda result: humid_air_condensation_point(
            result.gas.water_mass_fraction, p)))
        compare("native closure", [T, p], "chemical_potential_residual_J_kg", lambda: 0,
                lambda: obtain(saturated, lambda r: r.chemical_potential_residual_J_kg), 0, 1e-5)
        compare("native closure", [T, p], "pressure_Pa", lambda: p,
                lambda: obtain(saturated, lambda r: r.gas.pressure_Pa), 5e-11, 0)
        compare("native closure", [T, p], "condensation_point_K", lambda: T,
                lambda: obtain(dew, lambda r: r.temperature_K), 0, 2e-7)
        for rh in (.25, .6, 1.):
            state = evaluate(lambda rh=rh: humid_air_from_relative_humidity(T, p, rh))
            compare("native closure", [T, p, rh], "WMO_relative_humidity", lambda rh=rh: rh,
                    lambda state=state: obtain(state, lambda s: humid_air_relative_humidity(
                        s.water_mass_fraction, T, p)), 2e-9, 0)
            compare("native closure", [T, p, rh], "positive_supersaturation_mu_J_kg", lambda: 0,
                    lambda state=state: obtain(state, lambda s: obtain(saturated, lambda r: max(
                        0., s.water_chemical_potential_J_kg-r.condensate.gibbs_energy_J_kg))),
                    0, 1e-5)
        compare("CoolProp HA model comparison", [T, p], "W_kg_water_kg_dry_air",
                lambda: HAPropsSI("W", "T", T, "P", p, "R", 1),
                lambda: obtain(saturated, lambda r: r.gas.humidity_ratio_kg_water_kg_dry_air),
                2e-5, 1e-12)
        compare("CoolProp HA model comparison", [T, p], "density_kg_m3",
                lambda: 1/HAPropsSI("Vha", "T", T, "P", p, "R", 1),
                lambda: obtain(saturated, lambda r: r.gas.mass_density_kg_m3), 2e-5, 1e-10)
        print(f"humid equilibrium {len(rows)}/379 comparisons", flush=True)

    def liquid_point(T, p):
        state = evaluate(lambda: condensed_water_state(T, p, phase="liquid"))
        for field, key in (("mass_density_kg_m3", "Dmass"), ("enthalpy_J_kg", "Hmass"),
                           ("entropy_J_kg_K", "Smass"), ("gibbs_energy_J_kg", "Gmass"),
                           ("cp_J_kg_K", "Cpmass")):
            compare("CoolProp HEOS liquid water", [T, p], field,
                    lambda key=key: PropsSI(key, "T|liquid", T, "P", p, "HEOS::Water"),
                    lambda field=field: obtain(state, lambda s: getattr(s, field)), 2e-8, 1e-6)
        print(f"liquid water {len(rows)}/379 comparisons", flush=True)

    for T, p in GRID:
        gas_point(T, p)
    for T in (270, 273, 273.16, 300, 473):
        for p in (2e6, 5e6):
            liquid_point(T, p)
    assert len(rows) == 379
    return {"development_evidence_only": True, "comparisons": len(rows),
            "passed": sum(row["passed"] for row in rows),
            "failed": sum(not row["passed"] for row in rows), "rows": rows,
            "reference": {"CoolProp": "8.0.0", "model_difference":
                "G8 complete pure EOS and official cross virials, neglecting dissolved air; "
                "HA uses RP1485 truncation and its enhancement-factor conventions."},
            "scope": "Homogeneous humid gas / pure condensate equilibrium; no process inventory."}


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
