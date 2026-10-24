"""Fixed 24-state / 204-check native volume and analytic U response block."""

import argparse
import json
import math
from pathlib import Path

from chemworld.physchem import (
    humid_air_from_relative_humidity,
    humid_air_saturation,
    humid_air_volume,
)

INPUTS = tuple((T, phase, p) for T, phase in ((230., "ice"), (300., "liquid"), (400., "liquid"))
               for p in (1e6, 2e6))


def run():
    rows, cases = [], []

    def capture(function):
        try:
            return function()
        except Exception as exc:
            return exc

    def obtain(value, function):
        if isinstance(value, Exception):
            raise value
        return function(value)

    def point(kind, setting, T, phase, p):
        source = {"kind": kind, "T_K": T, "surface": phase, "p_Pa": p,
                  "excess_water_kg_kg_dry" if kind == "coexistence" else "RH": setting}

        def setup():
            a = .02
            if kind == "coexistence":
                sat = humid_air_saturation(T, p, condensed_phase=phase)
                gas, cond = sat.gas, sat.condensate
                c = a*setting
                uc = cond.enthalpy_J_kg-p/cond.mass_density_kg_m3
                vc = c/cond.mass_density_kg_m3
            else:
                gas = humid_air_from_relative_humidity(T, p, setting, condensed_phase=phase)
                c, uc, vc = 0., 0., 0.
            mg = a/gas.dry_air_mass_fraction
            b = mg*gas.water_mass_fraction+c
            volume = mg/gas.mass_density_kg_m3+vc
            expected = {"p": p, "rho": gas.mass_density_kg_m3,
                        "W": gas.humidity_ratio_kg_water_kg_dry_air,
                        "condensed": c, "U": mg*gas.internal_energy_J_kg+c*uc,
                        "mu": 0., "volume": volume}
            return a, b, volume, expected

        data = capture(setup)
        state = capture(lambda: obtain(data, lambda v: humid_air_volume(
            T, *v[:3], condensed_phase=phase)))
        cases.append({"inputs": source,
                      "result": str(state) if isinstance(state, Exception)
                      else state.to_dict()})

        def compare(name, actual, expected, relative, absolute):
            row = {"inputs": source, "check": name, "relative_tolerance": relative,
                   "absolute_tolerance": absolute}
            try:
                a, b = float(actual()), float(expected())
                row.update(actual=a, expected=b, absolute_error=abs(a-b),
                           passed=math.isclose(a, b, rel_tol=relative, abs_tol=absolute))
            except Exception as exc:
                row.update(passed=False, error=f"{type(exc).__name__}: {exc}")
            rows.append(row)

        fields = {"p": lambda s: s.gas.pressure_Pa,
                  "rho": lambda s: s.gas.mass_density_kg_m3,
                  "U": lambda s: s.internal_energy_J,
                  "condensed": lambda s: s.condensed_water_mass_kg}
        if kind == "coexistence":
            fields.update(W=lambda s: s.gas.humidity_ratio_kg_water_kg_dry_air,
                          mu=lambda s: s.chemical_potential_residual_J_kg,
                          volume=lambda s: s.gas_volume_m3+s.condensed_volume_m3)
        for name, function in fields.items():
            relative = 0 if name == "mu" else 2e-11 if name == "volume" else 2e-9
            absolute = {"U": 1e-6, "condensed": 1e-12, "mu": 1e-5}.get(name, 0.)
            compare(name, lambda f=function: obtain(state, f),
                    lambda name=name: obtain(data, lambda d: d[3][name]),
                    relative, absolute)

        def finite_difference(index):
            a, b, volume, _ = obtain(data, lambda d: d)
            values = [T, a, b]
            h = .001 if index == 0 else a*1e-5 if index == 1 else max(b*1e-5, 1e-10)
            forward = index == 2 and b == 0
            if forward:
                h = 1e-9
            upper, lower = list(values), list(values)
            upper[index] += h
            lower[index] -= 0 if forward else h
            energies = [humid_air_volume(*v, volume, condensed_phase=phase).internal_energy_J
                        for v in (upper, lower)]
            return (energies[0]-energies[1])/(h if forward else 2*h)

        for index, name in enumerate(("heat_capacity_J_K", "dair_internal_energy_J_kg",
                                       "dwater_internal_energy_J_kg")):
            compare(name, lambda name=name: obtain(state, lambda s: getattr(s, name)),
                    lambda index=index: finite_difference(index), 2e-4, .01)
        print(f"humid-air volume {len(cases)}/24 states, {len(rows)}/204 checks", flush=True)

    for kind, settings in (("coexistence", (.01, .2)), ("gas", (0., .4))):
        for T, phase, p in INPUTS:
            for setting in settings:
                point(kind, setting, T, phase, p)
    assert len(rows) == 204
    return {"development_evidence_only": True, "comparisons": 204, "states": 24,
            "passed": sum(r["passed"] for r in rows), "failed": sum(not r["passed"] for r in rows),
            "rows": rows, "cases": cases,
            "scope": "Native phase-volume/inventory closure and independent finite differences; "
                     "not an independent experimental validation of G8."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    path = parser.parse_args().output
    if path.exists():
        raise FileExistsError("retain earlier runs; select a fresh path")
    result = run()
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in {"rows", "cases"}}))
    raise SystemExit(bool(result["failed"]))
