"""Fixed NATIVE-CONV-25 ice block: all 177 comparisons, including failures."""

import argparse
import json
import math
from pathlib import Path

from chemworld.physchem.water_ice import ice_ih_gibbs_derivatives, ice_ih_state
from chemworld.physchem.water_ice_boundaries import (
    ice_sublimation_pressure,
    ice_sublimation_temperature,
    water_melting_pressure,
    water_melting_temperature,
)

# R10-06(2009), Table 6, all sixteen rows, mass-SI. The first six entries
# are g and its Hessian; the rest are h/f/u/s/Cp/rho/alpha/beta/kappaT/kappaS.
OFFICIAL_ICE = (
    (273.16, 611.657, (
        .611784135, .109085812737e-2, .122069433940e4, -.128495941571e-12,
        .174387964700e-6, -.767602985875e1, -.333444253966e6, -.55446875e-1,
        -.333444921197e6, -.122069433940e4, .209678431622e4, .916709492200e3,
        .159863102566e-3, .135714764659e7, .117793449348e-9, .114161597779e-9,
    )),
    (273.152519, 101325, (
        .10134274069e3, .109084388214e-2, .122076932550e4, -.128485364928e-12,
        .174362219972e-6, -.767598233365e1, -.333354873637e6, -.918701567e1,
        -.333465403393e6, -.122076932550e4, .209671391024e4, .916721463419e3,
        .159841589458e-3, .135705899321e7, .117785291765e-9, .114154442556e-9,
    )),
    (100, 100e6, (
        -.222296513088e6, .106193389260e-2, .261195122589e4, -.941807981761e-13,
        .274505162488e-7, -.866333195517e1, -.483491635676e6, -.328489902347e6,
        -.589685024936e6, -.261195122589e4, .866333195517e3, .941678203297e3,
        .258495528207e-4, .291466166994e6, .886880048115e-10, .886060982687e-10,
    )),
)
DERIVATIVES = ("value", "dx", "dy", "dxx", "dxy", "dyy")
PROPERTIES = ("enthalpy_J_kg", "helmholtz_energy_J_kg", "internal_energy_J_kg",
              "entropy_J_kg_K", "cp_J_kg_K", "mass_density_kg_m3",
              "thermal_expansion_K_inv", "pressure_coefficient_Pa_K",
              "isothermal_compressibility_Pa_inv", "isentropic_compressibility_Pa_inv")
MELTING_INPUTS = (("Ih", 260, 138.268e6), ("III", 254, 268.685e6),
                  ("V", 265, 479.640e6), ("VI", 320, 1356.76e6),
                  ("VII", 550, 6308.71e6))
MELTING_RANGES = (("Ih", 251.165, 273.16), ("III", 251.165, 256.164),
                  ("V", 256.164, 273.31), ("VI", 273.31, 355), ("VII", 355, 715))
SUBLIMATION_INPUTS = (50, 75, 100, 130, 180, 230, 260, 273.15, 273.16)


def run():
    import CoolProp
    from CoolProp.CoolProp import HAProps_Aux

    if CoolProp.__version__ != "8.0.0":
        raise RuntimeError("fixed block requires uv.lock CoolProp8.0.0")
    rows = []

    def compare(source, point, check, actual, expected, relative, absolute):
        row = {"source": source, "inputs": point, "check": check,
               "relative_tolerance": relative, "absolute_tolerance": absolute}
        try:
            a, e = float(actual()), float(expected())
            row.update(actual=a, expected=e, absolute_error=abs(a-e),
                       passed=math.isclose(a, e, rel_tol=relative, abs_tol=absolute))
        except Exception as exc:
            row.update(passed=False, error=f"{type(exc).__name__}: {exc}")
        rows.append(row)

    for T, p, values in OFFICIAL_ICE:
        fields = zip((*DERIVATIVES, *PROPERTIES), values, strict=True)
        for index, (name, value) in enumerate(fields):
            def actual(T=T, p=p, name=name, index=index):
                return getattr(ice_ih_gibbs_derivatives(T, p) if index < 6
                               else ice_ih_state(T, p), name)

            compare("IAPWS ice Table 6", [T, p], name, actual, lambda value=value: value,
                    2e-10, 1e-8 if index in (0, 6, 7, 8) else 0)
    for phase, T, p in MELTING_INPUTS:
        compare("IAPWS boundaries Table 3", [phase, T], "pressure_Pa",
                lambda T=T, phase=phase: water_melting_pressure(T, ice_phase=phase),
                lambda p=p: p, 5e-6, 0)
    compare("IAPWS boundaries Table 3", [230], "sublimation_pressure_Pa",
            lambda: ice_sublimation_pressure(230), lambda: 8.94735, 5e-6, 0)
    print(f"official {len(rows)}/177 comparisons", flush=True)
    keys = {"gibbs_energy_J_kg": "g_Ice", "enthalpy_J_kg": "h_Ice",
            "entropy_J_kg_K": "s_Ice", "mass_density_kg_m3": "rho_Ice",
            "isothermal_compressibility_Pa_inv": "kT"}
    for T in (50, 100, 180, 230, 260, 273.15):
        for p in (1000, 101325, 100e6):
            for name, key in keys.items():
                compare("CoolProp8 HAProps_Aux", [T, p], name,
                        lambda T=T, p=p, name=name: getattr(ice_ih_state(T, p), name),
                        lambda T=T, p=p, key=key: HAProps_Aux(key, T, p, 0)[0],
                        2e-10, 1e-20 if key == "kT" else 1e-7)
        print(f"ice implementation comparison {len(rows)}/177", flush=True)
    for T in SUBLIMATION_INPUTS:
        compare("CoolProp8 HAProps_Aux", [T], "sublimation_pressure_Pa",
                lambda T=T: ice_sublimation_pressure(T),
                lambda T=T: HAProps_Aux("psub_Ice", T, 101325, 0)[0], 2e-12, 0)
        compare("native sublimation round trip", [T], "temperature_K",
                lambda T=T: ice_sublimation_temperature(ice_sublimation_pressure(T)),
                lambda T=T: T, 0, 2e-8)
    for phase, low, high in MELTING_RANGES:
        for T in (low, (low+high)/2, high):
            compare("native melting round trip", [phase, T], "temperature_K",
                    lambda T=T, phase=phase: water_melting_temperature(
                        water_melting_pressure(T, ice_phase=phase), ice_phase=phase),
                    lambda T=T: T, 0, 2e-8)
    assert len(rows) == 177
    return {"development_evidence_only": True, "comparisons": 177,
            "passed": sum(r["passed"] for r in rows),
            "failed": sum(not r["passed"] for r in rows), "rows": rows,
            "reference_version": CoolProp.__version__, "reference_commit": CoolProp.__gitrevision__,
            "scope": "Official arithmetic and independent implementation checks, including "
                     "homogeneous metastable inputs; not global phase or humid-air validation."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    path = parser.parse_args().output
    if path.exists():
        raise FileExistsError("retain prior results; select a fresh path")
    result = run()
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}))
    raise SystemExit(bool(result["failed"]))
