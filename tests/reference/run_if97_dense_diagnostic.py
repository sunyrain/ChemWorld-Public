"""169 fixed Decimal checks; diagnostic arithmetic, not an independent EOS.

Explicit derivatives of official IF97 Eq.28/Table32; no production AD or
solver. Shared, previously verified Table30 coefficients remain a limitation.
"""

import argparse
import json
import math
from decimal import Decimal, localcontext
from pathlib import Path

from chemworld.physchem.water_if97_coefficients import REGION3, REGION3_LOG_COEFFICIENT
from chemworld.physchem.water_if97_solvers import water_if97_saturation, water_if97_tp
from chemworld.physchem.water_saturation import water_saturation_pressure

INPUTS = (
    (625, 17e6), (640, 20e6), (640, 22e6), (650, 22.3e6), (650, 25.6e6),
    (750, 78.31e6), (800, 80e6), (647.2, 22.1e6), (645, 22e6),
)
PROPERTIES = {
    "mass_density_kg_m3": "Dmass", "enthalpy_J_kg": "Hmass",
    "internal_energy_J_kg": "Umass", "entropy_J_kg_K": "Smass",
    "cp_J_kg_K": "Cpmass", "cv_J_kg_K": "Cvmass", "speed_of_sound_m_s": "A",
}


def decimal_state(temperature, density):
    T, rho = Decimal(str(temperature)), Decimal(str(density))
    delta, tau = rho / 322, Decimal("647.096") / T
    R, log_coefficient = Decimal("461.526"), Decimal(str(REGION3_LOG_COEFFICIENT))
    phi, dx, dy = log_coefficient * delta.ln(), log_coefficient / delta, Decimal(0)
    dxx, dxy, dyy = -log_coefficient / delta**2, Decimal(0), Decimal(0)
    for i, j, coefficient in REGION3:
        n = Decimal(str(coefficient))
        phi += n * delta**i * tau**j
        dx += n * i * delta**(i - 1) * tau**j
        dy += n * j * delta**i * tau**(j - 1)
        dxx += n * i * (i - 1) * delta**(i - 2) * tau**j
        dxy += n * i * j * delta**(i - 1) * tau**(j - 1)
        dyy += n * j * (j - 1) * delta**i * tau**(j - 2)
    stiffness = 2 * delta * dx + delta**2 * dxx
    coupling = delta * dx - delta * tau * dxy
    cv = -R * tau**2 * dyy
    return {
        "pressure_Pa": rho * R * T * delta * dx,
        "pressure_density_derivative": R * T * stiffness,
        "mass_density_kg_m3": rho,
        "enthalpy_J_kg": R * T * (tau * dy + delta * dx),
        "internal_energy_J_kg": R * T * tau * dy,
        "entropy_J_kg_K": R * (tau * dy - phi),
        "cp_J_kg_K": cv + R * coupling**2 / stiffness,
        "cv_J_kg_K": cv,
        "speed_of_sound_m_s": (R * T * (stiffness - coupling**2 / (tau**2 * dyy))).sqrt(),
    }


def decimal_pressure_root(temperature, pressure, seed):
    rho, target = Decimal(str(seed)), Decimal(str(pressure))
    for iteration in range(80):
        state = decimal_state(temperature, rho)
        residual = state["pressure_Pa"] - target
        if abs(residual) <= Decimal("1e-35"):
            return state, iteration + 1
        rho -= residual / state["pressure_density_derivative"]
    raise ValueError(f"Decimal pressure root did not converge: {residual}")


def run():
    import CoolProp
    from CoolProp.CoolProp import PropsSI

    if (CoolProp.__version__ != "8.0.0"
            or CoolProp.__gitrevision__ != "ae81610e7d23efc57f9d051c8e70a4d66e87537f"):
        raise ValueError("diagnostic requires the fixed CoolProp8 reference build")
    inputs = [(T, p, None) for T, p in INPUTS]
    inputs += [(T, water_saturation_pressure(T), q) for T in (640, 646.5) for q in (0, 1)]
    rows, diagnostics = [], []
    for T, p, quality in inputs:
        point = {"T": T, "P": p, "quality": quality}
        try:
            if quality is None:
                reference = {name: PropsSI(code, "T", T, "P", p, "IF97::Water")
                             for name, code in PROPERTIES.items()}
                native = water_if97_tp(T, p)
            else:
                T = PropsSI("T", "P", p, "Q", quality, "IF97::Water")
                reference = {name: PropsSI(code, "P", p, "Q", quality, "IF97::Water")
                             for name, code in PROPERTIES.items()}
                coexistence = water_if97_saturation(p, quality)
                native = coexistence.liquid if quality == 0 else coexistence.vapor
            with localcontext() as context:
                context.prec = 50
                seed = reference["mass_density_kg_m3"]
                at_reference_density = decimal_state(T, seed)
                exact, iterations = decimal_pressure_root(T, p, seed)
            diagnostics.append({**point, "reference_density_pressure_residual_Pa":
                                float(at_reference_density["pressure_Pa"]) - p,
                                "root_iterations": iterations,
                                "exact_density_kg_m3": float(exact["mass_density_kg_m3"]),
                                "reference_density_kg_m3": seed})
            for name in PROPERTIES:
                actual, expected = getattr(native, name), float(exact[name])
                rows.append({**point, "check": "native versus Decimal root:" + name,
                             "actual": actual, "expected": expected,
                             "relative_tolerance": 2e-7, "absolute_tolerance": 1e-6,
                             "passed": math.isclose(actual, expected, rel_tol=2e-7, abs_tol=1e-6)})
                if name == "mass_density_kg_m3":
                    continue
                actual, expected = reference[name], float(at_reference_density[name])
                rows.append({**point, "check": "reference at its own density:" + name,
                             "actual": actual, "expected": expected,
                             "relative_tolerance": 2e-9, "absolute_tolerance": 1e-6,
                             "passed": math.isclose(actual, expected, rel_tol=2e-9, abs_tol=1e-6)})
        except Exception as exc:
            rows.extend({**point, "check": name, "passed": False,
                         "error": f"{type(exc).__name__}: {exc}"}
                        for name in [*("native:" + n for n in PROPERTIES),
                                     *("reference:" + n for n in list(PROPERTIES)[1:])])
        print(f"dense diagnostic {len(diagnostics)}/13; {len(rows)}/169 checks", flush=True)
    assert len(rows) == 169
    return {"development_evidence_only": True, "inputs": 13, "comparisons": 169,
            "passed": sum(r["passed"] for r in rows),
            "failed": sum(not r["passed"] for r in rows), "rows": rows,
            "diagnostics": diagnostics,
            "limitation": "Shared official coefficients; independent arithmetic/derivatives/root, "
                          "not a third EOS or replacement of the original 82 differences."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    path = parser.parse_args().output
    if path.exists():
        raise FileExistsError("retain all diagnostic runs; choose a fresh path")
    result = run()
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("rows", "diagnostics")}))
    raise SystemExit(bool(result["failed"]))
