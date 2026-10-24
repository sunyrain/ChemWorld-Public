"""Complete fixed official tables: G8 150 + IAPWS-95 56 = 206 comparisons."""

import argparse
import json
import math
from pathlib import Path

from tests.reference import humid_air_official as official

from chemworld.physchem.humid_air import (
    _pure_energy,
    humid_air_cross_virials,
    humid_air_derivatives,
    humid_air_state,
)
from chemworld.physchem.humid_air_parameters import iapws95_water, iapws_dry_air
from chemworld.physchem.second_order_pair import SecondOrderPair


def run():
    rows = []

    def compare(source, inputs, name, value, function, relative, absolute=0):
        row = {"source": source, "inputs": inputs, "check": name, "expected": value,
               "relative_tolerance": relative, "absolute_tolerance": absolute}
        try:
            actual = float(function())
            row.update(actual=actual, absolute_error=abs(actual-value),
                       passed=math.isclose(actual, value, rel_tol=relative, abs_tol=absolute))
        except Exception as exc:
            row.update(passed=False, error=f"{type(exc).__name__}: {exc}")
        rows.append(row)

    for i, point in enumerate(official.G8_INPUTS):
        A, T, rho = point
        for field, expected in zip(official.DERIVATIVES,
                                   official.TOTAL_DERIVATIVES[i], strict=True):
            compare("G8 Table 13", point, field, expected,
                    lambda point=point, field=field: humid_air_derivatives(*point)[field],
                    2e-5 if field == "f_A" else 5e-7)
        for field, expected in zip(official.STATE_FIELDS, official.STATES[i], strict=True):
            compare("G8 Table 13", point, field, expected,
                    lambda point=point, field=field: getattr(humid_air_state(*point), field), 5e-7)
        for label, fluid, density, values in (
            ("air", iapws_dry_air(), A*rho, official.AIR[i]),
            ("vapor", iapws95_water(), (1-A)*rho, official.VAPOR[i]),
        ):
            compare("G8 Table 14", point, label+"_density", values[0],
                    lambda density=density: density, 5e-7)
            for field, expected in zip(("value", "dy", "dx", "dyy", "dxy", "dxx"),
                                       values[1:], strict=True):
                compare("G8 Table 14", point, label+"_"+field, expected,
                        lambda fluid=fluid, density=density, T=T, field=field: getattr(
                            _pure_energy(fluid, SecondOrderPair(T, dy=1),
                                         SecondOrderPair(density, dx=1)), field), 5e-7)
        for field, expected in zip(official.DERIVATIVES,
                                   official.CROSS_DERIVATIVES[i], strict=True):
            compare("G8 Table 15", point, field, expected,
                    lambda point=point, field=field: humid_air_derivatives(
                        *point, contribution="cross")[field], 2e-8)
        for j, expected in enumerate(official.CROSS_VIRIALS[i]):
            field = ("value", "dx", "dxx")[j % 3]
            compare("G8 Table 15", point, f"virial_{j//3}_{field}", expected,
                    lambda T=T, j=j, field=field: getattr(humid_air_cross_virials(
                        SecondOrderPair(T, dx=1))[j//3], field), 2e-8)
        print(f"G8 full tables {len(rows)}/206 comparisons", flush=True)
    water = iapws95_water()
    for kind, values in official.WATER_ALPHA.items():
        for orders, expected in zip(((0, 0), (1, 0), (2, 0), (0, 1), (0, 2), (1, 1)),
                                    values, strict=True):
            compare("R6 Table 6", [500, 838.025], f"{kind}_{orders}", expected,
                    lambda kind=kind, orders=orders: water.alpha_derivative(
                        kind, delta=838.025/322, tau=647.096/500,
                        delta_order=orders[0], tau_order=orders[1]), 2e-8, 1e-10)
    for T, rho, values in official.WATER_STATES:
        for field, expected in zip(("pressure_Pa", "cv_J_mol_K", "speed_of_sound_m_s",
                                    "entropy_J_mol_K"), values, strict=True):
            compare("R6 Table 7", [T, rho], field.replace("_J_mol_", "_J_kg_"), expected,
                    lambda T=T, rho=rho, field=field: getattr(water.state(
                        temperature_K=T, molar_density_mol_m3=rho/water.molar_mass_kg_mol), field)
                        / (water.molar_mass_kg_mol if field in ("cv_J_mol_K", "entropy_J_mol_K")
                           else 1), 2e-8, 1e-10)
        print(f"R6 single-phase tables {len(rows)}/206", flush=True)
    assert len(rows) == 206
    return {"development_evidence_only": True, "comparisons": 206,
            "passed": sum(row["passed"] for row in rows),
            "failed": sum(not row["passed"] for row in rows), "rows": rows,
            "sources": ["https://iapws.org/technical-guidance/release/SeaAir",
                        "https://iapws.org/technical-guidance/release/IAPWS-95"],
            "scope": "All G8 Tables 13--15 and R6 Tables 6--7; arithmetic qualification, "
                     "not real equipment or saturated-process validation."}


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
