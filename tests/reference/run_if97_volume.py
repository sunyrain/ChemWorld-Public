"""Fixed T/rho and energy-response block, 42 + 10 inputs and 230 checks."""

import argparse
import json
import math
from pathlib import Path

from tests.reference.run_if97_states import SAT_INPUTS, TP_INPUTS

from chemworld.physchem.water_if97_solvers import water_if97_saturation, water_if97_tp
from chemworld.physchem.water_if97_volume import (
    water_if97_internal_energy_derivatives,
    water_if97_trho,
)
from chemworld.physchem.water_saturation import water_saturation_pressure


def run():
    rows = []

    def compare(point, check, actual, expected, relative, absolute):
        row = {"inputs": point, "check": check, "relative_tolerance": relative,
               "absolute_tolerance": absolute}
        try:
            a, e = float(actual()), float(expected())
            row.update(actual=a, expected=e, passed=math.isclose(a, e, rel_tol=relative,
                                                                abs_tol=absolute))
        except Exception as exc:
            row.update(passed=False, error=f"{type(exc).__name__}: {exc}")
        rows.append(row)

    inputs = [(T, p, None) for T, p in TP_INPUTS]
    inputs += [(T, water_saturation_pressure(T), q) for T, q in SAT_INPUTS]
    for T, p, quality in inputs:
        def original(T=T, p=p, quality=quality):
            return water_if97_tp(T, p) if quality is None else water_if97_saturation(p, quality)

        def recovered(T=T):
            return water_if97_trho(T, original().mass_density_kg_m3)

        for key in ("pressure_Pa", "mass_density_kg_m3", "enthalpy_J_kg",
                    "internal_energy_J_kg", "entropy_J_kg_K"):
            compare({"T": T, "P": p, "quality": quality}, key,
                    lambda key=key: getattr(recovered(), key),
                    lambda key=key: getattr(original(), key), 2e-8, 1e-5)
        print(f"T/rho {len(rows)}/230 comparisons", flush=True)
    derivative_inputs = [(T, p, None) for T, p in (
        (300, 3e6), (700, 3500), (640, 22e6), (640, 19e6), (650, 25.6e6), (1500, .5e6))]
    derivative_inputs += [(T, water_saturation_pressure(T), q)
                          for T in (500, 640) for q in (.2, .8)]
    for T, p, quality in derivative_inputs:
        def base_state(T=T, p=p, quality=quality):
            return water_if97_tp(T, p) if quality is None else water_if97_saturation(p, quality)

        def analytic(T=T):
            return water_if97_internal_energy_derivatives(
                water_if97_trho(T, base_state().mass_density_kg_m3))

        def numerical(index, T=T):
            rho = base_state().mass_density_kg_m3
            if index == 0:
                left, right, step = (T - .001, rho), (T + .001, rho), .002
            else:
                left, right, step = (T, rho * (1 - 1e-5)), (T, rho * (1 + 1e-5)), 2e-5 * rho
            return (water_if97_trho(*right).internal_energy_J_kg
                    - water_if97_trho(*left).internal_energy_J_kg) / step

        for index, key in enumerate(("du/dT at rho", "du/drho at T")):
            compare({"T": T, "P": p, "quality": quality}, key,
                    lambda index=index: analytic()[index],
                    lambda index=index: numerical(index), 5e-6, 1e-4)
        print(f"isochoric derivatives {len(rows)}/230 comparisons", flush=True)
    assert len(rows) == 230
    return {"development_evidence_only": True, "state_inputs": 42, "derivative_inputs": 10,
            "comparisons": 230, "passed": sum(r["passed"] for r in rows),
            "failed": sum(not r["passed"] for r in rows), "rows": rows,
            "scope": "Fixed-volume inversion and derivative consistency; same official EOS, "
                     "not an independent model or real-equipment qualification."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    path = parser.parse_args().output
    if path.exists():
        raise FileExistsError("retain prior runs; choose a fresh path")
    result = run()
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}))
    raise SystemExit(bool(result["failed"]))
