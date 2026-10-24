"""Fixed 42-input/526-comparison IF97 pressure/caloric development block."""

import argparse
import json
import math
import time
from pathlib import Path

from chemworld.physchem.water_if97_solvers import (
    water_if97_at_pressure,
    water_if97_saturation,
    water_if97_tp,
)
from chemworld.physchem.water_saturation import water_saturation_pressure

TP_INPUTS = (
    (273.15, .1e6), (300, 3e6), (500, 3e6), (623.15, 50e6),
    (300, 3500), (700, 3500), (700, 30e6), (863.15, 100e6), (1073.15, 100e6),
    (625, 17e6), (640, 20e6), (640, 22e6), (650, 22.3e6), (650, 25.6e6),
    (750, 78.31e6), (800, 80e6), (647.2, 22.1e6), (645, 22e6),
    (1073.1501, 50e6), (1500, .5e6), (2000, 30e6), (2273.15, 50e6),
)
SAT_INPUTS = tuple((T, q) for T in (300, 500, 623.15, 640, 646.5) for q in (0, .2, .8, 1))
PROPERTIES = {
    "mass_density_kg_m3": "Dmass", "enthalpy_J_kg": "Hmass",
    "internal_energy_J_kg": "Umass", "entropy_J_kg_K": "Smass",
    "cp_J_kg_K": "Cpmass", "cv_J_kg_K": "Cvmass", "speed_of_sound_m_s": "A",
}


def run():
    import CoolProp
    from CoolProp.CoolProp import PropsSI

    if CoolProp.__version__ != "8.0.0":
        raise RuntimeError("fixed reference is uv.lock CoolProp8.0.0")
    rows = []
    start = time.monotonic()

    def compare(inputs, check, actual_fn, expected_fn, relative=2e-7, absolute=1e-6):
        row = {"inputs": inputs, "check": check, "relative_tolerance": relative,
               "absolute_tolerance": absolute}
        try:
            actual, expected = float(actual_fn()), float(expected_fn())
            if not math.isfinite(actual) or not math.isfinite(expected):
                raise ValueError("nonfinite comparison")
            row.update(actual=actual, expected=expected, absolute_error=abs(actual - expected),
                       passed=math.isclose(actual, expected, rel_tol=relative, abs_tol=absolute))
        except Exception as exc:
            row.update(passed=False, error=f"{type(exc).__name__}: {exc}")
        rows.append(row)

    for T, p in TP_INPUTS:
        def state(T=T, p=p):
            return water_if97_tp(T, p)

        inputs = {"T": T, "P": p}
        for name, code in PROPERTIES.items():
            compare(inputs, "reference:" + name,
                    lambda name=name: getattr(state(), name),
                    lambda code=code, T=T, p=p: PropsSI(code, "T", T, "P", p, "IF97::Water"))
        for target in ("enthalpy_J_kg", "entropy_J_kg_K"):
            def inverse(target=target, p=p):
                original = state()
                return water_if97_at_pressure(p, region=original.region,
                                              **{target: getattr(original, target)})

            for name in ("temperature_K", "mass_density_kg_m3", target):
                compare(inputs, target + " roundtrip:" + name,
                        lambda name=name: getattr(inverse(), name),
                        lambda name=name: getattr(state(), name),
                        0 if name == "temperature_K" else 2e-9)
        print(f"TP {T:g} K, {p:g} Pa; {len(rows)}/526", flush=True)
    for T, quality in SAT_INPUTS:
        p = water_saturation_pressure(T)

        def coexistence(p=p, quality=quality):
            return water_if97_saturation(p, quality)

        inputs = {"T": T, "P": p, "quality": quality}
        for name, code in {
            "temperature_K": "T", "mass_density_kg_m3": "Dmass",
            "enthalpy_J_kg": "Hmass", "internal_energy_J_kg": "Umass",
            "entropy_J_kg_K": "Smass", "specific_volume_m3_kg": "Dmass",
        }.items():
            def reference(name=name, code=code, p=p, quality=quality):
                value = PropsSI(code, "P", p, "Q", quality, "IF97::Water")
                return 1 / value if name == "specific_volume_m3_kg" else value

            compare(inputs, "reference:" + name,
                    lambda name=name: getattr(coexistence(), name), reference)
        for target in ("enthalpy_J_kg", "entropy_J_kg_K"):
            def coexistence_inverse(target=target, p=p):
                return water_if97_at_pressure(p, **{target: getattr(coexistence(), target)})

            compare(inputs, target + " roundtrip:T", lambda: coexistence_inverse().temperature_K,
                    lambda T=T: T, 0)
            compare(inputs, target + " roundtrip:quality",
                    lambda: coexistence_inverse().vapor_mass_fraction,
                    lambda quality=quality: quality, 0)
        compare(inputs, "h=u+Pv", lambda: coexistence().enthalpy_J_kg,
                lambda: coexistence().internal_energy_J_kg + coexistence().pressure_Pa
                * coexistence().specific_volume_m3_kg, 2e-10)
        compare(inputs, "mass volume lever rule", lambda: coexistence().specific_volume_m3_kg,
                lambda quality=quality: (1 - quality) * coexistence().liquid.specific_volume_m3_kg
                + quality * coexistence().vapor.specific_volume_m3_kg, 2e-10)
        print(f"coexistence {T:g} K, quality {quality:g}; {len(rows)}/526", flush=True)
    assert len(rows) == 526
    return {
        "development_evidence_only": True, "reference_version": CoolProp.__version__,
        "tp_inputs": 22, "coexistence_inputs": 20, "comparisons": len(rows),
        "passed": sum(r["passed"] for r in rows), "failed": sum(not r["passed"] for r in rows),
        "elapsed_s": time.monotonic() - start, "rows": rows,
        "boundary": "PH/PS roundtrips use the original homogeneous region at IF97 caloric joins; "
                    "two-phase Cp/Cv/acoustics remain undefined; "
                    "no runtime integration in this block",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    path = parser.parse_args().output
    if path.exists():
        raise FileExistsError("retain prior runs; choose a fresh output path")
    result = run()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}))
    raise SystemExit(bool(result["failed"]))
