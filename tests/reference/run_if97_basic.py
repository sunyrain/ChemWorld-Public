"""Fixed NATIVE-CONV-25 basic-equation block; official tables and CoolProp8.

Run with --output <path>. Retain each run under a new output path.
IAPWS R7-97(2012), tables 5/15/18/33/35/36/42. All values below use SI.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from pathlib import Path

from chemworld.physchem.water_if97 import (
    water_if97_gibbs_state,
    water_if97_region3_state,
)
from chemworld.physchem.water_saturation import (
    water_saturation_pressure,
    water_saturation_temperature,
)

FIELDS = (
    "specific_volume_m3_kg", "enthalpy_J_kg", "internal_energy_J_kg",
    "entropy_J_kg_K", "cp_J_kg_K", "speed_of_sound_m_s",
)

# region, T, P (Pa) or rho (kg/m3) for region 3, six printed reference values.
OFFICIAL = (
    (1, 300, 3e6, (.100215168e-2, .115331273e6, .112324818e6,
                  .392294792e3, .417301218e4, .150773921e4)),
    (1, 300, 80e6, (.971180894e-3, .184142828e6, .106448356e6,
                   .368563852e3, .401008987e4, .163469054e4)),
    (1, 500, 3e6, (.120241800e-2, .975542239e6, .971934985e6,
                  .258041912e4, .465580682e4, .124071337e4)),
    (2, 300, 3500, (.394913866e2, .254991145e7, .241169160e7,
                   .852238967e4, .191300162e4, .427920172e3)),
    (2, 700, 3500, (.923015898e2, .333568375e7, .301262819e7,
                   .101749996e5, .208141274e4, .644289068e3)),
    (2, 700, 30e6, (.542946619e-2, .263149474e7, .246861076e7,
                   .517540298e4, .103505092e5, .480386523e3)),
    ("2m", 450, 1e6, (.192516540, .276881115e7, .257629461e7,
                     .656660377e4, .276349265e4, .498408101e3)),
    ("2m", 440, 1e6, (.186212297, .274015123e7, .255393894e7,
                     .650218759e4, .298166443e4, .489363295e3)),
    ("2m", 450, 1.5e6, (.121685206, .272134539e7, .253881758e7,
                       .629170440e4, .362795578e4, .481941819e3)),
    (3, 650, 500, (.255837018e8, .186343019e7, .181226279e7,
                  .405427273e4, .138935717e5, .502005554e3)),
    (3, 650, 200, (.222930643e8, .237512401e7, .226365868e7,
                  .485438792e4, .446579342e5, .383444594e3)),
    (3, 750, 500, (.783095639e8, .225868845e7, .210206932e7,
                  .446971906e4, .634165359e4, .760696041e3)),
    (5, 1500, .5e6, (.138455090e1, .521976855e7, .452749310e7,
                    .965408875e4, .261609445e4, .917068690e3)),
    (5, 1500, 30e6, (.230761299e-1, .516723514e7, .447495124e7,
                    .772970133e4, .272724317e4, .928548002e3)),
    (5, 2000, 30e6, (.311385219e-1, .657122604e7, .563707038e7,
                    .853640523e4, .288569882e4, .106736948e4)),
)


def official_state(region, temperature, coordinate):
    if region == 3:
        return water_if97_region3_state(temperature, coordinate)
    return water_if97_gibbs_state(temperature, coordinate, metastable=region == "2m")


def run():
    import CoolProp
    from CoolProp.CoolProp import PropsSI

    if CoolProp.__version__ != "8.0.0":
        raise RuntimeError("the fixed block requires uv.lock CoolProp8.0.0")
    rows = []
    start = time.monotonic()

    def compare(source, inputs, name, expected, function, relative, absolute):
        row = {"source": source, "inputs": inputs, "property": name, "expected": expected,
                   "relative_tolerance": relative, "absolute_tolerance": absolute}
        try:
            actual = float(function())
            row.update(actual=actual, absolute_error=abs(actual - expected),
                       passed=math.isclose(actual, expected, rel_tol=relative, abs_tol=absolute))
        except Exception as exc:
            row.update(passed=False, error=f"{type(exc).__name__}: {exc}")
        rows.append(row)

    for region, T, coord, expected in OFFICIAL:
        fields = ("pressure_Pa", *FIELDS[1:]) if region == 3 else FIELDS
        for name, value in zip(fields, expected, strict=True):
            compare("IAPWS printed table", [region, T, coord], name, value,
                    lambda r=region, t=T, c=coord, n=name: getattr(official_state(r, t, c), n),
                    2e-8, 1e-10)
    for T, p in ((300, .353658941e4), (500, .263889776e7), (600, .123443146e8)):
        compare("IAPWS table 35", [T], "pressure_Pa", p,
                lambda t=T: water_saturation_pressure(t), 2e-8, 1e-10)
    for p, T in ((.1e6, 372.755919), (1e6, 453.035632), (10e6, 584.149488)):
        compare("IAPWS table 36", [p], "temperature_K", T,
                lambda pressure=p: water_saturation_temperature(pressure), 2e-8, 1e-10)

    properties = {
        "specific_volume_m3_kg": "Dmass", "enthalpy_J_kg": "Hmass",
        "internal_energy_J_kg": "Umass", "entropy_J_kg_K": "Smass",
        "cp_J_kg_K": "Cpmass", "cv_J_kg_K": "Cvmass",
        "speed_of_sound_m_s": "A", "gibbs_energy_J_kg": "Gmass",
    }
    for region, temperatures, pressures in (
        (1, (300, 500, 620), (20e6, 50e6, 100e6)),
        (2, (650, 800, 1000), (1e3, 1e4, 1e5)),
        (5, (1100, 1500, 2200), (1e5, 1e7, 5e7)),
    ):
        for T in temperatures:
            for p in pressures:
                for name, code in properties.items():
                    try:
                        if code == "Gmass":
                            value = (PropsSI("Hmass", "T", T, "P", p, "IF97::Water")
                                     - T * PropsSI("Smass", "T", T, "P", p, "IF97::Water"))
                        else:
                            value = PropsSI(code, "T", T, "P", p, "IF97::Water")
                        if code == "Dmass":
                            value = 1 / value
                    except Exception as exc:
                        rows.append({"source": "CoolProp8 IF97", "inputs": [region, T, p],
                                         "property": name, "passed": False,
                                         "reference_error": f"{type(exc).__name__}: {exc}"})
                        continue
                    compare("CoolProp8 IF97", [region, T, p], name, value,
                            lambda t=T, pressure=p, n=name: getattr(
                                water_if97_gibbs_state(t, pressure), n), 2e-9, 1e-7)
        print(f"completed region {region}; {len(rows)}/312 comparisons", flush=True)
    return {
        "development_evidence_only": True,
        "question": "IF97 basic equations and analytic responses",
        "python": platform.python_version(), "reference_version": CoolProp.__version__,
        "official_inputs": 21, "independent_inputs": 27, "comparisons": len(rows),
        "passed": sum(row["passed"] for row in rows),
        "failed": sum(not row["passed"] for row in rows),
        "elapsed_s": time.monotonic() - start, "rows": rows,
        "reference_gibbs": "27 G values derived from reference H-T*S; 189 direct backend outputs",
        "boundary": "R3 T/rho official-table coverage; no R3 pressure inversion, two-phase flash, "
                 "ice, humid-air, solution, runtime or release qualification in this block.",
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
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}))
    raise SystemExit(bool(result["failed"]))
