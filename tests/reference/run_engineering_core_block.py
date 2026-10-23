"""Fixed complete public core family coverage; reference imports stay here."""

import argparse
import inspect
import json
import time
from math import isclose
from pathlib import Path

import fluids
import fluids.core as ref

from chemworld.foundation.units import convert_value
from chemworld.physchem.engineering_numbers import (
    ENGINEERING_GROUP_PARAMETERS,
    convert_viscosity,
    current_fluid_similarity,
    engineering_group,
    engineering_resistance_conversion,
    ideal_gas_sound_speed,
    local_gravity,
    thermal_diffusivity_si,
)
from chemworld.physchem.transport import FluidState

TEMPERATURE_UNITS = {"C": "degC", "K": "K", "F": "degF", "R": "degR"}
BASE = {
    "L": 0.02,
    "rhof": 1000,
    "rhop": 2500,
    "mu": 0.001,
    "g": 9.80665,
    "dP": 300,
    "alpha": 1.5e-7,
    "K": 0.002,
    "h": 100,
    "k": 0.6,
    "G": 200,
    "q": 500,
    "Hvap": 2e6,
    "rhol": 1000,
    "rhog": 2,
    "sigma": 0.07,
    "V": 2,
    "P": 2e5,
    "Psat": 3000,
    "rho": 1000,
    "D": 0.01,
    "Re": 10000,
    "Di": 0.01,
    "F": 1,
    "A": 0.01,
    "Cp": 4000,
    "dT": 20,
    "t": 5,
    "rho1": 1000,
    "rho2": 800,
    "x": 0.1,
    "beta": 0.003,
    "T1": 320,
    "T2": 300,
    "nu": 1e-6,
    "fd": 0.02,
    "Te": 10,
    "path": 1e-8,
    "c": 340,
    "mul": 0.001,
    "N": 2,
    "Pr": 7,
    "Gr": 200,
    "Dp": 1e-5,
    "f": 10,
    "L_D": 100,
    "head": 10,
    "roughness": 1e-5,
    "latitude": 30,
    "H": 1000,
    "T": 300,
    "MW": 29,
}
SCALES = (0.5, 1, 2, 4)


def base_parameters(name, scale):
    signature = inspect.signature(getattr(ref, name))
    parameters = {key: BASE[key] * scale for key in signature.parameters if key in BASE}
    if name in ("Fourier_heat", "Graetz_heat", "Lewis", "Peclet_heat"):
        parameters.pop("alpha")
    if name in ("Grashof", "Reynolds", "Schmidt"):
        parameters.pop("nu")
    if name == "Prandtl":
        for key in ("nu", "rho", "alpha"):
            parameters.pop(key)
    if name == "nu_mu_converter":
        parameters.pop("nu")
    return parameters


def inputs():
    rows = []
    for name in ref.__all__:
        for index, scale in enumerate(SCALES):
            if name in {
                a + "2" + b for a in TEMPERATURE_UNITS for b in TEMPERATURE_UNITS if a != b
            }:
                parameters = {
                    name[0]: convert_value(
                        (200, 273.15, 300, 500)[index], "K", TEMPERATURE_UNITS[name[0]]
                    )
                }
            else:
                parameters = base_parameters(name, scale)
            rows.append({"family": "base", "method": name, "parameters": parameters})
    alternatives = (
        ("Reynolds", {"V": 2, "D": 0.01, "nu": 1e-6}),
        ("Prandtl", {"nu": 1e-6, "rho": 1000, "Cp": 4000, "k": 0.6}),
        ("Prandtl", {"nu": 1e-6, "alpha": 1.5e-7}),
        ("Fourier_heat", {"t": 5, "L": 0.02, "alpha": 1.5e-7}),
        ("Graetz_heat", {"V": 2, "D": 0.01, "x": 0.1, "alpha": 1.5e-7}),
        ("Peclet_heat", {"V": 2, "L": 0.02, "alpha": 1.5e-7}),
        ("Lewis", {"D": 1e-9, "alpha": 1.5e-7}),
        ("Grashof", {"L": 0.02, "beta": 0.003, "T1": 320, "T2": 300, "nu": 1e-6}),
        ("Schmidt", {"D": 1e-9, "nu": 1e-6}),
        ("Froude", {"V": 2, "L": 0.02, "squared": True}),
        ("Froude_densimetric", {"V": 2, "L": 0.02, "rho1": 1000, "rho2": 800, "heavy": False}),
        ("nu_mu_converter", {"rho": 1000, "nu": 1e-6}),
    )
    for name, parameters in alternatives:
        for scale in SCALES:
            rows.append(
                {
                    "family": "alternative",
                    "method": name,
                    "parameters": {
                        k: v if isinstance(v, bool) else v * scale for k, v in parameters.items()
                    },
                }
            )
    for name, key, negative in (
        ("Archimedes", "rhop", 800),
        ("Bond", "rhog", 1200),
        ("Eotvos", "rhog", 1200),
        ("Morton", "rhog", 1200),
        ("Grashof", "beta", -0.003),
        ("Euler", "dP", -300),
        ("Cavitation", "Psat", 3e5),
        ("Eckert", "dT", -20),
        ("Jakob", "Te", -10),
        ("Rayleigh", "Gr", -200),
    ):
        for scale in SCALES:
            rows.append(
                {
                    "family": "signed",
                    "method": name,
                    "parameters": {**base_parameters(name, scale), key: negative * scale},
                }
            )
    for scale in SCALES:
        rows.append({"family": "current_fluid", "parameters": {"scale": scale}})
    assert len(rows) == 360
    return rows


def native(name, p):
    if name in ENGINEERING_GROUP_PARAMETERS:
        return engineering_group(name, **p)
    if "2" in name:
        return convert_value(
            next(iter(p.values())), TEMPERATURE_UNITS[name[0]], TEMPERATURE_UNITS[name[2]]
        )
    if name == "thermal_diffusivity":
        return thermal_diffusivity_si(p["k"], p["rho"], p["Cp"])
    if name == "nu_mu_converter":
        return convert_viscosity(
            density_kg_m3=p["rho"], dynamic_Pa_s=p.get("mu"), kinematic_m2_s=p.get("nu")
        )
    if name == "c_ideal_gas":
        return ideal_gas_sound_speed(p["T"], p["k"], p["MW"] / 1000)
    if name == "gravity":
        return local_gravity(p["latitude"], p["H"])
    return engineering_resistance_conversion(name, **p)


def evaluate(row):
    if row["family"] != "current_fluid":
        name, p = row["method"], row["parameters"]
        original = getattr(ref, name)(*p.values()) if "2" in name else getattr(ref, name)(**p)
        return [(native(name, p), original)]
    scale = row["parameters"]["scale"]
    f = FluidState(1000, 0.001 * scale, 4000, 0.6)
    result = current_fluid_similarity(
        f, velocity_m_s=2 * scale, length_m=0.01, mass_diffusivity_m2_s=1e-9
    )
    common = {
        "rho": f.density_kg_m3,
        "mu": f.viscosity_Pa_s,
        "Cp": f.heat_capacity_J_kg_K,
        "k": f.thermal_conductivity_W_m_K,
    }
    expected = {
        "Reynolds": ref.Reynolds(2 * scale, 0.01, common["rho"], common["mu"]),
        "Prandtl": ref.Prandtl(Cp=common["Cp"], mu=common["mu"], k=common["k"]),
        "Peclet_heat": ref.Peclet_heat(
            2 * scale, 0.01, rho=common["rho"], Cp=common["Cp"], k=common["k"]
        ),
        "thermal_diffusivity_m2_s": ref.thermal_diffusivity(
            common["k"], common["rho"], common["Cp"]
        ),
        "Schmidt": ref.Schmidt(1e-9, mu=common["mu"], rho=common["rho"]),
        "Lewis": ref.Lewis(1e-9, rho=common["rho"], Cp=common["Cp"], k=common["k"]),
        "Peclet_mass": ref.Peclet_mass(2 * scale, 0.01, 1e-9),
    }
    return [(result[k], v) for k, v in expected.items()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    results = []
    completed = passed = 0
    for row in inputs():
        record = {"input": row, "comparisons": []}
        try:
            for value, reference in evaluate(row):
                good = isclose(value, reference, rel_tol=3e-12, abs_tol=2e-12)
                record["comparisons"].append(
                    {"native": value, "reference": reference, "passed": good}
                )
                completed += 1
                passed += good
        except Exception as error:
            record["exception"] = f"{type(error).__name__}: {error}"
        results.append(record)
    failures = [
        r for r in results if "exception" in r or not all(c["passed"] for c in r["comparisons"])
    ]
    report = {
        "inputs": 360,
        "expected_comparisons": 384,
        "completed_comparisons": completed,
        "passed": passed,
        "failed": completed - passed,
        "exceptions": sum("exception" in r for r in results),
        "failures": failures,
        "records": results,
        "elapsed_s": time.perf_counter() - started,
        "reference_version": fluids.__version__,
        "mode": "development",
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("records", "failures")}))
    return 0 if not failures and completed == 384 else 1


if __name__ == "__main__":
    raise SystemExit(main())
